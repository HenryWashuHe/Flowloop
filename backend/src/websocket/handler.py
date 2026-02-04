import logging
import random
import uuid
import base64
import io
from typing import Any

import numpy as np
from PIL import Image
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.inference.emotion_service import get_emotion_service
from src.websocket.messages import (
    ClientMessage,
    ConnectedMessage,
    ErrorMessage,
    PredictionMessage,
    TaskMessage,
    TaskResultMessage,
    TaskAnswerData,
)
from src.tasks.task_engine import (
    create_task_generator, 
    create_difficulty_controller, 
    TaskType,
    Task
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize emotion service (uses your trained model)
emotion_service = get_emotion_service()


class SimpleEMA:
    """Simple EMA for a single float value."""

    def __init__(self, alpha: float = 0.2) -> None:
        self.alpha = alpha
        self.value: float | None = None

    def update(self, new_value: float) -> float:
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.value = None


class SessionState:
    """Track state for a session."""

    def __init__(self) -> None:
        self.attention_ema = SimpleEMA(alpha=0.2)
        self.frustration_ema = SimpleEMA(alpha=0.2)
        self.frame_count = 0
        
        # Task Engine
        self.task_generator = create_task_generator()
        self.difficulty_controller = create_difficulty_controller()
        self.current_task: Task | None = None
        self.current_attention: float = 0.5
        self.current_frustration: float = 0.3


class ConnectionManager:
    """Manage active WebSocket connections."""

    def __init__(self) -> None:
        self.active_connections: dict[str, WebSocket] = {}
        self.session_states: dict[str, SessionState] = {}

    async def connect(self, websocket: WebSocket) -> str:
        """Accept connection and return session ID."""
        await websocket.accept()
        session_id = str(uuid.uuid4())
        self.active_connections[session_id] = websocket
        self.session_states[session_id] = SessionState()
        logger.info(f"Client connected: {session_id}")
        return session_id

    def disconnect(self, session_id: str) -> None:
        """Remove connection from active connections."""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.session_states:
            del self.session_states[session_id]
        logger.info(f"Client disconnected: {session_id}")

    async def send_message(self, session_id: str, message: dict[str, Any]) -> None:
        """Send message to specific client."""
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(message)

    def get_state(self, session_id: str) -> SessionState | None:
        """Get session state."""
        return self.session_states.get(session_id)


manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Main WebSocket endpoint for real-time communication."""
    session_id = await manager.connect(websocket)

    try:
        # Send connected message
        connected_msg = ConnectedMessage(session_id=session_id)
        await manager.send_message(session_id, connected_msg.model_dump())

        while True:
            # Receive and parse message
            data = await websocket.receive_json()

            try:
                message = ClientMessage.model_validate(data)
                await handle_message(session_id, message)
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                error_msg = ErrorMessage(code="INVALID_MESSAGE", message=str(e))
                await manager.send_message(session_id, error_msg.model_dump())

    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(session_id)


async def handle_message(session_id: str, message: ClientMessage) -> None:
    """Route and handle incoming messages."""
    logger.info(f"Handling message: {message.type} for {session_id}")
    if message.type == "frame":
        await handle_frame(session_id, message.data)
    elif message.type == "session_start":
        await handle_session_start(session_id, message.data)
    elif message.type == "session_end":
        await handle_session_end(session_id, message.data)
    elif message.type == "task_answer":
        await handle_task_answer(session_id, message.data)
    else:
        logger.warning(f"Unknown message type: {message.type}")


async def handle_frame(session_id: str, data: dict[str, Any]) -> None:
    """Process incoming frame with face data."""
    state = manager.get_state(session_id)
    if not state:
        return

    state.frame_count += 1
    landmarks = data.get("landmarks", [])
    timestamp = data.get("timestamp", 0)

    # Log every 30 frames
    if state.frame_count % 30 == 0:
        logger.info(f"Session {session_id[:8]}...: processed {state.frame_count} frames, {len(landmarks)} landmarks")

    # Compute attention and frustration
    # 1. Heuristics (always compute as baseline/fallback)
    attention_heuristic, frustration_heuristic = compute_cognitive_state(landmarks)
    
    # 2. ML Inference with your trained model (if image available)
    attention_ml = None
    frustration_ml = None
    emotions = None
    
    if "image" in data and data["image"]:
        attention_ml, frustration_ml, emotions = run_emotion_inference(data["image"])

    # 3. Use ML predictions directly when available, fallback to heuristics
    if attention_ml is not None and frustration_ml is not None:
        # Your model outputs engagement and frustration directly - use them
        attention_raw = attention_ml
        frustration_raw = frustration_ml
    else:
        # Fallback to heuristics when no face crop
        attention_raw = attention_heuristic
        frustration_raw = frustration_heuristic
        emotions = compute_emotions(attention_raw, frustration_raw)

    # Apply temporal smoothing
    attention = state.attention_ema.update(attention_raw)
    frustration = state.frustration_ema.update(frustration_raw)

    # Update state for task engine
    state.current_attention = attention
    state.current_frustration = frustration

    prediction = PredictionMessage(
        attention=round(attention, 3),
        frustration=round(frustration, 3),
        emotions=emotions,
        timestamp=float(timestamp),
    )
    await manager.send_message(session_id, prediction.model_dump())


def run_emotion_inference(image_base64: str) -> tuple[float, float, dict[str, float]]:
    """Run inference using your trained emotion model."""
    if not image_base64:
        return None, None, None
    
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        face_array = np.array(image)
        
        # Run inference with your trained model
        result = emotion_service.predict(face_array)
        
        return (
            result["engagement"],
            result["frustration"],
            result["emotions"]
        )
    except Exception as e:
        logger.warning(f"Inference failed: {e}")
        return None, None, None


def compute_cognitive_state(landmarks: list[float]) -> tuple[float, float]:
    """
    Compute attention and frustration from facial landmarks.

    Uses simplified heuristics based on:
    - Eye Aspect Ratio (EAR) for attention
    - Brow position for frustration

    Note: This is mock inference. Real implementation would use the ONNX model.
    """
    if not landmarks or len(landmarks) < 468 * 3:
        # No face detected - neutral state with some noise
        return 0.5 + random.uniform(-0.05, 0.05), 0.3 + random.uniform(-0.05, 0.05)

    try:
        # Eye Aspect Ratio calculation
        left_eye_outer = get_point(landmarks, 33)
        left_eye_inner = get_point(landmarks, 133)
        left_eye_top = get_point(landmarks, 159)
        left_eye_bottom = get_point(landmarks, 145)

        right_eye_outer = get_point(landmarks, 362)
        right_eye_inner = get_point(landmarks, 263)
        right_eye_top = get_point(landmarks, 386)
        right_eye_bottom = get_point(landmarks, 380)

        # Calculate EAR for both eyes
        left_ear = calculate_ear(left_eye_outer, left_eye_inner, left_eye_top, left_eye_bottom)
        right_ear = calculate_ear(right_eye_outer, right_eye_inner, right_eye_top, right_eye_bottom)
        avg_ear = (left_ear + right_ear) / 2

        # Map EAR to attention (higher EAR = more attentive)
        # Typical EAR range: 0.15 (closed) to 0.35 (open)
        attention = min(1.0, max(0.0, (avg_ear - 0.15) / 0.2))

        # Brow furrow for frustration
        left_brow = get_point(landmarks, 107)
        right_brow = get_point(landmarks, 336)

        # Distance between brows (normalized by face width)
        brow_dist = distance_2d(left_brow, right_brow)
        face_width = distance_2d(get_point(landmarks, 33), get_point(landmarks, 263))

        if face_width > 0:
            brow_ratio = brow_dist / face_width
            # Lower ratio = more furrowed = more frustrated
            frustration = min(1.0, max(0.0, 1.0 - (brow_ratio - 0.1) / 0.15))
        else:
            frustration = 0.3

        # Add small amount of noise for more natural variation
        attention = min(1.0, max(0.0, attention + random.uniform(-0.02, 0.02)))
        frustration = min(1.0, max(0.0, frustration + random.uniform(-0.02, 0.02)))

        return attention, frustration

    except Exception as e:
        logger.warning(f"Error computing cognitive state: {e}")
        return 0.5 + random.uniform(-0.05, 0.05), 0.3 + random.uniform(-0.05, 0.05)


def get_point(landmarks: list[float], idx: int) -> tuple[float, float]:
    """Get 2D point from flattened landmarks array."""
    base = idx * 3
    return (landmarks[base], landmarks[base + 1])


def distance_2d(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Calculate 2D Euclidean distance."""
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def calculate_ear(
    outer: tuple[float, float],
    inner: tuple[float, float],
    top: tuple[float, float],
    bottom: tuple[float, float],
) -> float:
    """Calculate Eye Aspect Ratio."""
    vertical = distance_2d(top, bottom)
    horizontal = distance_2d(outer, inner)
    if horizontal == 0:
        return 0.3
    return vertical / horizontal


def compute_emotions(attention: float, frustration: float) -> dict[str, float]:
    """
    Compute emotion distribution based on attention and frustration.

    This is a simplified heuristic. Real implementation would use the emotion model.
    """
    # Base emotions
    emotions = {
        "neutral": 0.0,
        "happy": 0.0,
        "sad": 0.0,
        "angry": 0.0,
        "fear": 0.0,
        "disgust": 0.0,
        "surprise": 0.0,
    }

    # High attention + low frustration = likely neutral or happy
    if attention > 0.6 and frustration < 0.3:
        emotions["neutral"] = 0.5
        emotions["happy"] = 0.3
        emotions["surprise"] = 0.1
    # High frustration = likely angry or disgust
    elif frustration > 0.6:
        emotions["angry"] = 0.4
        emotions["disgust"] = 0.2
        emotions["neutral"] = 0.2
        emotions["sad"] = 0.1
    # Low attention = likely bored/sad or neutral
    elif attention < 0.4:
        emotions["neutral"] = 0.4
        emotions["sad"] = 0.3
        emotions["fear"] = 0.1
    # Default: mostly neutral
    else:
        emotions["neutral"] = 0.6
        emotions["happy"] = 0.15
        emotions["sad"] = 0.1

    # Fill remaining with small values
    total = sum(emotions.values())
    if total < 1.0:
        remainder = 1.0 - total
        for emotion in emotions:
            if emotions[emotion] == 0:
                emotions[emotion] = remainder / 3
                remainder -= emotions[emotion]
                if remainder <= 0:
                    break

    # Normalize to sum to 1
    total = sum(emotions.values())
    if total > 0:
        emotions = {k: round(v / total, 3) for k, v in emotions.items()}

    return emotions


async def handle_session_start(session_id: str, data: dict[str, Any]) -> None:
    """Handle session start event."""
    logger.info(f"Session started: {session_id}, config: {data}")
    state = manager.get_state(session_id)
    if state:
        # Reset state
        state.attention_ema.reset()
        state.frustration_ema.reset()
        state.frame_count = 0
        state.difficulty_controller.reset()
        
        # Start first task
        await send_next_task(session_id, state)


async def handle_task_answer(session_id: str, data: dict[str, Any]) -> None:
    """Handle user's answer to a task."""
    state = manager.get_state(session_id)
    if not state or not state.current_task:
        return

    try:
        answer_data = TaskAnswerData.model_validate(data)
    except Exception as e:
        logger.error(f"Invalid task answer: {e}")
        return

    # Check answer
    task = state.current_task
    # Basic checking - for algebra we might need more complex parsing
    # For now assume answer is numeric or exact string match
    is_correct = str(answer_data.answer).strip() == str(task.answer).strip()
    
    # Update difficulty controller
    new_difficulty = state.difficulty_controller.record_result(
        correct=is_correct,
        response_time=answer_data.time_taken,
        attention_score=state.current_attention,
        frustration_score=state.current_frustration
    )
    
    # Send result
    result_msg = TaskResultMessage(
        task_id=task.id,
        correct=is_correct,
        user_answer=answer_data.answer,
        correct_answer=str(task.answer),
        time_taken=answer_data.time_taken,
        new_difficulty=new_difficulty
    )
    await manager.send_message(session_id, result_msg.model_dump())
    
    # Generate next task
    await send_next_task(session_id, state)


async def send_next_task(session_id: str, state: SessionState) -> None:
    """Generate and send the next task."""
    difficulty = state.difficulty_controller.get_difficulty()
    
    # Randomly choose task type mostly arithmetic for now
    task_type = TaskType.ARITHMETIC
    if random.random() > 0.7:
        task_type = TaskType.ALGEBRA
        
    task = state.task_generator.generate(task_type, difficulty)
    state.current_task = task
    
    msg = TaskMessage(
        task_id=task.id,
        task_type=task.task_type.value,
        question=task.question,
        difficulty=task.difficulty,
        time_limit=task.time_limit_seconds,
        hints=task.hints
    )
    await manager.send_message(session_id, msg.model_dump())


async def handle_session_end(session_id: str, data: dict[str, Any]) -> None:
    """Handle session end event."""
    state = manager.get_state(session_id)
    if state:
        logger.info(f"Session ended: {session_id}, total frames: {state.frame_count}")
    else:
        logger.info(f"Session ended: {session_id}")
