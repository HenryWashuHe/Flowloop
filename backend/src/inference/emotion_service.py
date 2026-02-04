"""
Emotion Inference Service using ONNX Runtime.

Provides real-time emotion detection from face crops using
the exported EmotionNet ONNX model.

Features:
- Lazy model loading
- GPU acceleration (optional)
- Batch inference support
- Performance monitoring
- Graceful fallback when ONNX not available
"""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.config import get_settings

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ort = None  # type: ignore
    ONNX_AVAILABLE = False
    logger.warning("onnxruntime not available - using mock inference")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore
    CV2_AVAILABLE = False
    logger.warning("opencv-python not available - preprocessing disabled")

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Emotion class labels (FER2013 order)
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


class EmotionInferenceService:
    """
    ONNX Runtime inference service for emotion detection.

    Singleton pattern to ensure single model instance.
    Falls back to mock predictions when ONNX is not available.
    """

    _instance: "EmotionInferenceService | None" = None
    _initialized: bool = False

    def __new__(cls) -> "EmotionInferenceService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self.settings = get_settings()
        self.session: Any = None
        self.input_name: str = ""
        self.output_names: list[str] = []
        self._inference_times: list[float] = []
        self._use_mock = not ONNX_AVAILABLE
        self._initialized = True

        if self._use_mock:
            logger.info("Using mock inference (ONNX not available)")

    def load_model(self) -> None:
        """Load ONNX model lazily on first inference."""
        if self._use_mock or self.session is not None:
            return

        model_path = self.settings.emotion_model_full_path

        if not model_path.exists():
            logger.warning(f"Model not found at {model_path}, using mock inference")
            self._use_mock = True
            return

        logger.info(f"Loading emotion model from {model_path}")

        # Configure providers
        providers = ["CPUExecutionProvider"]
        if self.settings.use_gpu:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        try:
            self.session = ort.InferenceSession(
                str(model_path),
                providers=providers,
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            logger.info(f"Model loaded successfully. Outputs: {self.output_names}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}, falling back to mock")
            self._use_mock = True

    def preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for inference.

        Args:
            face_image: BGR image from OpenCV (H, W, 3) or RGB

        Returns:
            Preprocessed tensor (1, 3, 224, 224)
        """
        if not CV2_AVAILABLE:
            # Return dummy tensor if cv2 not available
            return np.zeros((1, 3, 224, 224), dtype=np.float32)

        # Resize to 224x224
        if face_image.shape[:2] != (224, 224):
            face_image = cv2.resize(face_image, (224, 224), interpolation=cv2.INTER_LINEAR)

        # Convert BGR to RGB if needed
        if len(face_image.shape) == 3 and face_image.shape[2] == 3:
            # Assume BGR from OpenCV
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        image = face_image.astype(np.float32) / 255.0

        # Apply ImageNet normalization
        image = (image - IMAGENET_MEAN) / IMAGENET_STD

        # Convert to NCHW format
        image = np.transpose(image, (2, 0, 1))

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        return image.astype(np.float32)

    def predict(self, face_image: np.ndarray | None = None) -> dict[str, Any]:
        """
        Run inference on single face image.

        Args:
            face_image: Face crop image (BGR or RGB, any size), or None for mock

        Returns:
            Dictionary with:
                - emotions: dict of emotion -> probability
                - dominant_emotion: most likely emotion
                - frustration: frustration score (0-1)
                - engagement: engagement score (0-1)
                - features: backbone features (1280-dim)
                - inference_time_ms: inference duration
        """
        # Ensure model is loaded
        self.load_model()

        # Return mock prediction if using mock mode
        if self._use_mock or self.session is None:
            return self._mock_prediction()

        # Preprocess
        input_tensor = self.preprocess(face_image)

        # Run inference
        start_time = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        inference_time = (time.perf_counter() - start_time) * 1000

        # Track performance
        self._inference_times.append(inference_time)
        if len(self._inference_times) > 100:
            self._inference_times.pop(0)

        # Parse outputs
        emotion_logits = outputs[0][0]  # (7,)
        frustration = float(outputs[1][0][0])  # scalar
        engagement = float(outputs[2][0][0])  # scalar
        features = outputs[3][0]  # (1280,)

        # Softmax for emotion probabilities
        emotion_probs = self._softmax(emotion_logits)

        # Create emotion dict
        emotions = {label: float(prob) for label, prob in zip(EMOTION_LABELS, emotion_probs)}

        # Get dominant emotion
        dominant_idx = int(np.argmax(emotion_probs))
        dominant_emotion = EMOTION_LABELS[dominant_idx]

        return {
            "emotions": emotions,
            "dominant_emotion": dominant_emotion,
            "frustration": frustration,
            "engagement": engagement,
            "features": features,
            "inference_time_ms": inference_time,
        }

    def predict_batch(self, face_images: list[np.ndarray]) -> list[dict[str, Any]]:
        """
        Run batch inference on multiple face images.

        Args:
            face_images: List of face crop images

        Returns:
            List of prediction dictionaries
        """
        if not face_images:
            return []

        # For now, process sequentially
        return [self.predict(img) for img in face_images]

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax values."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def _mock_prediction(self) -> dict[str, Any]:
        """Return realistic mock prediction for development."""
        # Generate slightly varied mock data for realism
        base_probs = np.array([0.05, 0.02, 0.03, 0.15, 0.05, 0.10, 0.60])
        noise = np.random.normal(0, 0.02, 7)
        probs = np.clip(base_probs + noise, 0, 1)
        probs = probs / probs.sum()

        emotions = {label: float(prob) for label, prob in zip(EMOTION_LABELS, probs)}
        dominant_idx = int(np.argmax(probs))

        return {
            "emotions": emotions,
            "dominant_emotion": EMOTION_LABELS[dominant_idx],
            "frustration": float(np.clip(0.2 + np.random.normal(0, 0.1), 0, 1)),
            "engagement": float(np.clip(0.7 + np.random.normal(0, 0.1), 0, 1)),
            "features": np.random.randn(1280).astype(np.float32),
            "inference_time_ms": 0.5,
            "mock": True,
        }

    @property
    def average_inference_time(self) -> float:
        """Get average inference time in milliseconds."""
        if not self._inference_times:
            return 0.0
        return sum(self._inference_times) / len(self._inference_times)

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.session is not None

    @property
    def is_mock(self) -> bool:
        """Check if using mock inference."""
        return self._use_mock


def get_emotion_service() -> EmotionInferenceService:
    """Get singleton emotion service instance."""
    return EmotionInferenceService()
