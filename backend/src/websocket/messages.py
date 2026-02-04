from typing import Any, Literal

from pydantic import BaseModel, Field


# =============================================================================
# Client -> Server Messages
# =============================================================================


class FrameData(BaseModel):
    """Data for a single frame."""

    image: str = Field(..., description="Base64 encoded face crop image")
    landmarks: list[float] = Field(..., description="Flattened 468*3 face landmarks")
    timestamp: float = Field(..., description="Unix timestamp in milliseconds")


class SessionStartData(BaseModel):
    """Data for session start."""

    user_id: str | None = Field(None, description="Optional user identifier")
    task_type: str = Field("math", description="Type of cognitive task")
    is_adaptive_enabled: bool = Field(True, description="Whether adaptation is enabled")


class SessionEndData(BaseModel):
    """Data for session end."""

    session_id: str = Field(..., description="Session identifier")


class TaskAnswerData(BaseModel):
    """User answer to a task."""

    task_id: str = Field(..., description="Task identifier")
    answer: str = Field(..., description="User's answer")
    time_taken: float = Field(..., description="Time taken in seconds")


class ClientMessage(BaseModel):
    """Union of all client message types."""

    type: Literal["frame", "session_start", "session_end", "task_answer"]
    data: dict[str, Any]


# =============================================================================
# Server -> Client Messages
# =============================================================================


class PredictionMessage(BaseModel):
    """Prediction results from inference pipeline."""

    type: Literal["prediction"] = "prediction"
    attention: float = Field(..., ge=0, le=1, description="Attention score 0-1")
    frustration: float = Field(..., ge=0, le=1, description="Frustration score 0-1")
    emotions: dict[str, float] = Field(..., description="Emotion scores by class")
    timestamp: float = Field(..., description="Original frame timestamp")

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Serialize with nested data structure for WebSocket protocol."""
        return {
            "type": self.type,
            "data": {
                "attention": self.attention,
                "frustration": self.frustration,
                "emotions": self.emotions,
                "timestamp": self.timestamp,
            },
        }


class DifficultyMessage(BaseModel):
    """Difficulty adjustment notification."""

    type: Literal["difficulty"] = "difficulty"
    level: int = Field(..., ge=1, le=10, description="Difficulty level 1-10")
    reason: str = Field(..., description="Reason for adjustment")

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "type": self.type,
            "data": {
                "level": self.level,
                "reason": self.reason,
            },
        }


class ErrorMessage(BaseModel):
    """Error notification."""

    type: Literal["error"] = "error"
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "type": self.type,
            "data": {
                "code": self.code,
                "message": self.message,
            },
        }


class ConnectedMessage(BaseModel):
    """Connection confirmation."""

    type: Literal["connected"] = "connected"
    session_id: str = Field(..., description="Assigned session identifier")

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "type": self.type,
            "data": {
                "sessionId": self.session_id,
            },
        }


class TaskMessage(BaseModel):
    """New task for the user to solve."""

    type: Literal["task"] = "task"
    task_id: str = Field(..., description="Unique task identifier")
    task_type: str = Field(..., description="Type of task (arithmetic, algebra)")
    question: str = Field(..., description="The question to display")
    difficulty: int = Field(..., ge=1, le=10, description="Difficulty level")
    time_limit: float = Field(..., description="Time limit in seconds")
    hints: list[str] = Field(default_factory=list, description="Optional hints")

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "type": self.type,
            "data": {
                "taskId": self.task_id,
                "taskType": self.task_type,
                "question": self.question,
                "difficulty": self.difficulty,
                "timeLimit": self.time_limit,
                "hints": self.hints,
            },
        }


class TaskResultMessage(BaseModel):
    """Result of a completed task."""

    type: Literal["task_result"] = "task_result"
    task_id: str = Field(..., description="Task identifier")
    correct: bool = Field(..., description="Whether answer was correct")
    user_answer: str = Field(..., description="What the user answered")
    correct_answer: str = Field(..., description="The correct answer")
    time_taken: float = Field(..., description="Time taken in seconds")
    new_difficulty: int = Field(..., ge=1, le=10, description="New difficulty level")

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "type": self.type,
            "data": {
                "taskId": self.task_id,
                "correct": self.correct,
                "userAnswer": self.user_answer,
                "correctAnswer": self.correct_answer,
                "timeTaken": self.time_taken,
                "newDifficulty": self.new_difficulty,
            },
        }
