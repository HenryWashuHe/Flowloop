from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Server
    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1
    debug: bool = False

    # CORS
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    # Database
    database_url: str = "sqlite+aiosqlite:///./flowloop.db"

    # Model paths - points to ml/models where trained models are stored
    models_dir: Path = Path(__file__).parent.parent.parent / "ml" / "models"
    emotion_model_path: str = "emotion_net.onnx"
    fusion_model_path: str = "fusion_net.onnx"

    # Inference
    use_gpu: bool = False
    inference_batch_size: int = 1

    # Smoothing
    default_ema_alpha: float = 0.2

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        env_prefix = ""

    @property
    def emotion_model_full_path(self) -> Path:
        return self.models_dir / self.emotion_model_path

    @property
    def fusion_model_full_path(self) -> Path:
        return self.models_dir / self.fusion_model_path


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
