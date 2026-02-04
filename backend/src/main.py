import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_settings
from src.websocket.handler import router as websocket_router

# Configure logging
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup/shutdown events."""
    # Startup
    logger.info("Starting FlowLoop backend...")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"GPU enabled: {settings.use_gpu}")

    # Initialize services here (models, database, etc.)
    # TODO: Load ONNX models
    # TODO: Initialize database connection

    yield

    # Shutdown
    logger.info("Shutting down FlowLoop backend...")
    # Cleanup resources here


app = FastAPI(
    title="FlowLoop API",
    description="Backend API for FlowLoop adaptive cognitive engine",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(websocket_router)


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint for health check."""
    return {"status": "ok", "service": "flowloop-backend"}


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "0.1.0",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=settings.workers,
    )
