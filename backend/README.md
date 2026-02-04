# FlowLoop Backend

FastAPI backend for FlowLoop - handles WebSocket communication, ONNX inference, and session management.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

```bash
python -m src.main
```

## API Endpoints

- `GET /` - Health check
- `GET /health` - Detailed health status
- `WS /ws` - WebSocket for real-time communication
