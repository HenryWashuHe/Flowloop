# Project Handoff

## Current Status
The project has been successfully refactored to include a Fusion-based Frustration detection model and a comprehensive Frontend Dashboard.

### Core Features Working
1.  **ML Inference**:
    *   **Emotion Detection**: Uses HSEmotion ONNX model (`enet_b2_8` - ~70% accuracy).
    *   **Face Crop Extraction**: Frontend extracts 224x224 face crops from video using landmark bounding box.
    *   **Frustration Score**: Fuses ML predictions (60%) with Brow-Furrow Heuristics (40%) via Facial Landmarks.
    *   **Attention Score**: Fuses ML engagement (70%) with Eye Aspect Ratio heuristics (30%).
    *   **Sensitivity**: Adjusted to be more responsive to negative emotion spikes.

2.  **Frontend Dashboard**:
    *   **Cognitive State**: Real-time gauges for Attention and Frustration.
    *   **Emotion Analysis**: Live breakdown of all 7 detected emotions.
    *   **Task Zone**: Interactive Math tasks with immediate feedback.
    *   **Session Log**: Tracks usage and task completion.

### Known Issues
*   **Port Conflicts**: The development server ports often hang if not killed properly. Use the provided kill commands if needed.
*   **Startup Sequence**: ML server must be started before the Backend to ensure the connection allows for inference.

## Manual Startup Commands
The system consists of 3 services that must be run in separate terminals in this specific order:

**1. ML Inference Server (Port 8001)**
```bash
cd ml
source .venv/bin/activate
python inference_server.py
```

**2. Backend API (Port 8000)**
```bash
cd backend
source .venv/bin/activate
python -m src.main
```

**3. Frontend (Port 5173)**
```bash
cd frontend
pnpm dev
```

## Next Steps
*   **Refine Task Engine**: Add non-math tasks or adaptive difficulty logic beyond the placeholder.
*   **Session Persistence**: Save session logs to a database.
*   **User Settings**: Implement the settings page to configure camera selection and difficulty presets.
*   **Emotion Calibration**: Add per-user calibration to improve accuracy for individual differences.
