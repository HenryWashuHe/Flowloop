# FlowLoop

**Adaptive Cognitive Engine** - Real-time attention and emotion detection for adaptive learning.

FlowLoop uses webcam-based computer vision to estimate your attention level and emotional state, then dynamically adjusts task difficulty to maintain optimal cognitive engagement.

## Features

- **Real-time Detection**: 15-30 FPS attention and frustration tracking
- **Hybrid Architecture**: MediaPipe landmarks + EfficientNet emotion model
- **Adaptive Difficulty**: Tasks get harder when focused, easier when frustrated
- **Privacy-First**: All processing happens locally - no video leaves your device
- **Research-Ready**: Comprehensive logging and data export

## Architecture

```
┌─────────────┐
│  Webcam     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────┐
│ Face Detection/Tracking │ ← MediaPipe Face Mesh
└──────┬──────────────────┘
       │
       ├──────────────────┬─────────────────┐
       ▼                  ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌─────────────┐
│ Landmarks    │  │ Emotion Net  │  │ Gaze/Pose   │
│ (MediaPipe)  │  │ (EfficientNet)│  │ Estimation  │
└──────┬───────┘  └──────┬───────┘  └──────┬──────┘
       │                  │                  │
       └──────────┬───────┴──────────────────┘
                  ▼
        ┌─────────────────────┐
        │ Feature Fusion      │
        │ (Attention +        │
        │  Frustration Score) │
        └──────┬──────────────┘
               ▼
        ┌─────────────────────┐
        │ Adaptive Controller │
        │ (Rule-based)        │
        └──────┬──────────────┘
               ▼
        ┌─────────────────────┐
        │ Task Generator      │
        │ (Difficulty Param)  │
        └─────────────────────┘
```

## Quick Start

### Prerequisites

- Node.js 18+
- Python 3.10+
- pnpm 8+

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/flowloop.git
cd flowloop

# Install dependencies
pnpm install

# Setup Python environment for backend
cd backend
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
pip install -e ".[dev]"
cd ..

# Setup Python environment for ML
cd ml
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cd ..
```

### Running the Application

```bash
# Terminal 1: Start backend
cd backend
source .venv/bin/activate
python -m src.main

# Terminal 2: Start frontend
pnpm dev
```

Open http://localhost:5173 in your browser.

## Project Structure

```
flowloop/
├── frontend/           # React + TypeScript + Vite
│   ├── src/
│   │   ├── components/ # React components
│   │   ├── lib/        # Utilities, MediaPipe integration
│   │   ├── pages/      # Route pages
│   │   ├── stores/     # Zustand state stores
│   │   └── types/      # TypeScript types
│   └── ...
│
├── backend/            # Python + FastAPI
│   └── src/
│       ├── inference/  # ONNX model inference
│       ├── websocket/  # Real-time communication
│       ├── smoothing/  # Temporal smoothing (EMA)
│       └── db/         # Database models
│
├── ml/                 # PyTorch model training
│   ├── models/         # EmotionNet, FusionNet
│   ├── training/       # Training scripts
│   ├── data/           # Dataset preparation
│   └── export/         # ONNX export
│
├── shared/             # Shared TypeScript types
└── docs/               # Documentation
```

## ML Pipeline (Phase 2)

### 1. Prepare Dataset

```bash
cd ml
source .venv/bin/activate

# Download and prepare FER2013 (requires Kaggle API)
python data/prepare_fer2013.py --output_dir ./data/processed/fer2013
```

### 2. Train EmotionNet

```bash
python training/train_emotion.py \
  --data_dir ./data/processed/fer2013 \
  --epochs 50 \
  --batch_size 32
```

### 3. Export to ONNX

```bash
python export/export_onnx.py \
  --model emotion \
  --checkpoint ./outputs/best_model.pt \
  --output ../backend/models \
  --quantize
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
# Backend
BACKEND_HOST=127.0.0.1
BACKEND_PORT=8000

# Frontend
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/ws

# ML (for training)
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

## Tech Stack

### Frontend
- React 18 + TypeScript
- Vite (build tool)
- TailwindCSS (styling)
- Zustand (state management)
- Chart.js (visualization)
- MediaPipe Face Mesh (landmarks)

### Backend
- FastAPI (web framework)
- ONNX Runtime (inference)
- WebSockets (real-time)
- SQLite/PostgreSQL (storage)

### ML
- PyTorch (training)
- EfficientNet-B0 (backbone)
- ONNX (deployment)

## Datasets

- **FER2013**: 35,887 images, 7 emotions (baseline training)
- **AffectNet**: 450K+ images with valence/arousal (optional, better quality)
- **DAiSEE**: 9K videos with engagement labels (engagement-specific training)

## Performance Targets

| Metric | Target |
|--------|--------|
| End-to-end latency | <100ms |
| Frame rate | 15-30 FPS |
| Emotion accuracy | >65% |
| Model size | <50MB |

## Privacy

- All video processing happens locally in the browser
- No raw video is ever transmitted to servers
- Only extracted features are sent via WebSocket
- Session data stored locally with optional cloud backup

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for face mesh detection
- [EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch) for the backbone architecture
- [FER2013](https://www.kaggle.com/datasets/deadskull7/fer2013) dataset for emotion training
