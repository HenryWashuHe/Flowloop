# FlowLoop Handoff Document

## Quick Start (3 Terminals Required)

```bash
# Terminal 1 — ML Inference Server (Python 3.11, HSEmotion model)
cd ml && source .venv/bin/activate && python inference_server.py

# Terminal 2 — Backend API (Python 3.14, FastAPI)
cd backend && source .venv/bin/activate && python -m src.main

# Terminal 3 — Frontend (React + Vite)
cd frontend && pnpm dev
```

Open http://localhost:5173

**Note**: ML server must be running for real emotion predictions. Without it, backend uses mock data.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ WebcamCapture│→ │ FaceMesh     │→ │ FaceCropExtractor    │  │
│  │              │  │ (MediaPipe)  │  │ (224x224 crop)       │  │
│  └──────────────┘  └──────────────┘  └──────────┬───────────┘  │
│                                                  │              │
│                    WebSocket (landmarks + base64 image)        │
└──────────────────────────────────┬──────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Backend (FastAPI)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ WS Handler   │→ │ EmotionSvc   │→ │ Task Engine          │  │
│  │              │  │ (ONNX)       │  │ (Adaptive Difficulty)│  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Components

### Frontend (`/frontend`)

| File | Purpose |
|------|---------|
| `src/pages/SessionPage.tsx` | Main training interface |
| `src/pages/DashboardPage.tsx` | Performance charts + history |
| `src/pages/LoginPage.tsx` | User authentication |
| `src/pages/SignupPage.tsx` | User registration |
| `src/components/WebcamCapture/` | Camera capture + MediaPipe |
| `src/lib/features/faceCropExtractor.ts` | Extract face from video frame |
| `src/lib/websocket/WebSocketClient.ts` | Backend communication |
| `src/lib/supabase/` | Supabase client, auth, sessions |
| `src/index.css` | Design system (Tailwind) |
| `tailwind.config.js` | Color palette + typography |

### Backend (`/backend`)

| File | Purpose |
|------|---------|
| `src/main.py` | FastAPI app entry |
| `src/websocket/handler.py` | Frame processing + inference |
| `src/inference/emotion_service.py` | ONNX model wrapper |
| `src/tasks/task_engine.py` | Task generation + difficulty |
| `src/config.py` | Settings + model paths |

### ML (`/ml`)

| File | Purpose |
|------|---------|
| `models/emotion_net.onnx` | Trained emotion model |
| `models/emotion_net.py` | Model architecture (EfficientNet-B0) |
| `training/train_emotion.py` | Training script |
| `export/export_onnx.py` | ONNX conversion |

---

## Data Flow

1. **Frontend** captures video frame via webcam
2. **MediaPipe** extracts 468 face landmarks
3. **FaceCropExtractor** creates 224×224 padded face crop
4. **WebSocket** sends landmarks + base64 image to backend
5. **EmotionService** runs ONNX inference → emotions, frustration, engagement
6. **EMA Smoothing** stabilizes predictions over time
7. **Task Engine** adjusts difficulty based on cognitive state
8. **WebSocket** sends predictions + tasks back to frontend

---

## Design System

| Token | Value | Usage |
|-------|-------|-------|
| `canvas` | #f6ffff | Page background |
| `ink` | #1d2223 | Primary text |
| `accent` | #e43e33 | Alerts, critical states |
| `neutral-*` | gray scale | Secondary text, borders |
| `state-good` | #059669 | Positive feedback |

**Typography**: Inter (400, 500, 600 weights)

**Principles**:
- Calm, academic, research-tool aesthetic
- Generous whitespace, no clutter
- No animations unless meaningful
- Flat cards, subtle borders

---

## Environment Setup

### Backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Frontend
```bash
cd frontend
pnpm install
```

### ML (for training only)
```bash
cd ml
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Common Issues

| Issue | Solution |
|-------|----------|
| Port already in use | `lsof -ti:8000 \| xargs kill -9` |
| Model not found | Check `backend/src/config.py` → `models_dir` points to `ml/models` |
| WebSocket disconnects | Backend may have crashed — check terminal |
| Face not detected | Ensure good lighting, face centered |
| Emotions not updating | Verify face crop is being sent (check Network tab) |
| Predictions seem random | Check backend logs for `[INFERENCE PATH]` — if MOCK, model isn't loaded |
| Neutral always dominant | Expected with FER2013 dataset bias; engagement formula compensates |

---

## Inference Path Debugging

The backend logs which inference path is being used:

```
[INFERENCE] Using HSEmotion model - dominant: neutral  # Good - real model
[INFERENCE] MOCK predictions - start ML server...      # Bad - ML server not running
```

**To verify real model is loaded:**
1. Ensure ML server is running on port 8001: `curl http://localhost:8001/health`
2. Look for `[INFERENCE] Using HSEmotion model` in backend logs
3. Check ML server terminal for `POST /predict` requests

---

## Engagement & Frustration Formulas

### Engagement (Russell's Circumplex Model)
```python
arousal = surprise*1.0 + fear*0.9 + angry*0.85 + happy*0.7 + neutral*0.5 + ...
valence_boost = happy*0.3 + surprise*0.1
valence_penalty = sad*0.3 + disgust*0.15
engagement = arousal + valence_boost - valence_penalty
```

### Frustration
```python
frustration = angry*1.0 + disgust*0.7 + fear*0.5 + sad*0.4
```

### Emotion Mapping (HSEmotion 8 → 7 classes)
- Contempt → neutral (was causing inflated disgust)

---

## Supabase Setup

### 1. Configure Environment
Add your Supabase anon key to `frontend/.env`:
```bash
VITE_SUPABASE_URL=https://jafbsrgbawbibumuzojk.supabase.co
VITE_SUPABASE_ANON_KEY=your_anon_key_here  # Get from Supabase dashboard
```

### 2. Run Database Migration
Go to Supabase Dashboard → SQL Editor and run:
```
supabase/migrations/001_initial_schema.sql
```

This creates:
- `profiles` - User profile data (linked to auth.users)
- `sessions` - Training session metadata and aggregates
- `session_events` - Time-series events (emotions, tasks, difficulty changes)
- `task_results` - Individual task attempts with performance data

### 3. Enable Auth
In Supabase Dashboard → Authentication:
- Enable Email provider
- (Optional) Configure email templates

---

## Next Steps

### High Priority
- [x] ~~Implement session persistence~~ (Done - Supabase)
- [x] ~~Add chart visualizations to Dashboard~~ (Done - Recharts)
- [ ] Refine adaptive difficulty algorithm based on engagement/frustration
- [ ] Integrate session persistence into SessionPage

### Medium Priority
- [ ] Multiple task types (memory games, pattern recognition)
- [ ] User calibration for emotion baseline
- [ ] Export session data (CSV/JSON)
- [ ] Train custom EmotionNet on more diverse dataset

### Low Priority
- [ ] Dark mode toggle
- [ ] Mobile responsive layout
- [ ] Keyboard shortcuts
- [ ] WebSocket reconnection logic

---

## Session Notes (Feb 4, 2026)

**What was done:**
- Integrated HSEmotion pretrained model via separate ML server
- Fixed emotion mapping (Contempt→neutral instead of disgust)
- Improved engagement formula using arousal/valence model
- Added emotion distribution UI with colored bars
- Enhanced UI with gradients, glass morphism, micro-interactions

**What's working well:**
- Real-time emotion detection at ~15 FPS
- HSEmotion model provides reasonable accuracy (~70%)
- Dual-server architecture handles Python version constraints

**Known quirks:**
- Neutral tends to dominate (expected with most facial expressions)
- ML server must be started separately (port 8001)

---

## Repository

https://github.com/HenryWashuHe/Flowloop
