# FlowLoop Progress

## Project Overview
FlowLoop is an adaptive cognitive training platform that uses real-time emotion and attention detection to dynamically adjust task difficulty. All processing happens locally for privacy.

**Last Updated**: Feb 4, 2026

---

## Recent Session (Feb 4, 2026)

### ML Inference Improvements
- ✅ HSEmotion pretrained model integration (`enet_b2_8`, ~70% accuracy)
- ✅ Dual-server architecture: Backend (Python 3.14) + ML Server (Python 3.11)
- ✅ Fixed Contempt→Disgust mapping issue (was inflating disgust artificially)
- ✅ Engagement calculation now uses Russell's Circumplex Model (arousal + valence)
- ✅ Frustration formula uses direct emotion probabilities, no artificial caps
- ✅ EMA smoothing tuned to α=0.15 for balance between responsive and stable

### UI/UX Improvements
- ✅ Emotion distribution visualization (all 7 emotions with colored bars)
- ✅ Gradient gauges for engagement/frustration
- ✅ Status indicator with pulse animation
- ✅ Difficulty badge in header
- ✅ Cards with glass morphism effect (backdrop-blur)
- ✅ Improved button micro-interactions (scale on press)
- ✅ Better focus states on inputs

### Bug Fixes
- ✅ Removed erroneous BGR→RGB conversion (frontend already sends RGB)
- ✅ Added `requests` dependency to backend
- ✅ Fixed cv2 import error in external prediction path

---

## Completed Features

### Core Infrastructure
- [x] Monorepo setup with pnpm workspaces
- [x] Frontend: React 18 + Vite + TypeScript + TailwindCSS
- [x] Backend: FastAPI + WebSocket + Python 3.14
- [x] ML Pipeline: Custom trained EmotionNet (EfficientNet-B0 backbone)
- [x] Real-time WebSocket communication (<100ms latency target)

### ML & Computer Vision
- [x] MediaPipe Face Mesh integration (468 landmarks)
- [x] Custom EmotionNet model trained on FER2013
  - 7-class emotion classification (angry, disgust, fear, happy, sad, surprise, neutral)
  - Direct frustration regression head (0-1)
  - Direct engagement regression head (0-1)
- [x] Face crop extraction from video frames (224x224, padded)
- [x] ONNX export for production inference
- [x] Temporal smoothing with EMA (α=0.2)

### Frontend
- [x] Webcam capture with permissions handling
- [x] Real-time face landmark visualization
- [x] Cognitive state gauges (Engagement, Frustration)
- [x] Emotion breakdown display (top 4 emotions)
- [x] Task interface with math problems
- [x] Performance tracking (accuracy, completed tasks)
- [x] Session activity log
- [x] **UI Redesign**: Calm, academic aesthetic
  - Background: #f6ffff (light cyan tint)
  - Text: #1d2223 (near-black)
  - Accent: #e43e33 (red, used sparingly)
  - Inter font, generous whitespace, flat cards

### Backend
- [x] WebSocket session management
- [x] Frame processing pipeline
- [x] Emotion inference service (ONNX Runtime)
- [x] Heuristic fallback (Eye Aspect Ratio, Brow Furrow)
- [x] Task generation engine
- [x] Adaptive difficulty controller

### Pages
- [x] HomePage — Product introduction
- [x] SessionPage — Main training interface
- [x] DashboardPage — Performance overview (placeholder charts)
- [x] SettingsPage — Configuration options

---

## In Progress
- [ ] Adaptive difficulty logic refinement
- [x] Session persistence to database (Supabase)

---

## Recently Completed (Feb 4, 2026 - Session 2)

### Supabase Integration
- [x] Supabase client setup with TypeScript types
- [x] Database schema designed (profiles, sessions, session_events, task_results)
- [x] Row Level Security (RLS) policies for data isolation
- [x] Auth context with sign up/in/out functionality
- [x] Session persistence hook (`useSessionPersistence`)
- [x] Login and Signup pages

### Dashboard with Charts
- [x] Recharts integration for visualizations
- [x] Engagement over time (area chart)
- [x] Accuracy & Frustration trends (line chart)
- [x] Tasks per session (bar chart)
- [x] Overall metrics progress bars
- [x] Session history table with delete functionality

### New Pages
- [x] `/login` - Email/password authentication
- [x] `/signup` - New user registration
- [x] Protected dashboard with auth check

### Session UX Improvements
- [x] Real-time cognitive sparkline chart (engagement + frustration)
- [x] Session timer showing elapsed time (MM:SS)
- [x] Activity log with timestamped entries
- [x] Session summary modal on session end
- [x] Streak tracking with fire emoji (🔥)
- [x] Best streak displayed in summary
- [x] Keyboard shortcuts (Escape to end/close)

---

## Planned Features
- [ ] Session history with replay
- [ ] Multiple task types (memory, pattern recognition)
- [ ] User calibration for emotion baseline
- [ ] Export session data (CSV/JSON)

---

## Technical Debt
- Remove unused state variables in SessionPage
- Add proper error boundaries
- Implement reconnection logic for WebSocket
- Add unit tests for ML pipeline

---

## Known Issues & Mitigations

### Neutral Emotion Dominance
**Problem**: HSEmotion model tends to predict neutral frequently.
**Mitigation**: Engagement formula uses arousal-based calculation where neutral contributes moderate engagement (calm focus = engaged).

### Contempt → Neutral Mapping
**Changed**: HSEmotion has 8 classes, we use 7. Contempt is now mapped to neutral (subtle emotion) instead of disgust to avoid inflation.

### Inference Path Verification
Backend logs `[INFERENCE]` with clear indicators:
- `Using HSEmotion model - dominant: X` = real inference via ML server
- `MOCK predictions` = fallback (ML server not running)

### Architecture Note
Due to Python 3.14 not supporting ONNX Runtime, we use a dual-server architecture:
- **Backend (port 8000)**: FastAPI, handles WebSocket, calls ML server
- **ML Server (port 8001)**: Flask, runs HSEmotion ONNX model on Python 3.11

---

## Key Metrics (Targets)
| Metric | Target | Status |
|--------|--------|--------|
| Emotion Accuracy | >65% FER2013 | ✓ ~70% |
| Inference Latency | <50ms | ✓ |
| End-to-end Latency | <100ms | ✓ |
| Frame Rate | 15 FPS | ✓ |
