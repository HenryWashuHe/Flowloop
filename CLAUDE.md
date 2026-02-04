# FlowLoop Project Rules

## Project Overview
FlowLoop is an adaptive cognitive engine using computer vision for real-time attention and frustration detection. It dynamically adjusts task difficulty based on user cognitive state.

## Architecture
- **Frontend**: React + TypeScript + Vite + TailwindCSS
- **Backend**: Python + FastAPI + ONNX Runtime
- **ML**: PyTorch for training, ONNX for inference
- **Communication**: WebSocket for real-time data

## Critical Patterns

### Immutability
All state updates must be immutable. Never mutate objects directly.
```typescript
// CORRECT
const updated = { ...state, value: newValue }

// WRONG
state.value = newValue
```

### Error Handling
Wrap all async operations in try/catch with descriptive error messages.
```typescript
try {
  const result = await inference(frame)
  return result
} catch (error) {
  console.error('Inference failed:', error)
  throw new Error('Failed to process frame')
}
```

### Type Safety
- Frontend: Strict TypeScript, no `any` types
- Backend: Pydantic models for all API contracts

## ML Pipeline Rules

### Model Development
1. Always validate input shapes before inference
2. Log inference times for performance monitoring
3. Handle model loading failures gracefully
4. Use ONNX format for production inference

### Data Processing
1. Normalize images to ImageNet mean/std: `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`
2. Validate face detection confidence (>0.5) before processing
3. Handle missing landmarks gracefully (return neutral state)
4. Resize face crops to 224x224 for emotion model

### Temporal Smoothing
1. Always apply EMA to raw predictions
2. Use alpha=0.2 as default (configurable)
3. Reset smoother state on session start

## WebSocket Protocol

### Message Types
```typescript
// Client -> Server
type ClientMessage =
  | { type: 'frame'; data: { image: string; landmarks: number[]; timestamp: number } }
  | { type: 'session_start'; data: { userId?: string } }
  | { type: 'session_end'; data: { sessionId: string } }

// Server -> Client
type ServerMessage =
  | { type: 'prediction'; data: { attention: number; frustration: number; timestamp: number } }
  | { type: 'difficulty'; data: { level: number; reason: string } }
  | { type: 'error'; data: { code: string; message: string } }
```

### Latency Budget
- Face detection (MediaPipe): <20ms
- Emotion inference (ONNX): <30ms
- Fusion + smoothing: <10ms
- WebSocket round trip: <40ms
- **Total: <100ms**

## Testing Requirements

### Coverage Targets
- Unit tests: 80% minimum
- Integration tests: Critical paths covered
- E2E tests: Core user journeys

### Test Patterns
1. Test happy path first
2. Test edge cases (no face, low confidence, rapid motion)
3. Test error recovery
4. Mock external dependencies (webcam, models)

## Code Quality

### File Size Limits
- Components: <400 lines
- Utility modules: <200 lines
- Split large files proactively

### Function Length
- Maximum 50 lines per function
- Extract helpers for complex logic

### Naming Conventions
- Components: PascalCase (`AttentionChart.tsx`)
- Functions/hooks: camelCase (`useWebcam`)
- Constants: SCREAMING_SNAKE_CASE (`DEFAULT_ALPHA`)
- Files: kebab-case except components (`geometric-features.ts`)

## Performance Guidelines

### Frontend
- Lazy load heavy components (charts)
- Debounce rapid state updates (16ms for 60fps)
- Use `requestAnimationFrame` for animations
- Memoize expensive computations with `useMemo`

### Backend
- Use connection pooling for DB
- Batch predictions when possible
- Profile inference regularly
- Cache ONNX sessions (singleton pattern)

### Frame Processing
- Target 15-30 FPS for CV pipeline
- Skip frames under heavy load
- Use Web Workers for non-blocking processing

## Security

### Webcam Access
- Request permission explicitly with clear UI
- Show visual indicator when camera active
- Allow easy disable at any time
- Never transmit raw video to external servers

### Data Storage
- Hash user identifiers if stored
- Encrypt session data at rest
- Auto-delete old sessions (configurable retention)
- No PII in logs

## Git Workflow

### Branch Naming
- `feature/description` - New features
- `fix/description` - Bug fixes
- `refactor/description` - Code restructure
- `ml/description` - ML model changes

### Commit Messages
Follow conventional commits:
```
feat: add emotion model inference service
fix: handle missing face landmarks gracefully
refactor: extract geometric features to separate module
docs: add API documentation
test: add unit tests for fusion network
perf: optimize ONNX inference with INT8 quantization
```

## Dependencies

### Approved Libraries - Frontend
- react, react-dom
- vite, @vitejs/plugin-react
- tailwindcss, postcss, autoprefixer
- zustand (state management)
- chart.js, react-chartjs-2
- @mediapipe/face_mesh, @mediapipe/camera_utils

### Approved Libraries - Backend
- fastapi, uvicorn
- onnxruntime
- numpy, opencv-python
- pydantic, python-dotenv
- websockets

### Approved Libraries - ML
- torch, torchvision
- efficientnet_pytorch
- onnx, onnxruntime
- opencv-python
- pandas, scikit-learn

### Adding Dependencies
1. Check existing libraries first
2. Evaluate bundle size impact (frontend)
3. Verify license compatibility (MIT, Apache 2.0 preferred)
4. Document reason for addition

## Agents to Use

| Task | Agent |
|------|-------|
| Complex features | `planner` |
| New features/bugs | `tdd-guide` |
| After implementation | `code-reviewer` |
| Build failures | `build-error-resolver` |
| Security review | `security-reviewer` |

## Datasets

### Emotion Recognition
- **FER2013**: 35,887 images, 7 emotions (Kaggle)
- **AffectNet**: 450K+ images, valence/arousal (request access)
- **DAiSEE**: 9K videos, engagement/frustration labels

### Preprocessing
- FER2013: Grayscale 48x48 -> RGB 224x224
- Face crops: Align and resize to 224x224
- Augmentation: rotation (±15°), flip, brightness, color jitter

## Key Metrics

### Model Performance
- Emotion accuracy: >65% on FER2013 test set
- Attention/frustration MAE: <0.15
- Inference time: <30ms per frame

### System Performance
- End-to-end latency: <100ms
- Frame rate: 15-30 FPS sustained
- Memory usage: <500MB (frontend), <1GB (backend)

## File Structure Reference
```
Flowloop/
├── frontend/src/
│   ├── components/     # React components
│   ├── lib/            # Utilities, hooks
│   ├── stores/         # Zustand stores
│   └── types/          # TypeScript types
├── backend/src/
│   ├── inference/      # ONNX inference services
│   ├── websocket/      # WebSocket handlers
│   ├── smoothing/      # Temporal smoothing
│   └── db/             # Database models
├── ml/
│   ├── models/         # PyTorch model definitions
│   ├── training/       # Training scripts
│   ├── data/           # Data preparation
│   └── export/         # ONNX export scripts
└── docs/               # Documentation
```
