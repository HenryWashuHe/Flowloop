# FlowLoop - Product Requirements Document

## 1. Executive Summary

### Vision
FlowLoop is an adaptive cognitive training platform that uses real-time computer vision to optimize learning experiences by detecting user attention and emotional state, then dynamically adjusting task difficulty to maintain optimal engagement.

### Mission
To create a personalized, responsive learning environment that adapts in real-time to the learner's cognitive state, maximizing engagement and learning outcomes.

### Key Value Propositions
1. **Real-time Adaptation**: Difficulty adjusts within seconds based on detected cognitive state
2. **Privacy-First**: All processing happens locally - no video leaves the device
3. **Research-Ready**: Comprehensive logging enables scientific study of learning patterns
4. **Extensible**: Modular task system supports diverse cognitive challenges

## 2. Problem Statement

### Current Challenges
- Traditional learning platforms use static difficulty or simple performance-based progression
- Learners frequently experience frustration (too hard) or boredom (too easy)
- No real-time feedback loop between learner state and content delivery
- Existing emotion detection tools are not integrated with educational content

### Gap in Market
- Attention tracking exists (eye trackers) but is expensive and not integrated
- Emotion detection exists but is not applied to adaptive learning
- Adaptive learning exists but relies solely on performance, missing emotional signals

### Opportunity
Combine commodity webcams with modern CV/ML to create an accessible, real-time adaptive learning system that responds to both performance AND emotional state.

## 3. Goals and Success Metrics

### Primary Goals
1. Demonstrate real-time cognitive state detection (attention + frustration)
2. Implement closed-loop adaptation between detected state and task difficulty
3. Provide compelling visualizations of the adaptation process

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Detection Latency | <100ms | End-to-end time from frame to prediction |
| Frame Rate | 15-30 FPS | Sustained during task sessions |
| Emotion Accuracy | >65% | On FER2013 test set |
| User Engagement | >80% completion | Session completion rate |
| Adaptation Correlation | r>0.5 | Between attention score and performance |

## 4. User Personas

### Persona 1: Student Learner (Primary)
- **Name**: Alex, 22, Computer Science student
- **Goals**: Improve problem-solving skills, prepare for technical interviews
- **Pain Points**: Gets frustrated with too-hard problems, bored with easy ones
- **Tech Comfort**: High, comfortable with webcam-based apps

### Persona 2: Professional Skill Builder
- **Name**: Jordan, 35, Software Engineer
- **Goals**: Maintain sharp cognitive skills, learn new areas
- **Pain Points**: Limited practice time, needs efficient learning
- **Tech Comfort**: High

### Persona 3: Researcher/Educator
- **Name**: Dr. Chen, 45, Cognitive Science Professor
- **Goals**: Study attention patterns, evaluate adaptive learning effectiveness
- **Pain Points**: Needs data export, reproducible experiments
- **Tech Comfort**: Medium-high

## 5. User Stories and Requirements

### Epic 1: Webcam-Based Attention Tracking
```
As a user, I want the system to detect my attention level from my webcam
so that it can adapt to my current cognitive state.

Acceptance Criteria:
- [ ] Webcam permission requested clearly
- [ ] Visual indicator shows camera is active
- [ ] Attention score (0-1) displayed in real-time
- [ ] Works in Chrome, Firefox, Safari
- [ ] Graceful degradation if face not detected
```

### Epic 2: Emotion/Frustration Detection
```
As a user, I want the system to detect when I'm frustrated or confused
so that it can provide easier tasks or scaffolding.

Acceptance Criteria:
- [ ] Frustration score (0-1) computed in real-time
- [ ] Based on facial expressions AND geometric features
- [ ] Temporal smoothing prevents flickering
- [ ] Calibration option for individual differences
```

### Epic 3: Adaptive Task Difficulty
```
As a user, I want task difficulty to adjust based on my cognitive state
so that I stay in an optimal learning zone.

Acceptance Criteria:
- [ ] Difficulty increases when attention high + performance good
- [ ] Difficulty decreases when frustration detected
- [ ] Smooth transitions (no jarring jumps)
- [ ] Toggle to disable adaptation (control condition)
- [ ] Manual difficulty override available
```

### Epic 4: Progress Visualization
```
As a user, I want to see charts of my attention, performance, and difficulty
so that I can understand my learning patterns.

Acceptance Criteria:
- [ ] Real-time line chart of attention vs difficulty
- [ ] Scatter plot of performance vs emotional state
- [ ] Session summary statistics
- [ ] Export data as CSV
```

### Epic 5: Session Management
```
As a user, I want to start, pause, and review training sessions
so that I can manage my learning over time.

Acceptance Criteria:
- [ ] Start new session with task type selection
- [ ] Pause/resume without losing progress
- [ ] View history of past sessions
- [ ] Compare performance across sessions
```

## 6. Functional Requirements

### FR1: Real-Time Face Detection
- Detect face in webcam stream at 15-30 FPS
- Extract 468 facial landmarks using MediaPipe
- Handle multiple faces (use largest/closest)
- Report confidence score for detection quality

### FR2: Geometric Feature Extraction
- Compute gaze direction (pitch, yaw)
- Compute head pose (pitch, yaw, roll)
- Compute eye aspect ratio (blink detection)
- Compute brow position (furrow detection)
- Compute motion metrics (velocity, variance)

### FR3: Emotion Classification
- Classify 7 basic emotions from face crop
- Output frustration proxy score (0-1)
- Output engagement score (0-1)
- Use ONNX model for inference

### FR4: Feature Fusion
- Combine geometric and emotion features
- Output unified attention score (0-1)
- Output unified frustration score (0-1)
- Apply temporal smoothing (EMA)

### FR5: Task Generation
- Support multiple task types:
  - Math problems (arithmetic, algebra)
  - Logic puzzles (patterns, sequences)
  - Coding challenges (bug fixes, output prediction)
  - Memory tasks (n-back, sequence recall)
- Parameterize difficulty (1-10 scale)
- Generate unlimited unique instances

### FR6: Adaptive Controller
- Rule-based difficulty adjustment
- Inputs: attention, frustration, task performance
- Outputs: target difficulty level
- Hysteresis to prevent oscillation
- Configurable sensitivity

### FR7: Data Logging
- Log every frame's predictions
- Log task attempts and outcomes
- Log difficulty changes with reasons
- Store in local database
- Export to CSV/JSON

### FR8: Dashboard Visualization
- Real-time attention chart (rolling window)
- Real-time frustration chart
- Difficulty overlay on attention chart
- Performance vs state scatter plot
- Session statistics summary

## 7. Non-Functional Requirements

### Performance
| Requirement | Target |
|-------------|--------|
| End-to-end latency | <100ms |
| Frame rate | 15-30 FPS |
| Model load time | <5s |
| Memory (frontend) | <500MB |
| Memory (backend) | <1GB |

### Security
- All video processing local (no cloud transmission)
- No PII stored without explicit consent
- Session data encrypted at rest
- Webcam indicator always visible when active

### Accessibility
- Keyboard navigation for all controls
- Screen reader compatible UI elements
- Color-blind friendly charts
- Adjustable text sizes

### Browser Compatibility
| Browser | Minimum Version |
|---------|-----------------|
| Chrome | 90+ |
| Firefox | 88+ |
| Safari | 14+ |
| Edge | 90+ |

### Scalability
- Single-user local deployment (MVP)
- Future: Multi-user with centralized backend
- Future: Model updates via cloud sync

## 8. Technical Constraints

### Model Size
- Emotion model: <30MB (ONNX, quantized)
- Fusion model: <5MB (ONNX, quantized)
- Total download: <50MB

### Latency Budget
```
Face detection (MediaPipe): 15-20ms
Emotion inference (ONNX):   25-30ms
Fusion + smoothing:          5-10ms
WebSocket round trip:       30-40ms
─────────────────────────────────────
Total:                      75-100ms
```

### Privacy Requirements
- No raw video transmitted to server
- Only processed features sent via WebSocket
- Option to run fully client-side (TensorFlow.js)
- Clear privacy policy displayed

## 9. Dependencies

### External Libraries
- MediaPipe Face Mesh (Google)
- EfficientNet-B0 (pretrained weights)
- ONNX Runtime (inference)
- Chart.js (visualization)

### Datasets
- FER2013 (Kaggle, public)
- AffectNet (request access from authors)
- DAiSEE (request access from authors)

### Development Tools
- Node.js 18+
- Python 3.10+
- PyTorch 2.0+
- pnpm (package manager)

## 10. Timeline and Milestones

### Phase 1: Foundation (Weeks 1-2)
- [x] Project scaffolding
- [ ] Webcam capture component
- [ ] MediaPipe integration
- [ ] WebSocket setup
- [ ] Database schema

**Milestone**: Face landmarks displayed on webcam feed

### Phase 2: Core ML Pipeline (Weeks 3-5)
- [ ] Dataset preparation scripts
- [ ] EmotionNet training
- [ ] Fusion network training
- [ ] ONNX export + quantization
- [ ] Inference services
- [ ] Temporal smoothing
- [ ] Dashboard charts

**Milestone**: Real-time attention/frustration scores displayed

### Phase 3: Task Engine (Weeks 6-7)
- [ ] Task type implementations
- [ ] Difficulty parameterization
- [ ] Adaptive controller
- [ ] Performance tracking

**Milestone**: Tasks adapt based on cognitive state

### Phase 4: Dashboard & Polish (Weeks 8-9)
- [ ] Session management UI
- [ ] Settings panel
- [ ] Data export
- [ ] Error handling improvements
- [ ] Performance optimization

**Milestone**: Complete user experience

### Phase 5: Deployment (Week 10)
- [ ] Production build
- [ ] Documentation
- [ ] Demo video
- [ ] GitHub release

**Milestone**: Public release

## 11. Risks and Mitigations

### Technical Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Model accuracy on real users | High | Medium | Calibration phase, user feedback |
| Latency exceeds budget | High | Medium | INT8 quantization, frame skipping |
| Browser compatibility issues | Medium | Medium | Feature detection, fallbacks |
| WebGL not available | Medium | Low | CPU fallback for inference |

### User Adoption Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Privacy concerns | High | High | Clear policy, local processing |
| Webcam permission denied | High | Medium | Graceful degradation, manual mode |
| Adaptation feels wrong | Medium | Medium | Manual override, sensitivity settings |

### Data Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Dataset licensing issues | High | Low | Use only properly licensed data |
| Model bias on demographics | Medium | Medium | Test across diverse faces |
| Data loss | Medium | Low | Auto-save, export reminders |

## 12. Future Considerations

### Version 2.0 Features
- Multi-user collaborative sessions
- Mobile app (React Native)
- Advanced RL-based adaptation
- Custom task creation
- Leaderboards and gamification

### Research Extensions
- A/B testing framework
- Longitudinal study mode
- Integration with learning management systems
- Physiological sensor fusion (heart rate, GSR)

### Enterprise Features
- Team analytics dashboard
- SSO integration
- Custom branding
- SLA and support

## Appendix A: Wireframes

### Main Session View
```
┌────────────────────────────────────────────────────────────┐
│  FlowLoop                              [Settings] [Export] │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌──────────────┐  ┌────────────────────────────────────┐  │
│  │              │  │                                    │  │
│  │   Webcam     │  │         Current Task               │  │
│  │   Feed       │  │                                    │  │
│  │              │  │    What is 47 × 23?                │  │
│  │  [Attention] │  │                                    │  │
│  │    0.82      │  │    [_______________] [Submit]      │  │
│  │              │  │                                    │  │
│  │[Frustration] │  │    Difficulty: 6/10                │  │
│  │    0.23      │  │                                    │  │
│  └──────────────┘  └────────────────────────────────────┘  │
│                                                            │
│  ┌────────────────────────────────────────────────────────┐│
│  │  Attention ───  Difficulty ─ ─                        ││
│  │  1.0 │    ╱╲    ╱╲                              10    ││
│  │      │   ╱  ╲  ╱  ╲                                   ││
│  │  0.5 │  ╱    ╲╱    ╲    ─ ─ ─ ─ ─ ─              5    ││
│  │      │ ╱              ╲                               ││
│  │  0.0 └─────────────────────────────────────────  0    ││
│  │       -60s                                    now      ││
│  └────────────────────────────────────────────────────────┘│
│                                                            │
│  [Adaptive: ON]  [Sensitivity: ████░░░░░░]  [Pause] [End]  │
└────────────────────────────────────────────────────────────┘
```

### Dashboard View
```
┌────────────────────────────────────────────────────────────┐
│  Session Summary                         [Back] [Export]   │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Duration: 15:32    Tasks: 24    Accuracy: 79%             │
│                                                            │
│  ┌─────────────────────────┐  ┌─────────────────────────┐  │
│  │  Attention Over Time    │  │  Performance vs State   │  │
│  │                         │  │                         │  │
│  │  [Line Chart]           │  │  [Scatter Plot]         │  │
│  │                         │  │                         │  │
│  └─────────────────────────┘  └─────────────────────────┘  │
│                                                            │
│  ┌─────────────────────────┐  ┌─────────────────────────┐  │
│  │  Difficulty Progression │  │  Emotion Distribution   │  │
│  │                         │  │                         │  │
│  │  [Area Chart]           │  │  [Bar Chart]            │  │
│  │                         │  │                         │  │
│  └─────────────────────────┘  └─────────────────────────┘  │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

## Appendix B: API Contracts

### WebSocket Messages
See `shared/types/messages.ts` for full type definitions.

### REST Endpoints (Future)
```
GET  /api/sessions          - List sessions
GET  /api/sessions/:id      - Get session details
POST /api/sessions          - Create session
PUT  /api/sessions/:id      - Update session
GET  /api/sessions/:id/export - Export session data
```
