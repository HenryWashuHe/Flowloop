// =============================================================================
// Cognitive State Types
// =============================================================================

export interface CognitiveState {
  attention: number // 0-1, 1 = fully engaged
  frustration: number // 0-1, 1 = very frustrated
  timestamp: number // Unix timestamp in ms
}

export interface EmotionPrediction {
  emotions: EmotionScores
  dominantEmotion: Emotion
  confidence: number
}

export type Emotion =
  | 'neutral'
  | 'happy'
  | 'sad'
  | 'angry'
  | 'fear'
  | 'disgust'
  | 'surprise'

export type EmotionScores = Record<Emotion, number>

// =============================================================================
// Geometric Features Types
// =============================================================================

export interface GeometricFeatures {
  // Gaze estimation
  gazeDirection: Vector3
  gazeConfidence: number

  // Head pose (radians)
  headPitch: number // nodding up/down
  headYaw: number // turning left/right
  headRoll: number // tilting sideways

  // Eye metrics
  leftEyeAspectRatio: number
  rightEyeAspectRatio: number
  blinkRate: number // blinks per minute

  // Brow metrics (frustration proxy)
  browFurrowScore: number // 0-1
  browRaiseScore: number // 0-1

  // Mouth metrics
  mouthOpenness: number // 0-1
  lipCornerPull: number // -1 (frown) to 1 (smile)

  // Motion metrics
  headMotionVelocity: number
  headMotionVariance: number
}

export interface Vector3 {
  x: number
  y: number
  z: number
}

// =============================================================================
// Task Types
// =============================================================================

export type TaskType = 'math' | 'logic' | 'coding' | 'memory'

export interface Task {
  id: string
  type: TaskType
  difficulty: number // 1-10
  prompt: string
  expectedAnswer: string
  metadata?: Record<string, unknown>
}

export interface TaskAttempt {
  taskId: string
  userAnswer: string
  isCorrect: boolean
  responseTimeMs: number
  timestamp: number
}

export interface TaskPerformance {
  tasksCompleted: number
  correctAnswers: number
  averageResponseTime: number
  currentStreak: number
}

// =============================================================================
// Session Types
// =============================================================================

export interface Session {
  id: string
  startTime: number
  endTime?: number
  isAdaptiveEnabled: boolean
  taskType: TaskType
  frames: FrameLog[]
  tasks: TaskAttempt[]
}

export interface FrameLog {
  timestamp: number
  cognitiveState: CognitiveState
  geometricFeatures?: GeometricFeatures
  emotions?: EmotionPrediction
  currentDifficulty: number
}

export interface SessionSummary {
  id: string
  date: string
  duration: number // seconds
  tasksCompleted: number
  accuracy: number // 0-1
  averageAttention: number
  averageFrustration: number
}

// =============================================================================
// WebSocket Message Types
// =============================================================================

// Client -> Server
export type ClientMessage =
  | {
    type: 'frame'
    data: {
      image: string // base64 encoded face crop
      landmarks: number[] // flattened 468 * 3 landmarks
      timestamp: number
    }
  }
  | {
    type: 'session_start'
    data: {
      userId?: string
      taskType: TaskType
      isAdaptiveEnabled: boolean
    }
  }
  | {
    type: 'session_end'
    data: {
      sessionId: string
    }
  }
  | {
    type: 'task_answer'
    data: {
      task_id: string
      answer: string
      time_taken: number
    }
  }

// Server -> Client
export type ServerMessage =
  | {
    type: 'prediction'
    data: {
      attention: number
      frustration: number
      emotions: EmotionScores
      timestamp: number
    }
  }
  | {
    type: 'difficulty'
    data: {
      level: number
      reason: string
    }
  }
  | {
    type: 'task'
    data: {
      taskId: string
      taskType: string
      question: string
      difficulty: number
      timeLimit: number
      hints: string[]
    }
  }
  | {
    type: 'task_result'
    data: {
      taskId: string
      correct: boolean
      userAnswer: string
      correctAnswer: string
      timeTaken: number
      newDifficulty: number
    }
  }
  | {
    type: 'error'
    data: {
      code: string
      message: string
    }
  }
  | {
    type: 'connected'
    data: {
      sessionId: string
    }
  }

// =============================================================================
// Settings Types
// =============================================================================

export interface AppSettings {
  // Camera
  cameraDeviceId: string

  // ML Pipeline
  smoothingAlpha: number // EMA alpha, 0.05-0.5

  // Adaptation
  adaptationSensitivity: number // 1-10
  adaptationEnabled: boolean

  // Debug
  enableDebugOverlay: boolean
  enablePerformanceMonitoring: boolean

  // Data
  autoDeleteSessionsDays: number
}

export const DEFAULT_SETTINGS: AppSettings = {
  cameraDeviceId: '',
  smoothingAlpha: 0.2,
  adaptationSensitivity: 5,
  adaptationEnabled: true,
  enableDebugOverlay: true,
  enablePerformanceMonitoring: false,
  autoDeleteSessionsDays: 30,
}

// =============================================================================
// Chart Data Types
// =============================================================================

export interface TimeSeriesDataPoint {
  timestamp: number
  value: number
}

export interface AttentionChartData {
  attention: TimeSeriesDataPoint[]
  frustration: TimeSeriesDataPoint[]
  difficulty: TimeSeriesDataPoint[]
}

export interface PerformanceScatterPoint {
  x: number // emotional state composite
  y: number // performance score
  difficulty: number
  taskId: string
}
