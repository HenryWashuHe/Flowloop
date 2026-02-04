/**
 * WebSocket message types shared between frontend and backend.
 *
 * These types define the protocol for real-time communication
 * between the React frontend and FastAPI backend.
 */

// =============================================================================
// Emotion Types
// =============================================================================

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
// Client -> Server Messages
// =============================================================================

export interface FrameData {
  /** Base64 encoded face crop image (JPEG) */
  image: string
  /** Flattened MediaPipe landmarks (468 * 3 = 1404 floats) */
  landmarks: number[]
  /** Unix timestamp in milliseconds */
  timestamp: number
}

export interface SessionStartData {
  /** Optional user identifier */
  userId?: string
  /** Type of cognitive task */
  taskType: 'math' | 'logic' | 'coding' | 'memory'
  /** Whether adaptive difficulty is enabled */
  isAdaptiveEnabled: boolean
}

export interface SessionEndData {
  /** Session identifier */
  sessionId: string
}

export type ClientMessage =
  | { type: 'frame'; data: FrameData }
  | { type: 'session_start'; data: SessionStartData }
  | { type: 'session_end'; data: SessionEndData }

// =============================================================================
// Server -> Client Messages
// =============================================================================

export interface PredictionData {
  /** Attention score (0-1, 1 = fully engaged) */
  attention: number
  /** Frustration score (0-1, 1 = very frustrated) */
  frustration: number
  /** Emotion probability distribution */
  emotions: EmotionScores
  /** Original frame timestamp for latency tracking */
  timestamp: number
}

export interface DifficultyData {
  /** New difficulty level (1-10) */
  level: number
  /** Human-readable reason for adjustment */
  reason: string
}

export interface ErrorData {
  /** Error code for programmatic handling */
  code: string
  /** Human-readable error message */
  message: string
}

export interface ConnectedData {
  /** Assigned session identifier */
  sessionId: string
}

export type ServerMessage =
  | { type: 'prediction'; data: PredictionData }
  | { type: 'difficulty'; data: DifficultyData }
  | { type: 'error'; data: ErrorData }
  | { type: 'connected'; data: ConnectedData }

// =============================================================================
// Utility Types
// =============================================================================

export interface WebSocketConfig {
  /** WebSocket server URL */
  url: string
  /** Reconnection attempts before giving up */
  maxRetries: number
  /** Base delay between reconnection attempts (ms) */
  retryDelay: number
  /** Whether to use exponential backoff */
  useExponentialBackoff: boolean
}

export const DEFAULT_WS_CONFIG: WebSocketConfig = {
  url: 'ws://localhost:8000/ws',
  maxRetries: 5,
  retryDelay: 1000,
  useExponentialBackoff: true,
}
