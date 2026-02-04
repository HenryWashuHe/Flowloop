/**
 * WebSocket Client for FlowLoop Backend
 *
 * Handles real-time communication with FastAPI backend.
 * - Sends: landmarks, face crop, session control
 * - Receives: predictions, difficulty updates, errors
 *
 * Features:
 * - Auto-reconnect with exponential backoff
 * - Connection state management
 * - Type-safe message handling
 */

import type { ClientMessage, ServerMessage, TaskType, EmotionScores } from '../../types'

// =============================================================================
// Types
// =============================================================================

export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'reconnecting'

export interface PredictionData {
    attention: number
    frustration: number
    emotions: EmotionScores
    timestamp: number
}

export interface DifficultyData {
    level: number
    reason: string
}

export interface TaskData {
    taskId: string
    taskType: string
    question: string
    difficulty: number
    timeLimit: number
    hints: string[]
}

export interface TaskResultData {
    taskId: string
    correct: boolean
    userAnswer: string
    correctAnswer: string
    timeTaken: number
    newDifficulty: number
}

export interface WebSocketClientOptions {
    url?: string
    reconnectAttempts?: number
    reconnectDelayMs?: number
    maxReconnectDelayMs?: number
}

export interface WebSocketClientEvents {
    onConnectionChange?: (state: ConnectionState) => void
    onPrediction?: (data: PredictionData) => void
    onDifficultyChange?: (data: DifficultyData) => void
    onTask?: (data: TaskData) => void
    onTaskResult?: (data: TaskResultData) => void
    onSessionCreated?: (sessionId: string) => void
    onError?: (code: string, message: string) => void
}

const DEFAULT_OPTIONS: Required<WebSocketClientOptions> = {
    url: 'ws://localhost:8000/ws',
    reconnectAttempts: 5,
    reconnectDelayMs: 1000,
    maxReconnectDelayMs: 30000,
}

// =============================================================================
// WebSocketClient Class
// =============================================================================

export class WebSocketClient {
    private ws: WebSocket | null = null
    private options: Required<WebSocketClientOptions>
    private events: WebSocketClientEvents = {}
    private connectionState: ConnectionState = 'disconnected'
    private sessionId: string | null = null
    private reconnectAttempt = 0
    private reconnectTimeout: ReturnType<typeof setTimeout> | null = null
    private messageQueue: ClientMessage[] = []

    constructor(options: WebSocketClientOptions = {}) {
        this.options = { ...DEFAULT_OPTIONS, ...options }
    }

    /**
     * Set event handlers
     */
    setEventHandlers(events: WebSocketClientEvents): void {
        this.events = { ...this.events, ...events }
    }

    /**
     * Connect to WebSocket server
     */
    connect(): void {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            console.warn('WebSocket already connected')
            return
        }

        this.setConnectionState('connecting')

        try {
            this.ws = new WebSocket(this.options.url)
            this.setupEventListeners()
        } catch (error) {
            console.error('Failed to create WebSocket:', error)
            this.handleReconnect()
        }
    }

    /**
     * Disconnect from server
     */
    disconnect(): void {
        if (this.reconnectTimeout) {
            clearTimeout(this.reconnectTimeout)
            this.reconnectTimeout = null
        }

        if (this.ws) {
            this.ws.close(1000, 'Client disconnect')
            this.ws = null
        }

        this.sessionId = null
        this.reconnectAttempt = 0
        this.messageQueue = []
        this.setConnectionState('disconnected')
    }

    /**
     * Start a new session
     */
    startSession(taskType: TaskType, isAdaptiveEnabled: boolean, userId?: string): void {
        this.send({
            type: 'session_start',
            data: {
                userId,
                taskType,
                isAdaptiveEnabled,
            },
        })
    }

    /**
     * End current session
     */
    endSession(): void {
        if (!this.sessionId) {
            console.warn('No active session to end')
            return
        }

        this.send({
            type: 'session_end',
            data: {
                sessionId: this.sessionId,
            },
        })

        this.sessionId = null
    }

    /**
     * Send frame data (landmarks + face crop)
     */
    sendFrame(landmarks: number[], faceCropBase64: string): void {
        this.send({
            type: 'frame',
            data: {
                image: faceCropBase64,
                landmarks,
                timestamp: Date.now(),
            },
        })
    }

    /**
     * Send answer to a task
     */
    sendTaskAnswer(taskId: string, answer: string, timeTaken: number): void {
        this.send({
            type: 'task_answer',
            data: {
                task_id: taskId,
                answer,
                time_taken: timeTaken,
            },
        })
    }

    /**
     * Get current connection state
     */
    get state(): ConnectionState {
        return this.connectionState
    }

    /**
     * Get current session ID
     */
    get currentSessionId(): string | null {
        return this.sessionId
    }

    /**
     * Check if connected and ready
     */
    get isConnected(): boolean {
        return this.ws !== null && this.ws.readyState === WebSocket.OPEN
    }

    // ===========================================================================
    // Private Methods
    // ===========================================================================

    private setupEventListeners(): void {
        if (!this.ws) return

        this.ws.onopen = () => {
            console.log('WebSocket connected to', this.options.url)
            this.setConnectionState('connected')
            this.reconnectAttempt = 0
            this.flushMessageQueue()
        }

        this.ws.onclose = (event) => {
            console.log('WebSocket closed:', event.code, event.reason)

            if (event.code !== 1000) {
                // Abnormal close, attempt reconnect
                this.handleReconnect()
            } else {
                this.setConnectionState('disconnected')
            }
        }

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error)
            this.events.onError?.('WEBSOCKET_ERROR', 'Connection error occurred')
        }

        this.ws.onmessage = (event) => {
            this.handleMessage(event.data)
        }
    }

    private handleMessage(data: string): void {
        try {
            const message: ServerMessage = JSON.parse(data)

            switch (message.type) {
                case 'connected':
                    this.sessionId = message.data.sessionId
                    this.events.onSessionCreated?.(message.data.sessionId)
                    break

                case 'prediction':
                    this.events.onPrediction?.({
                        attention: message.data.attention,
                        frustration: message.data.frustration,
                        emotions: message.data.emotions,
                        timestamp: message.data.timestamp,
                    })
                    break

                case 'difficulty':
                    this.events.onDifficultyChange?.({
                        level: message.data.level,
                        reason: message.data.reason,
                    })
                    break

                case 'task':
                    this.events.onTask?.({
                        taskId: message.data.taskId,
                        taskType: message.data.taskType,
                        question: message.data.question,
                        difficulty: message.data.difficulty,
                        timeLimit: message.data.timeLimit,
                        hints: message.data.hints,
                    })
                    break

                case 'task_result':
                    this.events.onTaskResult?.({
                        taskId: message.data.taskId,
                        correct: message.data.correct,
                        userAnswer: message.data.userAnswer,
                        correctAnswer: message.data.correctAnswer,
                        timeTaken: message.data.timeTaken,
                        newDifficulty: message.data.newDifficulty,
                    })
                    break

                case 'error':
                    console.error('Server error:', message.data.code, message.data.message)
                    this.events.onError?.(message.data.code, message.data.message)
                    break

                default:
                    console.warn('Unknown message type:', (message as { type: string }).type)
            }
        } catch (error) {
            console.error('Failed to parse WebSocket message:', error)
        }
    }

    private send(message: ClientMessage): void {
        if (this.isConnected) {
            this.ws!.send(JSON.stringify(message))
        } else {
            // Queue message for later delivery
            this.messageQueue.push(message)
            console.debug('Message queued, WebSocket not connected')
        }
    }

    private flushMessageQueue(): void {
        while (this.messageQueue.length > 0 && this.isConnected) {
            const message = this.messageQueue.shift()
            if (message) {
                this.ws!.send(JSON.stringify(message))
            }
        }
    }

    private handleReconnect(): void {
        if (this.reconnectAttempt >= this.options.reconnectAttempts) {
            console.error('Max reconnect attempts reached')
            this.setConnectionState('disconnected')
            this.events.onError?.('MAX_RECONNECT', 'Failed to reconnect after multiple attempts')
            return
        }

        this.setConnectionState('reconnecting')
        this.reconnectAttempt++

        // Exponential backoff
        const delay = Math.min(
            this.options.reconnectDelayMs * Math.pow(2, this.reconnectAttempt - 1),
            this.options.maxReconnectDelayMs
        )

        console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempt}/${this.options.reconnectAttempts})`)

        this.reconnectTimeout = setTimeout(() => {
            this.connect()
        }, delay)
    }

    private setConnectionState(state: ConnectionState): void {
        if (this.connectionState !== state) {
            this.connectionState = state
            this.events.onConnectionChange?.(state)
        }
    }
}

// =============================================================================
// Singleton Instance
// =============================================================================

let sharedClient: WebSocketClient | null = null

/**
 * Get shared WebSocket client instance
 */
export function getSharedWebSocketClient(): WebSocketClient {
    if (!sharedClient) {
        sharedClient = new WebSocketClient()
    }
    return sharedClient
}

/**
 * Reset shared client (for testing)
 */
export function resetSharedWebSocketClient(): void {
    if (sharedClient) {
        sharedClient.disconnect()
        sharedClient = null
    }
}
