import { useState, useEffect, useCallback, useRef } from 'react'
import { Link } from 'react-router-dom'
import WebcamCapture from '../components/WebcamCapture'
import { FaceMeshResult } from '../lib/mediapipe/FaceMeshProcessor'
import { WebSocketClient, ConnectionState, PredictionData, TaskData, TaskResultData } from '../lib/websocket/WebSocketClient'
import { FaceCropExtractor } from '../lib/features/faceCropExtractor'
import { useSessionPersistence } from '../lib/supabase'
import type { CognitiveState, EmotionScores } from '../types'

// =============================================================================
// Default State
// =============================================================================

const DEFAULT_EMOTIONS: EmotionScores = {
  neutral: 1,
  happy: 0,
  sad: 0,
  angry: 0,
  fear: 0,
  disgust: 0,
  surprise: 0,
}

const DEFAULT_COGNITIVE_STATE: CognitiveState = {
  attention: 0,
  frustration: 0,
  timestamp: Date.now(),
}

// =============================================================================
// Component
// =============================================================================

export default function SessionPage() {
  // Session state
  const [isSessionActive, setIsSessionActive] = useState(false)
  const [isAdaptiveEnabled] = useState(true)

  // Connection state
  const [, setConnectionState] = useState<ConnectionState>('disconnected')
  const [, setSessionId] = useState<string | null>(null)

  // Cognitive state from backend
  const [cognitiveState, setCognitiveState] = useState<CognitiveState>(DEFAULT_COGNITIVE_STATE)
  const [emotions, setEmotions] = useState<EmotionScores>(DEFAULT_EMOTIONS)
  const [currentDifficulty, setCurrentDifficulty] = useState(5)

  // Face detection state
  const [faceDetected, setFaceDetected] = useState(false)

  // Task state
  const [currentTask, setCurrentTask] = useState<TaskData | null>(null)
  const [taskStartTime, setTaskStartTime] = useState<number>(0)
  const [lastResult, setLastResult] = useState<TaskResultData | null>(null)
  const [currentAnswer, setCurrentAnswer] = useState('')
  const [tasksCompleted, setTasksCompleted] = useState(0)
  const [correctAnswers, setCorrectAnswers] = useState(0)

  // Refs
  const wsClientRef = useRef<WebSocketClient | null>(null)
  const faceCropExtractorRef = useRef<FaceCropExtractor | null>(null)
  const emotionSampleCountRef = useRef(0)

  // Supabase session persistence
  const {
    isAuthenticated,
    startPersistentSession,
    endPersistentSession,
    recordEmotionEvent,
    recordTaskResult,
    recordDifficultyChange,
  } = useSessionPersistence()

  // ===========================================================================
  // WebSocket Setup
  // ===========================================================================

  useEffect(() => {
    // Initialize face crop extractor
    faceCropExtractorRef.current = new FaceCropExtractor(224, 0.3)
    
    const client = new WebSocketClient()

    client.setEventHandlers({
      onConnectionChange: (state) => {
        setConnectionState(state)
        console.log('WebSocket connection state:', state)
      },
      onSessionCreated: (id) => {
        setSessionId(id)
        console.log('Session created:', id)
      },
      onPrediction: (data: PredictionData) => {
        setCognitiveState({
          attention: data.attention,
          frustration: data.frustration,
          timestamp: data.timestamp,
        })
        setEmotions(data.emotions)
      },
      onDifficultyChange: (data) => {
        setCurrentDifficulty(data.level)
      },
      onTask: (data: TaskData) => {
        setCurrentTask(data)
        setTaskStartTime(Date.now())
        setCurrentAnswer('')
      },
      onTaskResult: (data: TaskResultData) => {
        setLastResult(data)
        setTasksCompleted((prev) => prev + 1)
        if (data.correct) {
          setCorrectAnswers((prev) => prev + 1)
        }
      },
      onError: (code, message) => {
        console.error('WebSocket error:', code, message)
      },
    })

    wsClientRef.current = client

    return () => {
      client.disconnect()
      faceCropExtractorRef.current?.dispose()
    }
  }, [])

  // ===========================================================================
  // Session Control
  // ===========================================================================

  const startSession = useCallback(async () => {
    // Start WebSocket session
    if (wsClientRef.current) {
      wsClientRef.current.connect()
      wsClientRef.current.startSession('math', isAdaptiveEnabled)
    }

    // Start persistent session if authenticated
    if (isAuthenticated) {
      await startPersistentSession('math', isAdaptiveEnabled)
    }

    setIsSessionActive(true)
    setTasksCompleted(0)
    setCorrectAnswers(0)
    emotionSampleCountRef.current = 0
  }, [isAdaptiveEnabled, isAuthenticated, startPersistentSession])

  const endSession = useCallback(async () => {
    // End WebSocket session
    if (wsClientRef.current) {
      wsClientRef.current.endSession()
      wsClientRef.current.disconnect()
    }

    // End persistent session if authenticated
    if (isAuthenticated) {
      await endPersistentSession()
    }

    setIsSessionActive(false)
    setSessionId(null)
    setCognitiveState(DEFAULT_COGNITIVE_STATE)
    setEmotions(DEFAULT_EMOTIONS)
  }, [isAuthenticated, endPersistentSession])

  // ===========================================================================
  // Persistence Effects
  // ===========================================================================

  // Record emotion events (sampled every 10 predictions to avoid data overload)
  const prevEmotionsRef = useRef<EmotionScores | null>(null)
  useEffect(() => {
    if (!isSessionActive || !isAuthenticated) return
    if (cognitiveState.timestamp === DEFAULT_COGNITIVE_STATE.timestamp) return

    // Sample every 10th prediction
    emotionSampleCountRef.current++
    if (emotionSampleCountRef.current % 10 !== 0) return

    // Only record if emotions have changed
    if (prevEmotionsRef.current === emotions) return
    prevEmotionsRef.current = emotions

    recordEmotionEvent(
      cognitiveState.attention,
      cognitiveState.frustration,
      emotions as Record<string, number>
    )
  }, [cognitiveState, emotions, isSessionActive, isAuthenticated, recordEmotionEvent])

  // Record task results when completed
  const prevLastResultRef = useRef<TaskResultData | null>(null)
  useEffect(() => {
    if (!isSessionActive || !isAuthenticated || !lastResult) return
    if (prevLastResultRef.current?.taskId === lastResult.taskId) return

    prevLastResultRef.current = lastResult

    recordTaskResult(
      currentTask?.taskType || 'math',
      currentTask?.difficulty || currentDifficulty,
      lastResult.correct,
      lastResult.timeTaken * 1000, // Convert to ms
      cognitiveState.attention,
      cognitiveState.frustration,
      currentTask?.question || '',
      lastResult.userAnswer,
      lastResult.correctAnswer
    )
  }, [lastResult, isSessionActive, isAuthenticated, currentTask, currentDifficulty, cognitiveState, recordTaskResult])

  // Record difficulty changes
  const prevDifficultyRef = useRef(currentDifficulty)
  useEffect(() => {
    if (!isSessionActive || !isAuthenticated) return
    if (prevDifficultyRef.current === currentDifficulty) return

    const prevDiff = prevDifficultyRef.current
    prevDifficultyRef.current = currentDifficulty

    const reason = currentDifficulty > prevDiff ? 'increased' : 'decreased'
    recordDifficultyChange(currentDifficulty, reason)
  }, [currentDifficulty, isSessionActive, isAuthenticated, recordDifficultyChange])

  // ===========================================================================
  // Frame Processing
  // ===========================================================================

  const handleFrame = useCallback((result: FaceMeshResult, video: HTMLVideoElement | null) => {
    if (!isSessionActive || !wsClientRef.current?.isConnected) return

    if (result.faceDetected && result.landmarks.length > 0 && video) {
      // Extract face crop for emotion model
      const faceCropBase64 = faceCropExtractorRef.current?.extract(video, result.landmarks) || ''
      
      // Send frame data to backend with face crop
      wsClientRef.current.sendFrame(result.landmarks, faceCropBase64)
    }
  }, [isSessionActive])

  const handleFaceDetectionChange = useCallback((detected: boolean) => {
    setFaceDetected(detected)
  }, [])

  // ===========================================================================
  // Task Handling (Placeholder)
  // ===========================================================================

  const handleSubmitAnswer = useCallback(() => {
    if (!currentTask || !wsClientRef.current) return

    const timeTaken = (Date.now() - taskStartTime) / 1000
    wsClientRef.current.sendTaskAnswer(currentTask.taskId, currentAnswer, timeTaken)

    // Optimistic UI update or wait?
    // Let's wait for onTaskResult to update stats, but clear input
    // Actually input clearing should happen on new task or result
  }, [currentTask, currentAnswer, taskStartTime])

  const accuracy = tasksCompleted > 0 ? Math.round((correctAnswers / tasksCompleted) * 100) : 0

  // ===========================================================================
  // Render
  // ===========================================================================

  return (
    <div className="min-h-screen bg-canvas text-ink p-6 lg:p-8">
      {/* Header */}
      <header className="flex items-center justify-between mb-8 max-w-7xl mx-auto">
        <div className="flex items-center gap-6">
          <Link to="/" className="text-xl font-semibold text-ink hover:text-neutral-600 transition-colors">
            FlowLoop
          </Link>
          <div className="h-5 w-px bg-neutral-300" />
          <span className="text-sm text-neutral-500">
            {isSessionActive ? 'Session in progress' : 'Ready'}
          </span>
          {isSessionActive && (
            <span className={`text-xs px-2 py-0.5 rounded-full ${
              isAuthenticated
                ? 'bg-green-100 text-green-700'
                : 'bg-neutral-100 text-neutral-500'
            }`}>
              {isAuthenticated ? 'Saving' : 'Demo mode'}
            </span>
          )}
        </div>

        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2 text-sm">
            <span className="text-neutral-500">Difficulty</span>
            <span className="font-mono font-medium">{currentDifficulty}</span>
          </div>
          <Link to="/dashboard" className="text-sm text-neutral-500 hover:text-ink transition-colors">
            Exit
          </Link>
        </div>
      </header>

      {/* Main Layout */}
      <div className="grid gap-8 lg:grid-cols-12 max-w-7xl mx-auto">

        {/* Left Panel - Monitoring */}
        <div className="lg:col-span-3 space-y-6">
          {/* Camera Feed */}
          <div className="card p-0 overflow-hidden">
            <WebcamCapture
              isActive={isSessionActive}
              onFrame={handleFrame}
              onFaceDetectionChange={handleFaceDetectionChange}
              showLandmarks={true}
              targetFps={15}
              className="w-full aspect-video"
            />
            {!faceDetected && isSessionActive && (
              <div className="absolute inset-0 flex items-center justify-center bg-white/90">
                <p className="text-sm text-accent">Face not detected</p>
              </div>
            )}
          </div>

          {/* Cognitive Metrics */}
          <div className="card">
            <h2 className="label mb-4">Cognitive State</h2>
            
            <div className="space-y-5">
              {/* Attention */}
              <div>
                <div className="flex justify-between items-baseline mb-2">
                  <span className="text-sm text-neutral-600">Engagement</span>
                  <span className="font-mono text-lg">{(cognitiveState.attention * 100).toFixed(0)}%</span>
                </div>
                <div className="gauge-track">
                  <div
                    className="gauge-fill bg-ink"
                    style={{ width: `${cognitiveState.attention * 100}%` }}
                  />
                </div>
              </div>

              {/* Frustration */}
              <div>
                <div className="flex justify-between items-baseline mb-2">
                  <span className="text-sm text-neutral-600">Frustration</span>
                  <span className={`font-mono text-lg ${cognitiveState.frustration > 0.6 ? 'text-accent' : ''}`}>
                    {(cognitiveState.frustration * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="gauge-track">
                  <div
                    className={`gauge-fill ${cognitiveState.frustration > 0.6 ? 'bg-accent' : 'bg-neutral-400'}`}
                    style={{ width: `${cognitiveState.frustration * 100}%` }}
                  />
                </div>
              </div>
            </div>

            {/* Emotion Breakdown */}
            <div className="mt-6 pt-5 border-t border-neutral-200">
              <h3 className="text-xs text-neutral-500 uppercase tracking-wide mb-3">Detected Emotions</h3>
              <div className="space-y-2">
                {Object.entries(emotions)
                  .sort(([, a], [, b]) => b - a)
                  .slice(0, 4)
                  .map(([emotion, value]) => (
                    <div key={emotion} className="flex items-center justify-between text-sm">
                      <span className="text-neutral-600 capitalize">{emotion}</span>
                      <span className="font-mono text-neutral-500">{(value * 100).toFixed(0)}%</span>
                    </div>
                  ))}
              </div>
            </div>
          </div>
        </div>

        {/* Center Panel - Task */}
        <div className="lg:col-span-6">
          <div className="card h-full flex flex-col">
            {isSessionActive ? (
              <>
                <div className="flex justify-between items-center mb-8">
                  <div>
                    <h2 className="text-lg font-medium">Current Task</h2>
                    {currentTask && (
                      <p className="text-sm text-neutral-500 mt-1">{currentTask.taskType}</p>
                    )}
                  </div>
                  <button
                    onClick={endSession}
                    className="text-sm text-neutral-500 hover:text-accent transition-colors"
                  >
                    End Session
                  </button>
                </div>

                <div className="flex-1 flex flex-col items-center justify-center py-12">
                  {currentTask ? (
                    <div className="w-full max-w-md text-center">
                      <p className="text-4xl md:text-5xl font-mono font-medium mb-8">
                        {currentTask.question}
                      </p>

                      {/* Feedback */}
                      {lastResult && lastResult.taskId !== currentTask.taskId && (
                        <p className={`text-sm mb-6 ${lastResult.correct ? 'text-state-good' : 'text-accent'}`}>
                          {lastResult.correct ? 'Correct' : `Incorrect — answer was ${lastResult.correctAnswer}`}
                        </p>
                      )}

                      <div className="flex items-center justify-center gap-3">
                        <input
                          type="text"
                          value={currentAnswer}
                          onChange={(e) => setCurrentAnswer(e.target.value)}
                          onKeyDown={(e) => e.key === 'Enter' && handleSubmitAnswer()}
                          className="input text-center text-2xl w-40 py-3"
                          placeholder="?"
                          autoFocus
                          autoComplete="off"
                        />
                        <button
                          onClick={handleSubmitAnswer}
                          disabled={!currentAnswer}
                          className="btn btn-primary px-6 py-3 disabled:opacity-40"
                        >
                          Submit
                        </button>
                      </div>

                      {currentTask.hints && currentTask.hints.length > 0 && (
                        <details className="mt-8 text-sm text-neutral-500">
                          <summary className="cursor-pointer hover:text-ink transition-colors">
                            Show hint
                          </summary>
                          <p className="mt-2 text-neutral-600">{currentTask.hints[0]}</p>
                        </details>
                      )}
                    </div>
                  ) : (
                    <p className="text-neutral-500">Loading next task...</p>
                  )}
                </div>
              </>
            ) : (
              <div className="flex-1 flex flex-col items-center justify-center py-16">
                <h2 className="text-2xl font-medium mb-3">Cognitive Training Session</h2>
                <p className="text-neutral-500 text-center max-w-sm mb-8">
                  Task difficulty adapts in real-time based on your cognitive state.
                  Position your face clearly in the camera frame.
                </p>
                <button
                  onClick={startSession}
                  className="btn btn-primary px-8 py-3"
                >
                  Begin Session
                </button>
                {!isAuthenticated && (
                  <p className="text-sm text-neutral-400 mt-4">
                    <Link to="/login" className="text-ink hover:underline">Sign in</Link>
                    {' '}to save your progress
                  </p>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Right Panel - Performance */}
        <div className="lg:col-span-3 space-y-6">
          {/* Session Stats */}
          <div className="card">
            <h2 className="label mb-4">Performance</h2>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="metric">{accuracy}%</div>
                <div className="text-xs text-neutral-500 mt-1">Accuracy</div>
              </div>
              <div>
                <div className="metric">{tasksCompleted}</div>
                <div className="text-xs text-neutral-500 mt-1">Completed</div>
              </div>
            </div>
            <div className="mt-4 pt-4 border-t border-neutral-200">
              <div className="text-sm text-neutral-600">
                {correctAnswers} of {tasksCompleted} correct
              </div>
            </div>
          </div>

          {/* Session Log */}
          <div className="card flex-1">
            <h2 className="label mb-4">Activity Log</h2>
            <div className="space-y-3 text-sm">
              {isSessionActive && tasksCompleted === 0 && (
                <p className="text-neutral-400 italic">Awaiting first response...</p>
              )}
              {lastResult && (
                <div className="flex justify-between">
                  <span className="text-neutral-600">Task {tasksCompleted}</span>
                  <span className={lastResult.correct ? 'text-state-good' : 'text-accent'}>
                    {lastResult.correct ? 'Correct' : 'Incorrect'}
                  </span>
                </div>
              )}
              {isSessionActive && (
                <div className="text-neutral-500">
                  Session started at {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
