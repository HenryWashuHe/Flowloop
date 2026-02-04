/**
 * Session Persistence Hook
 *
 * Manages saving session data to Supabase during training sessions.
 * Batches events and handles save on session end.
 */

import { useRef, useCallback, useEffect } from 'react'
import { useAuth } from './auth'
import {
  createSession,
  endSession,
  saveTaskResult,
  batchSaveSessionEvents,
} from './sessions'
import type { TaskResultInsert, SessionEventInsert } from './types'

// =============================================================================
// Types
// =============================================================================

interface SessionData {
  sessionId: string | null
  totalTasks: number
  correctTasks: number
  engagementSum: number
  frustrationSum: number
  engagementCount: number
  difficultyProgression: number[]
}

interface UseSessionPersistenceReturn {
  isAuthenticated: boolean
  startPersistentSession: (taskType: string, isAdaptive: boolean) => Promise<string | null>
  endPersistentSession: () => Promise<void>
  recordEmotionEvent: (engagement: number, frustration: number, emotions: Record<string, number>) => void
  recordTaskResult: (
    taskType: string,
    difficulty: number,
    isCorrect: boolean,
    timeSpentMs: number,
    engagement: number,
    frustration: number,
    question: string,
    userAnswer: string,
    correctAnswer: string
  ) => void
  recordDifficultyChange: (newDifficulty: number, reason: string) => void
  currentSessionId: string | null
}

// =============================================================================
// Constants
// =============================================================================

const EVENT_BATCH_SIZE = 10
const EVENT_FLUSH_INTERVAL_MS = 5000

// =============================================================================
// Hook
// =============================================================================

export function useSessionPersistence(): UseSessionPersistenceReturn {
  const { user } = useAuth()
  const isAuthenticated = !!user

  // Session state
  const sessionData = useRef<SessionData>({
    sessionId: null,
    totalTasks: 0,
    correctTasks: 0,
    engagementSum: 0,
    frustrationSum: 0,
    engagementCount: 0,
    difficultyProgression: [],
  })

  // Event buffer for batching
  const eventBuffer = useRef<SessionEventInsert[]>([])
  const flushTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Flush event buffer to database
  const flushEvents = useCallback(async () => {
    if (eventBuffer.current.length === 0) return

    const events = [...eventBuffer.current]
    eventBuffer.current = []

    if (sessionData.current.sessionId) {
      await batchSaveSessionEvents(events)
    }
  }, [])

  // Schedule flush
  const scheduleFlush = useCallback(() => {
    if (flushTimeoutRef.current) {
      clearTimeout(flushTimeoutRef.current)
    }
    flushTimeoutRef.current = setTimeout(flushEvents, EVENT_FLUSH_INTERVAL_MS)
  }, [flushEvents])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (flushTimeoutRef.current) {
        clearTimeout(flushTimeoutRef.current)
      }
      // Flush remaining events
      flushEvents()
    }
  }, [flushEvents])

  // Start a new persistent session
  const startPersistentSession = useCallback(
    async (taskType: string, isAdaptive: boolean): Promise<string | null> => {
      if (!user) return null

      const { data, error } = await createSession(user.id, taskType, isAdaptive)

      if (error || !data) {
        console.error('Failed to create session:', error)
        return null
      }

      sessionData.current = {
        sessionId: data.id,
        totalTasks: 0,
        correctTasks: 0,
        engagementSum: 0,
        frustrationSum: 0,
        engagementCount: 0,
        difficultyProgression: [],
      }

      return data.id
    },
    [user]
  )

  // End and save session
  const endPersistentSession = useCallback(async () => {
    const data = sessionData.current
    if (!data.sessionId) return

    // Flush any remaining events
    await flushEvents()

    // Calculate averages
    const avgEngagement = data.engagementCount > 0 ? data.engagementSum / data.engagementCount : 0
    const avgFrustration = data.engagementCount > 0 ? data.frustrationSum / data.engagementCount : 0

    // Save session end
    await endSession(data.sessionId, {
      totalTasks: data.totalTasks,
      correctTasks: data.correctTasks,
      avgEngagement,
      avgFrustration,
      difficultyProgression: data.difficultyProgression,
    })

    // Reset state
    sessionData.current = {
      sessionId: null,
      totalTasks: 0,
      correctTasks: 0,
      engagementSum: 0,
      frustrationSum: 0,
      engagementCount: 0,
      difficultyProgression: [],
    }
  }, [flushEvents])

  // Record emotion event (batched)
  const recordEmotionEvent = useCallback(
    (engagement: number, frustration: number, emotions: Record<string, number>) => {
      const data = sessionData.current
      if (!data.sessionId) return

      // Update running averages
      data.engagementSum += engagement
      data.frustrationSum += frustration
      data.engagementCount++

      // Add to event buffer
      eventBuffer.current.push({
        session_id: data.sessionId,
        event_type: 'emotion',
        data: { engagement, frustration, emotions },
        timestamp: new Date().toISOString(),
      })

      // Flush if buffer is full
      if (eventBuffer.current.length >= EVENT_BATCH_SIZE) {
        flushEvents()
      } else {
        scheduleFlush()
      }
    },
    [flushEvents, scheduleFlush]
  )

  // Record task result (immediate save)
  const recordTaskResult = useCallback(
    async (
      taskType: string,
      difficulty: number,
      isCorrect: boolean,
      timeSpentMs: number,
      engagement: number,
      frustration: number,
      question: string,
      userAnswer: string,
      correctAnswer: string
    ) => {
      const data = sessionData.current
      if (!data.sessionId) return

      // Update session stats
      data.totalTasks++
      if (isCorrect) {
        data.correctTasks++
      }

      // Save task result
      const result: TaskResultInsert = {
        session_id: data.sessionId,
        task_type: taskType,
        difficulty,
        is_correct: isCorrect,
        time_spent_ms: timeSpentMs,
        engagement_at_task: engagement,
        frustration_at_task: frustration,
        question,
        user_answer: userAnswer,
        correct_answer: correctAnswer,
      }

      await saveTaskResult(result)

      // Also add as event
      eventBuffer.current.push({
        session_id: data.sessionId,
        event_type: 'task_complete',
        data: { taskType, difficulty, isCorrect, timeSpentMs },
        timestamp: new Date().toISOString(),
      })

      scheduleFlush()
    },
    [scheduleFlush]
  )

  // Record difficulty change
  const recordDifficultyChange = useCallback(
    (newDifficulty: number, reason: string) => {
      const data = sessionData.current
      if (!data.sessionId) return

      // Track difficulty progression
      data.difficultyProgression.push(newDifficulty)

      // Add event
      eventBuffer.current.push({
        session_id: data.sessionId,
        event_type: 'difficulty_change',
        data: { newDifficulty, reason },
        timestamp: new Date().toISOString(),
      })

      scheduleFlush()
    },
    [scheduleFlush]
  )

  return {
    isAuthenticated,
    startPersistentSession,
    endPersistentSession,
    recordEmotionEvent,
    recordTaskResult,
    recordDifficultyChange,
    currentSessionId: sessionData.current.sessionId,
  }
}
