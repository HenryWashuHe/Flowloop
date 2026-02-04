/**
 * Session Service for Supabase
 *
 * Handles CRUD operations for training sessions.
 * Manages session events and task results.
 */

import { supabase } from './client'
import type {
  Session,
  TaskResult,
  TaskResultInsert,
  SessionEventInsert,
} from './types'

// =============================================================================
// Session Operations
// =============================================================================

/**
 * Create a new training session
 */
export async function createSession(
  userId: string,
  taskType: string,
  isAdaptive: boolean
): Promise<{ data: Session | null; error: Error | null }> {
  const { data, error } = await supabase
    .from('sessions')
    .insert({
      user_id: userId,
      task_type: taskType,
      is_adaptive: isAdaptive,
      started_at: new Date().toISOString(),
    })
    .select()
    .single()

  return { data: data as Session | null, error }
}

/**
 * Update session with final stats
 */
export async function endSession(
  sessionId: string,
  stats: {
    totalTasks: number
    correctTasks: number
    avgEngagement: number
    avgFrustration: number
    difficultyProgression: number[]
  }
): Promise<{ error: Error | null }> {
  const { error } = await supabase
    .from('sessions')
    .update({
      ended_at: new Date().toISOString(),
      total_tasks: stats.totalTasks,
      correct_tasks: stats.correctTasks,
      avg_engagement: stats.avgEngagement,
      avg_frustration: stats.avgFrustration,
      difficulty_progression: stats.difficultyProgression,
    })
    .eq('id', sessionId)

  return { error }
}

/**
 * Get user's sessions with pagination
 */
export async function getUserSessions(
  userId: string,
  options: { limit?: number; offset?: number } = {}
): Promise<{ data: Session[]; error: Error | null; count: number | null }> {
  const { limit = 20, offset = 0 } = options

  const { data, error, count } = await supabase
    .from('sessions')
    .select('*', { count: 'exact' })
    .eq('user_id', userId)
    .order('started_at', { ascending: false })
    .range(offset, offset + limit - 1)

  return { data: (data as Session[]) ?? [], error, count }
}

/**
 * Get a single session by ID
 */
export async function getSession(
  sessionId: string
): Promise<{ data: Session | null; error: Error | null }> {
  const { data, error } = await supabase
    .from('sessions')
    .select('*')
    .eq('id', sessionId)
    .single()

  return { data: data as Session | null, error }
}

/**
 * Delete a session and all related data
 */
export async function deleteSession(
  sessionId: string
): Promise<{ error: Error | null }> {
  const { error } = await supabase.from('sessions').delete().eq('id', sessionId)
  return { error }
}

// =============================================================================
// Task Results Operations
// =============================================================================

/**
 * Save a task result
 */
export async function saveTaskResult(
  result: TaskResultInsert
): Promise<{ data: TaskResult | null; error: Error | null }> {
  const { data, error } = await supabase
    .from('task_results')
    .insert(result)
    .select()
    .single()

  return { data: data as TaskResult | null, error }
}

/**
 * Get task results for a session
 */
export async function getSessionTaskResults(
  sessionId: string
): Promise<{ data: TaskResult[]; error: Error | null }> {
  const { data, error } = await supabase
    .from('task_results')
    .select('*')
    .eq('session_id', sessionId)
    .order('created_at', { ascending: true })

  return { data: (data as TaskResult[]) ?? [], error }
}

/**
 * Batch save task results
 */
export async function batchSaveTaskResults(
  results: TaskResultInsert[]
): Promise<{ error: Error | null }> {
  const { error } = await supabase.from('task_results').insert(results)
  return { error }
}

// =============================================================================
// Session Events Operations
// =============================================================================

/**
 * Save a session event (emotion, difficulty change, etc.)
 */
export async function saveSessionEvent(
  event: SessionEventInsert
): Promise<{ error: Error | null }> {
  const { error } = await supabase.from('session_events').insert(event)
  return { error }
}

/**
 * Batch save session events
 */
export async function batchSaveSessionEvents(
  events: SessionEventInsert[]
): Promise<{ error: Error | null }> {
  const { error } = await supabase.from('session_events').insert(events)
  return { error }
}

/**
 * Get session events for charting
 */
export async function getSessionEvents(
  sessionId: string,
  eventType?: 'emotion' | 'task_complete' | 'difficulty_change'
): Promise<{ data: Array<{ timestamp: string; data: Record<string, unknown> }>; error: Error | null }> {
  let query = supabase
    .from('session_events')
    .select('timestamp, data')
    .eq('session_id', sessionId)
    .order('timestamp', { ascending: true })

  if (eventType) {
    query = query.eq('event_type', eventType)
  }

  const { data, error } = await query

  return { data: (data as Array<{ timestamp: string; data: Record<string, unknown> }>) ?? [], error }
}

// =============================================================================
// Analytics Queries
// =============================================================================

interface SessionRow {
  total_tasks: number
  correct_tasks: number
  avg_engagement: number
  avg_frustration: number
  started_at: string
  ended_at: string | null
}

/**
 * Get user statistics aggregated
 */
export async function getUserStats(userId: string): Promise<{
  data: {
    totalSessions: number
    totalTasks: number
    totalCorrect: number
    totalTimeMs: number
    avgAccuracy: number
    avgEngagement: number
    avgFrustration: number
  } | null
  error: Error | null
}> {
  const { data: sessions, error } = await supabase
    .from('sessions')
    .select('total_tasks, correct_tasks, avg_engagement, avg_frustration, started_at, ended_at')
    .eq('user_id', userId)
    .not('ended_at', 'is', null)

  if (error) {
    return { data: null, error }
  }

  const rows = (sessions as SessionRow[]) ?? []

  if (rows.length === 0) {
    return {
      data: {
        totalSessions: 0,
        totalTasks: 0,
        totalCorrect: 0,
        totalTimeMs: 0,
        avgAccuracy: 0,
        avgEngagement: 0,
        avgFrustration: 0,
      },
      error: null,
    }
  }

  const totalSessions = rows.length
  const totalTasks = rows.reduce((sum, s) => sum + s.total_tasks, 0)
  const totalCorrect = rows.reduce((sum, s) => sum + s.correct_tasks, 0)
  const totalTimeMs = rows.reduce((sum, s) => {
    if (s.started_at && s.ended_at) {
      return sum + (new Date(s.ended_at).getTime() - new Date(s.started_at).getTime())
    }
    return sum
  }, 0)
  const avgAccuracy = totalTasks > 0 ? totalCorrect / totalTasks : 0
  const avgEngagement = rows.reduce((sum, s) => sum + s.avg_engagement, 0) / totalSessions
  const avgFrustration = rows.reduce((sum, s) => sum + s.avg_frustration, 0) / totalSessions

  return {
    data: {
      totalSessions,
      totalTasks,
      totalCorrect,
      totalTimeMs,
      avgAccuracy,
      avgEngagement,
      avgFrustration,
    },
    error: null,
  }
}

interface PerformanceRow {
  started_at: string
  total_tasks: number
  correct_tasks: number
  avg_engagement: number
  avg_frustration: number
}

/**
 * Get performance data for charts (last N sessions)
 */
export async function getPerformanceHistory(
  userId: string,
  limit: number = 30
): Promise<{
  data: Array<{
    date: string
    accuracy: number
    engagement: number
    frustration: number
    tasks: number
  }>
  error: Error | null
}> {
  const { data: sessions, error } = await supabase
    .from('sessions')
    .select('started_at, total_tasks, correct_tasks, avg_engagement, avg_frustration')
    .eq('user_id', userId)
    .not('ended_at', 'is', null)
    .order('started_at', { ascending: true })
    .limit(limit)

  if (error) {
    return { data: [], error }
  }

  const rows = (sessions as PerformanceRow[]) ?? []

  const chartData = rows.map((s) => ({
    date: new Date(s.started_at).toLocaleDateString(),
    accuracy: s.total_tasks > 0 ? s.correct_tasks / s.total_tasks : 0,
    engagement: s.avg_engagement,
    frustration: s.avg_frustration,
    tasks: s.total_tasks,
  }))

  return { data: chartData, error: null }
}
