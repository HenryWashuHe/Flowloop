/**
 * Supabase Module Exports
 */

export { supabase } from './client'

export { AuthProvider, useAuth, ProtectedRoute } from './auth'

export type {
  Profile,
  Session,
  SessionEvent,
  TaskResult,
  SessionInsert,
  SessionUpdate,
  TaskResultInsert,
  SessionEventInsert,
} from './types'

export {
  createSession,
  endSession,
  getUserSessions,
  getSession,
  deleteSession,
  saveTaskResult,
  getSessionTaskResults,
  batchSaveTaskResults,
  saveSessionEvent,
  batchSaveSessionEvents,
  getSessionEvents,
  getUserStats,
  getPerformanceHistory,
} from './sessions'

export { useSessionPersistence } from './useSessionPersistence'
