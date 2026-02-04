/**
 * Supabase Database Types
 *
 * TypeScript definitions for FlowLoop database schema.
 * Keep in sync with Supabase migrations.
 */

export interface Database {
  public: {
    Tables: {
      profiles: {
        Row: {
          id: string
          display_name: string | null
          avatar_url: string | null
          created_at: string
          updated_at: string
        }
        Insert: {
          id: string
          display_name?: string | null
          avatar_url?: string | null
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          display_name?: string | null
          avatar_url?: string | null
          updated_at?: string
        }
      }
      sessions: {
        Row: {
          id: string
          user_id: string
          started_at: string
          ended_at: string | null
          total_tasks: number
          correct_tasks: number
          avg_engagement: number
          avg_frustration: number
          difficulty_progression: number[]
          task_type: string
          is_adaptive: boolean
          created_at: string
        }
        Insert: {
          id?: string
          user_id: string
          started_at?: string
          ended_at?: string | null
          total_tasks?: number
          correct_tasks?: number
          avg_engagement?: number
          avg_frustration?: number
          difficulty_progression?: number[]
          task_type?: string
          is_adaptive?: boolean
          created_at?: string
        }
        Update: {
          ended_at?: string | null
          total_tasks?: number
          correct_tasks?: number
          avg_engagement?: number
          avg_frustration?: number
          difficulty_progression?: number[]
        }
      }
      session_events: {
        Row: {
          id: string
          session_id: string
          event_type: 'emotion' | 'task_complete' | 'difficulty_change'
          data: Record<string, unknown>
          timestamp: string
        }
        Insert: {
          id?: string
          session_id: string
          event_type: 'emotion' | 'task_complete' | 'difficulty_change'
          data: Record<string, unknown>
          timestamp?: string
        }
        Update: {
          data?: Record<string, unknown>
        }
      }
      task_results: {
        Row: {
          id: string
          session_id: string
          task_type: string
          difficulty: number
          is_correct: boolean
          time_spent_ms: number
          engagement_at_task: number
          frustration_at_task: number
          question: string
          user_answer: string
          correct_answer: string
          created_at: string
        }
        Insert: {
          id?: string
          session_id: string
          task_type: string
          difficulty: number
          is_correct: boolean
          time_spent_ms: number
          engagement_at_task?: number
          frustration_at_task?: number
          question: string
          user_answer: string
          correct_answer: string
          created_at?: string
        }
        Update: {
          is_correct?: boolean
        }
      }
    }
    Views: Record<string, never>
    Functions: Record<string, never>
    Enums: Record<string, never>
  }
}

// Convenience type aliases
export type Profile = Database['public']['Tables']['profiles']['Row']
export type Session = Database['public']['Tables']['sessions']['Row']
export type SessionEvent = Database['public']['Tables']['session_events']['Row']
export type TaskResult = Database['public']['Tables']['task_results']['Row']

export type SessionInsert = Database['public']['Tables']['sessions']['Insert']
export type SessionUpdate = Database['public']['Tables']['sessions']['Update']
export type TaskResultInsert = Database['public']['Tables']['task_results']['Insert']
export type SessionEventInsert = Database['public']['Tables']['session_events']['Insert']
