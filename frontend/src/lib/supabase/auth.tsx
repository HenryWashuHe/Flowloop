/**
 * Auth Context for Supabase Authentication
 *
 * Provides authentication state and methods throughout the app.
 * Handles session persistence and auth state changes.
 */

import {
  createContext,
  useContext,
  useEffect,
  useState,
  useCallback,
  type ReactNode,
} from 'react'
import type { User, Session, AuthError } from '@supabase/supabase-js'
import { supabase } from './client'
import type { Profile } from './types'

// =============================================================================
// Types
// =============================================================================

interface AuthState {
  user: User | null
  profile: Profile | null
  session: Session | null
  loading: boolean
}

interface AuthContextValue extends AuthState {
  signUp: (email: string, password: string, displayName?: string) => Promise<{ error: AuthError | null }>
  signIn: (email: string, password: string) => Promise<{ error: AuthError | null }>
  signOut: () => Promise<void>
  updateProfile: (updates: Partial<Profile>) => Promise<{ error: Error | null }>
}

// =============================================================================
// Context
// =============================================================================

const AuthContext = createContext<AuthContextValue | null>(null)

// =============================================================================
// Provider
// =============================================================================

interface AuthProviderProps {
  children: ReactNode
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [state, setState] = useState<AuthState>({
    user: null,
    profile: null,
    session: null,
    loading: true,
  })

  // Fetch user profile from database
  const fetchProfile = useCallback(async (userId: string) => {
    const { data, error } = await supabase
      .from('profiles')
      .select('*')
      .eq('id', userId)
      .single()

    if (error) {
      console.error('Error fetching profile:', error)
      return null
    }
    return data
  }, [])

  // Initialize auth state
  useEffect(() => {
    const initAuth = async () => {
      const { data: { session } } = await supabase.auth.getSession()

      if (session?.user) {
        const profile = await fetchProfile(session.user.id)
        setState({
          user: session.user,
          profile,
          session,
          loading: false,
        })
      } else {
        setState({
          user: null,
          profile: null,
          session: null,
          loading: false,
        })
      }
    }

    initAuth()

    // Listen for auth changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      async (event, session) => {
        if (event === 'SIGNED_IN' && session?.user) {
          const profile = await fetchProfile(session.user.id)
          setState({
            user: session.user,
            profile,
            session,
            loading: false,
          })
        } else if (event === 'SIGNED_OUT') {
          setState({
            user: null,
            profile: null,
            session: null,
            loading: false,
          })
        } else if (event === 'TOKEN_REFRESHED' && session) {
          setState((prev) => ({
            ...prev,
            session,
          }))
        }
      }
    )

    return () => {
      subscription.unsubscribe()
    }
  }, [fetchProfile])

  // Sign up with email/password
  const signUp = useCallback(
    async (email: string, password: string, displayName?: string) => {
      const { error } = await supabase.auth.signUp({
        email,
        password,
        options: {
          data: {
            display_name: displayName,
          },
        },
      })
      return { error }
    },
    []
  )

  // Sign in with email/password
  const signIn = useCallback(async (email: string, password: string) => {
    const { error } = await supabase.auth.signInWithPassword({
      email,
      password,
    })
    return { error }
  }, [])

  // Sign out
  const signOut = useCallback(async () => {
    await supabase.auth.signOut()
  }, [])

  // Update profile
  const updateProfile = useCallback(
    async (updates: Partial<Profile>) => {
      if (!state.user) {
        return { error: new Error('Not authenticated') }
      }

      const { error } = await supabase
        .from('profiles')
        .update(updates as Record<string, unknown>)
        .eq('id', state.user.id)

      if (!error) {
        setState((prev) => ({
          ...prev,
          profile: prev.profile ? { ...prev.profile, ...updates } : null,
        }))
      }

      return { error }
    },
    [state.user]
  )

  const value: AuthContextValue = {
    ...state,
    signUp,
    signIn,
    signOut,
    updateProfile,
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

// =============================================================================
// Hook
// =============================================================================

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

// =============================================================================
// Protected Route Component
// =============================================================================

interface ProtectedRouteProps {
  children: ReactNode
  fallback?: ReactNode
}

export function ProtectedRoute({ children, fallback }: ProtectedRouteProps) {
  const { user, loading } = useAuth()

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-canvas">
        <div className="animate-spin rounded-full h-8 w-8 border-2 border-ink border-t-transparent" />
      </div>
    )
  }

  if (!user) {
    return fallback ?? null
  }

  return <>{children}</>
}
