/**
 * Dashboard Page
 *
 * Displays user statistics, performance charts, and session history.
 * Uses Recharts for visualizations and Supabase for data.
 */

import { useState, useEffect, useCallback } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts'
import {
  useAuth,
  getUserStats,
  getUserSessions,
  getPerformanceHistory,
  deleteSession,
  type Session,
} from '../lib/supabase'

// =============================================================================
// Types
// =============================================================================

interface UserStats {
  totalSessions: number
  totalTasks: number
  totalCorrect: number
  totalTimeMs: number
  avgAccuracy: number
  avgEngagement: number
  avgFrustration: number
}

interface PerformanceDataPoint {
  date: string
  accuracy: number
  engagement: number
  frustration: number
  tasks: number
}

// =============================================================================
// Utility Functions
// =============================================================================

function formatDuration(ms: number): string {
  const hours = Math.floor(ms / 3600000)
  const minutes = Math.floor((ms % 3600000) / 60000)

  if (hours > 0) {
    return `${hours}h ${minutes}m`
  }
  return `${minutes}m`
}

function formatSessionDuration(startedAt: string, endedAt: string | null): string {
  if (!endedAt) return 'In Progress'

  const durationMs = new Date(endedAt).getTime() - new Date(startedAt).getTime()
  return formatDuration(durationMs)
}

// =============================================================================
// Component
// =============================================================================

export default function DashboardPage() {
  const { user, profile, loading: authLoading, signOut } = useAuth()
  const navigate = useNavigate()

  const [stats, setStats] = useState<UserStats | null>(null)
  const [sessions, setSessions] = useState<Session[]>([])
  const [performanceData, setPerformanceData] = useState<PerformanceDataPoint[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Fetch dashboard data
  const fetchData = useCallback(async () => {
    if (!user) return

    setLoading(true)
    setError(null)

    try {
      const [statsResult, sessionsResult, performanceResult] = await Promise.all([
        getUserStats(user.id),
        getUserSessions(user.id, { limit: 10 }),
        getPerformanceHistory(user.id, 30),
      ])

      if (statsResult.error) throw statsResult.error
      if (sessionsResult.error) throw sessionsResult.error
      if (performanceResult.error) throw performanceResult.error

      setStats(statsResult.data)
      setSessions(sessionsResult.data)
      setPerformanceData(performanceResult.data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data')
    } finally {
      setLoading(false)
    }
  }, [user])

  useEffect(() => {
    if (user && !authLoading) {
      fetchData()
    } else if (!authLoading && !user) {
      setLoading(false)
    }
  }, [user, authLoading, fetchData])

  // Handle session deletion
  const handleDeleteSession = async (sessionId: string) => {
    if (!confirm('Delete this session? This cannot be undone.')) return

    const { error: deleteError } = await deleteSession(sessionId)
    if (deleteError) {
      setError('Failed to delete session')
      return
    }

    setSessions((prev) => prev.filter((s) => s.id !== sessionId))
    fetchData() // Refresh stats
  }

  // Handle sign out
  const handleSignOut = async () => {
    await signOut()
    navigate('/')
  }

  // Show login prompt if not authenticated
  if (!authLoading && !user) {
    return (
      <div className="min-h-screen bg-canvas text-ink p-6 lg:p-8">
        <div className="max-w-md mx-auto text-center mt-20">
          <h1 className="text-2xl font-semibold mb-4">Dashboard</h1>
          <p className="text-neutral-500 mb-6">
            Sign in to view your training history and performance analytics.
          </p>
          <div className="flex gap-3 justify-center">
            <Link to="/login" className="btn btn-primary">
              Sign In
            </Link>
            <Link to="/signup" className="btn btn-secondary">
              Create Account
            </Link>
          </div>
          <div className="mt-8">
            <Link to="/session" className="text-sm text-neutral-400 hover:text-neutral-600">
              Or continue in demo mode
            </Link>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-canvas text-ink p-6 lg:p-8">
      {/* Header */}
      <header className="flex items-center justify-between mb-8 max-w-6xl mx-auto">
        <Link to="/" className="text-xl font-semibold">
          FlowLoop
        </Link>
        <div className="flex items-center gap-4">
          {profile && (
            <span className="text-sm text-neutral-500">
              {profile.display_name || user?.email}
            </span>
          )}
          <div className="flex gap-3">
            <Link to="/session" className="btn btn-primary">
              New Session
            </Link>
            <button onClick={handleSignOut} className="btn btn-secondary">
              Sign Out
            </button>
          </div>
        </div>
      </header>

      <div className="max-w-6xl mx-auto">
        {error && (
          <div className="mb-6 p-4 rounded bg-red-50 border border-red-200 text-red-700">
            {error}
          </div>
        )}

        {loading ? (
          <div className="flex items-center justify-center py-20">
            <div className="animate-spin rounded-full h-8 w-8 border-2 border-ink border-t-transparent" />
          </div>
        ) : (
          <>
            {/* Stats Overview */}
            <div className="grid gap-5 md:grid-cols-4 mb-8">
              <div className="card">
                <p className="text-sm text-neutral-500 mb-1">Total Sessions</p>
                <p className="metric">{stats?.totalSessions ?? 0}</p>
              </div>
              <div className="card">
                <p className="text-sm text-neutral-500 mb-1">Tasks Completed</p>
                <p className="metric">{stats?.totalTasks ?? 0}</p>
              </div>
              <div className="card">
                <p className="text-sm text-neutral-500 mb-1">Average Accuracy</p>
                <p className="metric">
                  {stats ? `${(stats.avgAccuracy * 100).toFixed(0)}%` : '0%'}
                </p>
              </div>
              <div className="card">
                <p className="text-sm text-neutral-500 mb-1">Total Time</p>
                <p className="metric">
                  {stats ? formatDuration(stats.totalTimeMs) : '0m'}
                </p>
              </div>
            </div>

            {/* Charts Grid */}
            <div className="grid gap-5 md:grid-cols-2 mb-8">
              {/* Engagement Over Time */}
              <div className="card">
                <h2 className="font-medium mb-4">Engagement Over Time</h2>
                {performanceData.length > 0 ? (
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={performanceData}>
                        <defs>
                          <linearGradient id="engagementGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#059669" stopOpacity={0.3} />
                            <stop offset="95%" stopColor="#059669" stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                        <XAxis
                          dataKey="date"
                          tick={{ fontSize: 12 }}
                          stroke="#9ca3af"
                        />
                        <YAxis
                          domain={[0, 1]}
                          tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                          tick={{ fontSize: 12 }}
                          stroke="#9ca3af"
                        />
                        <Tooltip
                          formatter={(value) => [`${(Number(value) * 100).toFixed(0)}%`, 'Engagement']}
                          contentStyle={{
                            backgroundColor: '#fff',
                            border: '1px solid #e5e7eb',
                            borderRadius: '6px',
                          }}
                        />
                        <Area
                          type="monotone"
                          dataKey="engagement"
                          stroke="#059669"
                          fill="url(#engagementGradient)"
                          strokeWidth={2}
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <div className="h-64 flex items-center justify-center bg-neutral-50 rounded">
                    <p className="text-neutral-400 text-sm">No data yet</p>
                  </div>
                )}
              </div>

              {/* Accuracy vs Frustration */}
              <div className="card">
                <h2 className="font-medium mb-4">Accuracy & Frustration</h2>
                {performanceData.length > 0 ? (
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={performanceData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                        <XAxis
                          dataKey="date"
                          tick={{ fontSize: 12 }}
                          stroke="#9ca3af"
                        />
                        <YAxis
                          domain={[0, 1]}
                          tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                          tick={{ fontSize: 12 }}
                          stroke="#9ca3af"
                        />
                        <Tooltip
                          formatter={(value, name) => [
                            `${(Number(value) * 100).toFixed(0)}%`,
                            name === 'accuracy' ? 'Accuracy' : 'Frustration',
                          ]}
                          contentStyle={{
                            backgroundColor: '#fff',
                            border: '1px solid #e5e7eb',
                            borderRadius: '6px',
                          }}
                        />
                        <Legend />
                        <Line
                          type="monotone"
                          dataKey="accuracy"
                          stroke="#3b82f6"
                          strokeWidth={2}
                          dot={{ r: 3 }}
                          name="Accuracy"
                        />
                        <Line
                          type="monotone"
                          dataKey="frustration"
                          stroke="#e43e33"
                          strokeWidth={2}
                          dot={{ r: 3 }}
                          name="Frustration"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <div className="h-64 flex items-center justify-center bg-neutral-50 rounded">
                    <p className="text-neutral-400 text-sm">No data yet</p>
                  </div>
                )}
              </div>

              {/* Tasks Per Session */}
              <div className="card">
                <h2 className="font-medium mb-4">Tasks Per Session</h2>
                {performanceData.length > 0 ? (
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={performanceData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                        <XAxis
                          dataKey="date"
                          tick={{ fontSize: 12 }}
                          stroke="#9ca3af"
                        />
                        <YAxis tick={{ fontSize: 12 }} stroke="#9ca3af" />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: '#fff',
                            border: '1px solid #e5e7eb',
                            borderRadius: '6px',
                          }}
                        />
                        <Bar dataKey="tasks" fill="#6366f1" radius={[4, 4, 0, 0]} name="Tasks" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <div className="h-64 flex items-center justify-center bg-neutral-50 rounded">
                    <p className="text-neutral-400 text-sm">No data yet</p>
                  </div>
                )}
              </div>

              {/* Average Metrics */}
              <div className="card">
                <h2 className="font-medium mb-4">Overall Metrics</h2>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-neutral-500">Average Engagement</span>
                      <span className="font-medium">
                        {stats ? `${(stats.avgEngagement * 100).toFixed(0)}%` : '0%'}
                      </span>
                    </div>
                    <div className="h-2 bg-neutral-100 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-green-500 rounded-full transition-all"
                        style={{ width: `${(stats?.avgEngagement ?? 0) * 100}%` }}
                      />
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-neutral-500">Average Frustration</span>
                      <span className="font-medium">
                        {stats ? `${(stats.avgFrustration * 100).toFixed(0)}%` : '0%'}
                      </span>
                    </div>
                    <div className="h-2 bg-neutral-100 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-red-500 rounded-full transition-all"
                        style={{ width: `${(stats?.avgFrustration ?? 0) * 100}%` }}
                      />
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-neutral-500">Accuracy Rate</span>
                      <span className="font-medium">
                        {stats ? `${(stats.avgAccuracy * 100).toFixed(0)}%` : '0%'}
                      </span>
                    </div>
                    <div className="h-2 bg-neutral-100 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-blue-500 rounded-full transition-all"
                        style={{ width: `${(stats?.avgAccuracy ?? 0) * 100}%` }}
                      />
                    </div>
                  </div>

                  <div className="pt-4 border-t border-neutral-200">
                    <div className="flex justify-between text-sm">
                      <span className="text-neutral-500">Total Correct</span>
                      <span className="font-medium">
                        {stats?.totalCorrect ?? 0} / {stats?.totalTasks ?? 0}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Session History */}
            <div className="card">
              <h2 className="font-medium mb-4">Session History</h2>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-sm">
                  <thead>
                    <tr className="border-b border-neutral-200">
                      <th className="pb-3 text-neutral-500 font-medium">Date</th>
                      <th className="pb-3 text-neutral-500 font-medium">Duration</th>
                      <th className="pb-3 text-neutral-500 font-medium">Tasks</th>
                      <th className="pb-3 text-neutral-500 font-medium">Accuracy</th>
                      <th className="pb-3 text-neutral-500 font-medium">Engagement</th>
                      <th className="pb-3 text-neutral-500 font-medium">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sessions.length > 0 ? (
                      sessions.map((session) => (
                        <tr key={session.id} className="border-b border-neutral-100 last:border-0">
                          <td className="py-3">
                            {new Date(session.started_at).toLocaleDateString('en-US', {
                              month: 'short',
                              day: 'numeric',
                              year: 'numeric',
                            })}
                          </td>
                          <td className="py-3">
                            {formatSessionDuration(session.started_at, session.ended_at)}
                          </td>
                          <td className="py-3">{session.total_tasks}</td>
                          <td className="py-3">
                            {session.total_tasks > 0
                              ? `${((session.correct_tasks / session.total_tasks) * 100).toFixed(0)}%`
                              : '-'}
                          </td>
                          <td className="py-3">
                            {`${(session.avg_engagement * 100).toFixed(0)}%`}
                          </td>
                          <td className="py-3">
                            <button
                              onClick={() => handleDeleteSession(session.id)}
                              className="text-red-500 hover:text-red-700 text-xs"
                            >
                              Delete
                            </button>
                          </td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan={6} className="py-12 text-center text-neutral-400">
                          No sessions yet. Start a session to see your history.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
