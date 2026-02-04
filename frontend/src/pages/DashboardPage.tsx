import { Link } from 'react-router-dom'

export default function DashboardPage() {
  return (
    <div className="min-h-screen bg-canvas text-ink p-6 lg:p-8">
      {/* Header */}
      <header className="flex items-center justify-between mb-8 max-w-6xl mx-auto">
        <Link to="/" className="text-xl font-semibold">
          FlowLoop
        </Link>
        <div className="flex gap-3">
          <button className="btn btn-secondary">Export Data</button>
          <Link to="/session" className="btn btn-primary">
            New Session
          </Link>
        </div>
      </header>

      <div className="max-w-6xl mx-auto">
        {/* Stats Overview */}
        <div className="grid gap-5 md:grid-cols-4 mb-8">
          <div className="card">
            <p className="text-sm text-neutral-500 mb-1">Total Sessions</p>
            <p className="metric">0</p>
          </div>
          <div className="card">
            <p className="text-sm text-neutral-500 mb-1">Tasks Completed</p>
            <p className="metric">0</p>
          </div>
          <div className="card">
            <p className="text-sm text-neutral-500 mb-1">Average Accuracy</p>
            <p className="metric">0%</p>
          </div>
          <div className="card">
            <p className="text-sm text-neutral-500 mb-1">Total Time</p>
            <p className="metric">0h 0m</p>
          </div>
        </div>

        {/* Charts Grid */}
        <div className="grid gap-5 md:grid-cols-2 mb-8">
          <div className="card">
            <h2 className="font-medium mb-4">Attention Over Time</h2>
            <div className="aspect-video bg-neutral-100 rounded flex items-center justify-center">
              <p className="text-neutral-400 text-sm">Chart placeholder</p>
            </div>
          </div>

          <div className="card">
            <h2 className="font-medium mb-4">Performance vs Emotional State</h2>
            <div className="aspect-video bg-neutral-100 rounded flex items-center justify-center">
              <p className="text-neutral-400 text-sm">Chart placeholder</p>
            </div>
          </div>

          <div className="card">
            <h2 className="font-medium mb-4">Difficulty Progression</h2>
            <div className="aspect-video bg-neutral-100 rounded flex items-center justify-center">
              <p className="text-neutral-400 text-sm">Chart placeholder</p>
            </div>
          </div>

          <div className="card">
            <h2 className="font-medium mb-4">Task Type Distribution</h2>
            <div className="aspect-video bg-neutral-100 rounded flex items-center justify-center">
              <p className="text-neutral-400 text-sm">Chart placeholder</p>
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
                  <th className="pb-3 text-neutral-500 font-medium">Avg Attention</th>
                  <th className="pb-3 text-neutral-500 font-medium">Actions</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td colSpan={6} className="py-12 text-center text-neutral-400">
                    No sessions yet. Start a session to see your history.
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}
