import { Link } from 'react-router-dom'

export default function DashboardPage() {
  return (
    <div className="min-h-screen p-4">
      {/* Header */}
      <header className="flex items-center justify-between mb-6">
        <Link to="/" className="text-2xl font-bold text-brand-primary">
          FlowLoop
        </Link>
        <div className="flex gap-2">
          <button className="btn btn-secondary">Export Data</button>
          <Link to="/session" className="btn btn-primary">
            New Session
          </Link>
        </div>
      </header>

      {/* Stats Overview */}
      <div className="grid gap-4 md:grid-cols-4 mb-6">
        <div className="card">
          <p className="text-sm text-gray-400">Total Sessions</p>
          <p className="text-3xl font-bold">0</p>
        </div>
        <div className="card">
          <p className="text-sm text-gray-400">Tasks Completed</p>
          <p className="text-3xl font-bold">0</p>
        </div>
        <div className="card">
          <p className="text-sm text-gray-400">Average Accuracy</p>
          <p className="text-3xl font-bold">0%</p>
        </div>
        <div className="card">
          <p className="text-sm text-gray-400">Total Time</p>
          <p className="text-3xl font-bold">0h 0m</p>
        </div>
      </div>

      {/* Charts Grid */}
      <div className="grid gap-4 md:grid-cols-2 mb-6">
        <div className="card">
          <h2 className="text-lg font-semibold mb-3">Attention Over Time</h2>
          <div className="aspect-video bg-gray-700 rounded-lg flex items-center justify-center">
            <p className="text-gray-500">Line chart placeholder</p>
          </div>
        </div>

        <div className="card">
          <h2 className="text-lg font-semibold mb-3">Performance vs Emotional State</h2>
          <div className="aspect-video bg-gray-700 rounded-lg flex items-center justify-center">
            <p className="text-gray-500">Scatter plot placeholder</p>
          </div>
        </div>

        <div className="card">
          <h2 className="text-lg font-semibold mb-3">Difficulty Progression</h2>
          <div className="aspect-video bg-gray-700 rounded-lg flex items-center justify-center">
            <p className="text-gray-500">Area chart placeholder</p>
          </div>
        </div>

        <div className="card">
          <h2 className="text-lg font-semibold mb-3">Task Type Distribution</h2>
          <div className="aspect-video bg-gray-700 rounded-lg flex items-center justify-center">
            <p className="text-gray-500">Pie chart placeholder</p>
          </div>
        </div>
      </div>

      {/* Session History */}
      <div className="card">
        <h2 className="text-lg font-semibold mb-3">Session History</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead>
              <tr className="border-b border-gray-700">
                <th className="pb-2 text-gray-400 font-medium">Date</th>
                <th className="pb-2 text-gray-400 font-medium">Duration</th>
                <th className="pb-2 text-gray-400 font-medium">Tasks</th>
                <th className="pb-2 text-gray-400 font-medium">Accuracy</th>
                <th className="pb-2 text-gray-400 font-medium">Avg Attention</th>
                <th className="pb-2 text-gray-400 font-medium">Actions</th>
              </tr>
            </thead>
            <tbody>
              <tr className="text-gray-500">
                <td colSpan={6} className="py-8 text-center">
                  No sessions yet. Start a session to see your history.
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
