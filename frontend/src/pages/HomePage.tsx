import { Link } from 'react-router-dom'

export default function HomePage() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-8">
      <div className="max-w-2xl text-center">
        <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-brand-primary to-brand-secondary bg-clip-text text-transparent">
          FlowLoop
        </h1>
        <p className="text-xl text-gray-400 mb-8">
          Adaptive cognitive training that responds to your attention and emotional state in real-time.
        </p>

        <div className="grid gap-4 md:grid-cols-2 mb-8">
          <div className="card">
            <h3 className="text-lg font-semibold mb-2">Real-Time Adaptation</h3>
            <p className="text-gray-400 text-sm">
              Tasks adjust difficulty based on your attention level and frustration signals.
            </p>
          </div>
          <div className="card">
            <h3 className="text-lg font-semibold mb-2">Privacy First</h3>
            <p className="text-gray-400 text-sm">
              All processing happens locally. Your video never leaves your device.
            </p>
          </div>
          <div className="card">
            <h3 className="text-lg font-semibold mb-2">Visualize Progress</h3>
            <p className="text-gray-400 text-sm">
              See real-time charts of your attention, performance, and difficulty.
            </p>
          </div>
          <div className="card">
            <h3 className="text-lg font-semibold mb-2">Research Ready</h3>
            <p className="text-gray-400 text-sm">
              Export detailed logs for analysis and scientific study.
            </p>
          </div>
        </div>

        <div className="flex gap-4 justify-center">
          <Link to="/session" className="btn btn-primary text-lg px-8 py-3">
            Start Session
          </Link>
          <Link to="/dashboard" className="btn btn-secondary text-lg px-8 py-3">
            View Dashboard
          </Link>
        </div>
      </div>
    </div>
  )
}
