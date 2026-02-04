import { Link } from 'react-router-dom'

export default function HomePage() {
  return (
    <div className="min-h-screen bg-canvas text-ink flex flex-col items-center justify-center p-8">
      <div className="max-w-2xl text-center">
        <h1 className="text-4xl font-semibold mb-4">
          FlowLoop
        </h1>
        <p className="text-lg text-neutral-500 mb-10 max-w-lg mx-auto">
          Adaptive cognitive training that responds to your attention and emotional state in real-time.
        </p>

        <div className="grid gap-5 md:grid-cols-2 mb-10">
          <div className="card text-left">
            <h3 className="font-medium mb-2">Real-Time Adaptation</h3>
            <p className="text-sm text-neutral-500">
              Tasks adjust difficulty based on your attention level and frustration signals.
            </p>
          </div>
          <div className="card text-left">
            <h3 className="font-medium mb-2">Privacy First</h3>
            <p className="text-sm text-neutral-500">
              All processing happens locally. Your video never leaves your device.
            </p>
          </div>
          <div className="card text-left">
            <h3 className="font-medium mb-2">Visualize Progress</h3>
            <p className="text-sm text-neutral-500">
              See real-time charts of your attention, performance, and difficulty.
            </p>
          </div>
          <div className="card text-left">
            <h3 className="font-medium mb-2">Research Ready</h3>
            <p className="text-sm text-neutral-500">
              Export detailed logs for analysis and scientific study.
            </p>
          </div>
        </div>

        <div className="flex gap-4 justify-center">
          <Link to="/session" className="btn btn-primary px-8 py-3">
            Start Session
          </Link>
          <Link to="/dashboard" className="btn btn-secondary px-8 py-3">
            View Dashboard
          </Link>
        </div>
      </div>
    </div>
  )
}
