import { Link } from 'react-router-dom'
import { useState } from 'react'

export default function SettingsPage() {
  const [settings, setSettings] = useState({
    cameraDevice: '',
    enableDebugOverlay: true,
    enablePerformanceMonitoring: false,
    smoothingAlpha: 0.2,
    adaptationSensitivity: 5,
    autoDeleteSessions: 30,
  })

  const updateSetting = <K extends keyof typeof settings>(
    key: K,
    value: (typeof settings)[K]
  ) => {
    setSettings((prev) => ({ ...prev, [key]: value }))
  }

  return (
    <div className="min-h-screen p-4">
      {/* Header */}
      <header className="flex items-center justify-between mb-6">
        <Link to="/" className="text-2xl font-bold text-brand-primary">
          FlowLoop
        </Link>
        <Link to="/session" className="btn btn-secondary">
          Back to Session
        </Link>
      </header>

      <div className="max-w-2xl mx-auto space-y-6">
        <h1 className="text-3xl font-bold">Settings</h1>

        {/* Camera Settings */}
        <div className="card">
          <h2 className="text-lg font-semibold mb-4">Camera</h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">Camera Device</label>
              <select
                className="input w-full"
                value={settings.cameraDevice}
                onChange={(e) => updateSetting('cameraDevice', e.target.value)}
              >
                <option value="">Default Camera</option>
              </select>
            </div>
          </div>
        </div>

        {/* ML Settings */}
        <div className="card">
          <h2 className="text-lg font-semibold mb-4">ML Pipeline</h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">
                Smoothing Alpha ({settings.smoothingAlpha})
              </label>
              <input
                type="range"
                min="0.05"
                max="0.5"
                step="0.05"
                value={settings.smoothingAlpha}
                onChange={(e) => updateSetting('smoothingAlpha', parseFloat(e.target.value))}
                className="w-full"
              />
              <p className="text-xs text-gray-500 mt-1">
                Higher = more responsive, Lower = smoother predictions
              </p>
            </div>
          </div>
        </div>

        {/* Adaptation Settings */}
        <div className="card">
          <h2 className="text-lg font-semibold mb-4">Adaptation</h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">
                Adaptation Sensitivity ({settings.adaptationSensitivity}/10)
              </label>
              <input
                type="range"
                min="1"
                max="10"
                value={settings.adaptationSensitivity}
                onChange={(e) => updateSetting('adaptationSensitivity', parseInt(e.target.value))}
                className="w-full"
              />
              <p className="text-xs text-gray-500 mt-1">
                How quickly difficulty changes in response to cognitive state
              </p>
            </div>
          </div>
        </div>

        {/* Debug Settings */}
        <div className="card">
          <h2 className="text-lg font-semibold mb-4">Developer</h2>
          <div className="space-y-4">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={settings.enableDebugOverlay}
                onChange={(e) => updateSetting('enableDebugOverlay', e.target.checked)}
                className="w-4 h-4 rounded"
              />
              <span>Show debug overlay on webcam</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={settings.enablePerformanceMonitoring}
                onChange={(e) => updateSetting('enablePerformanceMonitoring', e.target.checked)}
                className="w-4 h-4 rounded"
              />
              <span>Enable performance monitoring</span>
            </label>
          </div>
        </div>

        {/* Data Settings */}
        <div className="card">
          <h2 className="text-lg font-semibold mb-4">Data Management</h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">
                Auto-delete sessions after (days)
              </label>
              <input
                type="number"
                min="1"
                max="365"
                value={settings.autoDeleteSessions}
                onChange={(e) => updateSetting('autoDeleteSessions', parseInt(e.target.value))}
                className="input w-24"
              />
            </div>
            <div className="pt-4 border-t border-gray-700 flex gap-2">
              <button className="btn btn-secondary">Export All Data</button>
              <button className="btn btn-danger">Clear All Data</button>
            </div>
          </div>
        </div>

        {/* Privacy Notice */}
        <div className="card bg-gray-900 border border-gray-700">
          <h2 className="text-lg font-semibold mb-2">Privacy Notice</h2>
          <p className="text-sm text-gray-400">
            All video processing happens locally on your device. Your webcam feed is never
            transmitted to any server. Session data is stored locally and can be deleted
            at any time.
          </p>
        </div>
      </div>
    </div>
  )
}
