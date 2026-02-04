/**
 * WebcamCapture Component
 *
 * Captures webcam video feed, processes frames through MediaPipe Face Mesh,
 * and provides face detection results to parent components.
 *
 * Features:
 * - Camera preview with landmark overlay
 * - Face detection status indicator
 * - Integration with FaceMeshProcessor
 * - Canvas for frame extraction
 */

import { useEffect, useRef, useCallback, useState } from 'react'
import { useWebcam, isCameraSupported } from './useWebcam'
import { FaceMeshProcessor, FaceMeshResult } from '../../lib/mediapipe/FaceMeshProcessor'
// Types are imported via FaceMeshResult from FaceMeshProcessor

// =============================================================================
// Types
// =============================================================================

export interface WebcamCaptureProps {
    /** Called for each processed frame with face detection results */
    onFrame?: (result: FaceMeshResult, video: HTMLVideoElement | null) => void
    /** Called when face detection status changes */
    onFaceDetectionChange?: (detected: boolean) => void
    /** Camera device ID to use (optional) */
    deviceId?: string
    /** Whether to show landmark overlay on video */
    showLandmarks?: boolean
    /** Target frame rate for processing (default: 30) */
    targetFps?: number
    /** Whether the capture is active */
    isActive?: boolean
    /** Custom class name for container */
    className?: string
}

// =============================================================================
// Component
// =============================================================================

export default function WebcamCapture({
    onFrame,
    onFaceDetectionChange,
    deviceId,
    showLandmarks = true,
    targetFps = 30,
    isActive = true,
    className = '',
}: WebcamCaptureProps) {
    const videoRef = useRef<HTMLVideoElement>(null)
    const canvasRef = useRef<HTMLCanvasElement>(null)
    const overlayCanvasRef = useRef<HTMLCanvasElement>(null)
    const processorRef = useRef<FaceMeshProcessor | null>(null)
    const animationFrameRef = useRef<number | null>(null)
    const lastProcessTimeRef = useRef<number>(0)

    const [isProcessorReady, setIsProcessorReady] = useState(false)
    const [faceDetected, setFaceDetected] = useState(false)
    const [processingFps, setProcessingFps] = useState(0)

    const { stream, isLoading, error, isActive: cameraActive, startCamera, stopCamera } = useWebcam({
        deviceId,
        width: 640,
        height: 480,
    })

    // Frame interval based on target FPS
    const frameInterval = 1000 / targetFps

    // ===========================================================================
    // Initialize MediaPipe Processor
    // ===========================================================================

    useEffect(() => {
        const initProcessor = async () => {
            processorRef.current = new FaceMeshProcessor()
            await processorRef.current.initialize()
            setIsProcessorReady(true)
        }

        initProcessor().catch((err) => {
            console.error('Failed to initialize FaceMesh processor:', err)
        })

        return () => {
            if (processorRef.current) {
                processorRef.current.close()
                processorRef.current = null
            }
        }
    }, [])

    // ===========================================================================
    // Handle Face Mesh Results
    // ===========================================================================

    const handleFaceMeshResult = useCallback(
        (result: FaceMeshResult) => {
            // Update face detection state
            if (result.faceDetected !== faceDetected) {
                setFaceDetected(result.faceDetected)
                onFaceDetectionChange?.(result.faceDetected)
            }

            // Call parent callback with video element for face crop extraction
            onFrame?.(result, videoRef.current)

            // Draw landmarks overlay
            if (showLandmarks && result.faceDetected && overlayCanvasRef.current) {
                drawLandmarks(overlayCanvasRef.current, result.landmarks)
            } else if (overlayCanvasRef.current) {
                const ctx = overlayCanvasRef.current.getContext('2d')
                ctx?.clearRect(0, 0, overlayCanvasRef.current.width, overlayCanvasRef.current.height)
            }
        },
        [faceDetected, onFrame, onFaceDetectionChange, showLandmarks]
    )

    // ===========================================================================
    // Frame Processing Loop
    // ===========================================================================

    const processFrame = useCallback(() => {
        if (!isActive || !videoRef.current || !processorRef.current?.ready) {
            animationFrameRef.current = requestAnimationFrame(processFrame)
            return
        }

        const now = performance.now()
        const elapsed = now - lastProcessTimeRef.current

        if (elapsed >= frameInterval) {
            lastProcessTimeRef.current = now - (elapsed % frameInterval)

            // Calculate actual FPS
            if (elapsed > 0) {
                setProcessingFps(Math.round(1000 / elapsed))
            }

            processorRef.current.processFrame(videoRef.current)
        }

        animationFrameRef.current = requestAnimationFrame(processFrame)
    }, [isActive, frameInterval])

    // ===========================================================================
    // Connect Video Stream
    // ===========================================================================

    useEffect(() => {
        if (stream && videoRef.current) {
            videoRef.current.srcObject = stream
        }
    }, [stream])

    // ===========================================================================
    // Start/Stop Processing Loop
    // ===========================================================================

    useEffect(() => {
        if (isProcessorReady && processorRef.current) {
            processorRef.current.onResults(handleFaceMeshResult)
        }
    }, [isProcessorReady, handleFaceMeshResult])

    useEffect(() => {
        if (isActive && cameraActive && isProcessorReady) {
            animationFrameRef.current = requestAnimationFrame(processFrame)
        }

        return () => {
            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current)
                animationFrameRef.current = null
            }
        }
    }, [isActive, cameraActive, isProcessorReady, processFrame])

    // ===========================================================================
    // Auto-Start Camera
    // ===========================================================================

    useEffect(() => {
        if (isActive && !cameraActive && !isLoading && !error) {
            startCamera()
        } else if (!isActive && cameraActive) {
            stopCamera()
        }
    }, [isActive, cameraActive, isLoading, error, startCamera, stopCamera])

    // ===========================================================================
    // Render
    // ===========================================================================

    if (!isCameraSupported()) {
        return (
            <div className={`relative bg-gray-800 rounded-lg flex items-center justify-center ${className}`}>
                <div className="text-center p-4">
                    <p className="text-red-400 font-medium">Camera Not Supported</p>
                    <p className="text-gray-500 text-sm mt-1">
                        Your browser does not support camera access.
                    </p>
                </div>
            </div>
        )
    }

    return (
        <div className={`relative bg-gray-800 rounded-lg overflow-hidden ${className}`}>
            {/* Video Element */}
            <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover"
                style={{ transform: 'scaleX(-1)' }} // Mirror for user-facing camera
            />

            {/* Hidden Canvas for Frame Extraction */}
            <canvas ref={canvasRef} className="hidden" width={640} height={480} />

            {/* Overlay Canvas for Landmarks */}
            <canvas
                ref={overlayCanvasRef}
                className="absolute inset-0 w-full h-full pointer-events-none"
                width={640}
                height={480}
                style={{ transform: 'scaleX(-1)' }}
            />

            {/* Status Overlay */}
            <div className="absolute top-2 left-2 flex items-center gap-2">
                {/* Face Detection Indicator */}
                <div
                    className={`
            flex items-center gap-1.5 px-2 py-1 rounded-full text-xs font-medium
            ${faceDetected ? 'bg-green-500/20 text-green-400' : 'bg-yellow-500/20 text-yellow-400'}
          `}
                >
                    <div
                        className={`w-2 h-2 rounded-full ${faceDetected ? 'bg-green-400' : 'bg-yellow-400'}`}
                    />
                    {faceDetected ? 'Face Detected' : 'No Face'}
                </div>

                {/* FPS Counter (debug) */}
                {isActive && (
                    <div className="px-2 py-1 rounded-full bg-gray-900/70 text-gray-400 text-xs">
                        {processingFps} FPS
                    </div>
                )}
            </div>

            {/* Loading/Error States */}
            {isLoading && (
                <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80">
                    <div className="text-center">
                        <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto" />
                        <p className="text-gray-400 text-sm mt-2">Starting camera...</p>
                    </div>
                </div>
            )}

            {error && (
                <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80">
                    <div className="text-center p-4">
                        <p className="text-red-400 font-medium">Camera Error</p>
                        <p className="text-gray-500 text-sm mt-1">{error}</p>
                        <button
                            onClick={startCamera}
                            className="mt-3 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-lg"
                        >
                            Try Again
                        </button>
                    </div>
                </div>
            )}

            {!isProcessorReady && !isLoading && !error && (
                <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80">
                    <div className="text-center">
                        <div className="w-8 h-8 border-2 border-purple-500 border-t-transparent rounded-full animate-spin mx-auto" />
                        <p className="text-gray-400 text-sm mt-2">Loading face detection model...</p>
                    </div>
                </div>
            )}
        </div>
    )
}

// =============================================================================
// Helper: Draw Landmarks
// =============================================================================

function drawLandmarks(canvas: HTMLCanvasElement, landmarks: number[]): void {
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Draw connection lines for face outline
    ctx.strokeStyle = 'rgba(0, 255, 255, 0.3)'
    ctx.lineWidth = 1

    // Draw all landmarks as small dots
    ctx.fillStyle = 'rgba(0, 255, 255, 0.5)'

    for (let i = 0; i < landmarks.length; i += 3) {
        const x = landmarks[i] * canvas.width
        const y = landmarks[i + 1] * canvas.height

        // Skip every other point for performance
        if ((i / 3) % 2 === 0) {
            ctx.beginPath()
            ctx.arc(x, y, 1, 0, Math.PI * 2)
            ctx.fill()
        }
    }

    // Highlight key features with larger dots
    const keyPoints = [
        // Eyes
        33, 133, 362, 263,
        // Nose
        1, 4, 5, 195,
        // Mouth
        61, 291, 0, 17,
        // Eyebrows
        70, 107, 336, 300,
    ]

    ctx.fillStyle = 'rgba(255, 100, 100, 0.8)'
    for (const idx of keyPoints) {
        const x = landmarks[idx * 3] * canvas.width
        const y = landmarks[idx * 3 + 1] * canvas.height

        ctx.beginPath()
        ctx.arc(x, y, 3, 0, Math.PI * 2)
        ctx.fill()
    }
}

// =============================================================================
// Named Exports
// =============================================================================

export { useWebcam, isCameraSupported } from './useWebcam'
export type { WebcamDevice, UseWebcamReturn } from './useWebcam'
