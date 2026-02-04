/**
 * useWebcam Hook
 *
 * Custom hook for managing webcam stream acquisition and cleanup.
 * Handles permission requests, device enumeration, and error states.
 */

import { useState, useEffect, useRef, useCallback } from 'react'

// =============================================================================
// Types
// =============================================================================

export interface WebcamDevice {
    deviceId: string
    label: string
}

export interface UseWebcamOptions {
    deviceId?: string
    width?: number
    height?: number
    facingMode?: 'user' | 'environment'
}

export interface UseWebcamState {
    stream: MediaStream | null
    isLoading: boolean
    error: string | null
    isActive: boolean
    devices: WebcamDevice[]
}

export interface UseWebcamReturn extends UseWebcamState {
    startCamera: () => Promise<void>
    stopCamera: () => void
    switchDevice: (deviceId: string) => Promise<void>
    refreshDevices: () => Promise<void>
}

const DEFAULT_OPTIONS: Required<UseWebcamOptions> = {
    deviceId: '',
    width: 640,
    height: 480,
    facingMode: 'user',
}

// =============================================================================
// Hook Implementation
// =============================================================================

export function useWebcam(options: UseWebcamOptions = {}): UseWebcamReturn {
    const opts = { ...DEFAULT_OPTIONS, ...options }

    const [stream, setStream] = useState<MediaStream | null>(null)
    const [isLoading, setIsLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [devices, setDevices] = useState<WebcamDevice[]>([])

    const currentDeviceId = useRef<string>(opts.deviceId)

    /**
     * Get list of available video input devices
     */
    const refreshDevices = useCallback(async () => {
        try {
            const allDevices = await navigator.mediaDevices.enumerateDevices()
            const videoDevices = allDevices
                .filter((device) => device.kind === 'videoinput')
                .map((device) => ({
                    deviceId: device.deviceId,
                    label: device.label || `Camera ${device.deviceId.slice(0, 8)}`,
                }))
            setDevices(videoDevices)
        } catch (err) {
            console.error('Failed to enumerate devices:', err)
        }
    }, [])

    /**
     * Start camera stream
     */
    const startCamera = useCallback(async () => {
        setIsLoading(true)
        setError(null)

        try {
            // Build constraints
            const constraints: MediaStreamConstraints = {
                video: {
                    width: { ideal: opts.width },
                    height: { ideal: opts.height },
                    facingMode: currentDeviceId.current ? undefined : opts.facingMode,
                    deviceId: currentDeviceId.current
                        ? { exact: currentDeviceId.current }
                        : undefined,
                },
                audio: false,
            }

            const mediaStream = await navigator.mediaDevices.getUserMedia(constraints)
            setStream(mediaStream)

            // Update device list after permission granted
            await refreshDevices()

            // Store the actual device ID being used
            const videoTrack = mediaStream.getVideoTracks()[0]
            if (videoTrack) {
                const settings = videoTrack.getSettings()
                if (settings.deviceId) {
                    currentDeviceId.current = settings.deviceId
                }
            }
        } catch (err) {
            const errorMessage = getErrorMessage(err)
            setError(errorMessage)
            console.error('Failed to start camera:', err)
        } finally {
            setIsLoading(false)
        }
    }, [opts.width, opts.height, opts.facingMode, refreshDevices])

    /**
     * Stop camera stream
     */
    const stopCamera = useCallback(() => {
        if (stream) {
            stream.getTracks().forEach((track) => {
                track.stop()
            })
            setStream(null)
        }
    }, [stream])

    /**
     * Switch to a different camera device
     */
    const switchDevice = useCallback(async (deviceId: string) => {
        stopCamera()
        currentDeviceId.current = deviceId
        await startCamera()
    }, [stopCamera, startCamera])

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (stream) {
                stream.getTracks().forEach((track) => track.stop())
            }
        }
    }, [stream])

    // Initial device enumeration
    useEffect(() => {
        refreshDevices()
    }, [refreshDevices])

    return {
        stream,
        isLoading,
        error,
        isActive: stream !== null,
        devices,
        startCamera,
        stopCamera,
        switchDevice,
        refreshDevices,
    }
}

// =============================================================================
// Helpers
// =============================================================================

function getErrorMessage(error: unknown): string {
    if (error instanceof DOMException) {
        switch (error.name) {
            case 'NotAllowedError':
                return 'Camera permission denied. Please allow camera access in your browser settings.'
            case 'NotFoundError':
                return 'No camera device found. Please connect a camera and try again.'
            case 'NotReadableError':
                return 'Camera is in use by another application. Please close other apps using the camera.'
            case 'OverconstrainedError':
                return 'Camera does not support requested settings. Try with default settings.'
            case 'SecurityError':
                return 'Camera access is not allowed in this context. Ensure you are using HTTPS.'
            default:
                return `Camera error: ${error.message}`
        }
    }

    if (error instanceof Error) {
        return error.message
    }

    return 'An unknown error occurred while accessing the camera.'
}

/**
 * Check if camera is supported on this device/browser
 */
export function isCameraSupported(): boolean {
    return !!(
        navigator.mediaDevices &&
        typeof navigator.mediaDevices.getUserMedia === 'function'
    )
}

/**
 * Check camera permission status without triggering a prompt
 */
export async function checkCameraPermission(): Promise<'granted' | 'denied' | 'prompt'> {
    try {
        const result = await navigator.permissions.query({ name: 'camera' as PermissionName })
        return result.state
    } catch {
        // Permission API might not be available
        return 'prompt'
    }
}
