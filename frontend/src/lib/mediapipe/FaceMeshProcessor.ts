/**
 * MediaPipe Face Mesh Processor (Vision API)
 *
 * Uses the newer @mediapipe/tasks-vision package which is more stable
 * and compatible with modern bundlers like Vite.
 *
 * Usage:
 *   const processor = new FaceMeshProcessor()
 *   await processor.initialize()
 *   processor.onResults((landmarks, features) => { ... })
 *   processor.processFrame(videoElement)
 */

import { FaceLandmarker, FilesetResolver, FaceLandmarkerResult } from '@mediapipe/tasks-vision'
import type { GeometricFeatures } from '../../types'
import { extractGeometricFeatures, getDefaultFeatures, resetMotionTracking } from '../features/landmarkFeatures'

// =============================================================================
// Types
// =============================================================================

export interface FaceMeshResult {
    landmarks: number[] // Flattened 478 * 3 values (Vision API has more landmarks)
    features: GeometricFeatures
    faceDetected: boolean
    timestamp: number
}

export type FaceMeshCallback = (result: FaceMeshResult) => void

export interface FaceMeshProcessorOptions {
    numFaces?: number
    minDetectionConfidence?: number
    minTrackingConfidence?: number
    minPresenceConfidence?: number
}

const DEFAULT_OPTIONS: Required<FaceMeshProcessorOptions> = {
    numFaces: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
    minPresenceConfidence: 0.5,
}

// CDN URL for MediaPipe Vision WASM files
const VISION_WASM_CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm'

// =============================================================================
// FaceMeshProcessor Class
// =============================================================================

export class FaceMeshProcessor {
    private faceLandmarker: FaceLandmarker | null = null
    private callback: FaceMeshCallback | null = null
    private options: Required<FaceMeshProcessorOptions>
    private isInitialized = false
    private isProcessing = false
    private frameCount = 0
    private _lastProcessTime = 0
    private lastFpsUpdate = 0
    private fpsFrameCount = 0

    constructor(options: FaceMeshProcessorOptions = {}) {
        this.options = { ...DEFAULT_OPTIONS, ...options }
    }

    /**
     * Initialize MediaPipe Face Landmarker
     * Must be called before processing frames
     */
    async initialize(): Promise<void> {
        if (this.isInitialized) {
            console.warn('FaceMeshProcessor already initialized')
            return
        }

        try {
            console.log('Loading MediaPipe Vision files...')

            // Load the WASM fileset
            const vision = await FilesetResolver.forVisionTasks(VISION_WASM_CDN)

            console.log('Creating FaceLandmarker...')

            // Create the FaceLandmarker
            this.faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
                baseOptions: {
                    modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
                    delegate: 'GPU', // Use GPU if available, falls back to CPU
                },
                outputFaceBlendshapes: false, // We don't need blendshapes
                runningMode: 'VIDEO',
                numFaces: this.options.numFaces,
                minFaceDetectionConfidence: this.options.minDetectionConfidence,
                minFacePresenceConfidence: this.options.minPresenceConfidence,
                minTrackingConfidence: this.options.minTrackingConfidence,
            })

            this.isInitialized = true
            console.log('MediaPipe FaceLandmarker initialized successfully')
        } catch (error) {
            console.error('Failed to initialize MediaPipe FaceLandmarker:', error)
            throw new Error('Failed to initialize face detection')
        }
    }

    /**
     * Set callback for receiving results
     */
    onResults(callback: FaceMeshCallback): void {
        this.callback = callback
    }

    /**
     * Process a video frame
     */
    async processFrame(
        source: HTMLVideoElement | HTMLCanvasElement | HTMLImageElement
    ): Promise<void> {
        if (!this.isInitialized || !this.faceLandmarker) {
            console.warn('FaceMeshProcessor not initialized')
            return
        }

        // Skip if already processing (prevent frame accumulation)
        if (this.isProcessing) {
            return
        }

        // Check if video is ready
        if (source instanceof HTMLVideoElement) {
            if (source.readyState < 2) {
                return // Video not ready
            }
        }

        this.isProcessing = true
        const timestamp = performance.now()

        try {
            // Process the frame
            const results = this.faceLandmarker.detectForVideo(source, timestamp)
            this.handleResults(results, timestamp)
        } catch (error) {
            console.error('Error processing frame:', error)
        } finally {
            this.isProcessing = false
        }
    }

    /**
     * Handle results from MediaPipe
     */
    private handleResults(results: FaceLandmarkerResult, timestamp: number): void {
        this.frameCount++
        this.fpsFrameCount++

        // Calculate FPS every second
        if (timestamp - this.lastFpsUpdate >= 1000) {
            const fps = Math.round(this.fpsFrameCount * 1000 / (timestamp - this.lastFpsUpdate))
            if (this.frameCount % 30 === 0) {
                console.debug(`FaceMesh processing: ${fps} FPS`)
            }
            this.lastFpsUpdate = timestamp
            this.fpsFrameCount = 0
        }

        this._lastProcessTime = timestamp

        if (!this.callback) return

        if (!results.faceLandmarks || results.faceLandmarks.length === 0) {
            // No face detected
            this.callback({
                landmarks: [],
                features: getDefaultFeatures(),
                faceDetected: false,
                timestamp,
            })
            return
        }

        // Get first face landmarks
        const faceLandmarks = results.faceLandmarks[0]

        // Flatten landmarks to array: [x1, y1, z1, x2, y2, z2, ...]
        // Vision API provides 478 landmarks (more than old API's 468)
        const flatLandmarks: number[] = []
        for (const point of faceLandmarks) {
            flatLandmarks.push(point.x, point.y, point.z)
        }

        // For geometric features, we only use first 468 landmarks
        // to maintain compatibility with our feature extraction
        const landmarks468 = flatLandmarks.slice(0, 468 * 3)

        // Extract geometric features
        const features = extractGeometricFeatures(landmarks468, timestamp)

        this.callback({
            landmarks: flatLandmarks,
            features,
            faceDetected: true,
            timestamp,
        })
    }

    /**
     * Reset state (call when starting new session)
     */
    reset(): void {
        this.frameCount = 0
        this._lastProcessTime = 0
        this.lastFpsUpdate = 0
        this.fpsFrameCount = 0
        resetMotionTracking()
    }

    /**
     * Clean up resources
     */
    async close(): Promise<void> {
        if (this.faceLandmarker) {
            this.faceLandmarker.close()
            this.faceLandmarker = null
        }
        this.isInitialized = false
        this.callback = null
        this.reset()
    }

    /**
     * Check if processor is ready
     */
    get ready(): boolean {
        return this.isInitialized
    }

    /**
     * Get current frame count (for debugging)
     */
    get processedFrames(): number {
        return this.frameCount
    }

    /**
     * Get last process timestamp (for debugging)
     */
    get lastProcessTime(): number {
        return this._lastProcessTime
    }
}

// =============================================================================
// Singleton Instance (optional, for shared processor)
// =============================================================================

let sharedProcessor: FaceMeshProcessor | null = null

/**
 * Get shared FaceMesh processor instance
 * Useful when you want to reuse the same processor across components
 */
export async function getSharedProcessor(): Promise<FaceMeshProcessor> {
    if (!sharedProcessor) {
        sharedProcessor = new FaceMeshProcessor()
        await sharedProcessor.initialize()
    }
    return sharedProcessor
}

/**
 * Release shared processor (call on app unmount)
 */
export async function releaseSharedProcessor(): Promise<void> {
    if (sharedProcessor) {
        await sharedProcessor.close()
        sharedProcessor = null
    }
}
