/**
 * Geometric Feature Extraction from MediaPipe Face Mesh Landmarks
 *
 * MediaPipe Face Mesh provides 468 landmarks with normalized coordinates (0-1).
 * This module extracts attention/frustration proxies from these landmarks.
 *
 * Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
 */

import type { GeometricFeatures, Vector3 } from '../../types'

// =============================================================================
// Landmark Indices (MediaPipe Face Mesh)
// =============================================================================

// Eye landmarks for EAR calculation
const LEFT_EYE = {
    outer: 33,
    inner: 133,
    top1: 159,
    top2: 158,
    bottom1: 145,
    bottom2: 153,
}

const RIGHT_EYE = {
    outer: 362,
    inner: 263,
    top1: 386,
    top2: 385,
    bottom1: 380,
    bottom2: 374,
}

// Eyebrow landmarks
const LEFT_EYEBROW = {
    inner: 107,
    outer: 70,
    top: 105,
}

const RIGHT_EYEBROW = {
    inner: 336,
    outer: 300,
    top: 334,
}

// Key face points for head pose estimation
const FACE_POINTS = {
    noseTip: 1,
    chin: 152,
    leftEyeOuter: 33,
    rightEyeOuter: 263,
    leftMouth: 61,
    rightMouth: 291,
    foreheadTop: 10,
    noseBase: 168,
}

// Mouth landmarks
const MOUTH = {
    upperLipTop: 13,
    lowerLipBottom: 14,
    leftCorner: 61,
    rightCorner: 291,
    upperLipCenter: 0,
    lowerLipCenter: 17,
}

// =============================================================================
// Helper Functions
// =============================================================================

interface Landmark {
    x: number
    y: number
    z: number
}

/**
 * Calculate Euclidean distance between two 2D points
 */
function distance2D(p1: Landmark, p2: Landmark): number {
    const dx = p1.x - p2.x
    const dy = p1.y - p2.y
    return Math.sqrt(dx * dx + dy * dy)
}

/**
 * Calculate Euclidean distance between two 3D points
 */
function distance3D(p1: Landmark, p2: Landmark): number {
    const dx = p1.x - p2.x
    const dy = p1.y - p2.y
    const dz = p1.z - p2.z
    return Math.sqrt(dx * dx + dy * dy + dz * dz)
}

/**
 * Get landmark by index from flattened array
 */
function getLandmark(landmarks: number[], index: number): Landmark {
    const base = index * 3
    return {
        x: landmarks[base],
        y: landmarks[base + 1],
        z: landmarks[base + 2],
    }
}

// =============================================================================
// Feature Extraction Functions
// =============================================================================

/**
 * Calculate Eye Aspect Ratio (EAR)
 *
 * EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
 *
 * Where p1-p4 are horizontal eye landmarks and p2,p3,p5,p6 are vertical.
 * EAR drops significantly when eye closes (blink detection).
 *
 * @returns EAR value, typically 0.2-0.4 for open eyes, <0.2 for closed
 */
function calculateEAR(landmarks: number[], eye: typeof LEFT_EYE): number {
    const outer = getLandmark(landmarks, eye.outer)
    const inner = getLandmark(landmarks, eye.inner)
    const top1 = getLandmark(landmarks, eye.top1)
    const top2 = getLandmark(landmarks, eye.top2)
    const bottom1 = getLandmark(landmarks, eye.bottom1)
    const bottom2 = getLandmark(landmarks, eye.bottom2)

    // Vertical eye distances
    const v1 = distance2D(top1, bottom1)
    const v2 = distance2D(top2, bottom2)

    // Horizontal eye distance
    const h = distance2D(outer, inner)

    if (h === 0) return 0.3 // Avoid division by zero

    return (v1 + v2) / (2 * h)
}

/**
 * Estimate head pose (pitch, yaw, roll) from face landmarks
 *
 * Uses simplified geometric approach based on key face points.
 * Returns angles in radians.
 */
function estimateHeadPose(landmarks: number[]): { pitch: number; yaw: number; roll: number } {
    const noseTip = getLandmark(landmarks, FACE_POINTS.noseTip)
    const noseBase = getLandmark(landmarks, FACE_POINTS.noseBase)
    const chin = getLandmark(landmarks, FACE_POINTS.chin)
    const leftEye = getLandmark(landmarks, FACE_POINTS.leftEyeOuter)
    const rightEye = getLandmark(landmarks, FACE_POINTS.rightEyeOuter)
    const foreheadTop = getLandmark(landmarks, FACE_POINTS.foreheadTop)

    // Pitch (nodding up/down): angle of nose relative to vertical face axis
    // Positive = looking up, Negative = looking down
    const faceHeight = distance2D(foreheadTop, chin)
    const noseDrop = noseTip.y - noseBase.y
    const pitch = faceHeight > 0 ? Math.atan2(noseDrop * 2, faceHeight) : 0

    // Yaw (turning left/right): asymmetry of face width from nose
    // Positive = turning right, Negative = turning left
    const leftDist = distance2D(noseTip, leftEye)
    const rightDist = distance2D(noseTip, rightEye)
    const totalDist = leftDist + rightDist
    const yaw = totalDist > 0 ? Math.atan2(leftDist - rightDist, totalDist) * 2 : 0

    // Roll (tilting sideways): angle of eye line from horizontal
    // Positive = tilting right, Negative = tilting left
    const eyeDy = rightEye.y - leftEye.y
    const eyeDx = rightEye.x - leftEye.x
    const roll = Math.atan2(eyeDy, eyeDx)

    return { pitch, yaw, roll }
}

/**
 * Calculate brow furrow score (frustration proxy)
 *
 * Measures how close the inner eyebrows are to each other.
 * Higher score = more furrowed = more likely frustrated/concentrated.
 *
 * @returns Score 0-1, where 1 = maximum furrow
 */
function calculateBrowFurrow(landmarks: number[]): number {
    const leftInner = getLandmark(landmarks, LEFT_EYEBROW.inner)
    const rightInner = getLandmark(landmarks, RIGHT_EYEBROW.inner)
    const leftOuter = getLandmark(landmarks, LEFT_EYEBROW.outer)
    const rightOuter = getLandmark(landmarks, RIGHT_EYEBROW.outer)

    // Distance between inner eyebrows (normalized by face width)
    const innerDist = distance2D(leftInner, rightInner)
    const faceWidth = distance2D(leftOuter, rightOuter)

    if (faceWidth === 0) return 0

    // Normalize: smaller distance = higher furrow score
    // Typical ratio is 0.15-0.25 for neutral, <0.15 for furrowed
    const ratio = innerDist / faceWidth
    const normalized = 1 - Math.min(1, Math.max(0, (ratio - 0.1) / 0.2))

    return normalized
}

/**
 * Calculate brow raise score (surprise/attention proxy)
 *
 * Measures vertical position of eyebrows relative to eyes.
 * Higher score = raised eyebrows.
 *
 * @returns Score 0-1, where 1 = maximum raise
 */
function calculateBrowRaise(landmarks: number[]): number {
    const leftBrow = getLandmark(landmarks, LEFT_EYEBROW.top)
    const rightBrow = getLandmark(landmarks, RIGHT_EYEBROW.top)
    const leftEye = getLandmark(landmarks, LEFT_EYE.top1)
    const rightEye = getLandmark(landmarks, RIGHT_EYE.top1)

    // Average vertical distance between brow and eye
    const leftDist = leftBrow.y - leftEye.y
    const rightDist = rightBrow.y - rightEye.y
    const avgDist = (leftDist + rightDist) / 2

    // Normalize: more negative = higher brows (remember y increases downward)
    // Typical range: -0.03 to -0.08
    const normalized = Math.min(1, Math.max(0, (-avgDist - 0.02) / 0.06))

    return normalized
}

/**
 * Calculate mouth openness (yawning/surprise detection)
 *
 * @returns Score 0-1, where 1 = fully open
 */
function calculateMouthOpenness(landmarks: number[]): number {
    const upperLip = getLandmark(landmarks, MOUTH.upperLipTop)
    const lowerLip = getLandmark(landmarks, MOUTH.lowerLipBottom)
    const leftCorner = getLandmark(landmarks, MOUTH.leftCorner)
    const rightCorner = getLandmark(landmarks, MOUTH.rightCorner)

    const mouthHeight = distance2D(upperLip, lowerLip)
    const mouthWidth = distance2D(leftCorner, rightCorner)

    if (mouthWidth === 0) return 0

    // Normalize by mouth width
    const ratio = mouthHeight / mouthWidth

    // Typical range: 0.05-0.1 for closed, 0.3+ for open
    return Math.min(1, Math.max(0, (ratio - 0.05) / 0.3))
}

/**
 * Calculate lip corner pull (smile/frown detection)
 *
 * @returns Score -1 to 1, where 1 = smile, -1 = frown
 */
function calculateLipCornerPull(landmarks: number[]): number {
    const leftCorner = getLandmark(landmarks, MOUTH.leftCorner)
    const rightCorner = getLandmark(landmarks, MOUTH.rightCorner)
    const upperCenter = getLandmark(landmarks, MOUTH.upperLipCenter)
    const lowerCenter = getLandmark(landmarks, MOUTH.lowerLipCenter)

    // Average y position of corners relative to mouth center
    const mouthCenterY = (upperCenter.y + lowerCenter.y) / 2
    const avgCornerY = (leftCorner.y + rightCorner.y) / 2

    // Negative value = corners above center = smile
    // Positive value = corners below center = frown
    const pull = mouthCenterY - avgCornerY

    // Normalize to -1 to 1 range
    return Math.min(1, Math.max(-1, pull * 50))
}

/**
 * Estimate gaze direction from eye landmarks
 *
 * Simplified estimation based on iris position within eye.
 * For accurate gaze, would need iris landmarks (available with refineLandmarks=true).
 *
 * @returns Normalized gaze direction vector
 */
function estimateGaze(landmarks: number[]): { direction: Vector3; confidence: number } {
    // Use eye centers as proxy for gaze
    const leftEyeCenter = {
        x: (getLandmark(landmarks, LEFT_EYE.outer).x + getLandmark(landmarks, LEFT_EYE.inner).x) / 2,
        y: (getLandmark(landmarks, LEFT_EYE.top1).y + getLandmark(landmarks, LEFT_EYE.bottom1).y) / 2,
        z: getLandmark(landmarks, LEFT_EYE.outer).z,
    }

    const rightEyeCenter = {
        x: (getLandmark(landmarks, RIGHT_EYE.outer).x + getLandmark(landmarks, RIGHT_EYE.inner).x) / 2,
        y: (getLandmark(landmarks, RIGHT_EYE.top1).y + getLandmark(landmarks, RIGHT_EYE.bottom1).y) / 2,
        z: getLandmark(landmarks, RIGHT_EYE.outer).z,
    }

    // Average eye position
    const avgX = (leftEyeCenter.x + rightEyeCenter.x) / 2 - 0.5 // Center at 0
    const avgY = (leftEyeCenter.y + rightEyeCenter.y) / 2 - 0.5
    const avgZ = (leftEyeCenter.z + rightEyeCenter.z) / 2

    // Normalize to unit vector
    const magnitude = Math.sqrt(avgX * avgX + avgY * avgY + avgZ * avgZ)

    return {
        direction: {
            x: magnitude > 0 ? avgX / magnitude : 0,
            y: magnitude > 0 ? avgY / magnitude : 0,
            z: magnitude > 0 ? avgZ / magnitude : 1,
        },
        confidence: 0.7, // Fixed confidence for simplified estimation
    }
}

// =============================================================================
// Motion Tracking (requires previous frame)
// =============================================================================

let previousHeadPosition: Landmark | null = null
let previousTimestamp = 0
const headPositionHistory: number[] = []
const MOTION_HISTORY_SIZE = 30 // ~1 second at 30fps

/**
 * Calculate head motion metrics
 *
 * @returns velocity (units/second) and variance (restlessness measure)
 */
function calculateHeadMotion(
    landmarks: number[],
    timestamp: number
): { velocity: number; variance: number } {
    const noseTip = getLandmark(landmarks, FACE_POINTS.noseTip)

    const deltaTime = (timestamp - previousTimestamp) / 1000 // Convert to seconds

    let velocity = 0
    if (previousHeadPosition && deltaTime > 0) {
        const distance = distance3D(noseTip, previousHeadPosition)
        velocity = distance / deltaTime
    }

    // Track motion history for variance calculation
    headPositionHistory.push(velocity)
    if (headPositionHistory.length > MOTION_HISTORY_SIZE) {
        headPositionHistory.shift()
    }

    // Calculate variance
    let variance = 0
    if (headPositionHistory.length > 1) {
        const mean = headPositionHistory.reduce((a, b) => a + b, 0) / headPositionHistory.length
        variance =
            headPositionHistory.reduce((sum, v) => sum + (v - mean) ** 2, 0) /
            headPositionHistory.length
    }

    previousHeadPosition = { ...noseTip }
    previousTimestamp = timestamp

    return { velocity, variance }
}

/**
 * Reset motion tracking state (call on session start)
 */
export function resetMotionTracking(): void {
    previousHeadPosition = null
    previousTimestamp = 0
    headPositionHistory.length = 0
}

// =============================================================================
// Main Export
// =============================================================================

/**
 * Extract all geometric features from MediaPipe face landmarks
 *
 * @param landmarks Flattened array of 468 landmarks (x, y, z) = 1404 values
 * @param timestamp Current frame timestamp in milliseconds
 * @returns GeometricFeatures object with all calculated values
 */
export function extractGeometricFeatures(
    landmarks: number[],
    timestamp: number
): GeometricFeatures {
    // Validate input
    if (landmarks.length !== 468 * 3) {
        console.warn(`Invalid landmark count: expected 1404, got ${landmarks.length}`)
        return getDefaultFeatures()
    }

    const leftEAR = calculateEAR(landmarks, LEFT_EYE)
    const rightEAR = calculateEAR(landmarks, RIGHT_EYE)
    const { pitch, yaw, roll } = estimateHeadPose(landmarks)
    const { direction, confidence } = estimateGaze(landmarks)
    const { velocity, variance } = calculateHeadMotion(landmarks, timestamp)

    return {
        gazeDirection: direction,
        gazeConfidence: confidence,
        headPitch: pitch,
        headYaw: yaw,
        headRoll: roll,
        leftEyeAspectRatio: leftEAR,
        rightEyeAspectRatio: rightEAR,
        blinkRate: 0, // Requires temporal tracking, computed elsewhere
        browFurrowScore: calculateBrowFurrow(landmarks),
        browRaiseScore: calculateBrowRaise(landmarks),
        mouthOpenness: calculateMouthOpenness(landmarks),
        lipCornerPull: calculateLipCornerPull(landmarks),
        headMotionVelocity: velocity,
        headMotionVariance: variance,
    }
}

/**
 * Get default/neutral features (used when face not detected)
 */
export function getDefaultFeatures(): GeometricFeatures {
    return {
        gazeDirection: { x: 0, y: 0, z: 1 },
        gazeConfidence: 0,
        headPitch: 0,
        headYaw: 0,
        headRoll: 0,
        leftEyeAspectRatio: 0.3,
        rightEyeAspectRatio: 0.3,
        blinkRate: 0,
        browFurrowScore: 0,
        browRaiseScore: 0,
        mouthOpenness: 0,
        lipCornerPull: 0,
        headMotionVelocity: 0,
        headMotionVariance: 0,
    }
}

/**
 * Convert eye aspect ratios to a simple "engaged gaze" score
 *
 * @returns Score 0-1, where 1 = eyes fully open and engaged
 */
export function calculateEngagementFromEyes(leftEAR: number, rightEAR: number): number {
    const avgEAR = (leftEAR + rightEAR) / 2

    // EAR thresholds:
    // < 0.15: eyes closed (blink or drowsy)
    // 0.15-0.25: partially open
    // > 0.25: fully open

    if (avgEAR < 0.15) return 0
    if (avgEAR > 0.35) return 1

    return (avgEAR - 0.15) / 0.2
}
