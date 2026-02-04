/**
 * Face Crop Extractor
 * 
 * Extracts a face crop from video frame using MediaPipe landmarks.
 * The crop is padded and resized to 224x224 for emotion model input.
 */

export interface FaceCropResult {
  base64: string  // Base64 encoded JPEG image (no data: prefix)
  width: number
  height: number
}

/**
 * Extract face crop from video element using landmarks bounding box.
 * 
 * @param video - Video element to capture from
 * @param landmarks - Flattened landmarks array [x1, y1, z1, x2, y2, z2, ...]
 * @param padding - Padding ratio around face (default 0.2 = 20%)
 * @param outputSize - Output image size (default 224 for emotion models)
 * @returns Face crop as base64 JPEG or null if extraction fails
 */
export function extractFaceCrop(
  video: HTMLVideoElement,
  landmarks: number[],
  padding: number = 0.3,
  outputSize: number = 224
): FaceCropResult | null {
  if (!video || landmarks.length < 468 * 3) {
    return null
  }

  const videoWidth = video.videoWidth
  const videoHeight = video.videoHeight

  if (videoWidth === 0 || videoHeight === 0) {
    return null
  }

  // Find bounding box from landmarks
  let minX = Infinity, minY = Infinity
  let maxX = -Infinity, maxY = -Infinity

  for (let i = 0; i < landmarks.length; i += 3) {
    const x = landmarks[i] * videoWidth
    const y = landmarks[i + 1] * videoHeight

    minX = Math.min(minX, x)
    maxX = Math.max(maxX, x)
    minY = Math.min(minY, y)
    maxY = Math.max(maxY, y)
  }

  // Calculate face dimensions
  const faceWidth = maxX - minX
  const faceHeight = maxY - minY

  // Add padding
  const padX = faceWidth * padding
  const padY = faceHeight * padding

  // Calculate crop region with padding
  const cropX = Math.max(0, minX - padX)
  const cropY = Math.max(0, minY - padY)
  const cropWidth = Math.min(videoWidth - cropX, faceWidth + 2 * padX)
  const cropHeight = Math.min(videoHeight - cropY, faceHeight + 2 * padY)

  // Make it square (use larger dimension)
  const squareSize = Math.max(cropWidth, cropHeight)
  const centerX = cropX + cropWidth / 2
  const centerY = cropY + cropHeight / 2

  // Adjust to square crop centered on face
  let finalX = centerX - squareSize / 2
  let finalY = centerY - squareSize / 2
  let finalSize = squareSize

  // Clamp to video bounds
  if (finalX < 0) {
    finalX = 0
  }
  if (finalY < 0) {
    finalY = 0
  }
  if (finalX + finalSize > videoWidth) {
    finalSize = videoWidth - finalX
  }
  if (finalY + finalSize > videoHeight) {
    finalSize = videoHeight - finalY
  }

  try {
    // Create canvas for cropping
    const canvas = document.createElement('canvas')
    canvas.width = outputSize
    canvas.height = outputSize
    const ctx = canvas.getContext('2d')

    if (!ctx) {
      return null
    }

    // Draw cropped and resized face
    ctx.drawImage(
      video,
      finalX, finalY, finalSize, finalSize,  // Source rect
      0, 0, outputSize, outputSize            // Dest rect
    )

    // Convert to base64 JPEG (smaller than PNG)
    const dataUrl = canvas.toDataURL('image/jpeg', 0.85)
    
    // Remove the data:image/jpeg;base64, prefix
    const base64 = dataUrl.split(',')[1]

    return {
      base64,
      width: outputSize,
      height: outputSize,
    }
  } catch (error) {
    console.error('Failed to extract face crop:', error)
    return null
  }
}

/**
 * Create a reusable face crop extractor with canvas pooling.
 * More efficient for continuous frame processing.
 */
export class FaceCropExtractor {
  private canvas: HTMLCanvasElement
  private ctx: CanvasRenderingContext2D | null
  private outputSize: number
  private padding: number

  constructor(outputSize: number = 224, padding: number = 0.3) {
    this.outputSize = outputSize
    this.padding = padding
    this.canvas = document.createElement('canvas')
    this.canvas.width = outputSize
    this.canvas.height = outputSize
    this.ctx = this.canvas.getContext('2d')
  }

  extract(video: HTMLVideoElement, landmarks: number[]): string | null {
    if (!this.ctx || !video || landmarks.length < 468 * 3) {
      return null
    }

    const videoWidth = video.videoWidth
    const videoHeight = video.videoHeight

    if (videoWidth === 0 || videoHeight === 0) {
      return null
    }

    // Find bounding box from landmarks
    let minX = Infinity, minY = Infinity
    let maxX = -Infinity, maxY = -Infinity

    for (let i = 0; i < landmarks.length; i += 3) {
      const x = landmarks[i] * videoWidth
      const y = landmarks[i + 1] * videoHeight

      minX = Math.min(minX, x)
      maxX = Math.max(maxX, x)
      minY = Math.min(minY, y)
      maxY = Math.max(maxY, y)
    }

    // Calculate face dimensions with padding
    const faceWidth = maxX - minX
    const faceHeight = maxY - minY
    const padX = faceWidth * this.padding
    const padY = faceHeight * this.padding

    // Make square crop
    const size = Math.max(faceWidth + 2 * padX, faceHeight + 2 * padY)
    const centerX = (minX + maxX) / 2
    const centerY = (minY + maxY) / 2

    let srcX = Math.max(0, centerX - size / 2)
    let srcY = Math.max(0, centerY - size / 2)
    let srcSize = size

    // Clamp to bounds
    if (srcX + srcSize > videoWidth) srcSize = videoWidth - srcX
    if (srcY + srcSize > videoHeight) srcSize = videoHeight - srcY

    try {
      this.ctx.drawImage(
        video,
        srcX, srcY, srcSize, srcSize,
        0, 0, this.outputSize, this.outputSize
      )

      const dataUrl = this.canvas.toDataURL('image/jpeg', 0.85)
      return dataUrl.split(',')[1]
    } catch {
      return null
    }
  }

  dispose(): void {
    this.ctx = null
  }
}
