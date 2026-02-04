"""
ML Inference Server - Runs separately from backend due to Python version constraints.

The backend uses Python 3.14 which doesn't support ONNX Runtime.
This service runs on Python 3.11 and communicates via HTTP.

Run with: 
    cd ml && source .venv/bin/activate && python inference_server.py

Endpoints:
    POST /predict - Predict emotions from base64-encoded face image
    GET /health - Health check
"""

import base64
import io
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global emotion service
emotion_service = None


def get_emotion_service():
    """Lazy load emotion service."""
    global emotion_service
    if emotion_service is None:
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
        # Use enet_b2_8 for better accuracy (larger model, ~70% vs ~66%)
        emotion_service = HSEmotionRecognizer(model_name='enet_b2_8')
        logger.info("Loaded pretrained emotion model: enet_b2_8")
    return emotion_service


# Emotion mapping
EMOTION_MAPPING = {
    'Anger': 'angry',
    'Contempt': 'disgust',
    'Disgust': 'disgust', 
    'Fear': 'fear',
    'Happiness': 'happy',
    'Neutral': 'neutral',
    'Sadness': 'sad',
    'Surprise': 'surprise',
}


def softmax(logits):
    """Convert logits to probabilities."""
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()


def compute_frustration(scores: dict) -> float:
    """Derive frustration from emotions."""
    # Sum of negative emotions
    negative = scores.get('angry', 0) + scores.get('disgust', 0) + scores.get('fear', 0) + scores.get('sad', 0)
    # Sum of positive emotions
    positive = scores.get('happy', 0) + scores.get('neutral', 0) * 0.8 + scores.get('surprise', 0) * 0.5
    
    # Frustration increases with negative emotions and lack of positive ones
    # New formula: more sensitive to negative spikes
    frustration = negative + 0.3 * (1.0 - positive) * (1.0 - scores.get('neutral', 0))
    
    return float(np.clip(frustration, 0, 1))


def compute_engagement(scores: dict) -> float:
    """Derive engagement from emotions."""
    engaged = scores.get('happy', 0) * 0.9 + scores.get('surprise', 0) * 0.8 + scores.get('fear', 0) * 0.5
    disengaged = scores.get('neutral', 0) * 0.3 + scores.get('sad', 0) * 0.6
    engagement = 0.5 + 0.5 * (engaged - disengaged)
    return float(np.clip(engagement, 0, 1))


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'model': 'enet_b2_8'})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict emotions from face image.
    
    Request body:
        - image_base64: Base64-encoded face image (RGB)
        
    Response:
        - dominant_emotion: string
        - emotion_scores: dict of emotion -> probability
        - frustration_score: float 0-1
        - engagement_score: float 0-1
    """
    try:
        data = request.get_json()
        
        if 'image_base64' not in data:
            return jsonify({'error': 'missing image_base64'}), 400
        
        # Decode image
        image_bytes = base64.b64decode(data['image_base64'])
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        face_array = np.array(image)
        
        # Get prediction
        recognizer = get_emotion_service()
        emotion, scores = recognizer.predict_emotions(face_array, logits=True)
        
        # Process scores
        probs = softmax(scores)
        emotion_classes = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
        
        # Map to our format
        emotion_scores = {}
        for emo_name, prob in zip(emotion_classes, probs):
            mapped = EMOTION_MAPPING[emo_name]
            emotion_scores[mapped] = emotion_scores.get(mapped, 0) + float(prob)
        
        # Normalize
        total = sum(emotion_scores.values())
        emotion_scores = {k: v / total for k, v in emotion_scores.items()}
        
        result = {
            'dominant_emotion': EMOTION_MAPPING[emotion],
            'emotion_scores': emotion_scores,
            'frustration_score': compute_frustration(emotion_scores),
            'engagement_score': compute_engagement(emotion_scores),
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Preload model
    get_emotion_service()
    
    # Run server
    print("Starting ML Inference Server on http://localhost:8001")
    app.run(host='0.0.0.0', port=8001, debug=False)
