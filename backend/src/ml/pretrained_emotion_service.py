"""
Pretrained Emotion Recognition Service using HSEmotion.

Uses pretrained models from HSE lab (https://github.com/HSE-asavchenko/face-emotion-recognition)
trained on AffectNet dataset with ~66% accuracy.

Models available:
- enet_b0_8_best_afew: EfficientNet-B0, 8 emotions, trained on AFEW
- enet_b0_8_best_vgaf: EfficientNet-B0, 8 emotions, trained on VGAF
- enet_b2_8: EfficientNet-B2, 8 emotions (larger, more accurate)
"""

import logging
from typing import TypedDict
import numpy as np

logger = logging.getLogger(__name__)


class EmotionPrediction(TypedDict):
    """Emotion prediction result."""
    dominant_emotion: str
    emotion_scores: dict[str, float]
    frustration_score: float
    engagement_score: float


class HSEmotionService:
    """
    Pretrained emotion recognition using HSEmotion ONNX models.
    
    Provides ~66% accuracy out of the box on emotion classification.
    """
    
    # Emotion classes from HSEmotion (8 classes including Contempt)
    EMOTION_CLASSES = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
    
    # Map to our FER2013 format (7 classes)
    EMOTION_MAPPING = {
        'Anger': 'angry',
        'Contempt': 'disgust',  # Map contempt to disgust
        'Disgust': 'disgust',
        'Fear': 'fear',
        'Happiness': 'happy',
        'Neutral': 'neutral',
        'Sadness': 'sad',
        'Surprise': 'surprise',
    }
    
    def __init__(self, model_name: str = 'enet_b0_8_best_afew'):
        """
        Initialize with pretrained model.
        
        Args:
            model_name: One of 'enet_b0_8_best_afew', 'enet_b0_8_best_vgaf', 'enet_b2_8'
        """
        self.model_name = model_name
        self._recognizer = None
        
    def _ensure_loaded(self) -> None:
        """Lazy load the model on first use."""
        if self._recognizer is None:
            try:
                from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
                self._recognizer = HSEmotionRecognizer(model_name=self.model_name)
                logger.info(f"Loaded pretrained model: {self.model_name}")
            except ImportError:
                logger.error("hsemotion-onnx not installed. Run: pip install hsemotion-onnx")
                raise
            except Exception as e:
                logger.error(f"Failed to load pretrained model: {e}")
                raise
    
    def predict(self, face_image: np.ndarray) -> EmotionPrediction:
        """
        Predict emotions from a face image.
        
        Args:
            face_image: RGB face crop, any size (will be resized)
        
        Returns:
            EmotionPrediction with dominant emotion, scores, and derived metrics
        """
        self._ensure_loaded()
        
        # Ensure correct format
        if face_image.dtype != np.uint8:
            face_image = (face_image * 255).astype(np.uint8)
        
        # Get prediction
        emotion, scores = self._recognizer.predict_emotions(face_image, logits=True)
        
        # Convert logits to probabilities
        probs = self._softmax(scores)
        
        # Create score dict mapped to our format
        emotion_scores = {}
        for i, (emo_name, prob) in enumerate(zip(self.EMOTION_CLASSES, probs)):
            mapped_name = self.EMOTION_MAPPING[emo_name]
            if mapped_name in emotion_scores:
                emotion_scores[mapped_name] += prob  # Combine contempt + disgust
            else:
                emotion_scores[mapped_name] = prob
        
        # Normalize after combining
        total = sum(emotion_scores.values())
        emotion_scores = {k: v / total for k, v in emotion_scores.items()}
        
        # Derive frustration and engagement scores
        frustration_score = self._compute_frustration(emotion_scores)
        engagement_score = self._compute_engagement(emotion_scores)
        
        return EmotionPrediction(
            dominant_emotion=self.EMOTION_MAPPING[emotion],
            emotion_scores=emotion_scores,
            frustration_score=frustration_score,
            engagement_score=engagement_score,
        )
    
    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Convert logits to probabilities."""
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / exp_logits.sum()
    
    def _compute_frustration(self, scores: dict[str, float]) -> float:
        """
        Derive frustration score from emotion probabilities.
        
        Frustration = high negative emotions + low positive emotions
        """
        negative = scores.get('angry', 0) + scores.get('disgust', 0) + scores.get('fear', 0) + scores.get('sad', 0)
        positive = scores.get('happy', 0) + scores.get('neutral', 0) * 0.5
        
        frustration = 0.6 * negative + 0.4 * (1 - positive)
        return float(np.clip(frustration, 0, 1))
    
    def _compute_engagement(self, scores: dict[str, float]) -> float:
        """
        Derive engagement score from emotion probabilities.
        
        Engaged = alert, responsive (surprise, happy, some fear = attention)
        Disengaged = neutral, sad
        """
        engaged = scores.get('happy', 0) * 0.9 + scores.get('surprise', 0) * 0.8 + scores.get('fear', 0) * 0.5
        disengaged = scores.get('neutral', 0) * 0.3 + scores.get('sad', 0) * 0.6
        
        engagement = 0.5 + 0.5 * (engaged - disengaged)
        return float(np.clip(engagement, 0, 1))


# Factory function
def create_emotion_service(model_name: str = 'enet_b0_8_best_afew') -> HSEmotionService:
    """Create emotion recognition service with pretrained model."""
    return HSEmotionService(model_name=model_name)


if __name__ == "__main__":
    # Test
    import logging
    logging.basicConfig(level=logging.INFO)
    
    service = create_emotion_service()
    
    # Test with random image
    dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    result = service.predict(dummy)
    
    print(f"Dominant emotion: {result['dominant_emotion']}")
    print(f"Scores: {result['emotion_scores']}")
    print(f"Frustration: {result['frustration_score']:.2f}")
    print(f"Engagement: {result['engagement_score']:.2f}")
