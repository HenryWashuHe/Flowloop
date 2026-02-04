"""
EmotionNet: EfficientNet-B0 backbone with multi-task heads for emotion detection.

Architecture:
- Backbone: EfficientNet-B0 (pretrained on ImageNet)
- Emotion Head: 7-class classification (FER2013 emotions)
- Frustration Head: Regression (0-1 score)
- Engagement Head: Regression (0-1 score)

Input: RGB image tensor [B, 3, 224, 224]
Output: dict with 'emotions', 'frustration', 'engagement', 'features'
"""

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class EmotionNet(nn.Module):
    """
    EfficientNet-B0 based emotion detection network.

    Multi-task learning with shared backbone:
    1. Emotion classification (7 classes)
    2. Frustration regression (0-1)
    3. Engagement regression (0-1)
    """

    EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    def __init__(
        self,
        num_emotions: int = 7,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False,
    ) -> None:
        """
        Initialize EmotionNet.

        Args:
            num_emotions: Number of emotion classes (default: 7 for FER2013)
            pretrained: Whether to use ImageNet pretrained weights
            dropout_rate: Dropout probability for regularization
            freeze_backbone: Whether to freeze EfficientNet backbone
        """
        super().__init__()

        # Load EfficientNet-B0 backbone
        if pretrained:
            self.backbone = EfficientNet.from_pretrained("efficientnet-b0")
        else:
            self.backbone = EfficientNet.from_name("efficientnet-b0")

        # Get feature dimension (1280 for EfficientNet-B0)
        self.feature_dim = self.backbone._fc.in_features

        # Remove original classifier
        self.backbone._fc = nn.Identity()

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Emotion classification head
        self.emotion_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.66),
            nn.Linear(512, num_emotions),
        )

        # Frustration regression head (0-1 score)
        self.frustration_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # Engagement regression head (0-1 score)
        self.engagement_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self._initialize_heads()

    def _initialize_heads(self) -> None:
        """Initialize head weights with Xavier initialization."""
        for module in [self.emotion_head, self.frustration_head, self.engagement_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor [B, 3, 224, 224]

        Returns:
            Dictionary with:
                - 'emotions': [B, num_emotions] emotion logits
                - 'frustration': [B, 1] frustration score
                - 'engagement': [B, 1] engagement score
                - 'features': [B, feature_dim] backbone features for fusion
        """
        # Extract features from backbone
        features = self.backbone(x)

        # Task-specific predictions
        emotions = self.emotion_head(features)
        frustration = self.frustration_head(features)
        engagement = self.engagement_head(features)

        return {
            "emotions": emotions,
            "frustration": frustration,
            "engagement": engagement,
            "features": features,
        }

    def predict_emotions(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get emotion predictions with probabilities.

        Args:
            x: Input tensor [B, 3, 224, 224]

        Returns:
            Tuple of (predicted_classes, probabilities)
        """
        with torch.no_grad():
            outputs = self.forward(x)
            probs = torch.softmax(outputs["emotions"], dim=1)
            predicted = torch.argmax(probs, dim=1)
        return predicted, probs

    def get_frustration_proxy(
        self, emotions_probs: torch.Tensor, engagement: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute frustration proxy from emotion predictions.

        Frustration is approximated by:
        - High negative emotions (angry, fear, sad, disgust)
        - Low engagement
        - Low neutral/happy

        Args:
            emotions_probs: Softmax emotion probabilities [B, 7]
            engagement: Engagement score [B, 1]

        Returns:
            Frustration proxy score [B, 1]
        """
        # Emotion indices: angry=0, disgust=1, fear=2, happy=3, sad=4, surprise=5, neutral=6
        negative_emotions = (
            emotions_probs[:, 0]  # angry
            + emotions_probs[:, 1]  # disgust
            + emotions_probs[:, 2]  # fear
            + emotions_probs[:, 4]  # sad
        )

        positive_emotions = emotions_probs[:, 3] + emotions_probs[:, 6]  # happy + neutral

        frustration_proxy = (
            0.5 * negative_emotions
            + 0.3 * (1 - engagement.squeeze())
            + 0.2 * (1 - positive_emotions)
        )

        return frustration_proxy.unsqueeze(1).clamp(0, 1)


def create_emotion_net(
    pretrained: bool = True,
    freeze_backbone: bool = False,
    device: str = "cpu",
) -> EmotionNet:
    """
    Factory function to create EmotionNet.

    Args:
        pretrained: Use ImageNet pretrained weights
        freeze_backbone: Freeze backbone for fine-tuning heads only
        device: Target device

    Returns:
        Configured EmotionNet instance
    """
    model = EmotionNet(pretrained=pretrained, freeze_backbone=freeze_backbone)
    return model.to(device)


if __name__ == "__main__":
    # Test model
    model = create_emotion_net(pretrained=True)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    outputs = model(dummy_input)

    print(f"Emotions shape: {outputs['emotions'].shape}")
    print(f"Frustration shape: {outputs['frustration'].shape}")
    print(f"Engagement shape: {outputs['engagement'].shape}")
    print(f"Features shape: {outputs['features'].shape}")
