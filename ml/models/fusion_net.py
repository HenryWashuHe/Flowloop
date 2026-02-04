"""
FeatureFusionNet: Combines geometric features with emotion features.

Architecture:
- Geometric branch: landmarks/pose features -> encoded representation
- Emotion branch: EfficientNet features -> encoded representation
- Fusion: concatenate + MLP -> attention/frustration scores

Input:
- geometric_features: [B, 20] from MediaPipe landmarks
- emotion_features: [B, 1280] from EmotionNet backbone

Output: dict with 'attention', 'frustration', 'fused_features'
"""

import torch
import torch.nn as nn


class FeatureFusionNet(nn.Module):
    """
    Feature fusion network combining geometric and emotion features.

    The network encodes each modality separately, then fuses them
    to produce final attention and frustration scores.
    """

    def __init__(
        self,
        geometric_dim: int = 20,
        emotion_dim: int = 1280,
        hidden_dim: int = 64,
        dropout_rate: float = 0.2,
    ) -> None:
        """
        Initialize FeatureFusionNet.

        Args:
            geometric_dim: Dimension of geometric features (default: 20)
            emotion_dim: Dimension of emotion features from EmotionNet (default: 1280)
            hidden_dim: Hidden layer dimension
            dropout_rate: Dropout probability
        """
        super().__init__()

        self.geometric_dim = geometric_dim
        self.emotion_dim = emotion_dim
        self.hidden_dim = hidden_dim

        # Geometric feature encoder
        self.geometric_encoder = nn.Sequential(
            nn.Linear(geometric_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
        )

        # Emotion feature encoder
        self.emotion_encoder = nn.Sequential(
            nn.Linear(emotion_dim, hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 1.5),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
        )

        # Fusion network
        fusion_input_dim = hidden_dim  # geometric (h/2) + emotion (h/2)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
        )

        # Output heads
        output_dim = hidden_dim // 2

        self.attention_head = nn.Sequential(
            nn.Linear(output_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        self.frustration_head = nn.Sequential(
            nn.Linear(output_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights with Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        geometric_features: torch.Tensor,
        emotion_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass combining geometric and emotion features.

        Args:
            geometric_features: [B, geometric_dim] from MediaPipe
            emotion_features: [B, emotion_dim] from EmotionNet

        Returns:
            Dictionary with:
                - 'attention': [B, 1] attention score (0-1)
                - 'frustration': [B, 1] frustration score (0-1)
                - 'fused_features': [B, hidden_dim//2] fused representation
        """
        # Encode each modality
        geo_encoded = self.geometric_encoder(geometric_features)
        emo_encoded = self.emotion_encoder(emotion_features)

        # Concatenate and fuse
        fused = torch.cat([geo_encoded, emo_encoded], dim=-1)
        fused_features = self.fusion(fused)

        # Output predictions
        attention = self.attention_head(fused_features)
        frustration = self.frustration_head(fused_features)

        return {
            "attention": attention,
            "frustration": frustration,
            "fused_features": fused_features,
        }


class GeometricFeatureExtractor:
    """
    Helper class to extract and normalize geometric features for fusion.

    Expected input: dict with raw MediaPipe outputs
    Output: normalized tensor ready for fusion network
    """

    # Feature indices and normalization constants
    FEATURE_NAMES = [
        "gaze_x",
        "gaze_y",
        "gaze_z",
        "head_pitch",
        "head_yaw",
        "head_roll",
        "left_ear",
        "right_ear",
        "blink_rate",
        "brow_furrow",
        "brow_raise",
        "mouth_open",
        "lip_corner",
        "head_velocity",
        "head_variance",
        "gaze_confidence",
        "left_ear_smooth",
        "right_ear_smooth",
        "brow_furrow_smooth",
        "engagement_proxy",
    ]

    def __init__(self) -> None:
        self.feature_dim = len(self.FEATURE_NAMES)
        self._running_mean: torch.Tensor | None = None
        self._running_std: torch.Tensor | None = None

    def extract(self, raw_features: dict[str, float]) -> torch.Tensor:
        """
        Extract and normalize features from raw MediaPipe output.

        Args:
            raw_features: Dictionary with raw feature values

        Returns:
            Normalized feature tensor [1, feature_dim]
        """
        features = []
        for name in self.FEATURE_NAMES:
            value = raw_features.get(name, 0.0)
            features.append(value)

        tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        # Apply normalization if available
        if self._running_mean is not None and self._running_std is not None:
            tensor = (tensor - self._running_mean) / (self._running_std + 1e-8)

        return tensor

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Set normalization parameters from training data."""
        self._running_mean = mean
        self._running_std = std


def create_fusion_net(
    geometric_dim: int = 20,
    emotion_dim: int = 1280,
    device: str = "cpu",
) -> FeatureFusionNet:
    """
    Factory function to create FeatureFusionNet.

    Args:
        geometric_dim: Dimension of geometric features
        emotion_dim: Dimension of emotion features
        device: Target device

    Returns:
        Configured FeatureFusionNet instance
    """
    model = FeatureFusionNet(geometric_dim=geometric_dim, emotion_dim=emotion_dim)
    return model.to(device)


if __name__ == "__main__":
    # Test model
    model = create_fusion_net()
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Test forward pass
    geometric = torch.randn(2, 20)
    emotion = torch.randn(2, 1280)
    outputs = model(geometric, emotion)

    print(f"Attention shape: {outputs['attention'].shape}")
    print(f"Frustration shape: {outputs['frustration'].shape}")
    print(f"Fused features shape: {outputs['fused_features'].shape}")
