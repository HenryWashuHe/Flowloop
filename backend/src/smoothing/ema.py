from dataclasses import dataclass


@dataclass
class CognitiveState:
    """Represents the current cognitive state estimate."""

    attention: float
    frustration: float
    timestamp: float

    def to_dict(self) -> dict[str, float]:
        return {
            "attention": self.attention,
            "frustration": self.frustration,
            "timestamp": self.timestamp,
        }


class ExponentialMovingAverage:
    """
    Temporal smoothing using Exponential Moving Average.

    EMA formula: EMA_t = alpha * value_t + (1 - alpha) * EMA_{t-1}

    Higher alpha = more responsive to recent values
    Lower alpha = smoother, more stable output

    Recommended alpha values:
    - 0.1: Very smooth, slow to respond
    - 0.2: Balanced (default)
    - 0.3: More responsive
    - 0.5: Very responsive, less smoothing
    """

    def __init__(self, alpha: float = 0.2) -> None:
        """
        Initialize EMA smoother.

        Args:
            alpha: Smoothing factor between 0 and 1
        """
        if not 0 < alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1")

        self.alpha = alpha
        self.current_state: CognitiveState | None = None

    def update(self, new_state: CognitiveState) -> CognitiveState:
        """
        Update EMA with new observation.

        Args:
            new_state: New cognitive state observation

        Returns:
            Smoothed cognitive state
        """
        if self.current_state is None:
            # First observation, no smoothing
            self.current_state = new_state
            return new_state

        smoothed = CognitiveState(
            attention=self._ema(self.current_state.attention, new_state.attention),
            frustration=self._ema(self.current_state.frustration, new_state.frustration),
            timestamp=new_state.timestamp,
        )

        self.current_state = smoothed
        return smoothed

    def _ema(self, prev: float, curr: float) -> float:
        """Apply EMA formula."""
        return self.alpha * curr + (1 - self.alpha) * prev

    def reset(self) -> None:
        """Reset smoother state."""
        self.current_state = None

    @property
    def is_initialized(self) -> bool:
        """Check if smoother has received at least one observation."""
        return self.current_state is not None


class AdaptiveEMA:
    """
    EMA with adaptive alpha based on prediction confidence.

    Uses higher alpha (more responsive) when confidence is high,
    lower alpha (more smoothing) when confidence is low.
    """

    def __init__(
        self,
        base_alpha: float = 0.2,
        min_alpha: float = 0.1,
        max_alpha: float = 0.4,
    ) -> None:
        self.base_alpha = base_alpha
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.current_state: CognitiveState | None = None

    def update(self, new_state: CognitiveState, confidence: float = 1.0) -> CognitiveState:
        """
        Update with adaptive alpha based on confidence.

        Args:
            new_state: New cognitive state observation
            confidence: Prediction confidence (0-1), affects smoothing

        Returns:
            Smoothed cognitive state
        """
        if self.current_state is None:
            self.current_state = new_state
            return new_state

        # Scale alpha by confidence
        alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * confidence

        smoothed = CognitiveState(
            attention=alpha * new_state.attention + (1 - alpha) * self.current_state.attention,
            frustration=alpha * new_state.frustration
            + (1 - alpha) * self.current_state.frustration,
            timestamp=new_state.timestamp,
        )

        self.current_state = smoothed
        return smoothed

    def reset(self) -> None:
        """Reset smoother state."""
        self.current_state = None
