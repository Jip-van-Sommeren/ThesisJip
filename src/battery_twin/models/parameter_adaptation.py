"""
Parameter Adaptation for Battery Models

Implements online parameter estimation and adaptation algorithms:

1. Recursive Least Squares (RLS)
   - Online parameter estimation
   - Exponential forgetting factor for time-varying parameters
   - Covariance matrix tracking for confidence estimation

2. Exponentially Weighted Moving Average (EWMA)
   - Simple parameter smoothing
   - Configurable forgetting factor

These algorithms enable the battery models to adapt to:
- Aging and degradation effects
- Temperature variations
- Changes in operating conditions
- Individual cell differences

The RLS algorithm minimizes:
    J = Σ λ^(k-i) × (y[i] - θᵀφ[i])²

where:
    λ = forgetting factor (0 < λ ≤ 1)
    y = measurement
    θ = parameter vector
    φ = regressor vector
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class RLSConfig:
    """
    Configuration for Recursive Least Squares estimator.

    Attributes:
        forgetting_factor: λ ∈ (0, 1], controls adaptation speed
                          λ = 1: no forgetting (standard RLS)
                          λ < 1: exponential forgetting for time-varying parameters
        initial_covariance: Initial P matrix diagonal value
        min_covariance: Minimum covariance to prevent numerical issues
        max_covariance: Maximum covariance to prevent divergence
    """
    forgetting_factor: float = 0.98
    initial_covariance: float = 1000.0
    min_covariance: float = 1e-6
    max_covariance: float = 1e6

    def __post_init__(self):
        """Validate configuration."""
        if not (0 < self.forgetting_factor <= 1.0):
            raise ValueError(f"Forgetting factor must be in (0, 1], got {self.forgetting_factor}")
        if self.initial_covariance <= 0:
            raise ValueError(f"Initial covariance must be positive, got {self.initial_covariance}")


class RecursiveLeastSquares:
    """
    Recursive Least Squares parameter estimator with forgetting factor.

    The RLS algorithm provides online parameter estimation with
    exponential weighting of past data. The forgetting factor λ
    determines how quickly old data is discounted:

    - λ = 1.0: All data weighted equally (standard RLS)
    - λ = 0.95-0.99: Exponential forgetting (typical for adaptive systems)
    - λ < 0.95: Rapid adaptation to changes

    Update equations:
        K[k] = P[k-1] × φ[k] / (λ + φ[k]ᵀ × P[k-1] × φ[k])
        θ[k] = θ[k-1] + K[k] × (y[k] - θ[k-1]ᵀ × φ[k])
        P[k] = (P[k-1] - K[k] × φ[k]ᵀ × P[k-1]) / λ

    Example:
        >>> rls = RecursiveLeastSquares(n_parameters=2, forgetting_factor=0.98)
        >>>
        >>> # Update with new measurement
        >>> params = rls.update(
        ...     regressor=np.array([1.0, 2.5]),
        ...     measurement=5.3
        ... )
        >>> print(f"Estimated parameters: {params}")
    """

    def __init__(
        self,
        n_parameters: int,
        config: Optional[RLSConfig] = None,
        initial_parameters: Optional[np.ndarray] = None
    ):
        """
        Initialize RLS estimator.

        Args:
            n_parameters: Number of parameters to estimate
            config: RLS configuration
            initial_parameters: Initial parameter guess (zeros if None)
        """
        self.n_params = n_parameters
        self.config = config or RLSConfig()

        # Parameter vector
        if initial_parameters is not None:
            if len(initial_parameters) != n_parameters:
                raise ValueError(f"Initial parameters must have length {n_parameters}")
            self.theta = initial_parameters.copy()
        else:
            self.theta = np.zeros(n_parameters)

        # Covariance matrix (P)
        self.P = np.eye(n_parameters) * self.config.initial_covariance

        # Statistics
        self.n_updates = 0
        self.residuals: List[float] = []
        self.parameter_history: List[np.ndarray] = []

        logger.info(
            f"Initialized RLS: n_params={n_parameters}, "
            f"λ={self.config.forgetting_factor}, P0={self.config.initial_covariance}"
        )

    def update(
        self,
        regressor: np.ndarray,
        measurement: float
    ) -> np.ndarray:
        """
        Update parameter estimates with new measurement.

        Args:
            regressor: Regressor vector φ[k] (feature vector)
            measurement: Measured output y[k]

        Returns:
            Updated parameter vector θ[k]
        """
        if len(regressor) != self.n_params:
            raise ValueError(f"Regressor must have length {self.n_params}, got {len(regressor)}")

        # Reshape to column vector
        phi = regressor.reshape(-1, 1)
        lam = self.config.forgetting_factor

        # Compute Kalman gain: K = P×φ / (λ + φᵀ×P×φ)
        numerator = self.P @ phi
        denominator = lam + phi.T @ self.P @ phi
        K = numerator / denominator

        # Prediction error: e = y - θᵀ×φ
        prediction = self.theta.T @ phi
        error = measurement - float(prediction.flat[0])  # Handle both 1D and 2D

        # Update parameters: θ = θ + K×e
        self.theta = self.theta + (K * error).flatten()

        # Update covariance: P = (P - K×φᵀ×P) / λ
        self.P = (self.P - K @ phi.T @ self.P) / lam

        # Clip covariance to prevent numerical issues
        self.P = np.clip(
            self.P,
            self.config.min_covariance,
            self.config.max_covariance
        )

        # Track statistics
        self.n_updates += 1
        self.residuals.append(float(error))
        self.parameter_history.append(self.theta.copy())

        if self.n_updates % 100 == 0:
            logger.debug(
                f"RLS update {self.n_updates}: "
                f"error={error:.4f}, params={self.theta}"
            )

        return self.theta.copy()

    def get_parameters(self) -> np.ndarray:
        """Get current parameter estimates."""
        return self.theta.copy()

    def get_covariance(self) -> np.ndarray:
        """Get current covariance matrix."""
        return self.P.copy()

    def get_parameter_uncertainty(self) -> np.ndarray:
        """
        Get parameter uncertainty (standard deviations).

        Returns:
            Array of standard deviations for each parameter
        """
        return np.sqrt(np.diag(self.P))

    def get_confidence_intervals(self, confidence: float = 0.95) -> np.ndarray:
        """
        Get confidence intervals for parameters.

        Args:
            confidence: Confidence level (e.g., 0.95 for 95% CI)

        Returns:
            Array of shape (n_params, 2) with [lower, upper] bounds
        """
        from scipy import stats

        # Z-score for confidence level
        z = stats.norm.ppf((1 + confidence) / 2)

        # Standard errors
        std_errors = self.get_parameter_uncertainty()

        # Confidence intervals
        intervals = np.zeros((self.n_params, 2))
        intervals[:, 0] = self.theta - z * std_errors  # Lower bound
        intervals[:, 1] = self.theta + z * std_errors  # Upper bound

        return intervals

    def reset(self, initial_parameters: Optional[np.ndarray] = None):
        """
        Reset estimator to initial state.

        Args:
            initial_parameters: New initial parameters (zeros if None)
        """
        if initial_parameters is not None:
            self.theta = initial_parameters.copy()
        else:
            self.theta = np.zeros(self.n_params)

        self.P = np.eye(self.n_params) * self.config.initial_covariance
        self.n_updates = 0
        self.residuals = []
        self.parameter_history = []

        logger.info("RLS estimator reset")

    def get_statistics(self) -> Dict:
        """Get estimator statistics."""
        if len(self.residuals) == 0:
            return {
                'n_updates': 0,
                'rmse': 0.0,
                'mae': 0.0,
                'last_residual': 0.0
            }

        residuals = np.array(self.residuals)

        return {
            'n_updates': self.n_updates,
            'rmse': float(np.sqrt(np.mean(residuals ** 2))),
            'mae': float(np.mean(np.abs(residuals))),
            'last_residual': float(residuals[-1]) if len(residuals) > 0 else 0.0,
            'parameter_uncertainty': self.get_parameter_uncertainty().tolist()
        }


class ExponentialSmoothing:
    """
    Exponentially Weighted Moving Average (EWMA) for parameter smoothing.

    Simpler alternative to RLS for parameter tracking when
    covariance information is not needed.

    Update equation:
        θ[k] = α × θ[k-1] + (1 - α) × θ_new

    where:
        α = smoothing factor (0 < α < 1)
        α ≈ 1: Slow adaptation (more smoothing)
        α ≈ 0: Fast adaptation (less smoothing)

    Example:
        >>> smoother = ExponentialSmoothing(n_parameters=3, alpha=0.9)
        >>> smoothed = smoother.update(new_parameters=np.array([1.0, 2.0, 3.0]))
    """

    def __init__(
        self,
        n_parameters: int,
        alpha: float = 0.9,
        initial_parameters: Optional[np.ndarray] = None
    ):
        """
        Initialize exponential smoother.

        Args:
            n_parameters: Number of parameters
            alpha: Smoothing factor (0 < α < 1)
            initial_parameters: Initial values (zeros if None)
        """
        if not (0 < alpha < 1):
            raise ValueError(f"Alpha must be in (0, 1), got {alpha}")

        self.n_params = n_parameters
        self.alpha = alpha

        if initial_parameters is not None:
            self.theta = initial_parameters.copy()
        else:
            self.theta = np.zeros(n_parameters)

        self.n_updates = 0
        logger.info(f"Initialized EWMA: n_params={n_parameters}, α={alpha}")

    def update(self, new_parameters: np.ndarray) -> np.ndarray:
        """
        Update smoothed parameters.

        Args:
            new_parameters: New parameter values

        Returns:
            Smoothed parameter vector
        """
        if len(new_parameters) != self.n_params:
            raise ValueError(f"Parameters must have length {self.n_params}")

        # EWMA update
        self.theta = self.alpha * self.theta + (1 - self.alpha) * new_parameters
        self.n_updates += 1

        return self.theta.copy()

    def get_parameters(self) -> np.ndarray:
        """Get current smoothed parameters."""
        return self.theta.copy()

    def reset(self, initial_parameters: Optional[np.ndarray] = None):
        """Reset to initial state."""
        if initial_parameters is not None:
            self.theta = initial_parameters.copy()
        else:
            self.theta = np.zeros(self.n_params)
        self.n_updates = 0


class DriftDetector:
    """
    Detect parameter drift using statistical tests.

    Monitors parameter changes over time and triggers alerts
    when significant drift is detected.

    Uses CUSUM (Cumulative Sum) for drift detection.
    """

    def __init__(
        self,
        n_parameters: int,
        threshold: float = 5.0,
        drift_rate_threshold: float = 0.01
    ):
        """
        Initialize drift detector.

        Args:
            n_parameters: Number of parameters to monitor
            threshold: CUSUM threshold for drift detection
            drift_rate_threshold: Minimum drift rate to trigger alert
        """
        self.n_params = n_parameters
        self.threshold = threshold
        self.drift_rate_threshold = drift_rate_threshold

        # CUSUM statistics
        self.cusum_pos = np.zeros(n_parameters)
        self.cusum_neg = np.zeros(n_parameters)

        # Reference parameters
        self.reference_params = None
        self.drift_detected = np.zeros(n_parameters, dtype=bool)

        logger.info(f"Initialized DriftDetector: threshold={threshold}")

    def set_reference(self, parameters: np.ndarray):
        """Set reference parameters for drift detection."""
        self.reference_params = parameters.copy()
        self.cusum_pos = np.zeros(self.n_params)
        self.cusum_neg = np.zeros(self.n_params)
        self.drift_detected = np.zeros(self.n_params, dtype=bool)

    def update(self, parameters: np.ndarray) -> np.ndarray:
        """
        Update drift detection with new parameters.

        Args:
            parameters: Current parameter values

        Returns:
            Boolean array indicating drift for each parameter
        """
        if self.reference_params is None:
            self.set_reference(parameters)
            return self.drift_detected

        # Compute deviation from reference
        deviation = parameters - self.reference_params

        # CUSUM update
        self.cusum_pos = np.maximum(0, self.cusum_pos + deviation - self.drift_rate_threshold)
        self.cusum_neg = np.maximum(0, self.cusum_neg - deviation - self.drift_rate_threshold)

        # Check thresholds
        self.drift_detected = (self.cusum_pos > self.threshold) | (self.cusum_neg > self.threshold)

        if np.any(self.drift_detected):
            drift_indices = np.where(self.drift_detected)[0]
            logger.warning(f"Parameter drift detected in parameters: {drift_indices}")

        return self.drift_detected.copy()

    def reset(self):
        """Reset drift detector."""
        self.cusum_pos = np.zeros(self.n_params)
        self.cusum_neg = np.zeros(self.n_params)
        self.drift_detected = np.zeros(self.n_params, dtype=bool)


__all__ = [
    'RecursiveLeastSquares',
    'RLSConfig',
    'ExponentialSmoothing',
    'DriftDetector',
]
