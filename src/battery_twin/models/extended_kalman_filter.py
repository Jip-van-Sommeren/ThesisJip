"""
Extended Kalman Filter for Battery State Estimation.

This module implements an Extended Kalman Filter (EKF) for estimating battery
internal states including State of Charge (SoC), State of Health (SoH), and
equivalent circuit model parameters.

The EKF estimates a state vector:
    x = [SoC, SoH, R0, R1, C1, V1]

Where:
    - SoC: State of Charge (0-1)
    - SoH: State of Health (0-1)
    - R0: Ohmic resistance (Ω)
    - R1: Polarization resistance (Ω)
    - C1: Polarization capacitance (F)
    - V1: RC voltage across C1 (V)

Measurements:
    - voltage: Terminal voltage (V)
    - current: Current (A, positive = discharge)
    - temperature: Temperature (°C)
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from loguru import logger


@dataclass
class EKFState:
    """Container for EKF state and covariance."""

    # State vector [SoC, SoH, R0, R1, C1, V1]
    x: np.ndarray

    # Covariance matrix (6x6)
    P: np.ndarray

    # Timestamp
    timestamp: float

    def __post_init__(self):
        """Validate dimensions."""
        assert self.x.shape == (6,), f"State vector must be shape (6,), got {self.x.shape}"
        assert self.P.shape == (6, 6), f"Covariance must be shape (6, 6), got {self.P.shape}"


@dataclass
class EKFMeasurement:
    """Container for EKF measurements."""

    voltage: float  # Terminal voltage (V)
    current: float  # Current (A, positive = discharge)
    temperature: float  # Temperature (°C)
    timestamp: float
    dt: Optional[float] = None  # Time step (s), computed if not provided


@dataclass
class EKFConfig:
    """Configuration for Extended Kalman Filter."""

    # Initial state
    initial_soc: float = 0.8
    initial_soh: float = 1.0
    initial_r0: float = 0.05  # Ohms
    initial_r1: float = 0.03  # Ohms
    initial_c1: float = 1000.0  # Farads
    initial_v1: float = 0.0  # Volts

    # Initial covariance (diagonal)
    initial_cov_soc: float = 0.01
    initial_cov_soh: float = 0.01
    initial_cov_r0: float = 0.001
    initial_cov_r1: float = 0.001
    initial_cov_c1: float = 10.0  # Reduced from 100.0
    initial_cov_v1: float = 0.01

    # Process noise (Q matrix diagonal)
    q_soc: float = 1e-6  # SoC process noise
    q_soh: float = 1e-8  # SoH process noise (very slow drift)
    q_r0: float = 1e-8  # R0 process noise
    q_r1: float = 1e-8  # R1 process noise
    q_c1: float = 1e-4  # C1 process noise
    q_v1: float = 1e-4  # V1 process noise

    # Measurement noise (R matrix)
    r_voltage: float = 0.01  # Voltage measurement noise (V)

    # Battery parameters
    capacity_nominal: float = 2.0  # Nominal capacity (Ah)
    voltage_min: float = 2.5  # Minimum voltage (V)
    voltage_max: float = 4.2  # Maximum voltage (V)

    # Numerical stability
    min_soc: float = 0.0
    max_soc: float = 1.0
    min_soh: float = 0.5
    max_soh: float = 1.0
    min_resistance: float = 0.001  # Minimum resistance (Ω)
    max_resistance: float = 1.0  # Maximum resistance (Ω)
    min_capacitance: float = 100.0  # Minimum capacitance (F)
    max_capacitance: float = 10000.0  # Maximum capacitance (F)

    # Divergence detection
    max_covariance_trace: float = 100.0  # Increased threshold
    max_innovation: float = 1.0  # Maximum innovation (V)


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for battery state estimation.

    This class implements the prediction-update cycle of an EKF for estimating
    battery states from voltage, current, and temperature measurements.

    The EKF operates in two phases:
    1. Prediction: Propagate state and covariance forward in time
    2. Update: Correct state estimate using measurements

    Example:
        >>> config = EKFConfig()
        >>> ekf = ExtendedKalmanFilter(config)
        >>>
        >>> # Process measurements
        >>> measurement = EKFMeasurement(
        ...     voltage=3.7, current=1.0, temperature=25.0, timestamp=1.0
        ... )
        >>> state = ekf.predict(measurement.current, measurement.dt)
        >>> state = ekf.update(measurement)
        >>>
        >>> soc = ekf.get_soc()
        >>> soh = ekf.get_soh()
    """

    def __init__(self, config: Optional[EKFConfig] = None):
        """
        Initialize Extended Kalman Filter.

        Args:
            config: EKF configuration parameters
        """
        self.config = config or EKFConfig()

        # Initialize state
        self.state = self._create_initial_state()

        # Last timestamp
        self.last_timestamp: Optional[float] = None

        # Statistics
        self.num_predictions = 0
        self.num_updates = 0
        self.last_innovation: Optional[float] = None
        self.divergence_detected = False

        logger.info("Extended Kalman Filter initialized")

    def _create_initial_state(self) -> EKFState:
        """Create initial state with configured values."""
        x = np.array([
            self.config.initial_soc,
            self.config.initial_soh,
            self.config.initial_r0,
            self.config.initial_r1,
            self.config.initial_c1,
            self.config.initial_v1
        ])

        P = np.diag([
            self.config.initial_cov_soc,
            self.config.initial_cov_soh,
            self.config.initial_cov_r0,
            self.config.initial_cov_r1,
            self.config.initial_cov_c1,
            self.config.initial_cov_v1
        ])

        return EKFState(x=x, P=P, timestamp=0.0)

    def reset(self):
        """Reset filter to initial state."""
        self.state = self._create_initial_state()
        self.last_timestamp = None
        self.num_predictions = 0
        self.num_updates = 0
        self.last_innovation = None
        self.divergence_detected = False
        logger.info("EKF reset to initial state")

    def predict(self, current: float, dt: float) -> EKFState:
        """
        Prediction step: Propagate state and covariance forward in time.

        State transition model:
            SoC_{k+1} = SoC_k - (I * dt) / (Q_nominal * SoH * 3600)
            SoH_{k+1} = SoH_k  (slow drift, modeled by process noise)
            R0_{k+1} = R0_k  (modeled by process noise)
            R1_{k+1} = R1_k  (modeled by process noise)
            C1_{k+1} = C1_k  (modeled by process noise)
            V1_{k+1} = V1_k * exp(-dt / (R1 * C1)) + I * R1 * (1 - exp(-dt / (R1 * C1)))

        Args:
            current: Current in Amperes (positive = discharge)
            dt: Time step in seconds

        Returns:
            Updated state after prediction
        """
        if dt <= 0:
            logger.warning(f"Invalid dt={dt}, skipping prediction")
            return self.state

        # Extract current state
        soc, soh, r0, r1, c1, v1 = self.state.x

        # State transition function f(x, u)
        soc_new = soc - (current * dt) / (self.config.capacity_nominal * soh * 3600.0)
        soh_new = soh  # SoH changes very slowly
        r0_new = r0
        r1_new = r1
        c1_new = c1

        # RC circuit dynamics: V1(t) = V1(0)*exp(-t/RC) + I*R*(1-exp(-t/RC))
        tau = r1 * c1  # Time constant
        if tau > 0:
            exp_term = np.exp(-dt / tau)
            v1_new = v1 * exp_term + current * r1 * (1 - exp_term)
        else:
            v1_new = v1

        # Apply state constraints
        soc_new = np.clip(soc_new, self.config.min_soc, self.config.max_soc)
        soh_new = np.clip(soh_new, self.config.min_soh, self.config.max_soh)
        r0_new = np.clip(r0_new, self.config.min_resistance, self.config.max_resistance)
        r1_new = np.clip(r1_new, self.config.min_resistance, self.config.max_resistance)
        c1_new = np.clip(c1_new, self.config.min_capacitance, self.config.max_capacitance)

        x_pred = np.array([soc_new, soh_new, r0_new, r1_new, c1_new, v1_new])

        # Jacobian of state transition (F = ∂f/∂x)
        F = self._compute_state_jacobian(current, dt, r1, c1, v1)

        # Process noise covariance (Q)
        Q = np.diag([
            self.config.q_soc * dt,
            self.config.q_soh * dt,
            self.config.q_r0 * dt,
            self.config.q_r1 * dt,
            self.config.q_c1 * dt,
            self.config.q_v1 * dt
        ])

        # Covariance prediction: P = F * P * F^T + Q
        P_pred = F @ self.state.P @ F.T + Q

        # Ensure P is symmetric
        P_pred = 0.5 * (P_pred + P_pred.T)

        # Update state
        self.state = EKFState(x=x_pred, P=P_pred, timestamp=self.state.timestamp + dt)
        self.num_predictions += 1

        return self.state

    def update(self, measurement: EKFMeasurement) -> EKFState:
        """
        Update step: Correct state estimate using voltage measurement.

        Measurement model:
            V_terminal = OCV(SoC) - I * R0 - V1

        Kalman gain:
            K = P * H^T * (H * P * H^T + R)^{-1}

        State update:
            x = x + K * (z - h(x))

        Covariance update:
            P = (I - K * H) * P

        Args:
            measurement: Measurement object containing voltage, current, temperature

        Returns:
            Updated state after correction
        """
        # Predict OCV from SoC
        soc = self.state.x[0]
        ocv = self._ocv_model(soc, measurement.temperature)

        # Extract parameters
        r0 = self.state.x[2]
        v1 = self.state.x[5]

        # Measurement prediction: h(x) = OCV(SoC) - I*R0 - V1
        voltage_pred = ocv - measurement.current * r0 - v1

        # Innovation (residual)
        innovation = measurement.voltage - voltage_pred
        self.last_innovation = innovation

        # Jacobian of measurement function (H = ∂h/∂x)
        H = self._compute_measurement_jacobian(soc, measurement.current, measurement.temperature)

        # Measurement noise covariance (R)
        R = np.array([[self.config.r_voltage**2]])

        # Innovation covariance: S = H * P * H^T + R
        S = H @ self.state.P @ H.T + R

        # Kalman gain: K = P * H^T * S^{-1}
        K = self.state.P @ H.T @ np.linalg.inv(S)

        # State update: x = x + K * innovation
        x_updated = self.state.x + (K @ np.array([[innovation]])).flatten()

        # Apply state constraints
        x_updated[0] = np.clip(x_updated[0], self.config.min_soc, self.config.max_soc)  # SoC
        x_updated[1] = np.clip(x_updated[1], self.config.min_soh, self.config.max_soh)  # SoH
        x_updated[2] = np.clip(x_updated[2], self.config.min_resistance, self.config.max_resistance)  # R0
        x_updated[3] = np.clip(x_updated[3], self.config.min_resistance, self.config.max_resistance)  # R1
        x_updated[4] = np.clip(x_updated[4], self.config.min_capacitance, self.config.max_capacitance)  # C1

        # Covariance update: P = (I - K * H) * P
        I = np.eye(6)
        P_updated = (I - K @ H) @ self.state.P

        # Ensure P is symmetric and positive definite
        P_updated = 0.5 * (P_updated + P_updated.T)

        # Update state
        self.state = EKFState(x=x_updated, P=P_updated, timestamp=measurement.timestamp)
        self.num_updates += 1

        # Check for divergence
        self._check_divergence(innovation)

        return self.state

    def process_measurement(self, measurement: EKFMeasurement) -> EKFState:
        """
        Process a measurement by running prediction and update steps.

        Args:
            measurement: Measurement object

        Returns:
            Updated state after prediction and correction
        """
        # Compute dt
        if self.last_timestamp is not None:
            dt = measurement.timestamp - self.last_timestamp
        elif measurement.dt is not None:
            dt = measurement.dt
        else:
            dt = 1.0  # Default 1 second

        self.last_timestamp = measurement.timestamp

        # Prediction step
        self.predict(measurement.current, dt)

        # Update step
        return self.update(measurement)

    def _compute_state_jacobian(self, current: float, dt: float, r1: float, c1: float, v1: float) -> np.ndarray:
        """
        Compute Jacobian of state transition function.

        F = ∂f/∂x where f is the state transition function

        Returns:
            6x6 Jacobian matrix
        """
        F = np.eye(6)

        # ∂SoC/∂SoH = (I * dt) / (Q_nominal * SoH^2 * 3600)
        soh = self.state.x[1]
        F[0, 1] = (current * dt) / (self.config.capacity_nominal * soh**2 * 3600.0)

        # ∂V1/∂R1 and ∂V1/∂C1 (RC circuit dynamics)
        tau = r1 * c1
        if tau > 0:
            exp_term = np.exp(-dt / tau)

            # ∂V1/∂R1
            dV1_dR1 = (v1 * dt * c1 / (tau**2)) * exp_term + current * (1 - exp_term) + current * r1 * (dt * c1 / (tau**2)) * exp_term
            F[5, 3] = dV1_dR1

            # ∂V1/∂C1
            dV1_dC1 = (v1 * dt * r1 / (tau**2)) * exp_term + current * r1 * (dt * r1 / (tau**2)) * exp_term
            F[5, 4] = dV1_dC1

            # ∂V1/∂V1
            F[5, 5] = exp_term

        return F

    def _compute_measurement_jacobian(self, soc: float, current: float, temperature: float) -> np.ndarray:
        """
        Compute Jacobian of measurement function.

        H = ∂h/∂x where h is the measurement function
        h(x) = OCV(SoC) - I*R0 - V1

        Returns:
            1x6 Jacobian matrix
        """
        H = np.zeros((1, 6))

        # ∂h/∂SoC = ∂OCV/∂SoC
        H[0, 0] = self._docv_dsoc(soc, temperature)

        # ∂h/∂SoH = 0
        H[0, 1] = 0.0

        # ∂h/∂R0 = -I
        H[0, 2] = -current

        # ∂h/∂R1 = 0
        H[0, 3] = 0.0

        # ∂h/∂C1 = 0
        H[0, 4] = 0.0

        # ∂h/∂V1 = -1
        H[0, 5] = -1.0

        return H

    def _ocv_model(self, soc: float, temperature: float) -> float:
        """
        Open Circuit Voltage (OCV) as a function of SoC and temperature.

        This uses a simplified polynomial model. In practice, this should be
        calibrated from experimental data.

        Args:
            soc: State of Charge (0-1)
            temperature: Temperature (°C)

        Returns:
            OCV in Volts
        """
        # Polynomial coefficients (simplified model)
        # OCV = a0 + a1*SoC + a2*SoC^2 + a3*SoC^3 + a4*SoC^4
        a0 = 3.2
        a1 = 0.5
        a2 = 0.8
        a3 = -0.3
        a4 = 0.05

        soc_clipped = np.clip(soc, 0.0, 1.0)

        ocv = a0 + a1*soc_clipped + a2*soc_clipped**2 + a3*soc_clipped**3 + a4*soc_clipped**4

        # Temperature correction (simplified: -0.5mV/°C)
        temp_correction = -0.0005 * (temperature - 25.0)
        ocv += temp_correction

        return ocv

    def _docv_dsoc(self, soc: float, temperature: float) -> float:
        """
        Derivative of OCV with respect to SoC.

        Args:
            soc: State of Charge (0-1)
            temperature: Temperature (°C)

        Returns:
            dOCV/dSoC in V per unit SoC
        """
        # Derivative of polynomial
        a1 = 0.5
        a2 = 0.8
        a3 = -0.3
        a4 = 0.05

        soc_clipped = np.clip(soc, 0.0, 1.0)

        docv = a1 + 2*a2*soc_clipped + 3*a3*soc_clipped**2 + 4*a4*soc_clipped**3

        return docv

    def _check_divergence(self, innovation: float):
        """
        Check for filter divergence based on innovation and covariance.

        Args:
            innovation: Measurement innovation (residual)
        """
        # Check innovation magnitude
        if abs(innovation) > self.config.max_innovation:
            logger.warning(f"Large innovation detected: {innovation:.4f} V")
            self.divergence_detected = True

        # Check covariance trace
        trace = np.trace(self.state.P)
        if trace > self.config.max_covariance_trace:
            logger.warning(f"Large covariance trace: {trace:.4f}")
            self.divergence_detected = True

    # Getter methods for convenience

    def get_soc(self) -> float:
        """Get current State of Charge estimate."""
        return float(self.state.x[0])

    def get_soh(self) -> float:
        """Get current State of Health estimate."""
        return float(self.state.x[1])

    def get_r0(self) -> float:
        """Get current R0 (ohmic resistance) estimate."""
        return float(self.state.x[2])

    def get_r1(self) -> float:
        """Get current R1 (polarization resistance) estimate."""
        return float(self.state.x[3])

    def get_c1(self) -> float:
        """Get current C1 (polarization capacitance) estimate."""
        return float(self.state.x[4])

    def get_v1(self) -> float:
        """Get current V1 (RC voltage) estimate."""
        return float(self.state.x[5])

    def get_state_vector(self) -> np.ndarray:
        """Get complete state vector [SoC, SoH, R0, R1, C1, V1]."""
        return self.state.x.copy()

    def get_covariance(self) -> np.ndarray:
        """Get covariance matrix (6x6)."""
        return self.state.P.copy()

    def get_soc_uncertainty(self) -> float:
        """Get SoC uncertainty (standard deviation)."""
        return float(np.sqrt(self.state.P[0, 0]))

    def get_soh_uncertainty(self) -> float:
        """Get SoH uncertainty (standard deviation)."""
        return float(np.sqrt(self.state.P[1, 1]))

    def get_confidence_interval(self, state_index: int, num_std: float = 2.0) -> Tuple[float, float]:
        """
        Get confidence interval for a state variable.

        Args:
            state_index: Index of state variable (0=SoC, 1=SoH, etc.)
            num_std: Number of standard deviations (default 2.0 for ~95% CI)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        mean = self.state.x[state_index]
        std = np.sqrt(self.state.P[state_index, state_index])

        lower = mean - num_std * std
        upper = mean + num_std * std

        return (float(lower), float(upper))

    def is_diverged(self) -> bool:
        """Check if filter has diverged."""
        return self.divergence_detected

    def get_statistics(self) -> dict:
        """Get filter statistics."""
        return {
            'num_predictions': self.num_predictions,
            'num_updates': self.num_updates,
            'last_innovation': self.last_innovation,
            'covariance_trace': float(np.trace(self.state.P)),
            'divergence_detected': self.divergence_detected,
            'soc': self.get_soc(),
            'soh': self.get_soh(),
            'soc_uncertainty': self.get_soc_uncertainty(),
            'soh_uncertainty': self.get_soh_uncertainty()
        }
