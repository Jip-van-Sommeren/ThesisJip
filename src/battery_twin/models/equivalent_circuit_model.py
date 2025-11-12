"""
Equivalent Circuit Model for Battery Voltage Response

Implements a first-order RC equivalent circuit model (Thevenin model):

Circuit Model:
    V_terminal = V_oc - I × R0 - V1

where:
    V_oc = Open circuit voltage (function of SoC)
    R0 = Series resistance (Ohmic resistance)
    V1 = Voltage across RC parallel branch
    I = Current (positive for discharge, negative for charge)

RC Dynamics:
    dV1/dt = -V1/(R1×C1) + I/C1

where:
    R1 = Polarization resistance
    C1 = Polarization capacitance

This model captures:
- Instantaneous voltage drop (R0)
- Transient voltage response (R1-C1 branch)
- Open circuit voltage vs. SoC relationship

The model is widely used for battery state estimation and control.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class CircuitParameters:
    """
    Parameters for equivalent circuit model.

    Attributes:
        R0: Series resistance in Ohms (typical: 0.01-0.1Ω)
        R1: Polarization resistance in Ohms (typical: 0.01-0.05Ω)
        C1: Polarization capacitance in Farads (typical: 1000-10000F)
        V_oc_nominal: Nominal open circuit voltage in V (e.g., 3.7V for Li-ion)
    """
    R0: float = 0.05  # Ohms
    R1: float = 0.03  # Ohms
    C1: float = 2000.0  # Farads
    V_oc_nominal: float = 3.7  # Volts

    def __post_init__(self):
        """Validate parameters."""
        if self.R0 < 0:
            raise ValueError(f"R0 must be non-negative, got {self.R0}")
        if self.R1 < 0:
            raise ValueError(f"R1 must be non-negative, got {self.R1}")
        if self.C1 <= 0:
            raise ValueError(f"C1 must be positive, got {self.C1}")
        if not (2.0 <= self.V_oc_nominal <= 5.0):
            raise ValueError(f"V_oc_nominal out of range [2, 5]V, got {self.V_oc_nominal}")

    @property
    def tau(self) -> float:
        """Time constant τ = R1 × C1 in seconds."""
        return self.R1 * self.C1

    def to_dict(self) -> Dict[str, float]:
        """Convert parameters to dictionary."""
        return {
            'R0': self.R0,
            'R1': self.R1,
            'C1': self.C1,
            'V_oc_nominal': self.V_oc_nominal
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'CircuitParameters':
        """Create parameters from dictionary."""
        return cls(
            R0=data.get('R0', 0.05),
            R1=data.get('R1', 0.03),
            C1=data.get('C1', 2000.0),
            V_oc_nominal=data.get('V_oc_nominal', 3.7)
        )


class EquivalentCircuitModel:
    """
    First-order RC equivalent circuit model for battery dynamics.

    This model simulates battery voltage response to current inputs,
    capturing both instantaneous (resistive) and transient (RC) behavior.

    The model uses:
    - R0: Captures instantaneous voltage drop
    - R1-C1: Captures transient voltage response with time constant τ = R1×C1
    - V_oc(SoC): Open circuit voltage as function of state of charge

    Example:
        >>> params = CircuitParameters(R0=0.05, R1=0.03, C1=2000.0)
        >>> model = EquivalentCircuitModel(params)
        >>>
        >>> # Compute voltage response to 2A discharge current
        >>> voltage = model.compute_voltage(
        ...     current=2.0,
        ...     soc=0.8,
        ...     V1_prev=0.0,
        ...     dt=1.0
        ... )
        >>> print(f"Terminal voltage: {voltage:.3f} V")
    """

    def __init__(self, parameters: Optional[CircuitParameters] = None):
        """
        Initialize equivalent circuit model.

        Args:
            parameters: Circuit parameters (uses defaults if None)
        """
        self.params = parameters or CircuitParameters()

        # State variables
        self.V1 = 0.0  # Voltage across RC branch
        self.last_update_time = 0.0

        logger.info(
            f"Initialized EquivalentCircuitModel: R0={self.params.R0}Ω, "
            f"R1={self.params.R1}Ω, C1={self.params.C1}F, τ={self.params.tau:.1f}s"
        )

    def compute_open_circuit_voltage(self, soc: float) -> float:
        """
        Compute open circuit voltage as function of SoC.

        Uses a polynomial approximation of typical Li-ion OCV-SoC curve:
        V_oc(SoC) ≈ a0 + a1×SoC + a2×SoC² + a3×SoC³

        Args:
            soc: State of charge (0 to 1)

        Returns:
            Open circuit voltage in Volts
        """
        if not (0 <= soc <= 1):
            logger.warning(f"SoC {soc} out of range [0, 1], clipping")
            soc = np.clip(soc, 0, 1)

        # Polynomial coefficients for typical Li-ion OCV curve
        # These approximate a curve from ~3.0V (0% SoC) to ~4.2V (100% SoC)
        a0 = 3.0
        a1 = 1.5
        a2 = -0.6
        a3 = 0.3

        V_oc = a0 + a1 * soc + a2 * soc**2 + a3 * soc**3

        return float(V_oc)

    def update_V1(self, current: float, dt: float) -> float:
        """
        Update voltage across RC branch using discretized dynamics.

        Discretized equation:
        V1[k+1] = V1[k] × exp(-dt/τ) + I × R1 × (1 - exp(-dt/τ))

        Args:
            current: Current in Amperes (positive for discharge)
            dt: Time step in seconds

        Returns:
            Updated V1 voltage
        """
        tau = self.params.tau

        if tau <= 0:
            logger.warning("Time constant τ ≤ 0, setting V1 = I×R1")
            self.V1 = current * self.params.R1
            return self.V1

        # Discretized update
        exp_term = np.exp(-dt / tau)
        self.V1 = self.V1 * exp_term + current * self.params.R1 * (1 - exp_term)

        return self.V1

    def compute_voltage(
        self,
        current: float,
        soc: float,
        V1_prev: Optional[float] = None,
        dt: float = 1.0
    ) -> float:
        """
        Compute terminal voltage for given current and state.

        V_terminal = V_oc(SoC) - I × R0 - V1

        Args:
            current: Current in Amperes (positive for discharge)
            soc: State of charge (0 to 1)
            V1_prev: Previous V1 value (uses internal state if None)
            dt: Time step for V1 update in seconds

        Returns:
            Terminal voltage in Volts
        """
        # Update V1 if previous value provided, otherwise use internal state
        if V1_prev is not None:
            self.V1 = V1_prev

        # Update V1 dynamics
        V1_new = self.update_V1(current, dt)

        # Compute OCV
        V_oc = self.compute_open_circuit_voltage(soc)

        # Compute terminal voltage
        V_terminal = V_oc - current * self.params.R0 - V1_new

        return float(V_terminal)

    def compute_voltage_batch(
        self,
        currents: np.ndarray,
        socs: np.ndarray,
        dt: float = 1.0,
        V1_init: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute voltage response for a sequence of currents (time series).

        Args:
            currents: Array of current values in Amperes
            socs: Array of SoC values (0 to 1)
            dt: Time step in seconds
            V1_init: Initial V1 value

        Returns:
            Tuple of (voltages, V1_history)
        """
        if len(currents) != len(socs):
            raise ValueError("currents and socs must have same length")

        n = len(currents)
        voltages = np.zeros(n)
        V1_history = np.zeros(n)

        # Initialize V1
        self.V1 = V1_init

        # Simulate time series
        for i in range(n):
            voltage = self.compute_voltage(currents[i], socs[i], dt=dt)
            voltages[i] = voltage
            V1_history[i] = self.V1

        return voltages, V1_history

    def identify_parameters(
        self,
        currents: np.ndarray,
        voltages: np.ndarray,
        socs: np.ndarray,
        dt: float = 1.0
    ) -> Dict[str, float]:
        """
        Identify circuit parameters from measured data using least squares.

        This is a simplified identification that assumes:
        - V_oc can be estimated from OCV-SoC curve
        - Parameters are time-invariant

        Args:
            currents: Measured current time series
            voltages: Measured terminal voltage time series
            socs: Estimated SoC time series
            dt: Time step

        Returns:
            Dictionary with identified parameters and fit quality
        """
        from scipy.optimize import minimize

        logger.info("Identifying circuit parameters from data")

        n = len(currents)
        if not (n == len(voltages) == len(socs)):
            raise ValueError("Input arrays must have same length")

        # Objective function: minimize voltage prediction error
        def objective(params):
            R0, R1, C1 = params

            # Create temporary model with these parameters
            temp_params = CircuitParameters(
                R0=R0,
                R1=R1,
                C1=C1,
                V_oc_nominal=self.params.V_oc_nominal
            )
            temp_model = EquivalentCircuitModel(temp_params)

            # Simulate
            pred_voltages, _ = temp_model.compute_voltage_batch(
                currents, socs, dt=dt, V1_init=0.0
            )

            # Compute RMSE
            rmse = np.sqrt(np.mean((voltages - pred_voltages) ** 2))
            return rmse

        # Initial guess and bounds
        x0 = [self.params.R0, self.params.R1, self.params.C1]
        bounds = [
            (0.001, 1.0),    # R0: 1mΩ to 1Ω
            (0.001, 0.5),    # R1: 1mΩ to 0.5Ω
            (100.0, 50000.0) # C1: 100F to 50kF
        ]

        # Optimize
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

        if result.success:
            # Update parameters
            self.params.R0 = result.x[0]
            self.params.R1 = result.x[1]
            self.params.C1 = result.x[2]

            logger.info(
                f"Parameter identification successful: "
                f"R0={self.params.R0:.5f}Ω, R1={self.params.R1:.5f}Ω, "
                f"C1={self.params.C1:.1f}F"
            )
        else:
            logger.warning(f"Parameter identification failed: {result.message}")

        # Compute final fit quality
        pred_voltages, _ = self.compute_voltage_batch(currents, socs, dt=dt, V1_init=0.0)
        rmse = np.sqrt(np.mean((voltages - pred_voltages) ** 2))
        mae = np.mean(np.abs(voltages - pred_voltages))

        return {
            'R0': self.params.R0,
            'R1': self.params.R1,
            'C1': self.params.C1,
            'rmse': float(rmse),
            'mae': float(mae),
            'success': result.success
        }

    def reset_state(self):
        """Reset internal state variables."""
        self.V1 = 0.0
        self.last_update_time = 0.0
        logger.debug("Reset circuit model state")

    def update_parameters(self, **kwargs):
        """
        Update circuit parameters.

        Args:
            **kwargs: Parameters to update (R0, R1, C1, V_oc_nominal)
        """
        if 'R0' in kwargs:
            self.params.R0 = kwargs['R0']
            logger.info(f"Updated R0 to {self.params.R0}Ω")

        if 'R1' in kwargs:
            self.params.R1 = kwargs['R1']
            logger.info(f"Updated R1 to {self.params.R1}Ω")

        if 'C1' in kwargs:
            self.params.C1 = kwargs['C1']
            logger.info(f"Updated C1 to {self.params.C1}F")

        if 'V_oc_nominal' in kwargs:
            self.params.V_oc_nominal = kwargs['V_oc_nominal']
            logger.info(f"Updated V_oc_nominal to {self.params.V_oc_nominal}V")

    def get_parameters(self) -> CircuitParameters:
        """Get current circuit parameters."""
        return self.params

    def export_state(self) -> Dict:
        """Export model state for serialization."""
        return {
            'parameters': self.params.to_dict(),
            'V1': self.V1,
            'last_update_time': self.last_update_time
        }

    def import_state(self, state: Dict):
        """Import model state from serialization."""
        self.params = CircuitParameters.from_dict(state['parameters'])
        self.V1 = state.get('V1', 0.0)
        self.last_update_time = state.get('last_update_time', 0.0)
        logger.info(f"Imported circuit model state: R0={self.params.R0}Ω, V1={self.V1}V")


__all__ = [
    'EquivalentCircuitModel',
    'CircuitParameters',
]
