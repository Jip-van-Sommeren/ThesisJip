"""
Physics-Based Battery Degradation Model

Hybrid-aware version that wraps the embedded HybridDigitalTwin physics model.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.battery_twin.hybrid.models.physics_model import PhysicsBasedModel


@dataclass
class DegradationParameters:
    """Parameters for the physics-based degradation model."""

    k: float = 0.13
    Q0: float = 2.0
    T_ref: float = 25.0

    def __post_init__(self):
        if self.k <= 0:
            raise ValueError(f"Degradation coefficient k must be positive, got {self.k}")
        if self.Q0 <= 0:
            raise ValueError(f"Initial capacity Q0 must be positive, got {self.Q0}")
        if not (-50 <= self.T_ref <= 100):
            raise ValueError(
                f"Reference temperature out of range [-50, 100]°C, got {self.T_ref}"
            )

    def to_dict(self) -> Dict[str, float]:
        return {"k": self.k, "Q0": self.Q0, "T_ref": self.T_ref}

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "DegradationParameters":
        return cls(
            k=data.get("k", 0.13),
            Q0=data.get("Q0", 2.0),
            T_ref=data.get("T_ref", 25.0),
        )


class PhysicsDegradationModel:
    """
    Wrapper around the embedded HybridDigitalTwin physics model so the rest of the
    codebase can keep using the legacy interface.
    """

    def __init__(self, parameters: Optional[DegradationParameters] = None):
        self.params = parameters or DegradationParameters()
        self.model = PhysicsBasedModel(
            config={"physics_k": self.params.k, "temperature_ref": self.params.T_ref}
        )
        self.model.params.initial_capacity = self.params.Q0
        self.model.is_fitted = True
        self.is_fitted = False
        self.fit_history: Dict = {}

        logger.info(
            "Initialized PhysicsDegradationModel (hybrid-backed) with k=%s, Q0=%s",
            self.params.k,
            self.params.Q0,
        )

    def compute_degradation_factor(
        self, cycle: int, temperature: float, charge_time: float
    ) -> float:
        if charge_time <= 0:
            charge_time = 1e-6
            logger.warning("Invalid charge_time provided; using 1e-6 for stability")

        if cycle < 0:
            cycle = 0
            logger.warning("Negative cycle number encountered; using 0")

        return self.params.k * temperature * cycle / charge_time

    def _build_dataframe(
        self, cycles: np.ndarray, temperatures: np.ndarray, charge_times: np.ndarray
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "id_cycle": cycles,
                "Temperature_measured": temperatures,
                "Time": charge_times,
                "Capacity": self.model.params.initial_capacity,
            }
        )

    def predict_capacity(
        self, cycle: int, temperature: float, charge_time: float
    ) -> float:
        df = self._build_dataframe(
            np.array([cycle]),
            np.array([temperature]),
            np.array([max(charge_time, 1e-6)]),
        )
        prediction = self.model.predict(df)[0]
        return float(prediction)

    def predict_capacity_batch(
        self,
        cycles: np.ndarray,
        temperatures: np.ndarray,
        charge_times: np.ndarray,
    ) -> np.ndarray:
        if not (len(cycles) == len(temperatures) == len(charge_times)):
            raise ValueError("Input arrays must have the same length")

        df = self._build_dataframe(
            cycles,
            temperatures,
            np.where(charge_times <= 0, 1e-6, charge_times),
        )
        predictions = self.model.predict(df)
        return predictions

    def fit(
        self,
        cycles: np.ndarray,
        temperatures: np.ndarray,
        charge_times: np.ndarray,
        actual_capacities: np.ndarray,
        **_: Dict,
    ) -> Dict[str, float]:
        df = pd.DataFrame(
            {
                "id_cycle": cycles,
                "Temperature_measured": temperatures,
                "Time": np.where(charge_times <= 0, 1e-6, charge_times),
                "Capacity": actual_capacities,
            }
        )

        _ = self.model.fit(df, target_column="Capacity")
        self.params.Q0 = self.model.params.initial_capacity or self.params.Q0
        self.params.k = self.model.params.k
        self.is_fitted = True
        predictions = self.predict_capacity_batch(cycles, temperatures, charge_times)
        metrics = self._compute_metrics(actual_capacities, predictions)
        self.fit_history = {"metrics": metrics, "n_samples": len(df)}

        return metrics

    def predict_end_of_life(
        self,
        temperature: float,
        charge_time: float,
        capacity_threshold: float = 0.8,
        max_cycles: int = 1000,
    ) -> Tuple[int, float]:
        for cycle in range(1, max_cycles + 1):
            capacity = self.predict_capacity(cycle, temperature, charge_time)
            normalized_capacity = capacity / self.params.Q0

            if normalized_capacity < capacity_threshold:
                logger.info(
                    "End of life predicted at cycle %d (capacity: %.4f Ah, %.2f%% of Q0)",
                    cycle,
                    capacity,
                    normalized_capacity * 100,
                )
                return cycle, capacity

        final_capacity = self.predict_capacity(max_cycles, temperature, charge_time)
        logger.warning(
            "End of life not reached within %d cycles. Final capacity %.4f Ah",
            max_cycles,
            final_capacity,
        )
        return max_cycles, final_capacity

    def update_parameters(self, **kwargs):
        if "k" in kwargs:
            self.params.k = kwargs["k"]
            self.model.params.k = kwargs["k"]
            logger.info("Updated k to %s", self.params.k)

        if "Q0" in kwargs:
            self.params.Q0 = kwargs["Q0"]
            self.model.params.initial_capacity = kwargs["Q0"]
            logger.info("Updated Q0 to %s Ah", self.params.Q0)

        if "T_ref" in kwargs:
            self.params.T_ref = kwargs["T_ref"]
            logger.info("Updated T_ref to %s°C", self.params.T_ref)

        self.is_fitted = True
        self.model.is_fitted = True

    def get_parameters(self) -> DegradationParameters:
        return self.params

    def export_state(self) -> Dict:
        return {
            "parameters": self.params.to_dict(),
            "is_fitted": self.is_fitted,
            "fit_history": self.fit_history,
        }

    def import_state(self, state: Dict):
        self.params = DegradationParameters.from_dict(state["parameters"])
        self.is_fitted = state.get("is_fitted", False)
        self.fit_history = state.get("fit_history", {})

        self.model.params.k = self.params.k
        self.model.params.initial_capacity = self.params.Q0
        self.model.is_fitted = self.is_fitted

        logger.info(
            "Imported model state: k=%s, Q0=%s", self.params.k, self.params.Q0
        )

    def _compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(
            np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-6, y_true))
        ) * 100

        return {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "mape": float(mape),
        }

        # Initial parameter guess
        x0 = []
        param_names = []

        if optimize_Q0:
            x0.append(actual_capacities[0])  # Use first capacity as Q0 guess
            param_names.append('Q0')

        if optimize_k:
            x0.append(self.params.k)
            param_names.append('k')

        if not x0:
            logger.warning("No parameters to optimize, returning current predictions")
            predictions = self.predict_capacity_batch(cycles, temperatures, charge_times)
            metrics = self._compute_metrics(actual_capacities, predictions)
            return metrics

        # Objective function: minimize RMSE
        def objective(params):
            idx = 0
            if optimize_Q0:
                Q0 = params[idx]
                idx += 1
            else:
                Q0 = self.params.Q0

            if optimize_k:
                k = params[idx]
                idx += 1
            else:
                k = self.params.k

            # Predict with current parameters
            f_d = k * temperatures * cycles / np.where(charge_times <= 0, 1e-6, charge_times)
            predictions = Q0 * np.exp(-f_d)

            # Compute RMSE
            rmse = np.sqrt(np.mean((actual_capacities - predictions) ** 2))
            return rmse

        # Bounds for parameters
        bounds = []
        if optimize_Q0:
            bounds.append((0.1, 10.0))  # Q0 reasonable range
        if optimize_k:
            bounds.append((0.01, 1.0))  # k reasonable range

        # Optimize
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

        # Update parameters
        idx = 0
        if optimize_Q0:
            self.params.Q0 = result.x[idx]
            idx += 1
            logger.info(f"Optimized Q0: {self.params.Q0:.4f} Ah")

        if optimize_k:
            self.params.k = result.x[idx]
            idx += 1
            logger.info(f"Optimized k: {self.params.k:.6f}")

        self.is_fitted = True

        # Compute final metrics
        predictions = self.predict_capacity_batch(cycles, temperatures, charge_times)
        metrics = self._compute_metrics(actual_capacities, predictions)

        self.fit_history = {
            'parameters': self.params.to_dict(),
            'metrics': metrics,
            'n_samples': len(cycles),
            'optimization_success': result.success
        }

        logger.info(f"Fit complete. RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")

        return metrics

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute prediction metrics."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # MAPE (handle zero values)
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-6, y_true))) * 100

        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape)
        }

    def predict_end_of_life(
        self,
        temperature: float,
        charge_time: float,
        capacity_threshold: float = 0.8,
        max_cycles: int = 1000
    ) -> Tuple[int, float]:
        """
        Predict cycle number when capacity drops below threshold.

        Args:
            temperature: Operating temperature in °C
            charge_time: Charge time per cycle in seconds
            capacity_threshold: Threshold as fraction of Q0 (default: 0.8)
            max_cycles: Maximum cycles to check

        Returns:
            Tuple of (end_of_life_cycle, final_capacity)
        """
        for cycle in range(1, max_cycles + 1):
            capacity = self.predict_capacity(cycle, temperature, charge_time)
            normalized_capacity = capacity / self.params.Q0

            if normalized_capacity < capacity_threshold:
                logger.info(
                    f"End of life predicted at cycle {cycle} "
                    f"(capacity: {capacity:.4f} Ah, {normalized_capacity:.2%} of Q0)"
                )
                return cycle, capacity

        # Didn't reach threshold within max_cycles
        final_capacity = self.predict_capacity(max_cycles, temperature, charge_time)
        logger.warning(
            f"End of life not reached within {max_cycles} cycles. "
            f"Final capacity: {final_capacity:.4f} Ah"
        )
        return max_cycles, final_capacity

    def update_parameters(self, **kwargs):
        """
        Update model parameters.

        Args:
            **kwargs: Parameters to update (k, Q0, T_ref)
        """
        if 'k' in kwargs:
            self.params.k = kwargs['k']
            logger.info(f"Updated k to {self.params.k}")

        if 'Q0' in kwargs:
            self.params.Q0 = kwargs['Q0']
            logger.info(f"Updated Q0 to {self.params.Q0} Ah")

        if 'T_ref' in kwargs:
            self.params.T_ref = kwargs['T_ref']
            logger.info(f"Updated T_ref to {self.params.T_ref}°C")

    def get_parameters(self) -> DegradationParameters:
        """Get current model parameters."""
        return self.params

    def export_state(self) -> Dict:
        """Export model state for serialization."""
        return {
            'parameters': self.params.to_dict(),
            'is_fitted': self.is_fitted,
            'fit_history': self.fit_history
        }

    def import_state(self, state: Dict):
        """Import model state from serialization."""
        self.params = DegradationParameters.from_dict(state['parameters'])
        self.is_fitted = state.get('is_fitted', False)
        self.fit_history = state.get('fit_history', {})
        logger.info(f"Imported model state: k={self.params.k}, Q0={self.params.Q0}")


__all__ = [
    'PhysicsDegradationModel',
    'DegradationParameters',
]
