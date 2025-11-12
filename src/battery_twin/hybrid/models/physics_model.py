"""
Physics-based battery degradation model (embedded copy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

from ..utils.exceptions import InvalidParameterError, ModelError


@dataclass
class PhysicsModelParameters:
    """Parameters for the physics-based model."""

    k: float = 0.13
    initial_capacity: Optional[float] = None
    temperature_ref: float = 25.0

    def __post_init__(self):
        if self.k <= 0:
            raise InvalidParameterError("Degradation coefficient k must be positive")
        if not (-50 <= self.temperature_ref <= 100):
            raise InvalidParameterError("Reference temperature out of realistic range")


class PhysicsBasedModel:
    """
    Physics-based Li-ion battery degradation model based on Xu et al. (2016).
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        config = config or {}
        self.params = PhysicsModelParameters(
            k=config.get("physics_k", 0.13),
            temperature_ref=config.get("temperature_ref", 25.0),
        )
        self.is_fitted = False
        self.fit_metrics: Dict = {}

        logger.debug("Initialized PhysicsBasedModel with %s", self.params)

    def fit(
        self, data: pd.DataFrame, target_column: str = "Capacity"
    ) -> Dict[str, float]:
        """Fit the physics model to training data."""
        try:
            logger.info("Fitting physics-based model")

            required_cols = ["id_cycle", "Temperature_measured", "Time", target_column]
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ModelError(f"Missing required columns: {missing_cols}")

            self.params.initial_capacity = data[target_column].iloc[0]
            logger.debug(
                "Estimated initial capacity: %.4f Ah", self.params.initial_capacity
            )

            physics_pred = self._predict_physics(data)
            actual_capacity = data[target_column].values

            self.fit_metrics = self._calculate_physics_metrics(
                actual_capacity, physics_pred
            )

            self.is_fitted = True
            logger.success(
                "Physics model fitted successfully. RMSE: %.4f",
                self.fit_metrics["rmse"],
            )

            return self.fit_metrics

        except Exception as exc:
            logger.error("Physics model fitting failed: %s", exc)
            raise ModelError(f"Physics model fitting failed: {exc}") from exc

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Generate physics-based capacity predictions."""
        if not self.is_fitted:
            raise ModelError("Model must be fitted before making predictions")

        try:
            return self._predict_physics(data)
        except Exception as exc:
            logger.error("Physics prediction failed: %s", exc)
            raise ModelError(f"Physics prediction failed: {exc}") from exc

    def _predict_physics(self, data: pd.DataFrame) -> np.ndarray:
        """Internal method to compute physics-based predictions."""
        cycles = data["id_cycle"].values
        temperature = data["Temperature_measured"].values
        charge_time = data["Time"].values

        charge_time = np.where(charge_time <= 0, 1e-6, charge_time)

        f_d = self.params.k * temperature * cycles / charge_time
        capacity_predictions = self.params.initial_capacity * np.exp(-f_d)

        return capacity_predictions

    def _calculate_physics_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate physics model performance metrics."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape": mape,
            "n_samples": len(y_true),
        }

    def get_degradation_factor(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate the degradation factor f_d for given conditions."""
        cycles = data["id_cycle"].values
        temperature = data["Temperature_measured"].values
        charge_time = data["Time"].values
        charge_time = np.where(charge_time <= 0, 1e-6, charge_time)
        return self.params.k * temperature * cycles / charge_time

    def predict_lifetime(
        self,
        max_cycles: int,
        temperature: float,
        charge_time: float,
        capacity_threshold: float = 0.8,
    ) -> Dict[str, float]:
        """Predict battery lifetime until capacity drops below threshold."""
        if not self.is_fitted:
            raise ModelError("Model must be fitted before lifetime prediction")

        cycles = np.arange(1, max_cycles + 1)
        data = pd.DataFrame(
            {
                "id_cycle": cycles,
                "Temperature_measured": temperature,
                "Time": charge_time,
            }
        )

        predicted_capacities = self.predict(data)
        normalized_capacities = predicted_capacities / self.params.initial_capacity

        below_threshold = normalized_capacities < capacity_threshold

        if np.any(below_threshold):
            end_of_life_cycle = int(cycles[np.where(below_threshold)[0][0]])
            end_of_life_capacity = float(predicted_capacities[below_threshold][0])
        else:
            end_of_life_cycle = max_cycles
            end_of_life_capacity = float(predicted_capacities[-1])

        return {
            "end_of_life_cycle": end_of_life_cycle,
            "end_of_life_capacity": end_of_life_capacity,
            "capacity_threshold": capacity_threshold,
            "final_normalized_capacity": float(normalized_capacities[-1]),
            "total_cycles_simulated": max_cycles,
        }

    def sensitivity_analysis(
        self, data: pd.DataFrame, parameter_ranges: Dict[str, tuple]
    ) -> Dict[str, np.ndarray]:
        """Perform sensitivity analysis on model parameters."""
        results = {}
        base_prediction = self.predict(data)

        for param_name, (min_val, max_val) in parameter_ranges.items():
            if param_name == "k":
                original_k = self.params.k
                k_values = np.linspace(min_val, max_val, 10)
                predictions = []

                for k_val in k_values:
                    self.params.k = k_val
                    predictions.append(self._predict_physics(data))

                results[param_name] = {
                    "values": k_values,
                    "predictions": np.array(predictions),
                    "rmse_vs_base": [
                        np.sqrt(np.mean((pred - base_prediction) ** 2))
                        for pred in predictions
                    ],
                }

                self.params.k = original_k

        return results

    def export_parameters(self) -> Dict:
        """Export model parameters for serialization."""
        return {
            "k": self.params.k,
            "initial_capacity": self.params.initial_capacity,
            "temperature_ref": self.params.temperature_ref,
            "is_fitted": self.is_fitted,
            "fit_metrics": self.fit_metrics,
        }

    def import_parameters(self, params: Dict) -> None:
        """Import model parameters from serialization."""
        self.params.k = params["k"]
        self.params.initial_capacity = params["initial_capacity"]
        self.params.temperature_ref = params["temperature_ref"]
        self.is_fitted = params["is_fitted"]
        self.fit_metrics = params["fit_metrics"]

