"""
Main Hybrid Digital Twin implementation (embedded copy).

This module mirrors the original HybridDigitalTwin class from the
Digital-Twin-in-python project so that the multi-agent system can leverage the
same physics + ML modelling capabilities without an external dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from ..models.ml_model import MLCorrectionModel
from ..models.physics_model import PhysicsBasedModel
from ..utils.exceptions import (
    DigitalTwinError,
    InvalidDataError,
    ModelNotTrainedError,
)
from ..utils.metrics import ModelMetrics
from ..utils.validators import validate_input_data


@dataclass
class PredictionResult:
    """Container for prediction results."""

    physics_prediction: np.ndarray
    ml_correction: np.ndarray
    hybrid_prediction: np.ndarray
    uncertainty: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None


class HybridDigitalTwin:
    """
    Hybrid Digital Twin for Li-ion Battery Capacity Prediction.

    Combines a physics-based degradation model with an ML residual model to
    deliver accurate battery capacity forecasts across the lifecycle.
    """

    def __init__(
        self,
        physics_model: Optional[PhysicsBasedModel] = None,
        ml_model: Optional[MLCorrectionModel] = None,
        config: Optional[Dict] = None,
    ) -> None:
        self.config = config or {}
        self.physics_model = physics_model or PhysicsBasedModel(config=self.config)
        self.ml_model = ml_model or MLCorrectionModel(config=self.config)
        self.is_trained = False
        self.metrics = ModelMetrics()
        self.training_history: Dict = {}

        logger.info("Initialized Hybrid Digital Twin")

    def fit(
        self,
        data: pd.DataFrame,
        target_column: str = "Capacity",
        validation_split: float = 0.2,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Train the hybrid digital twin on battery data.
        """
        try:
            logger.info("Starting hybrid digital twin training")

            validate_input_data(data, required_columns=[target_column])

            split_idx = int(len(data) * (1 - validation_split))
            train_data = data.iloc[:split_idx]
            val_data = data.iloc[split_idx:]

            logger.info("Training physics-based model")
            physics_metrics = self.physics_model.fit(train_data, target_column)

            train_physics_pred = self.physics_model.predict(train_data)
            val_physics_pred = self.physics_model.predict(val_data)

            train_residuals = train_data[target_column].values - train_physics_pred
            val_residuals = val_data[target_column].values - val_physics_pred

            logger.info("Training ML correction model")
            ml_features_train = self._extract_ml_features(
                train_data, train_physics_pred
            )
            ml_features_val = self._extract_ml_features(val_data, val_physics_pred)

            ml_metrics = self.ml_model.fit(
                ml_features_train,
                train_residuals,
                validation_data=(ml_features_val, val_residuals),
                **kwargs,
            )

            hybrid_pred_train = train_physics_pred + self.ml_model.predict(
                ml_features_train
            )
            hybrid_pred_val = val_physics_pred + self.ml_model.predict(ml_features_val)

            training_metrics = self._calculate_metrics(
                train_data[target_column].values,
                val_data[target_column].values,
                hybrid_pred_train,
                hybrid_pred_val,
            )

            self.training_history = {
                "physics_metrics": physics_metrics,
                "ml_metrics": ml_metrics,
                "hybrid_metrics": training_metrics,
                "training_size": len(train_data),
                "validation_size": len(val_data),
            }

            self.is_trained = True
            logger.success("Hybrid digital twin training completed")

            return training_metrics

        except Exception as exc:
            logger.error(f"Training failed: {exc}")
            raise DigitalTwinError(f"Training failed: {exc}") from exc

    def predict(
        self,
        data: pd.DataFrame,
        return_uncertainty: bool = False,
        return_components: bool = False,
    ) -> Union[np.ndarray, PredictionResult]:
        """
        Make predictions using the hybrid digital twin.
        """
        if not self.is_trained:
            raise ModelNotTrainedError("Models must be trained before prediction")

        try:
            validate_input_data(data)

            physics_pred = self.physics_model.predict(data)
            ml_features = self._extract_ml_features(data, physics_pred)
            ml_correction = self.ml_model.predict(ml_features)

            hybrid_pred = physics_pred + ml_correction

            uncertainty = None
            if return_uncertainty:
                uncertainty = self._estimate_uncertainty(ml_features)

            if return_components:
                return PredictionResult(
                    physics_prediction=physics_pred,
                    ml_correction=ml_correction,
                    hybrid_prediction=hybrid_pred,
                    uncertainty=uncertainty,
                    metadata={
                        "n_samples": len(data),
                        "feature_dimensions": ml_features.shape[1]
                        if ml_features.ndim > 1
                        else 1,
                    },
                )

            return hybrid_pred

        except Exception as exc:
            logger.error(f"Prediction failed: {exc}")
            raise DigitalTwinError(f"Prediction failed: {exc}") from exc

    def predict_future(
        self,
        cycles: np.ndarray,
        temperature: float,
        charge_time: float,
        initial_capacity: float,
        return_uncertainty: bool = False,
    ) -> Union[np.ndarray, PredictionResult]:
        """
        Predict future capacity for upcoming cycles using synthetic input data.
        """
        if not self.is_trained:
            raise ModelNotTrainedError("Models must be trained before prediction")

        future_data = pd.DataFrame(
            {
                "id_cycle": cycles,
                "Temperature_measured": temperature,
                "Time": charge_time,
                "Capacity": initial_capacity,
            }
        )

        return self.predict(
            future_data, return_uncertainty=return_uncertainty, return_components=True
        )

    def evaluate(
        self, test_data: pd.DataFrame, target_column: str = "Capacity"
    ) -> Dict[str, float]:
        """
        Evaluate the hybrid model on hold-out data.
        """
        if not self.is_trained:
            raise ModelNotTrainedError("Models must be trained before evaluation")

        predictions = self.predict(test_data)
        actual = test_data[target_column].values
        return self.metrics.calculate_all_metrics(actual, predictions)

    def save_model(self, filepath: Union[str, Path]) -> None:
        """Persist the trained model to disk."""
        if not self.is_trained:
            raise ModelNotTrainedError("Cannot save untrained model")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "physics_model": self.physics_model,
            "ml_model": self.ml_model,
            "training_history": self.training_history,
            "config": self.config,
            "version": "1.0.0",
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: Union[str, Path]) -> "HybridDigitalTwin":
        """Load a trained model from disk."""
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = joblib.load(filepath)

        instance = cls(
            physics_model=model_data["physics_model"],
            ml_model=model_data["ml_model"],
            config=model_data.get("config", {}),
        )

        instance.training_history = model_data.get("training_history", {})
        instance.is_trained = True

        logger.info(f"Model loaded from {filepath}")
        return instance

    def _extract_ml_features(
        self, data: pd.DataFrame, physics_pred: np.ndarray
    ) -> np.ndarray:
        """Build feature matrix for the ML correction model."""
        features = [physics_pred.reshape(-1, 1)]

        if "Temperature_measured" in data.columns:
            features.append(data["Temperature_measured"].values.reshape(-1, 1))

        if "id_cycle" in data.columns:
            features.append(data["id_cycle"].values.reshape(-1, 1))

        if "Time" in data.columns:
            features.append(data["Time"].values.reshape(-1, 1))

        return np.hstack(features)

    def _estimate_uncertainty(self, features: np.ndarray) -> np.ndarray:
        """Placeholder uncertainty estimation."""
        return np.ones(len(features)) * 0.01

    def _calculate_metrics(
        self,
        y_train_true: np.ndarray,
        y_val_true: np.ndarray,
        y_train_pred: np.ndarray,
        y_val_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate metrics for training and validation sets."""
        metrics = {}

        train_metrics = self.metrics.calculate_all_metrics(y_train_true, y_train_pred)
        for key, value in train_metrics.items():
            metrics[f"train_{key}"] = value

        val_metrics = self.metrics.calculate_all_metrics(y_val_true, y_val_pred)
        for key, value in val_metrics.items():
            metrics[f"val_{key}"] = value

        return metrics

