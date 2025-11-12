"""
Machine learning residual correction model (embedded copy).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

try:
    from typing import Literal
except ImportError:  # pragma: no cover
    from typing_extensions import Literal  # type: ignore

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler

try:  # pragma: no cover - heavy optional dependency
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import callbacks, layers, optimizers
except ImportError:  # pragma: no cover
    tf = None
    keras = None
    callbacks = None
    layers = None
    optimizers = None

from ..utils.exceptions import InvalidParameterError, ModelError


@dataclass
class MLModelConfig:
    """Configuration for the ML correction model."""

    hidden_layers: List[int] = None
    dropout_rate: float = 0.1
    activation: str = "relu"
    output_activation: Optional[str] = None
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    l1_regularization: float = 0.0
    l2_regularization: float = 0.001
    optimizer: str = "adam"
    loss_function: str = "mse"

    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [64, 64]
        if not 0 <= self.dropout_rate <= 1:
            raise InvalidParameterError("Dropout rate must be between 0 and 1")
        if self.learning_rate <= 0:
            raise InvalidParameterError("Learning rate must be positive")
        if self.batch_size <= 0:
            raise InvalidParameterError("Batch size must be positive")


class MLCorrectionModel:
    """
    Neural network that learns residual corrections for physics predictions.
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        if keras is None or tf is None:
            raise ImportError(
                "TensorFlow is required for MLCorrectionModel but is not installed. "
                "Install tensorflow>=2.0 to enable the hybrid ML backend."
            )
        self.config = MLModelConfig(
            **({} if config is None else config.get("ml_model", {}))
        )
        self.model: Optional[keras.Model] = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.training_history: Dict = {}
        self.feature_names: List[str] = []

        np.random.seed(42)
        tf.random.set_seed(42)

        logger.debug("Initialized MLCorrectionModel with config: %s", self.config)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        feature_names: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Train the ML correction model."""
        try:
            logger.info("Training ML correction model")

            self.feature_names = feature_names or [
                f"feature_{i}" for i in range(X.shape[1])
            ]

            X_scaled = self.scaler.fit_transform(X)

            if validation_data is not None:
                X_val, y_val = validation_data
                X_val_scaled = self.scaler.transform(X_val)
                validation_data_scaled = (X_val_scaled, y_val)
            else:
                validation_data_scaled = None

            self.model = self._build_model(input_dim=X_scaled.shape[1])
            callbacks_list = self._setup_callbacks()

            history = self.model.fit(
                X_scaled,
                y,
                batch_size=self.config.batch_size,
                epochs=self.config.epochs,
                validation_data=validation_data_scaled,
                callbacks=callbacks_list,
                verbose=1,
                **kwargs,
            )

            self.training_history = {
                "loss": history.history["loss"],
                "val_loss": history.history.get("val_loss", []),
                "mae": history.history.get("mae", []),
                "val_mae": history.history.get("val_mae", []),
            }

            train_pred = self.model.predict(X_scaled, verbose=0)
            training_metrics = self._calculate_metrics(y, train_pred.flatten())

            if validation_data_scaled is not None:
                X_val_scaled, y_val = validation_data_scaled
                val_pred = self.model.predict(X_val_scaled, verbose=0)
                val_metrics = self._calculate_metrics(y_val, val_pred.flatten())
                training_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

            self.is_fitted = True
            logger.success(
                "ML model training completed. Train RMSE: %.4f",
                training_metrics["rmse"],
            )

            return training_metrics

        except Exception as exc:
            logger.error("ML model training failed: %s", exc)
            raise ModelError(f"ML model training failed: {exc}") from exc

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate ML correction predictions."""
        if not self.is_fitted:
            raise ModelError("Model must be fitted before making predictions")

        try:
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled, verbose=0)
            return predictions.flatten()
        except Exception as exc:
            logger.error("ML prediction failed: %s", exc)
            raise ModelError(f"ML prediction failed: {exc}") from exc

    def predict_with_uncertainty(
        self, X: np.ndarray, n_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with Monte Carlo dropout-based uncertainty."""
        if not self.is_fitted:
            raise ModelError("Model must be fitted before making predictions")

        X_scaled = self.scaler.transform(X)
        predictions = []

        for _ in range(n_samples):
            pred = self.model(X_scaled, training=True)
            predictions.append(pred.numpy().flatten())

        predictions = np.array(predictions)
        return np.mean(predictions, axis=0), np.std(predictions, axis=0)

    def _build_model(self, input_dim: int) -> keras.Model:
        """Build the neural network architecture."""
        inputs = keras.Input(shape=(input_dim,), name="features")
        x = inputs

        for i, units in enumerate(self.config.hidden_layers):
            x = layers.Dense(
                units,
                activation=self.config.activation,
                kernel_regularizer=keras.regularizers.L1L2(
                    l1=self.config.l1_regularization,
                    l2=self.config.l2_regularization,
                ),
                name=f"dense_{i+1}",
            )(x)

            if self.config.dropout_rate > 0:
                x = layers.Dropout(self.config.dropout_rate, name=f"dropout_{i+1}")(x)

        outputs = layers.Dense(
            1, activation=self.config.output_activation, name="output"
        )(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="ml_correction_model")
        model.compile(
            optimizer=self._get_optimizer(),
            loss=self.config.loss_function,
            metrics=["mae", "mse"],
        )

        logger.debug("Built model with %d parameters", model.count_params())
        return model

    def _get_optimizer(self) -> keras.optimizers.Optimizer:
        """Return configured optimizer."""
        name = self.config.optimizer.lower()
        if name == "adam":
            return optimizers.Adam(learning_rate=self.config.learning_rate)
        if name == "sgd":
            return optimizers.SGD(
                learning_rate=self.config.learning_rate, momentum=0.9
            )
        if name == "rmsprop":
            return optimizers.RMSprop(learning_rate=self.config.learning_rate)
        raise InvalidParameterError(f"Unknown optimizer: {self.config.optimizer}")

    def _setup_callbacks(self) -> List[keras.callbacks.Callback]:
        """Configure training callbacks."""
        callbacks_list: List[keras.callbacks.Callback] = []

        if self.config.early_stopping_patience > 0:
            callbacks_list.append(
                callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True,
                    verbose=1,
                )
            )

        if self.config.reduce_lr_patience > 0:
            callbacks_list.append(
                callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=self.config.reduce_lr_patience,
                    min_lr=1e-7,
                    verbose=1,
                )
            )

        return callbacks_list

    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate regression metrics."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        max_error = np.max(np.abs(y_true - y_pred))

        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape": mape,
            "max_error": max_error,
        }

    def get_feature_importance(
        self, X: np.ndarray, method: Literal["permutation", "gradient"] = "permutation"
    ) -> Dict[str, float]:
        """Calculate feature importance."""
        if not self.is_fitted:
            raise ModelError("Model must be fitted before calculating importance")

        if method == "permutation":
            return self._permutation_importance(X)
        if method == "gradient":
            return self._gradient_importance(X)
        raise InvalidParameterError(f"Unknown importance method: {method}")

    def _permutation_importance(self, X: np.ndarray) -> Dict[str, float]:
        baseline_pred = self.predict(X)
        baseline_score = np.mean(baseline_pred**2)

        importance_scores: Dict[str, float] = {}

        for i, feature_name in enumerate(self.feature_names):
            X_permuted = X.copy()
            X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
            permuted_pred = self.predict(X_permuted)
            permuted_score = np.mean(permuted_pred**2)
            importance_scores[feature_name] = permuted_score - baseline_score

        return importance_scores

    def _gradient_importance(self, X: np.ndarray) -> Dict[str, float]:
        X_scaled = self.scaler.transform(X)
        X_tensor = tf.constant(X_scaled, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(X_tensor)
            predictions = self.model(X_tensor)
            loss = tf.reduce_mean(tf.square(predictions))

        gradients = tape.gradient(loss, X_tensor)
        importance_scores = tf.reduce_mean(tf.abs(gradients), axis=0).numpy()

        return dict(zip(self.feature_names, importance_scores))

    def save_model(self, filepath: Union[str, Path]) -> None:
        """Persist the ML model and scaler."""
        if not self.is_fitted:
            raise ModelError("Cannot save unfitted model")

        filepath = Path(filepath)
        filepath.mkdir(parents=True, exist_ok=True)

        self.model.save(filepath / "keras_model")

        import joblib

        joblib.dump(self.scaler, filepath / "scaler.joblib")

        metadata = {
            "config": self.config.__dict__,
            "feature_names": self.feature_names,
            "training_history": self.training_history,
        }

        with open(filepath / "metadata.json", "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        logger.info("ML model saved to %s", filepath)

    @classmethod
    def load_model(cls, filepath: Union[str, Path]) -> "MLCorrectionModel":
        """Load a previously saved model."""
        filepath = Path(filepath)

        with open(filepath / "metadata.json", "r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        instance = cls({"ml_model": metadata["config"]})
        instance.model = keras.models.load_model(filepath / "keras_model")

        import joblib

        instance.scaler = joblib.load(filepath / "scaler.joblib")
        instance.feature_names = metadata["feature_names"]
        instance.training_history = metadata["training_history"]
        instance.is_fitted = True

        logger.info("ML model loaded from %s", filepath)
        return instance
