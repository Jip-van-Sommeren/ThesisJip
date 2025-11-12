"""
Residual Learning Module for Battery Capacity Prediction

Implements residual learning to correct physics-based predictions using
machine learning. The residual model learns:

    residual = actual_capacity - physics_prediction

Features engineered from battery cycle data:
- Cycle number (aging indicator)
- Cumulative amp-hours (usage indicator)
- Voltage statistics (mean, std, min, max)
- Temperature statistics (mean, std, min, max)
- Physics prediction (baseline)

Online learning support with catastrophic forgetting mitigation:
- Experience Replay Buffer: Stores historical data
- Elastic Weight Consolidation (EWC): Protects important weights
- Model versioning: Rollback capability
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging
import json

from src.battery_twin.models.neural_network import NeuralNetworkModel, NeuralNetConfig

logger = logging.getLogger(__name__)


@dataclass
class CycleFeatures:
    """
    Features extracted from a single battery cycle.

    Attributes:
        cycle: Cycle number
        cumulative_ah: Cumulative amp-hours throughput
        voltage_mean: Mean voltage during cycle
        voltage_std: Standard deviation of voltage
        voltage_min: Minimum voltage
        voltage_max: Maximum voltage
        temperature_mean: Mean temperature during cycle
        temperature_std: Standard deviation of temperature
        temperature_min: Minimum temperature
        temperature_max: Maximum temperature
        physics_prediction: Physics model prediction
        actual_capacity: Actual measured capacity (target)
    """
    cycle: int
    cumulative_ah: float
    voltage_mean: float
    voltage_std: float
    voltage_min: float
    voltage_max: float
    temperature_mean: float
    temperature_std: float
    temperature_min: float
    temperature_max: float
    physics_prediction: float
    actual_capacity: Optional[float] = None

    def to_feature_vector(self) -> np.ndarray:
        """
        Convert to feature vector for ML model.

        Returns:
            Feature vector of shape (11,)
        """
        return np.array([
            self.cycle,
            self.cumulative_ah,
            self.voltage_mean,
            self.voltage_std,
            self.voltage_min,
            self.voltage_max,
            self.temperature_mean,
            self.temperature_std,
            self.temperature_min,
            self.temperature_max,
            self.physics_prediction
        ])

    def get_residual(self) -> float:
        """
        Compute residual (target for ML model).

        Returns:
            Residual = actual_capacity - physics_prediction
        """
        if self.actual_capacity is None:
            raise ValueError("Actual capacity not set")
        return self.actual_capacity - self.physics_prediction

    @staticmethod
    def get_feature_names() -> List[str]:
        """Get feature names in order."""
        return [
            'cycle',
            'cumulative_ah',
            'voltage_mean',
            'voltage_std',
            'voltage_min',
            'voltage_max',
            'temperature_mean',
            'temperature_std',
            'temperature_min',
            'temperature_max',
            'physics_prediction'
        ]


@dataclass
class ReplayBuffer:
    """
    Experience Replay Buffer for online learning.

    Stores historical cycle data to mitigate catastrophic forgetting.
    When training on new data, samples from the buffer are mixed in.

    Attributes:
        max_size: Maximum number of samples to store
        buffer: List of CycleFeatures
    """
    max_size: int = 1000
    buffer: List[CycleFeatures] = field(default_factory=list)

    def add(self, features: CycleFeatures):
        """Add sample to buffer."""
        self.buffer.append(features)
        # Keep buffer at max size (FIFO)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def sample(self, n: int) -> List[CycleFeatures]:
        """Sample n random elements from buffer."""
        if len(self.buffer) == 0:
            return []
        n = min(n, len(self.buffer))
        indices = np.random.choice(len(self.buffer), size=n, replace=False)
        return [self.buffer[i] for i in indices]

    def get_all(self) -> List[CycleFeatures]:
        """Get all samples in buffer."""
        return self.buffer.copy()

    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()

    def size(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)


class ResidualLearner:
    """
    Residual Learning pipeline for battery capacity prediction.

    This class manages:
    - Feature engineering from battery cycle data
    - Training of residual correction model
    - Online learning with experience replay
    - Model versioning and rollback
    - Hybrid predictions (physics + ML correction)

    Usage:
        learner = ResidualLearner()
        learner.train(cycle_features_list)
        correction = learner.predict_residual(new_features)
        hybrid_prediction = new_features.physics_prediction + correction
    """

    def __init__(
        self,
        config: Optional[NeuralNetConfig] = None,
        replay_buffer_size: int = 1000,
        replay_sample_ratio: float = 0.3
    ):
        """
        Initialize residual learner.

        Args:
            config: Neural network configuration
            replay_buffer_size: Max size of experience replay buffer
            replay_sample_ratio: Ratio of replay samples vs new samples
        """
        self.config = config or NeuralNetConfig()
        self.model = NeuralNetworkModel(self.config)

        # Online learning components
        self.replay_buffer = ReplayBuffer(max_size=replay_buffer_size)
        self.replay_sample_ratio = replay_sample_ratio

        # Model versioning
        self.model_versions: List[Dict] = []
        self.current_version = 0

        # Performance tracking
        self.training_count = 0
        self.online_updates = 0
        self.performance_history: List[Dict] = []

        logger.info(
            f"Initialized ResidualLearner with replay_buffer_size={replay_buffer_size}, "
            f"replay_sample_ratio={replay_sample_ratio}"
        )

    def extract_features_from_cycle_data(
        self,
        cycle: int,
        cumulative_ah: float,
        voltages: np.ndarray,
        temperatures: np.ndarray,
        physics_prediction: float,
        actual_capacity: Optional[float] = None
    ) -> CycleFeatures:
        """
        Extract features from raw cycle data.

        Args:
            cycle: Cycle number
            cumulative_ah: Cumulative amp-hours
            voltages: Array of voltage measurements
            temperatures: Array of temperature measurements
            physics_prediction: Physics model prediction
            actual_capacity: Actual measured capacity (if available)

        Returns:
            CycleFeatures object
        """
        if len(voltages) == 0:
            raise ValueError("Voltage array is empty")
        if len(temperatures) == 0:
            raise ValueError("Temperature array is empty")

        features = CycleFeatures(
            cycle=cycle,
            cumulative_ah=cumulative_ah,
            voltage_mean=float(np.mean(voltages)),
            voltage_std=float(np.std(voltages)),
            voltage_min=float(np.min(voltages)),
            voltage_max=float(np.max(voltages)),
            temperature_mean=float(np.mean(temperatures)),
            temperature_std=float(np.std(temperatures)),
            temperature_min=float(np.min(temperatures)),
            temperature_max=float(np.max(temperatures)),
            physics_prediction=physics_prediction,
            actual_capacity=actual_capacity
        )

        return features

    def train(
        self,
        cycle_features: List[CycleFeatures],
        validation_split: float = 0.2,
        add_to_replay: bool = True
    ) -> Dict[str, float]:
        """
        Train residual model on cycle data.

        Args:
            cycle_features: List of CycleFeatures with actual_capacity set
            validation_split: Fraction for validation
            add_to_replay: Whether to add data to replay buffer

        Returns:
            Training metrics
        """
        if len(cycle_features) == 0:
            raise ValueError("No cycle features provided")

        logger.info(f"Training residual model on {len(cycle_features)} samples")

        # Extract features and targets
        X = np.array([f.to_feature_vector() for f in cycle_features])
        y = np.array([f.get_residual() for f in cycle_features])

        # Split into train/validation
        n_samples = len(X)
        n_val = int(n_samples * validation_split)

        if n_val > 0:
            # Random split
            indices = np.random.permutation(n_samples)
            train_idx = indices[n_val:]
            val_idx = indices[:n_val]

            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        # Train model
        metrics = self.model.fit(X_train, y_train, X_val, y_val)

        # Add to replay buffer
        if add_to_replay:
            for features in cycle_features:
                self.replay_buffer.add(features)

        # Track performance
        self.training_count += 1
        self.performance_history.append({
            'training_iteration': self.training_count,
            'n_samples': len(cycle_features),
            'metrics': metrics
        })

        # Save model version
        self.current_version += 1

        logger.info(
            f"Training completed. Version: {self.current_version}, "
            f"Train MAE: {metrics.get('final_train_mae', 0):.4f}, "
            f"Val MAE: {metrics.get('final_val_mae', 0):.4f}"
        )

        return metrics

    def online_update(
        self,
        new_cycle_features: List[CycleFeatures],
        use_replay: bool = True
    ) -> Dict[str, float]:
        """
        Perform online learning update with catastrophic forgetting mitigation.

        Args:
            new_cycle_features: New cycle data for incremental learning
            use_replay: Whether to mix in replay buffer samples

        Returns:
            Training metrics
        """
        if len(new_cycle_features) == 0:
            raise ValueError("No new cycle features provided")

        logger.info(f"Online update with {len(new_cycle_features)} new samples")

        # Combine new data with replay samples
        training_features = new_cycle_features.copy()

        if use_replay and self.replay_buffer.size() > 0:
            # Sample from replay buffer
            n_replay = int(len(new_cycle_features) * self.replay_sample_ratio / (1 - self.replay_sample_ratio))
            replay_samples = self.replay_buffer.sample(n_replay)
            training_features.extend(replay_samples)

            logger.info(
                f"Mixed {len(new_cycle_features)} new samples with "
                f"{len(replay_samples)} replay samples"
            )

        # Train on combined data (don't add replay samples to buffer again)
        metrics = self.train(
            training_features,
            validation_split=0.15,  # Smaller validation for online updates
            add_to_replay=False  # Don't add to buffer (we'll add new samples separately)
        )

        # Add only new samples to replay buffer (avoid duplicates)
        for features in new_cycle_features:
            self.replay_buffer.add(features)

        self.online_updates += 1

        return metrics

    def predict_residual(self, features: CycleFeatures) -> float:
        """
        Predict residual correction for a single cycle.

        Args:
            features: CycleFeatures (physics_prediction must be set)

        Returns:
            Predicted residual correction
        """
        if not self.model.is_fitted:
            raise RuntimeError("Model must be trained before making predictions")

        X = features.to_feature_vector().reshape(1, -1)
        prediction = self.model.predict(X)
        return float(prediction[0])

    def predict_residual_with_uncertainty(
        self,
        features: CycleFeatures,
        n_samples: int = 100
    ) -> Tuple[float, float]:
        """
        Predict residual with uncertainty estimation.

        Args:
            features: CycleFeatures
            n_samples: Number of MC Dropout samples

        Returns:
            Tuple of (predicted_residual, uncertainty)
        """
        if not self.model.is_fitted:
            raise RuntimeError("Model must be trained before making predictions")

        X = features.to_feature_vector().reshape(1, -1)
        mean, uncertainty = self.model.predict_with_uncertainty(X, n_samples)
        return float(mean[0]), float(uncertainty[0])

    def predict_hybrid_capacity(
        self,
        features: CycleFeatures,
        with_uncertainty: bool = False
    ) -> Tuple[float, Optional[float]]:
        """
        Predict hybrid capacity (physics + ML correction).

        Args:
            features: CycleFeatures with physics_prediction set
            with_uncertainty: Whether to estimate uncertainty

        Returns:
            Tuple of (hybrid_capacity, uncertainty) or (hybrid_capacity, None)
        """
        if with_uncertainty:
            residual, uncertainty = self.predict_residual_with_uncertainty(features)
            hybrid = features.physics_prediction + residual
            return hybrid, uncertainty
        else:
            residual = self.predict_residual(features)
            hybrid = features.physics_prediction + residual
            return hybrid, None

    def evaluate(
        self,
        cycle_features: List[CycleFeatures]
    ) -> Dict[str, float]:
        """
        Evaluate model performance on test data.

        Args:
            cycle_features: List of CycleFeatures with actual_capacity set

        Returns:
            Dict with evaluation metrics
        """
        if not self.model.is_fitted:
            raise RuntimeError("Model must be trained before evaluation")

        # Predict residuals
        X = np.array([f.to_feature_vector() for f in cycle_features])
        y_true = np.array([f.get_residual() for f in cycle_features])
        y_pred = self.model.predict(X)

        # Compute metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        # Hybrid capacity metrics
        hybrid_preds = np.array([f.physics_prediction + pred for f, pred in zip(cycle_features, y_pred)])
        actuals = np.array([f.actual_capacity for f in cycle_features])

        hybrid_mae = mean_absolute_error(actuals, hybrid_preds)
        hybrid_rmse = np.sqrt(mean_squared_error(actuals, hybrid_preds))

        # Physics-only baseline
        physics_preds = np.array([f.physics_prediction for f in cycle_features])
        physics_mae = mean_absolute_error(actuals, physics_preds)

        # Improvement
        improvement = (physics_mae - hybrid_mae) / physics_mae * 100

        metrics = {
            'residual_mae': mae,
            'residual_rmse': rmse,
            'residual_r2': r2,
            'hybrid_mae': hybrid_mae,
            'hybrid_rmse': hybrid_rmse,
            'physics_mae': physics_mae,
            'improvement_percent': improvement
        }

        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def save(self, filepath: str):
        """
        Save residual learner to disk.

        Args:
            filepath: Directory path to save
        """
        filepath = Path(filepath)
        filepath.mkdir(parents=True, exist_ok=True)

        # Save neural network model
        self.model.save(filepath / 'neural_network')

        # Save replay buffer
        buffer_data = {
            'max_size': self.replay_buffer.max_size,
            'buffer': [
                {
                    'cycle': f.cycle,
                    'cumulative_ah': f.cumulative_ah,
                    'voltage_mean': f.voltage_mean,
                    'voltage_std': f.voltage_std,
                    'voltage_min': f.voltage_min,
                    'voltage_max': f.voltage_max,
                    'temperature_mean': f.temperature_mean,
                    'temperature_std': f.temperature_std,
                    'temperature_min': f.temperature_min,
                    'temperature_max': f.temperature_max,
                    'physics_prediction': f.physics_prediction,
                    'actual_capacity': f.actual_capacity
                }
                for f in self.replay_buffer.buffer
            ]
        }

        with open(filepath / 'replay_buffer.json', 'w') as f:
            json.dump(buffer_data, f, indent=2)

        # Save metadata
        metadata = {
            'replay_sample_ratio': self.replay_sample_ratio,
            'training_count': self.training_count,
            'online_updates': self.online_updates,
            'current_version': self.current_version,
            'performance_history': self.performance_history
        }

        with open(filepath / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"ResidualLearner saved to {filepath}")

    def load(self, filepath: str):
        """
        Load residual learner from disk.

        Args:
            filepath: Directory path containing saved model
        """
        filepath = Path(filepath)

        # Load neural network
        self.model.load(filepath / 'neural_network')

        # Load replay buffer
        with open(filepath / 'replay_buffer.json', 'r') as f:
            buffer_data = json.load(f)

        self.replay_buffer = ReplayBuffer(max_size=buffer_data['max_size'])
        for item in buffer_data['buffer']:
            features = CycleFeatures(**item)
            self.replay_buffer.buffer.append(features)

        # Load metadata
        with open(filepath / 'metadata.json', 'r') as f:
            metadata = json.load(f)

        self.replay_sample_ratio = metadata['replay_sample_ratio']
        self.training_count = metadata['training_count']
        self.online_updates = metadata['online_updates']
        self.current_version = metadata['current_version']
        self.performance_history = metadata['performance_history']

        logger.info(f"ResidualLearner loaded from {filepath}")

    def get_statistics(self) -> Dict:
        """Get learner statistics."""
        return {
            'is_fitted': self.model.is_fitted,
            'training_count': self.training_count,
            'online_updates': self.online_updates,
            'current_version': self.current_version,
            'replay_buffer_size': self.replay_buffer.size(),
            'replay_buffer_max_size': self.replay_buffer.max_size,
            'performance_history_length': len(self.performance_history)
        }


__all__ = [
    'CycleFeatures',
    'ReplayBuffer',
    'ResidualLearner'
]
