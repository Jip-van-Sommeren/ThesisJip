"""
ML Residual Agent - BDI Agent for Machine Learning Model Management

This agent implements a BDI (Belief-Desire-Intention) architecture for managing
machine learning models that correct physics-based predictions.

Formal Definition:
A_ml = ⟨
  Id: "model.ml.{N}",
  Beliefs: {model_performance, data_quality, training_status, data_available},
  Desires: {high_accuracy, low_uncertainty, generalization},
  Intentions: {train_model, retrain_model, evaluate_model},
  Perception: {MQTT: battery/{battery_id}/prediction/physics, battery/{battery_id}/capacity},
  Actions: {train(), predict_residual(), evaluate(), publish_hybrid()}
⟩

BDI Reasoning:
- Beliefs: Track model performance, available training data, training status
- Desires: Achieve high prediction accuracy, minimize uncertainty, ensure generalization
- Intentions: Plans for training, retraining, and evaluation

Deliberative Decisions:
1. When to start initial training? (enough_data AND not_trained)
2. When to retrain? (performance_degraded OR significant_new_data)
3. When to switch models? (new_model_better_than_old)
4. When to request more data? (high_uncertainty OR poor_generalization)
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.abstract_agent import AgentId, Goal, GoalType
from src.battery_twin.agents.battery_agent_types import BatteryBDIAgent
from src.battery_twin.communication.mqtt_bridge import MqttBridge, MqttConfig
from src.battery_twin.communication.message_schemas import (
    PredictionMessage,
    CapacityMessage,
    MessageFactory
)
from src.battery_twin.storage.battery_storage_manager import BatteryStorageManager
from src.battery_twin.models.residual_learner import (
    ResidualLearner,
    CycleFeatures,
    CycleFeatures,
)
from src.battery_twin.models.neural_network import NeuralNetConfig

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Status of ML model."""
    NOT_TRAINED = "not_trained"
    TRAINING = "training"
    TRAINED = "trained"
    RETRAINING = "retraining"
    FAILED = "failed"


class PerformanceLevel(Enum):
    """Model performance level."""
    EXCELLENT = "excellent"  # MAE < 0.02 Ah
    GOOD = "good"           # MAE < 0.05 Ah
    DEGRADED = "degraded"   # MAE < 0.10 Ah
    POOR = "poor"           # MAE >= 0.10 Ah


@dataclass
class TrainingDataPoint:
    """
    Single training data point for ML model.

    Combines physics prediction with actual measurement for residual learning.
    """
    cycle: int
    timestamp: float
    physics_prediction: float
    actual_capacity: Optional[float] = None
    cumulative_ah: float = 0.0
    voltages: List[float] = field(default_factory=list)
    temperatures: List[float] = field(default_factory=list)

    def is_complete(self) -> bool:
        """Check if data point has all required information."""
        return (
            self.actual_capacity is not None and
            len(self.voltages) > 0 and
            len(self.temperatures) > 0
        )


class MLResidualAgent(BatteryBDIAgent):
    """
    ML Residual Agent - BDI agent for managing ML model training and predictions.

    This agent:
    1. Accumulates training data (physics predictions + actual measurements)
    2. Deliberates on when to train/retrain the model
    3. Trains ResidualLearner to predict corrections to physics model
    4. Publishes hybrid predictions (physics + ML correction)
    5. Monitors model performance and triggers retraining when needed

    BDI Components:
    - Beliefs: model_status, data_count, model_performance, last_training_time
    - Desires: high_accuracy, sufficient_data, timely_training
    - Intentions: train_model_plan, retrain_model_plan, evaluate_model_plan
    """

    def __init__(
        self,
        agent_id: AgentId,
        battery_id: str,
        mqtt_bridge: Optional[MqttBridge] = None,
        storage_manager: Optional[BatteryStorageManager] = None,
        ml_config: Optional[NeuralNetConfig] = None,
        min_training_samples: int = 30,
        retrain_threshold_mae: float = 0.10,
        retrain_interval_cycles: int = 50,
        hybrid_service: Optional[object] = None,
    ):
        """
        Initialize ML Residual Agent.

        Args:
            agent_id: Unique agent identifier
            battery_id: Battery identifier
            mqtt_bridge: MQTT communication bridge
            storage_manager: Storage manager for persistence
            ml_config: Neural network configuration
            min_training_samples: Minimum samples needed for initial training
            retrain_threshold_mae: MAE threshold to trigger retraining
            retrain_interval_cycles: Cycles between retraining checks
        """
        # Store battery_id before super init
        self.battery_id = battery_id

        super().__init__(
            agent_id=agent_id,
            observable_properties={'ml_predictions', 'model_performance'},
            mqtt_bridge=mqtt_bridge,
            storage_manager=storage_manager
        )

        # ML Model
        self.ml_config = ml_config or NeuralNetConfig(epochs=50, batch_size=16, device='cpu')
        self.learner = ResidualLearner(self.ml_config, replay_buffer_size=1000)
        self.hybrid_service = hybrid_service

        # Training data buffer
        self.training_buffer: Dict[int, TrainingDataPoint] = {}  # cycle -> data
        self.min_training_samples = min_training_samples
        self.retrain_threshold_mae = retrain_threshold_mae
        self.retrain_interval_cycles = retrain_interval_cycles

        # Model state
        self.model_status = ModelStatus.NOT_TRAINED
        self.last_training_cycle = 0
        self.last_evaluation_metrics: Dict = {}
        self.training_history: List[Dict] = []

        # Performance tracking
        self.predictions_made = 0
        self.hybrid_errors: List[float] = []
        self.physics_errors: List[float] = []

        # Statistics
        self.total_trainings = 0
        self.total_retrainings = 0

        # Initialize beliefs and goals
        self._initialize_beliefs()
        self._initialize_goals()

        logger.info(
            f"Initialized MLResidualAgent for battery {battery_id}: "
            f"min_samples={min_training_samples}, retrain_mae={retrain_threshold_mae}"
        )

    def _initialize_beliefs(self):
        """Initialize BDI beliefs about model state and performance."""
        # Model status beliefs
        self.state.update_belief(
            key='model_status',
            proposition=f"status_{self.model_status.value}",
            confidence=1.0,
            is_internal=True
        )

        self.state.update_belief(
            key='model_trained',
            proposition='false',
            confidence=1.0,
            is_internal=True
        )

        # Data availability beliefs
        self.state.update_belief(
            key='training_data_count',
            proposition=f"count_{len(self.training_buffer)}",
            confidence=1.0,
            is_internal=True
        )

        self.state.update_belief(
            key='enough_data_for_training',
            proposition='false',
            confidence=1.0,
            is_internal=True
        )

        # Performance beliefs
        self.state.update_belief(
            key='model_performance',
            proposition='unknown',
            confidence=0.0,
            is_internal=True
        )

    def _initialize_goals(self):
        """Initialize BDI goals (desires) for ML model management."""
        # Goal: Train model when enough data available
        self.add_goal(Goal(
            condition="model_trained",
            goal_type=GoalType.INTRINSIC,
            priority=1.0
        ))

        # Goal: Maintain high prediction accuracy
        self.add_goal(Goal(
            condition="high_accuracy",
            goal_type=GoalType.PERFORMANCE,
            priority=0.9
        ))

        # Goal: Keep model up-to-date with recent data
        self.add_goal(Goal(
            condition="model_fresh",
            goal_type=GoalType.INTRINSIC,
            priority=0.7
        ))

    def _agent_setup(self) -> bool:
        """Agent-specific setup."""
        try:
            # Register MQTT action handlers
            self.register_action(
                action_id="process_physics_prediction",
                handler=self._handle_physics_prediction,
                topic_pattern=f"battery/{self.battery_id}/prediction/physics",
                description="Process physics model predictions"
            )

            self.register_action(
                action_id="process_actual_capacity",
                handler=self._handle_actual_capacity,
                topic_pattern=f"battery/{self.battery_id}/capacity",
                description="Process actual capacity measurements"
            )

            # Try to load existing model from storage
            if self.storage_manager:
                self._load_model_from_storage()

            logger.info(f"MLResidualAgent setup complete for battery {self.battery_id}")
            return True

        except Exception as e:
            logger.error(f"MLResidualAgent setup failed: {e}")
            return False

    def _agent_teardown(self):
        """Agent-specific teardown."""
        # Save model to storage
        if self.storage_manager and self.learner.model.is_fitted:
            self._save_model_to_storage()

        logger.info(f"MLResidualAgent teardown complete for battery {self.battery_id}")

    # ========================================================================
    # Message Handling (Perception)
    # ========================================================================

    def _handle_physics_prediction(self, topic: str, payload: str):
        """
        Handle incoming physics prediction.

        Creates/updates training data point for the cycle.
        """
        try:
            msg = MessageFactory.parse_message('prediction', payload)

            # Get or create training data point for this cycle
            if msg.cycle not in self.training_buffer:
                self.training_buffer[msg.cycle] = TrainingDataPoint(
                    cycle=msg.cycle,
                    timestamp=msg.timestamp,
                    physics_prediction=msg.predicted_capacity
                )
            else:
                # Update existing data point
                self.training_buffer[msg.cycle].physics_prediction = msg.predicted_capacity

            # Update belief about data availability
            self._update_data_beliefs()

            logger.debug(f"Received physics prediction for cycle {msg.cycle}: {msg.predicted_capacity:.4f} Ah")

        except Exception as e:
            logger.error(f"Failed to handle physics prediction: {e}")

    def _handle_actual_capacity(self, topic: str, payload: str):
        """
        Handle actual capacity measurement.

        Completes training data point and triggers deliberation on training.
        """
        try:
            msg = MessageFactory.parse_message('capacity', payload)

            # Get or create training data point
            if msg.cycle not in self.training_buffer:
                # Physics prediction not received yet, create placeholder
                self.training_buffer[msg.cycle] = TrainingDataPoint(
                    cycle=msg.cycle,
                    timestamp=msg.timestamp,
                    physics_prediction=0.0,  # Will be updated when physics prediction arrives
                    actual_capacity=msg.capacity
                )
            else:
                # Update existing data point
                self.training_buffer[msg.cycle].actual_capacity = msg.capacity

            # Update cumulative Ah (approximate)
            self.training_buffer[msg.cycle].cumulative_ah = msg.cycle * msg.capacity

            # Set default voltages and temperatures if not present
            if len(self.training_buffer[msg.cycle].voltages) == 0:
                self.training_buffer[msg.cycle].voltages = [3.8]  # Default voltage
            if len(self.training_buffer[msg.cycle].temperatures) == 0:
                self.training_buffer[msg.cycle].temperatures = [25.0]  # Default temperature

            # Check if we have physics prediction for this cycle
            data_point = self.training_buffer[msg.cycle]
            if data_point.is_complete() and data_point.physics_prediction > 0:
                # Data point is complete, can be used for training
                logger.info(
                    f"Complete training data for cycle {msg.cycle}: "
                    f"physics={data_point.physics_prediction:.4f}, actual={msg.capacity:.4f}"
                )

                # Update belief about data availability
                self._update_data_beliefs()

                # Deliberate on whether to train/retrain
                self._deliberate_on_training()

        except Exception as e:
            logger.error(f"Failed to handle actual capacity: {e}")

    def _update_data_beliefs(self):
        """Update beliefs about available training data."""
        # Count complete data points
        complete_count = sum(
            1 for dp in self.training_buffer.values()
            if dp.is_complete() and dp.physics_prediction > 0
        )

        # Update belief about data count
        self.state.update_belief(
            key='training_data_count',
            proposition=f"count_{complete_count}",
            confidence=1.0,
            is_internal=True
        )

        # Update belief about having enough data
        enough_data = complete_count >= self.min_training_samples
        self.state.update_belief(
            key='enough_data_for_training',
            proposition=str(enough_data).lower(),
            confidence=1.0,
            is_internal=True
        )

    # ========================================================================
    # BDI Deliberation
    # ========================================================================

    def _deliberate_on_training(self):
        """
        Deliberate on whether to train or retrain the model.

        BDI Decision Logic:
        1. If model not trained AND enough data → TRAIN
        2. If model trained AND performance degraded → RETRAIN
        3. If model trained AND many new cycles since last training → RETRAIN
        """
        try:
            complete_data = [
                dp for dp in self.training_buffer.values()
                if dp.is_complete() and dp.physics_prediction > 0
            ]

            # Decision 1: Initial training
            if self.model_status == ModelStatus.NOT_TRAINED:
                if len(complete_data) >= self.min_training_samples:
                    logger.info(
                        f"Decision: Initial training triggered. "
                        f"Data available: {len(complete_data)} >= {self.min_training_samples}"
                    )
                    self._train_model(complete_data)
                else:
                    logger.debug(
                        f"Decision: Waiting for more data. "
                        f"Have {len(complete_data)}/{self.min_training_samples}"
                    )

            # Decision 2: Retraining due to performance degradation
            elif self.model_status == ModelStatus.TRAINED:
                # Check if performance has degraded
                if len(self.hybrid_errors) > 10:
                    recent_mae = np.mean(self.hybrid_errors[-10:])

                    if recent_mae > self.retrain_threshold_mae:
                        logger.warning(
                            f"Decision: Retraining triggered due to performance degradation. "
                            f"Recent MAE: {recent_mae:.4f} > {self.retrain_threshold_mae:.4f}"
                        )
                        self._retrain_model(complete_data)

                # Check if many cycles passed since last training
                if complete_data:
                    latest_cycle = max(dp.cycle for dp in complete_data)
                    cycles_since_training = latest_cycle - self.last_training_cycle

                    if cycles_since_training >= self.retrain_interval_cycles:
                        logger.info(
                            f"Decision: Retraining triggered due to staleness. "
                            f"Cycles since last training: {cycles_since_training}"
                        )
                        self._retrain_model(complete_data)

        except Exception as e:
            logger.error(f"Deliberation on training failed: {e}")

    # ========================================================================
    # Actions (ML Model Training and Prediction)
    # ========================================================================

    def _train_model(self, training_data: List[TrainingDataPoint]):
        """
        Train ML model for the first time.

        Args:
            training_data: List of complete training data points
        """
        try:
            logger.info(f"Starting initial model training with {len(training_data)} samples")

            # Update status
            self.model_status = ModelStatus.TRAINING

            # Convert to CycleFeatures
            features = self._convert_to_features(training_data)

            # Train model
            metrics = self.learner.train(features, validation_split=0.2)

            # Update status and beliefs
            self.model_status = ModelStatus.TRAINED
            self.last_training_cycle = max(dp.cycle for dp in training_data)
            self.total_trainings += 1

            self.state.update_belief(
                key='model_trained',
                proposition='true',
                confidence=1.0,
                is_internal=True
            )

            self.state.update_belief(
                key='model_status',
                proposition=f"status_{self.model_status.value}",
                confidence=1.0,
                is_internal=True
            )

            # Store training history
            self.training_history.append({
                'type': 'initial_training',
                'cycle': self.last_training_cycle,
                'n_samples': len(training_data),
                'metrics': metrics
            })

            logger.info(
                f"Initial training completed. Train MAE: {metrics.get('final_train_mae', 0):.4f}, "
                f"Val MAE: {metrics.get('final_val_mae', 0):.4f}"
            )

            self._submit_hybrid_training(training_data)

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            self.model_status = ModelStatus.FAILED

    def _retrain_model(self, training_data: List[TrainingDataPoint]):
        """
        Retrain ML model with new data (online learning).

        Args:
            training_data: List of complete training data points
        """
        try:
            # Get new data since last training
            new_data = [
                dp for dp in training_data
                if dp.cycle > self.last_training_cycle
            ]

            if len(new_data) == 0:
                logger.warning("No new data for retraining")
                return

            logger.info(f"Starting model retraining with {len(new_data)} new samples")

            # Update status
            self.model_status = ModelStatus.RETRAINING

            # Convert to CycleFeatures
            features = self._convert_to_features(new_data)

            # Online update (uses replay buffer)
            metrics = self.learner.online_update(features, use_replay=True)

            # Update status
            self.model_status = ModelStatus.TRAINED
            self.last_training_cycle = max(dp.cycle for dp in training_data)
            self.total_retrainings += 1

            # Store training history
            self.training_history.append({
                'type': 'retraining',
                'cycle': self.last_training_cycle,
                'n_new_samples': len(new_data),
                'metrics': metrics
            })

            logger.info(
                f"Retraining completed. Train MAE: {metrics.get('final_train_mae', 0):.4f}"
            )

            self._submit_hybrid_training(new_data)

        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
            self.model_status = ModelStatus.FAILED

    def _convert_to_features(self, training_data: List[TrainingDataPoint]) -> List[CycleFeatures]:
        """Convert training data points to CycleFeatures."""
        features = []

        for dp in training_data:
            # Use simple voltage/temperature if not available
            voltages = np.array(dp.voltages) if dp.voltages else np.array([3.8])
            temperatures = np.array(dp.temperatures) if dp.temperatures else np.array([25.0])

            feature = self.learner.extract_features_from_cycle_data(
                cycle=dp.cycle,
                cumulative_ah=dp.cumulative_ah,
                voltages=voltages,
                temperatures=temperatures,
                physics_prediction=dp.physics_prediction,
                actual_capacity=dp.actual_capacity
            )
            features.append(feature)

        return features

    def _submit_hybrid_training(self, training_data: List[TrainingDataPoint]):
        """Send training samples to shared hybrid twin service if available."""
        if not self.hybrid_service or not hasattr(self.hybrid_service, "train_hybrid_twin"):
            return

        samples = []
        for dp in training_data:
            if dp.actual_capacity is None:
                continue
            avg_temp = float(np.mean(dp.temperatures)) if dp.temperatures else 25.0
            duration = float(len(dp.voltages)) if dp.voltages else 1.0
            samples.append(
                {
                    "id_cycle": dp.cycle,
                    "Temperature_measured": avg_temp,
                    "Time": duration,
                    "Capacity": dp.actual_capacity,
                }
            )

        if samples:
            try:
                self.hybrid_service.train_hybrid_twin(samples)
            except Exception as exc:
                logger.debug("Hybrid training submission failed: %s", exc)

    def _predict_via_hybrid_service(
        self, cycle: int, physics_prediction: float
    ) -> Optional[Tuple[float, Optional[float]]]:
        if not self.hybrid_service or not hasattr(
            self.hybrid_service, "predict_hybrid_capacity"
        ):
            return None

        df = pd.DataFrame(
            [
                {
                    "id_cycle": cycle,
                    "Temperature_measured": 25.0,
                    "Time": 1.0,
                    "Capacity": physics_prediction,
                }
            ]
        )

        try:
            result = self.hybrid_service.predict_hybrid_capacity(df)
        except Exception as exc:
            logger.debug("Hybrid service prediction failed: %s", exc)
            return None

        if result is None:
            return None

        try:
            from src.battery_twin.hybrid.core.digital_twin import PredictionResult
        except ImportError:  # pragma: no cover
            PredictionResult = None  # type: ignore

        if PredictionResult and isinstance(result, PredictionResult):
            value = float(result.hybrid_prediction[0])
            uncertainty = (
                float(result.uncertainty[0]) if result.uncertainty is not None else None
            )
            return value, uncertainty

        if isinstance(result, np.ndarray):
            return float(result[0]), None

        return None


    def predict_hybrid_capacity(
        self,
        cycle: int,
        physics_prediction: float,
        with_uncertainty: bool = True
    ) -> Tuple[float, Optional[float]]:
        """
        Predict hybrid capacity (physics + ML correction).

        Args:
            cycle: Cycle number
            physics_prediction: Physics model prediction
            with_uncertainty: Whether to estimate uncertainty

        Returns:
            Tuple of (hybrid_capacity, uncertainty)
        """
        if self.model_status != ModelStatus.TRAINED:
            # Model not trained yet, return physics prediction
            return physics_prediction, None

        try:
            service_prediction = self._predict_via_hybrid_service(
                cycle, physics_prediction
            )
            if service_prediction:
                self.predictions_made += 1
                return service_prediction

            # Create features (use defaults for voltage/temperature)
            features = CycleFeatures(
                cycle=cycle,
                cumulative_ah=cycle * physics_prediction,
                voltage_mean=3.8,
                voltage_std=0.1,
                voltage_min=3.6,
                voltage_max=4.0,
                temperature_mean=25.0,
                temperature_std=2.0,
                temperature_min=22.0,
                temperature_max=28.0,
                physics_prediction=physics_prediction
            )

            # Predict hybrid capacity
            hybrid, uncertainty = self.learner.predict_hybrid_capacity(
                features,
                with_uncertainty=with_uncertainty
            )

            self.predictions_made += 1

            return hybrid, uncertainty

        except Exception as e:
            logger.error(f"Hybrid prediction failed: {e}")
            return physics_prediction, None

    def publish_hybrid_prediction(
        self,
        cycle: int,
        hybrid_capacity: float,
        uncertainty: Optional[float] = None
    ):
        """
        Publish hybrid prediction to MQTT.

        Args:
            cycle: Cycle number
            hybrid_capacity: Predicted capacity
            uncertainty: Prediction uncertainty (optional)
        """
        try:
            prediction_msg = PredictionMessage(
                battery_id=self.battery_id,
                timestamp=time.time(),
                cycle=cycle,
                prediction_type="hybrid",
                predicted_capacity=float(hybrid_capacity),
                uncertainty=float(uncertainty) if uncertainty is not None else None,
                horizon=0,
                agent_id=str(self.id)
            )

            self.publish_message(
                topic_name="battery_prediction_ml",
                message=prediction_msg,
                battery_id=self.battery_id
            )

            logger.debug(f"Published hybrid prediction for cycle {cycle}: {hybrid_capacity:.4f} Ah")

        except Exception as e:
            logger.error(f"Failed to publish hybrid prediction: {e}")

    # ========================================================================
    # Storage Integration
    # ========================================================================

    def _save_model_to_storage(self):
        """Save ML model to storage."""
        if not self.storage_manager or not self.learner.model.is_fitted:
            return

        try:
            # Save model to filesystem
            model_dir = Path(f"/tmp/ml_models/{self.battery_id}")
            self.learner.save(str(model_dir))

            logger.info(f"ML model saved to {model_dir}")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def _load_model_from_storage(self):
        """Load ML model from storage."""
        if not self.storage_manager:
            return

        try:
            model_dir = Path(f"/tmp/ml_models/{self.battery_id}")
            if model_dir.exists():
                self.learner.load(str(model_dir))
                self.model_status = ModelStatus.TRAINED

                self.state.update_belief(
                    key='model_trained',
                    proposition='true',
                    confidence=1.0,
                    is_internal=True
                )

                logger.info(f"ML model loaded from {model_dir}")

        except Exception as e:
            logger.warning(f"Failed to load model (may not exist yet): {e}")

    # ========================================================================
    # Statistics and Monitoring
    # ========================================================================

    def get_statistics(self) -> Dict:
        """Get agent statistics."""
        complete_data_count = sum(
            1 for dp in self.training_buffer.values()
            if dp.is_complete() and dp.physics_prediction > 0
        )

        stats = {
            'model_status': self.model_status.value,
            'model_trained': self.learner.model.is_fitted,
            'training_data_count': len(self.training_buffer),
            'complete_data_count': complete_data_count,
            'min_training_samples': self.min_training_samples,
            'last_training_cycle': self.last_training_cycle,
            'total_trainings': self.total_trainings,
            'total_retrainings': self.total_retrainings,
            'predictions_made': self.predictions_made,
            'training_history_length': len(self.training_history)
        }

        # Add model stats if trained
        if self.learner.model.is_fitted:
            stats.update(self.learner.get_statistics())

        # Add recent performance if available
        if len(self.hybrid_errors) > 0:
            stats['recent_hybrid_mae'] = float(np.mean(self.hybrid_errors[-10:]))
        if len(self.physics_errors) > 0:
            stats['recent_physics_mae'] = float(np.mean(self.physics_errors[-10:]))

        return stats

    def get_performance_level(self) -> PerformanceLevel:
        """Get current model performance level."""
        if len(self.hybrid_errors) < 5:
            return PerformanceLevel.EXCELLENT  # Not enough data

        recent_mae = np.mean(self.hybrid_errors[-10:])

        if recent_mae < 0.02:
            return PerformanceLevel.EXCELLENT
        elif recent_mae < 0.05:
            return PerformanceLevel.GOOD
        elif recent_mae < 0.10:
            return PerformanceLevel.DEGRADED
        else:
            return PerformanceLevel.POOR


__all__ = [
    'MLResidualAgent',
    'ModelStatus',
    'PerformanceLevel',
    'TrainingDataPoint'
]
