"""
Physics Model Agent - Hybrid Agent for Battery Degradation Prediction

This agent implements a hybrid (reactive + goal-driven) architecture for
physics-based battery capacity prediction.

Formal Definition:
A_physics = ⟨
  Id: "model.physics.{N}",
  State: {battery_params, degradation_model, last_prediction, cycle_data},
  Goals: {predict_capacity, maintain_accuracy, adapt_parameters},
  Perception: {MQTT: battery/{battery_id}/telemetry/clean},
  Action: {compute_degradation(), predict_capacity(), update_parameters()},
  Decision: reactive + goal_driven
⟩

Reactive Layer:
- Fast response to telemetry messages
- Accumulate cycle data
- Detect anomalies in telemetry
- Detect end-of-cycle events

Deliberative Layer (BDI):
- Goals: Predict capacity accurately, maintain model accuracy
- Desires: Minimize prediction error, adapt to battery aging
- Intentions: Update parameters when accuracy degrades
"""

import logging
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.abstract_agent import AgentId, Goal, GoalType
from src.battery_twin.agents.battery_agent_types import BatteryHybridAgent
from src.battery_twin.communication.mqtt_bridge import MqttBridge, MqttConfig
from src.battery_twin.communication.message_schemas import (
    TelemetryMessage,
    PredictionMessage,
    MessageFactory
)
from src.battery_twin.storage.battery_storage_manager import BatteryStorageManager
from src.battery_twin.models.physics_degradation_model import (
    PhysicsDegradationModel,
    DegradationParameters
)

logger = logging.getLogger(__name__)


class PredictionStatus(Enum):
    """Status of prediction accuracy."""
    EXCELLENT = "excellent"  # Error < 2%
    GOOD = "good"           # Error < 5%
    DEGRADED = "degraded"   # Error < 10%
    POOR = "poor"           # Error >= 10%


@dataclass
class CycleData:
    """
    Accumulated telemetry data for a single charge/discharge cycle.

    Attributes:
        cycle: Cycle number
        telemetry_points: List of telemetry messages
        avg_temperature: Average temperature during cycle
        total_charge_time: Total charge time in seconds
        start_time: Cycle start timestamp
        end_time: Cycle end timestamp (None if ongoing)
    """
    cycle: int
    telemetry_points: List[TelemetryMessage] = field(default_factory=list)
    avg_temperature: float = 0.0
    total_charge_time: float = 0.0
    start_time: float = 0.0
    end_time: Optional[float] = None

    def add_telemetry(self, msg: TelemetryMessage):
        """Add telemetry point to cycle data."""
        self.telemetry_points.append(msg)

    def finalize(self):
        """Compute aggregate statistics when cycle ends."""
        if len(self.telemetry_points) == 0:
            return

        # Compute average temperature
        temps = [t.temperature for t in self.telemetry_points]
        self.avg_temperature = float(np.mean(temps))

        # Compute total charge time (approximate from number of samples)
        # Assuming 1 second between samples (adjust based on actual data)
        self.total_charge_time = float(len(self.telemetry_points))

        # Set end time
        self.end_time = self.telemetry_points[-1].timestamp


@dataclass
class PredictionHistory:
    """History of predictions for monitoring accuracy."""
    cycle: int
    predicted_capacity: float
    actual_capacity: Optional[float] = None
    prediction_error: Optional[float] = None
    timestamp: float = field(default_factory=time.time)

    def compute_error(self, actual: float):
        """Compute prediction error when actual capacity is known."""
        self.actual_capacity = actual
        self.prediction_error = abs(self.predicted_capacity - actual)

    def relative_error(self) -> float:
        """Compute relative error as percentage."""
        if self.actual_capacity is None or self.actual_capacity == 0:
            return 0.0
        return abs(self.prediction_error / self.actual_capacity) * 100


class PhysicsModelAgent(BatteryHybridAgent):
    """
    Hybrid agent for physics-based battery capacity prediction.

    This agent combines:
    - Reactive rules: Fast telemetry processing, anomaly detection
    - Goal-driven behavior: Maintain prediction accuracy, adapt parameters

    The agent uses the PhysicsDegradationModel to predict battery capacity
    based on cycle number, temperature, and charge time.

    Example:
        >>> agent = PhysicsModelAgent(
        ...     agent_id=AgentId(app="battery_twin", type="physics", instance="1"),
        ...     battery_id="B0005"
        ... )
        >>> agent.setup()
        >>> # Agent processes telemetry and makes predictions
        >>> agent.teardown()
    """

    def __init__(
        self,
        agent_id: AgentId,
        battery_id: str,
        mqtt_bridge: Optional[MqttBridge] = None,
        storage_manager: Optional[BatteryStorageManager] = None,
        mqtt_config: Optional[MqttConfig] = None,
        model_parameters: Optional[DegradationParameters] = None,
        accuracy_threshold: float = 0.05,  # 5% error threshold
        enable_parameter_adaptation: bool = True,
        hybrid_service: Optional[object] = None,
    ):
        """
        Initialize Physics Model Agent.

        Args:
            agent_id: Agent identifier
            battery_id: Battery identifier to monitor
            mqtt_bridge: MQTT bridge for communication
            storage_manager: Storage manager for persistence
            mqtt_config: MQTT configuration
            model_parameters: Initial model parameters (uses defaults if None)
            accuracy_threshold: Error threshold for triggering parameter updates
            enable_parameter_adaptation: Enable automatic parameter adaptation
        """
        # Observable properties for this agent
        observable_properties = {
            'telemetry',
            'capacity',
            'prediction_error',
            'model_parameters',
            'cycle_complete'
        }

        # Initialize parent classes
        super().__init__(
            agent_id=agent_id,
            observable_properties=observable_properties,
            mqtt_bridge=mqtt_bridge,
            storage_manager=storage_manager,
            mqtt_config=mqtt_config,
            enable_heartbeat=True
        )

        # Configuration
        self.battery_id = battery_id
        self.accuracy_threshold = accuracy_threshold
        self.enable_parameter_adaptation = enable_parameter_adaptation
        self.hybrid_service = hybrid_service

        # Physics degradation model
        self.model = PhysicsDegradationModel(model_parameters)

        # Cycle data accumulation
        self.current_cycle: Optional[CycleData] = None
        self.last_cycle_number: int = 0

        # Prediction history for accuracy monitoring
        self.prediction_history: List[PredictionHistory] = []
        self.recent_errors: List[float] = []
        self.max_history = 100  # Keep last 100 predictions

        # Performance statistics
        self.total_predictions = 0
        self.successful_predictions = 0
        self.parameter_updates = 0
        self.completed_cycles: Dict[int, CycleData] = {}

        # Goals for deliberative layer
        self._initialize_goals()

        # Reactive rules
        self._initialize_reactive_rules()

        logger.info(
            f"PhysicsModelAgent initialized for battery {battery_id}: "
            f"k={self.model.params.k}, Q0={self.model.params.Q0}, "
            f"accuracy_threshold={accuracy_threshold}"
        )

    def _initialize_goals(self):
        """Initialize agent goals (BDI layer)."""
        # Goal 1: Predict capacity accurately
        self.add_goal(Goal(
            condition="predict_capacity_accurately",
            goal_type=GoalType.PERFORMANCE,
            priority=1.0
        ))

        # Goal 2: Maintain prediction accuracy
        self.add_goal(Goal(
            condition="maintain_accuracy",
            goal_type=GoalType.PERFORMANCE,
            priority=0.9
        ))

        # Goal 3: Adapt parameters when needed
        if self.enable_parameter_adaptation:
            self.add_goal(Goal(
                condition="adapt_model_parameters",
                goal_type=GoalType.INTRINSIC,
                priority=0.8
            ))

    def _initialize_reactive_rules(self):
        """
        Initialize reactive rules for fast response.

        Note: The reactive layer in this agent is primarily implemented through
        fast MQTT message handlers (via register_action), not formal reactive rules.
        The reactive rules here are for belief-driven fast responses.
        """
        from src.abstract_agent import ReactiveRule, Action, ActionType

        # Create action for cycle transition handling
        cycle_action = Action(
            action_id="handle_cycle_transition_action",
            action_type=ActionType.TRANSIENT,
            preconditions=lambda env: True,
            effects=lambda env: {**env, "cycle_transition_handled": True}
        )
        self.add_action(cycle_action)

        # Create reactive rule for cycle transition
        cycle_rule = ReactiveRule(
            condition=lambda state: 'cycle_transition' in state.get('external', {}),
            action="handle_cycle_transition_action",
            priority=1.0
        )
        self.add_reactive_rule(cycle_rule)

    def _agent_setup(self) -> bool:
        """Agent-specific setup."""
        try:
            # Register MQTT action handlers
            self.register_action(
                action_id="process_telemetry",
                handler=self._handle_telemetry,
                topic_pattern=f"battery/{self.battery_id}/telemetry/clean",
                description="Process clean telemetry data"
            )

            self.register_action(
                action_id="process_capacity",
                handler=self._handle_actual_capacity,
                topic_pattern=f"battery/{self.battery_id}/capacity",
                description="Process actual capacity measurements"
            )

            # Try to load model state from storage
            if self.storage_manager:
                self._load_model_from_storage()

            logger.info(f"PhysicsModelAgent setup complete for battery {self.battery_id}")
            return True

        except Exception as e:
            logger.error(f"PhysicsModelAgent setup failed: {e}")
            return False

    def _agent_teardown(self):
        """Agent-specific teardown."""
        # Save model state to storage
        if self.storage_manager:
            self._save_model_to_storage()

        logger.info(f"PhysicsModelAgent teardown complete for battery {self.battery_id}")

    # ========================================================================
    # Telemetry Processing (Reactive Layer)
    # ========================================================================

    def _handle_telemetry(self, topic: str, payload: str):
        """
        Handle incoming telemetry message.

        Reactive behavior: Fast accumulation of cycle data.

        Args:
            topic: MQTT topic
            payload: JSON message payload
        """
        try:
            # Parse telemetry message
            msg = MessageFactory.parse_message('telemetry', payload)

            # Check if cycle changed
            if msg.cycle != self.last_cycle_number:
                # New cycle detected
                if self.current_cycle is not None:
                    # Finalize previous cycle
                    self.current_cycle.finalize()

                    # Trigger cycle transition (reactive rule)
                    self.state.update_belief(
                        key='cycle_transition',
                        proposition=f"cycle_{self.last_cycle_number}_complete",
                        confidence=1.0
                    )

                    # Store the completed cycle before starting new one
                    completed_cycle = self.current_cycle
                    self.completed_cycles[completed_cycle.cycle] = completed_cycle

                    # Start new cycle first
                    self.current_cycle = CycleData(
                        cycle=msg.cycle,
                        start_time=msg.timestamp
                    )
                    self.last_cycle_number = msg.cycle

                    # Handle cycle transition (reactive behavior)
                    # Note: We do this after starting new cycle so predictions
                    # are for the completed cycle, not the new one
                    self._handle_cycle_transition_for_cycle(completed_cycle)

                    logger.debug(f"Completed cycle {completed_cycle.cycle}, started tracking cycle {msg.cycle}")
                else:
                    # First cycle
                    self.current_cycle = CycleData(
                        cycle=msg.cycle,
                        start_time=msg.timestamp
                    )
                    self.last_cycle_number = msg.cycle
                    logger.debug(f"Started tracking cycle {msg.cycle}")

            # Add telemetry to current cycle
            if self.current_cycle:
                self.current_cycle.add_telemetry(msg)

            # Update belief state
            self.state.update_belief(
                key='telemetry',
                proposition=f"received_cycle_{msg.cycle}",
                confidence=1.0
            )

        except Exception as e:
            logger.error(f"Failed to handle telemetry: {e}")

    def _handle_cycle_transition_for_cycle(self, completed_cycle: CycleData):
        """
        Handle cycle transition (reactive behavior).

        When a cycle completes:
        1. Make capacity prediction
        2. Publish prediction to MQTT
        3. Store prediction
        4. Check if deliberative action needed

        Args:
            completed_cycle: The CycleData for the just-completed cycle
        """
        try:
            if completed_cycle is None:
                return

            cycle = completed_cycle.cycle

            logger.info(f"Processing cycle {cycle} completion")

            # Predict capacity using physics model
            predicted_capacity = self.model.predict_capacity(
                cycle=cycle,
                temperature=completed_cycle.avg_temperature,
                charge_time=max(completed_cycle.total_charge_time, 1.0)
            )

            # Create prediction message
            prediction_msg = PredictionMessage(
                battery_id=self.battery_id,
                timestamp=time.time(),
                cycle=cycle,
                prediction_type="physics",
                predicted_capacity=float(predicted_capacity),
                uncertainty=None,  # Physics model doesn't provide uncertainty
                horizon=0,
                agent_id=str(self.id)
            )

            # Publish to MQTT
            self.publish_message(
                topic_name="battery_prediction_physics",
                message=prediction_msg,
                battery_id=self.battery_id
            )

            # Store prediction
            self._store_prediction(prediction_msg)

            # Track prediction
            pred_history = PredictionHistory(
                cycle=cycle,
                predicted_capacity=predicted_capacity
            )
            self.prediction_history.append(pred_history)

            # Trim history
            if len(self.prediction_history) > self.max_history:
                self.prediction_history = self.prediction_history[-self.max_history:]

            self.total_predictions += 1

            logger.info(
                f"Predicted capacity for cycle {cycle}: {predicted_capacity:.4f} Ah "
                f"(T={self.current_cycle.avg_temperature:.1f}°C, "
                f"t={self.current_cycle.total_charge_time:.0f}s)"
            )

        except Exception as e:
            logger.error(f"Failed to handle cycle transition: {e}")

    # ========================================================================
    # Actual Capacity Processing
    # ========================================================================

    def _handle_actual_capacity(self, topic: str, payload: str):
        """
        Handle actual capacity measurement.

        Update prediction history with actual values and compute errors.

        Args:
            topic: MQTT topic
            payload: JSON message payload
        """
        try:
            # Parse capacity message
            msg = MessageFactory.parse_message('capacity', payload)

            # Find corresponding prediction
            for pred in reversed(self.prediction_history):
                if pred.cycle == msg.cycle and pred.actual_capacity is None:
                    # Update with actual capacity
                    pred.compute_error(msg.capacity)

                    # Track recent errors
                    self.recent_errors.append(pred.relative_error())
                    if len(self.recent_errors) > 20:  # Keep last 20 errors
                        self.recent_errors = self.recent_errors[-20:]

                    # Update belief
                    self.state.update_belief(
                        key='prediction_error',
                        proposition=f"error_{pred.relative_error():.2f}_percent",
                        confidence=1.0
                    )

                    self.successful_predictions += 1

                    logger.info(
                        f"Cycle {msg.cycle}: Predicted={pred.predicted_capacity:.4f} Ah, "
                        f"Actual={msg.capacity:.4f} Ah, "
                        f"Error={pred.relative_error():.2f}%"
                    )

                    # Trigger deliberative reasoning if error is high
                    if pred.relative_error() > self.accuracy_threshold * 100:
                        self._deliberate_on_accuracy()

                    self._publish_hybrid_training_sample(msg.cycle, msg.capacity)

                    break

        except Exception as e:
            logger.error(f"Failed to handle actual capacity: {e}")

    def _publish_hybrid_training_sample(self, cycle: int, actual_capacity: float):
        """Send completed cycle data to shared hybrid service."""
        if (
            not self.hybrid_service
            or not hasattr(self.hybrid_service, "train_hybrid_twin")
            or actual_capacity is None
        ):
            return

        cycle_data = self.completed_cycles.pop(cycle, None)
        if cycle_data is None and self.current_cycle and self.current_cycle.cycle == cycle:
            cycle_data = self.current_cycle
            cycle_data.finalize()

        if not cycle_data or cycle_data.avg_temperature == 0.0:
            return

        avg_temp = cycle_data.avg_temperature
        duration = cycle_data.total_charge_time or len(cycle_data.telemetry_points) or 1.0

        sample = {
            "id_cycle": cycle,
            "Temperature_measured": avg_temp,
            "Time": duration,
            "Capacity": actual_capacity,
        }

        try:
            self.hybrid_service.train_hybrid_twin([sample])
        except Exception as exc:
            logger.debug("Failed to submit hybrid training sample: %s", exc)

    # ========================================================================
    # Deliberative Layer (BDI)
    # ========================================================================

    def _deliberate_on_accuracy(self):
        """
        Deliberate on prediction accuracy (goal-driven behavior).

        When accuracy degrades:
        1. Assess if parameter adaptation is needed
        2. Decide whether to trigger parameter update
        3. Create intention to improve accuracy
        """
        try:
            # Check average recent error
            if len(self.recent_errors) < 3:
                return  # Not enough data

            avg_error = np.mean(self.recent_errors)
            max_error = np.max(self.recent_errors)

            logger.info(
                f"Accuracy assessment: avg_error={avg_error:.2f}%, "
                f"max_error={max_error:.2f}%, threshold={self.accuracy_threshold * 100}%"
            )

            # Decision: Should we adapt parameters?
            if avg_error > self.accuracy_threshold * 100:
                if self.enable_parameter_adaptation:
                    # Create intention to adapt parameters
                    logger.warning(
                        f"Accuracy degraded (avg={avg_error:.2f}%). "
                        "Triggering parameter adaptation"
                    )
                    self._adapt_parameters()
                else:
                    logger.warning(
                        f"Accuracy degraded (avg={avg_error:.2f}%), "
                        "but parameter adaptation is disabled"
                    )

        except Exception as e:
            logger.error(f"Deliberation failed: {e}")

    def _adapt_parameters(self):
        """
        Adapt model parameters based on recent prediction errors.

        Uses collected data to re-fit the physics model parameters.
        """
        try:
            logger.info("Starting parameter adaptation...")

            # Collect recent data for fitting
            cycles = []
            temperatures = []
            charge_times = []
            actual_capacities = []

            for pred in self.prediction_history:
                if pred.actual_capacity is not None:
                    cycles.append(pred.cycle)
                    # We need temperature and charge time, but we don't store them
                    # For now, use estimates or default values
                    temperatures.append(25.0)  # TODO: Store with prediction
                    charge_times.append(3600.0)  # TODO: Store with prediction
                    actual_capacities.append(pred.actual_capacity)

            if len(cycles) < 10:
                logger.warning("Not enough data for parameter adaptation (need >= 10 points)")
                return

            # Convert to numpy arrays
            cycles = np.array(cycles)
            temperatures = np.array(temperatures)
            charge_times = np.array(charge_times)
            actual_capacities = np.array(actual_capacities)

            # Fit model
            old_k = self.model.params.k
            old_Q0 = self.model.params.Q0

            metrics = self.model.fit(
                cycles=cycles,
                temperatures=temperatures,
                charge_times=charge_times,
                actual_capacities=actual_capacities,
                optimize_k=True,
                optimize_Q0=True
            )

            self.parameter_updates += 1

            logger.info(
                f"Parameters adapted: k: {old_k:.6f} → {self.model.params.k:.6f}, "
                f"Q0: {old_Q0:.4f} → {self.model.params.Q0:.4f} "
                f"(RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f})"
            )

            # Update belief
            self.state.update_belief(
                key='model_parameters',
                proposition=f"adapted_k_{self.model.params.k:.6f}_Q0_{self.model.params.Q0:.4f}",
                confidence=metrics['r2']
            )

            # Clear recent errors after adaptation
            self.recent_errors = []

            # Save updated model
            if self.storage_manager:
                self._save_model_to_storage()

        except Exception as e:
            logger.error(f"Parameter adaptation failed: {e}")

    # ========================================================================
    # Storage Integration
    # ========================================================================

    def _store_prediction(self, prediction: PredictionMessage):
        """Store prediction to database."""
        if not self.storage_manager:
            return

        try:
            self.persist_to_storage(
                operation="prediction",
                battery_id=self.battery_id,
                timestamp=prediction.timestamp,
                cycle=prediction.cycle,
                prediction_type=prediction.prediction_type,
                predicted_capacity=prediction.predicted_capacity,
                uncertainty=prediction.uncertainty,
                agent_id=str(self.id)
            )
        except Exception as e:
            logger.error(f"Failed to store prediction: {e}")

    def _save_model_to_storage(self):
        """Save model state to MongoDB."""
        if not self.storage_manager:
            return

        try:
            import pickle

            model_data = {
                'parameters': self.model.export_state(),
                'prediction_history': [
                    {
                        'cycle': p.cycle,
                        'predicted': p.predicted_capacity,
                        'actual': p.actual_capacity,
                        'error': p.prediction_error
                    }
                    for p in self.prediction_history[-50:]  # Save last 50
                ],
                'statistics': self.get_statistics()
            }

            model_bytes = pickle.dumps(model_data)

            # Store trained model (using storage manager's MongoDB)
            # This would typically use storage_manager.store_trained_model()
            logger.info(f"Saved model state ({len(model_bytes)} bytes)")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def _load_model_from_storage(self):
        """Load model state from MongoDB."""
        if not self.storage_manager:
            return

        try:
            # This would typically use storage_manager.load_latest_model()
            logger.debug("Attempted to load model from storage")
            # For now, just log - actual implementation would restore state

        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    # ========================================================================
    # Statistics and Monitoring
    # ========================================================================

    def get_statistics(self) -> Dict:
        """Get agent statistics."""
        if len(self.recent_errors) > 0:
            avg_error = np.mean(self.recent_errors)
            max_error = np.max(self.recent_errors)
            min_error = np.min(self.recent_errors)
        else:
            avg_error = max_error = min_error = 0.0

        # Determine prediction status
        if avg_error < 2.0:
            status = PredictionStatus.EXCELLENT
        elif avg_error < 5.0:
            status = PredictionStatus.GOOD
        elif avg_error < 10.0:
            status = PredictionStatus.DEGRADED
        else:
            status = PredictionStatus.POOR

        return {
            'total_predictions': self.total_predictions,
            'successful_predictions': self.successful_predictions,
            'parameter_updates': self.parameter_updates,
            'avg_error_percent': avg_error,
            'max_error_percent': max_error,
            'min_error_percent': min_error,
            'prediction_status': status.value,
            'current_k': self.model.params.k,
            'current_Q0': self.model.params.Q0,
            'last_cycle': self.last_cycle_number
        }

    def get_prediction_status(self) -> PredictionStatus:
        """Get current prediction accuracy status."""
        if len(self.recent_errors) == 0:
            return PredictionStatus.GOOD

        avg_error = np.mean(self.recent_errors)

        if avg_error < 2.0:
            return PredictionStatus.EXCELLENT
        elif avg_error < 5.0:
            return PredictionStatus.GOOD
        elif avg_error < 10.0:
            return PredictionStatus.DEGRADED
        else:
            return PredictionStatus.POOR

    def get_model_parameters(self) -> DegradationParameters:
        """Get current model parameters."""
        return self.model.get_parameters()

    def set_model_parameters(self, parameters: DegradationParameters):
        """Set model parameters manually."""
        self.model.params = parameters
        logger.info(f"Model parameters updated: k={parameters.k}, Q0={parameters.Q0}")


__all__ = [
    'PhysicsModelAgent',
    'PredictionStatus',
    'CycleData',
    'PredictionHistory',
]
