"""
State Estimator Agent for Battery Digital Twin.

This module implements a BDI (Belief-Desire-Intention) agent that performs
real-time battery state estimation using an Extended Kalman Filter.

The agent estimates:
    - SoC (State of Charge)
    - SoH (State of Health)
    - R0, R1, C1 (Equivalent circuit parameters)
    - V1 (RC voltage)

BDI Components:
    - Beliefs: current_state, uncertainty, filter_health, divergence_status
    - Desires: accurate_estimation, low_uncertainty, robust_filtering
    - Intentions: filter_tuning_plan, outlier_handling, reset_plan

Deliberation:
    - Decides when to reset filter (divergence detected)
    - Decides when to adjust process noise (high innovation)
    - Decides when to flag low confidence (large covariance)
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
import time
import numpy as np
from loguru import logger

from src.abstract_agent import AgentId, Goal, GoalType
from src.battery_twin.agents.battery_agent_types import BatteryBDIAgent
from src.battery_twin.communication.mqtt_bridge import MqttBridge
from src.battery_twin.storage.battery_storage_manager import BatteryStorageManager
from src.battery_twin.models.extended_kalman_filter import (
    ExtendedKalmanFilter,
    EKFConfig,
    EKFMeasurement
)
from src.battery_twin.communication.message_schemas import (
    TelemetryMessage,
    StateEstimateMessage,
    MessageFactory
)
from src.battery_twin.models.physics_degradation_model import (
    DegradationParameters,
    PhysicsDegradationModel,
)


class FilterHealth(Enum):
    """Filter health status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    DIVERGED = "diverged"
    RESET_REQUIRED = "reset_required"


class ConfidenceLevel(Enum):
    """Confidence level for state estimates."""
    HIGH = "high"          # Uncertainty < threshold_low
    MEDIUM = "medium"      # threshold_low <= uncertainty < threshold_high
    LOW = "low"            # uncertainty >= threshold_high


@dataclass
class StateEstimate:
    """Container for state estimates with metadata."""
    cycle: int
    timestamp: float
    soc: float
    soh: float
    r0: float
    r1: float
    c1: float
    v1: float
    soc_uncertainty: float
    soh_uncertainty: float
    confidence_level: ConfidenceLevel
    filter_health: FilterHealth


class StateEstimatorAgent(BatteryBDIAgent):
    """
    BDI Agent for real-time battery state estimation using Extended Kalman Filter.

    This agent integrates the EKF from Step 12 with BDI reasoning to provide
    intelligent filter management, adaptive tuning, and robust state estimation.

    BDI Architecture:
        - Beliefs: State estimates, uncertainty, filter health
        - Desires: Accurate estimation, low uncertainty, robust operation
        - Intentions: Filter tuning plan, reset plan, outlier handling

    Example:
        >>> agent = StateEstimatorAgent(
        ...     agent_id=AgentId("battery", "state_estimator", "001"),
        ...     battery_id="B0005",
        ...     mqtt_bridge=bridge
        ... )
        >>> agent.start()
        >>> # Agent subscribes to telemetry and publishes state estimates
    """

    def __init__(
        self,
        agent_id: AgentId,
        battery_id: str,
        mqtt_bridge: Optional[MqttBridge] = None,
        storage_manager: Optional[BatteryStorageManager] = None,
        ekf_config: Optional[EKFConfig] = None,
        soc_uncertainty_threshold_low: float = 0.03,  # 3%
        soc_uncertainty_threshold_high: float = 0.10,  # 10%
        soh_uncertainty_threshold_low: float = 0.05,  # 5%
        soh_uncertainty_threshold_high: float = 0.15,  # 15%
        max_innovation: float = 0.5,  # Volts
        process_noise_adjustment_factor: float = 1.5,
        reset_threshold_innovation: float = 1.0,  # Volts
        publish_interval: float = 1.0  # Publish every measurement
    ):
        """
        Initialize State Estimator Agent.

        Args:
            agent_id: Unique agent identifier
            battery_id: Battery identifier
            mqtt_bridge: MQTT communication bridge
            storage_manager: Storage manager for persistence
            ekf_config: EKF configuration (uses default if None)
            soc_uncertainty_threshold_low: Low uncertainty threshold for SoC
            soc_uncertainty_threshold_high: High uncertainty threshold for SoC
            soh_uncertainty_threshold_low: Low uncertainty threshold for SoH
            soh_uncertainty_threshold_high: High uncertainty threshold for SoH
            max_innovation: Maximum acceptable innovation before warning
            process_noise_adjustment_factor: Factor for increasing process noise
            reset_threshold_innovation: Innovation threshold for filter reset
            publish_interval: Interval for publishing state estimates
        """
        # Store battery_id before super init
        self.battery_id = battery_id

        super().__init__(
            agent_id=agent_id,
            observable_properties={'state_estimates', 'filter_health', 'uncertainty'},
            mqtt_bridge=mqtt_bridge,
            storage_manager=storage_manager
        )

        # Initialize Extended Kalman Filter
        self.ekf_config = ekf_config or EKFConfig()
        self.ekf = ExtendedKalmanFilter(self.ekf_config)

        self.physics_model = PhysicsDegradationModel(
            DegradationParameters(Q0=self.ekf_config.capacity_nominal)
        )
        self.physics_model.is_fitted = True
        self._physics_cycle_stats: Dict[int, Dict[str, float]] = defaultdict(
            lambda: {"temp_sum": 0.0, "count": 0, "duration": 0.0}
        )
        self._physics_current_cycle: Optional[int] = None

        # Thresholds and parameters
        self.soc_uncertainty_threshold_low = soc_uncertainty_threshold_low
        self.soc_uncertainty_threshold_high = soc_uncertainty_threshold_high
        self.soh_uncertainty_threshold_low = soh_uncertainty_threshold_low
        self.soh_uncertainty_threshold_high = soh_uncertainty_threshold_high
        self.max_innovation = max_innovation
        self.process_noise_adjustment_factor = process_noise_adjustment_factor
        self.reset_threshold_innovation = reset_threshold_innovation
        self.publish_interval = publish_interval

        # State tracking
        self.latest_estimate: Optional[StateEstimate] = None
        self.estimate_history: List[StateEstimate] = []
        self.last_measurement_time: Optional[float] = None
        self.last_publish_time: float = 0.0

        # Statistics
        self.total_measurements_processed = 0
        self.total_resets = 0
        self.total_adjustments = 0
        self.total_warnings = 0

        # Subscribe to telemetry
        if self.mqtt_bridge:
            self._subscribe_to_topics()

        # Initialize BDI components
        self._initialize_beliefs()
        self._initialize_goals()

        logger.info(f"StateEstimatorAgent initialized for battery {battery_id}")

    def _subscribe_to_topics(self):
        """Subscribe to relevant MQTT topics."""
        # Subscribe to clean telemetry data
        telemetry_topic = f"battery/{self.battery_id}/telemetry/clean"
        self.mqtt_bridge.subscribe(telemetry_topic, self._handle_telemetry)

        logger.info(f"Subscribed to {telemetry_topic}")

    def _initialize_beliefs(self):
        """Initialize BDI beliefs about state and filter health."""
        # Current state beliefs
        self.state.update_belief(
            key='current_soc',
            proposition=f"soc_{self.ekf.get_soc():.2f}",
            confidence=1.0,
            is_internal=True
        )

        self.state.update_belief(
            key='current_soh',
            proposition=f"soh_{self.ekf.get_soh():.2f}",
            confidence=1.0,
            is_internal=True
        )

        # Uncertainty beliefs
        self.state.update_belief(
            key='soc_uncertainty',
            proposition=f"uncertainty_{self.ekf.get_soc_uncertainty():.4f}",
            confidence=1.0,
            is_internal=True
        )

        self.state.update_belief(
            key='soh_uncertainty',
            proposition=f"uncertainty_{self.ekf.get_soh_uncertainty():.4f}",
            confidence=1.0,
            is_internal=True
        )

        # Filter health beliefs
        self.state.update_belief(
            key='filter_health',
            proposition='status_healthy',
            confidence=1.0,
            is_internal=True
        )

        self.state.update_belief(
            key='divergence_detected',
            proposition='false',
            confidence=1.0,
            is_internal=True
        )

        # Confidence level belief
        self.state.update_belief(
            key='confidence_level',
            proposition='level_high',
            confidence=1.0,
            is_internal=True
        )

        logger.debug("State estimator beliefs initialized")

    def _initialize_goals(self):
        """Initialize BDI goals for state estimation."""
        # Goal: Maintain accurate state estimation
        self.add_goal(Goal(
            condition="accurate_estimation",
            goal_type=GoalType.PERFORMANCE,
            priority=1.0
        ))

        # Goal: Maintain low uncertainty
        self.add_goal(Goal(
            condition="low_uncertainty",
            goal_type=GoalType.PERFORMANCE,
            priority=0.9
        ))

        # Goal: Robust filtering (no divergence)
        self.add_goal(Goal(
            condition="robust_filtering",
            goal_type=GoalType.PERFORMANCE,
            priority=1.0
        ))

        logger.debug("State estimator goals initialized")

    def _handle_telemetry(self, topic: str, payload: str):
        """
        Handle incoming telemetry message.

        Args:
            topic: MQTT topic
            payload: JSON message payload
        """
        try:
            # Parse telemetry message
            msg = MessageFactory.parse_message('telemetry', payload)

            self._update_physics_cycle_stats(msg)

            # Create EKF measurement
            measurement = EKFMeasurement(
                voltage=msg.voltage,
                current=msg.current,
                temperature=msg.temperature,
                timestamp=msg.timestamp
            )

            # Process measurement through EKF
            self._process_measurement(measurement, msg.cycle)

            # Deliberate on filter management
            self._deliberate_on_filter_management()

            # Publish state estimate if interval elapsed
            self._maybe_publish_estimate(msg.cycle, msg.timestamp)

            self.total_measurements_processed += 1

        except Exception as e:
            logger.error(f"Error handling telemetry: {e}")

    def _update_physics_cycle_stats(self, telemetry: TelemetryMessage):
        """Accumulate telemetry statistics for physics-based priors."""
        try:
            stats = self._physics_cycle_stats.setdefault(
                telemetry.cycle,
                {"temp_sum": 0.0, "count": 0, "duration": 0.0},
            )

            if self.last_measurement_time is None:
                dt = 0.0
            else:
                dt = max(telemetry.timestamp - self.last_measurement_time, 0.0)

            stats["temp_sum"] += telemetry.temperature
            stats["count"] += 1
            stats["duration"] += dt

            if self._physics_current_cycle is None:
                self._physics_current_cycle = telemetry.cycle
            elif telemetry.cycle != self._physics_current_cycle:
                self._apply_physics_prior(self._physics_current_cycle)
                self._physics_current_cycle = telemetry.cycle
        except Exception as exc:
            logger.debug("Failed to update physics stats: %s", exc)

    def _apply_physics_prior(self, cycle: Optional[int]):
        """Blend physics-based SoH predictions into the EKF state."""
        if cycle is None:
            return

        stats = self._physics_cycle_stats.pop(cycle, None)
        if not stats or stats["count"] == 0:
            return

        avg_temp = stats["temp_sum"] / stats["count"]
        duration = max(stats["duration"], 1.0)

        try:
            capacity = self.physics_model.predict_capacity(
                cycle=cycle, temperature=avg_temp, charge_time=duration
            )
        except Exception as exc:
            logger.debug("Physics prior skipped for cycle %s: %s", cycle, exc)
            return

        predicted_soh = capacity / max(self.ekf_config.capacity_nominal, 1e-6)
        current_soh = float(self.ekf.state.x[1])
        blended_soh = 0.85 * current_soh + 0.15 * predicted_soh
        clipped_soh = float(
            np.clip(
                blended_soh,
                self.ekf_config.min_soh,
                self.ekf_config.max_soh,
            )
        )
        self.ekf.state.x[1] = clipped_soh

        logger.debug(
            "Physics prior applied for cycle %s: predicted SoH %.3f, blended %.3f",
            cycle,
            predicted_soh,
            clipped_soh,
        )

    def _process_measurement(self, measurement: EKFMeasurement, cycle: int):
        """
        Process measurement through Extended Kalman Filter.

        Args:
            measurement: EKF measurement
            cycle: Battery cycle number
        """
        # Run EKF prediction-update cycle
        self.ekf.process_measurement(measurement)

        # Extract state estimates
        soc = self.ekf.get_soc()
        soh = self.ekf.get_soh()
        r0 = self.ekf.get_r0()
        r1 = self.ekf.get_r1()
        c1 = self.ekf.get_c1()
        v1 = self.ekf.get_v1()

        # Extract uncertainties
        soc_uncertainty = self.ekf.get_soc_uncertainty()
        soh_uncertainty = self.ekf.get_soh_uncertainty()

        # Determine confidence level
        confidence_level = self._compute_confidence_level(soc_uncertainty, soh_uncertainty)

        # Determine filter health
        filter_health = self._assess_filter_health()

        # Create state estimate
        estimate = StateEstimate(
            cycle=cycle,
            timestamp=measurement.timestamp,
            soc=soc,
            soh=soh,
            r0=r0,
            r1=r1,
            c1=c1,
            v1=v1,
            soc_uncertainty=soc_uncertainty,
            soh_uncertainty=soh_uncertainty,
            confidence_level=confidence_level,
            filter_health=filter_health
        )

        # Store estimate
        self.latest_estimate = estimate
        self.estimate_history.append(estimate)

        # Update last measurement time
        self.last_measurement_time = measurement.timestamp

        # Update beliefs
        self._update_beliefs_from_estimate(estimate)

        logger.debug(
            f"Processed measurement: SoC={soc:.3f}±{soc_uncertainty:.3f}, "
            f"SoH={soh:.3f}±{soh_uncertainty:.3f}, confidence={confidence_level.value}"
        )

    def _compute_confidence_level(
        self,
        soc_uncertainty: float,
        soh_uncertainty: float
    ) -> ConfidenceLevel:
        """
        Compute confidence level based on uncertainties.

        Args:
            soc_uncertainty: SoC uncertainty (standard deviation)
            soh_uncertainty: SoH uncertainty (standard deviation)

        Returns:
            Confidence level
        """
        # Check if either exceeds high threshold
        if (soc_uncertainty >= self.soc_uncertainty_threshold_high or
            soh_uncertainty >= self.soh_uncertainty_threshold_high):
            return ConfidenceLevel.LOW

        # Check if both below low threshold
        if (soc_uncertainty < self.soc_uncertainty_threshold_low and
            soh_uncertainty < self.soh_uncertainty_threshold_low):
            return ConfidenceLevel.HIGH

        return ConfidenceLevel.MEDIUM

    def _assess_filter_health(self) -> FilterHealth:
        """
        Assess Extended Kalman Filter health.

        Returns:
            Filter health status
        """
        # Check if diverged
        if self.ekf.is_diverged():
            return FilterHealth.DIVERGED

        # Check innovation magnitude
        if self.ekf.last_innovation is not None:
            innovation_abs = abs(self.ekf.last_innovation)

            if innovation_abs > self.reset_threshold_innovation:
                return FilterHealth.RESET_REQUIRED

            if innovation_abs > self.max_innovation:
                return FilterHealth.WARNING

        return FilterHealth.HEALTHY

    def _update_beliefs_from_estimate(self, estimate: StateEstimate):
        """
        Update BDI beliefs from state estimate.

        Args:
            estimate: State estimate
        """
        # Update state beliefs
        self.state.update_belief(
            key='current_soc',
            proposition=f"soc_{estimate.soc:.2f}",
            confidence=1.0 - estimate.soc_uncertainty,
            is_internal=True
        )

        self.state.update_belief(
            key='current_soh',
            proposition=f"soh_{estimate.soh:.2f}",
            confidence=1.0 - estimate.soh_uncertainty,
            is_internal=True
        )

        # Update uncertainty beliefs
        self.state.update_belief(
            key='soc_uncertainty',
            proposition=f"uncertainty_{estimate.soc_uncertainty:.4f}",
            confidence=1.0,
            is_internal=True
        )

        self.state.update_belief(
            key='soh_uncertainty',
            proposition=f"uncertainty_{estimate.soh_uncertainty:.4f}",
            confidence=1.0,
            is_internal=True
        )

        # Update filter health belief
        self.state.update_belief(
            key='filter_health',
            proposition=f"status_{estimate.filter_health.value}",
            confidence=1.0,
            is_internal=True
        )

        # Update divergence belief
        self.state.update_belief(
            key='divergence_detected',
            proposition='true' if estimate.filter_health == FilterHealth.DIVERGED else 'false',
            confidence=1.0,
            is_internal=True
        )

        # Update confidence level belief
        self.state.update_belief(
            key='confidence_level',
            proposition=f"level_{estimate.confidence_level.value}",
            confidence=1.0,
            is_internal=True
        )

    def _deliberate_on_filter_management(self):
        """
        BDI deliberation: Decide on filter management actions.

        Decisions:
            1. Reset filter if diverged or innovation too large
            2. Adjust process noise if high innovation
            3. Flag low confidence warnings
        """
        if not self.latest_estimate:
            return

        filter_health = self.latest_estimate.filter_health

        # Decision 1: Reset filter if diverged or reset required
        if filter_health in [FilterHealth.DIVERGED, FilterHealth.RESET_REQUIRED]:
            logger.warning(f"Filter health is {filter_health.value}, resetting filter")
            self._reset_filter()
            return

        # Decision 2: Adjust process noise if warning state
        if filter_health == FilterHealth.WARNING:
            logger.warning("Filter in warning state, considering process noise adjustment")
            self._maybe_adjust_process_noise()

        # Decision 3: Flag low confidence
        if self.latest_estimate.confidence_level == ConfidenceLevel.LOW:
            logger.warning(
                f"Low confidence in state estimates: "
                f"SoC uncertainty={self.latest_estimate.soc_uncertainty:.3f}, "
                f"SoH uncertainty={self.latest_estimate.soh_uncertainty:.3f}"
            )
            self.total_warnings += 1

    def _reset_filter(self):
        """Reset Extended Kalman Filter to initial state."""
        logger.info("Resetting Extended Kalman Filter")

        # Reset EKF
        self.ekf.reset()

        # Update statistics
        self.total_resets += 1

        # Update beliefs
        self.state.update_belief(
            key='filter_health',
            proposition='status_healthy',
            confidence=1.0,
            is_internal=True
        )

        self.state.update_belief(
            key='divergence_detected',
            proposition='false',
            confidence=1.0,
            is_internal=True
        )

        logger.info(f"Filter reset complete (total resets: {self.total_resets})")

    def _maybe_adjust_process_noise(self):
        """
        Adjust process noise if high innovation detected.

        This is a simple adaptive strategy: increase process noise to allow
        more flexibility in state tracking.
        """
        # Check recent innovation history
        if self.ekf.last_innovation is not None:
            innovation_abs = abs(self.ekf.last_innovation)

            # Only adjust if innovation is consistently high
            if innovation_abs > self.max_innovation:
                logger.info(
                    f"Adjusting process noise due to high innovation: {innovation_abs:.4f}V"
                )

                # Increase process noise for SoC and SoH
                self.ekf.config.q_soc *= self.process_noise_adjustment_factor
                self.ekf.config.q_soh *= self.process_noise_adjustment_factor

                self.total_adjustments += 1

                logger.info(
                    f"Process noise adjusted (total adjustments: {self.total_adjustments})"
                )

    def _maybe_publish_estimate(self, cycle: int, timestamp: float):
        """
        Publish state estimate if interval elapsed.

        Args:
            cycle: Current cycle
            timestamp: Current timestamp
        """
        if not self.latest_estimate:
            return

        # Check if publish interval elapsed
        if timestamp - self.last_publish_time < self.publish_interval:
            return

        # Publish estimate
        self._publish_state_estimate(self.latest_estimate)
        self.last_publish_time = timestamp

    def _publish_state_estimate(self, estimate: StateEstimate):
        """
        Publish state estimate to MQTT.

        Args:
            estimate: State estimate to publish
        """
        if not self.mqtt_bridge:
            return

        # Create state estimate message using existing schema
        message = StateEstimateMessage(
            battery_id=self.battery_id,
            timestamp=estimate.timestamp,
            soc=estimate.soc,
            soh=estimate.soh,
            internal_resistance={
                'R0': estimate.r0,
                'R1': estimate.r1,
                'C1': estimate.c1
            },
            uncertainty={
                'soc': estimate.soc_uncertainty,
                'soh': estimate.soh_uncertainty
            },
            agent_id=str(self.id)
        )

        # Publish to state topic
        state_topic = f"battery/{self.battery_id}/state/estimate"
        self.mqtt_bridge.publish(state_topic, message.model_dump_json())

        logger.debug(f"Published state estimate to {state_topic}")

    # Getter methods for external access

    def get_latest_estimate(self) -> Optional[StateEstimate]:
        """Get latest state estimate."""
        return self.latest_estimate

    def get_current_soc(self) -> float:
        """Get current SoC estimate."""
        return self.ekf.get_soc()

    def get_current_soh(self) -> float:
        """Get current SoH estimate."""
        return self.ekf.get_soh()

    def get_soc_with_uncertainty(self) -> tuple:
        """Get SoC with uncertainty (mean, std)."""
        return (self.ekf.get_soc(), self.ekf.get_soc_uncertainty())

    def get_soh_with_uncertainty(self) -> tuple:
        """Get SoH with uncertainty (mean, std)."""
        return (self.ekf.get_soh(), self.ekf.get_soh_uncertainty())

    def get_filter_health(self) -> FilterHealth:
        """Get current filter health status."""
        if self.latest_estimate:
            return self.latest_estimate.filter_health
        return FilterHealth.HEALTHY

    def get_confidence_level(self) -> ConfidenceLevel:
        """Get current confidence level."""
        if self.latest_estimate:
            return self.latest_estimate.confidence_level
        return ConfidenceLevel.HIGH

    def get_statistics(self) -> dict:
        """Get agent statistics."""
        ekf_stats = self.ekf.get_statistics()

        return {
            'total_measurements_processed': self.total_measurements_processed,
            'total_resets': self.total_resets,
            'total_adjustments': self.total_adjustments,
            'total_warnings': self.total_warnings,
            'estimate_history_size': len(self.estimate_history),
            'filter_health': self.get_filter_health().value,
            'confidence_level': self.get_confidence_level().value,
            'ekf_stats': ekf_stats
        }

    def _agent_teardown(self):
        """Agent-specific teardown."""
        self._apply_physics_prior(self._physics_current_cycle)
        logger.info(f"StateEstimatorAgent {self.agent_id} shutting down")
        logger.info(
            "Final statistics: total_measurements=%s, resets=%s, adjustments=%s",
            self.total_measurements_processed,
            self.total_resets,
            self.total_adjustments,
        )
