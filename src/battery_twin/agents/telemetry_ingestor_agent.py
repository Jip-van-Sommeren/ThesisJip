"""
Telemetry Ingestor Agent

A Reactive agent that:
- Subscribes to raw telemetry data from replay engine or physical sensors
- Validates and cleans telemetry data
- Detects outliers and missing data
- Publishes cleaned telemetry for downstream processing
- Stores telemetry in InfluxDB time-series database

Agent Type: Reactive (fast, stimulus-response behavior)
Input: battery/{battery_id}/raw
Output: battery/{battery_id}/telemetry
Storage: InfluxDB (time-series)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from abstract_agent import AgentId
from src.battery_twin.agents.battery_agent_types import BatteryReactiveAgent
from src.battery_twin.communication.mqtt_bridge import MqttBridge, MqttConfig
from src.battery_twin.communication.message_schemas import (
    MessageFactory,
    TelemetryMessage,
)
from src.battery_twin.data.telemetry_cleaner import TelemetryCleaner
from src.battery_twin.storage.battery_storage_manager import BatteryStorageManager

logger = logging.getLogger(__name__)


@dataclass
class TelemetryStats:
    """Statistics for telemetry ingestion."""
    messages_received: int = 0
    messages_validated: int = 0
    messages_rejected: int = 0
    outliers_detected: int = 0
    missing_data_count: int = 0

    # Validation failures by type
    voltage_failures: int = 0
    current_failures: int = 0
    temperature_failures: int = 0

    # Timing
    first_message_time: float = 0.0
    last_message_time: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'messages_received': self.messages_received,
            'messages_validated': self.messages_validated,
            'messages_rejected': self.messages_rejected,
            'outliers_detected': self.outliers_detected,
            'missing_data_count': self.missing_data_count,
            'voltage_failures': self.voltage_failures,
            'current_failures': self.current_failures,
            'temperature_failures': self.temperature_failures,
            'uptime_seconds': self.uptime if self.first_message_time > 0 else 0.0,
            'throughput_msg_per_sec': self.throughput
        }

    @property
    def uptime(self) -> float:
        """Get uptime in seconds."""
        if self.first_message_time == 0:
            return 0.0
        return self.last_message_time - self.first_message_time

    @property
    def throughput(self) -> float:
        """Get throughput in messages per second."""
        uptime = self.uptime
        if uptime > 0:
            return self.messages_received / uptime
        return 0.0


@dataclass
class ValidationRules:
    """Validation rules for telemetry data."""
    # Voltage range (V)
    min_voltage: float = 2.0
    max_voltage: float = 5.0

    # Current range (A)
    min_current: float = -5.0
    max_current: float = 5.0

    # Temperature range (°C)
    min_temperature: float = -10.0
    max_temperature: float = 60.0

    # Outlier detection (using simple z-score approach)
    outlier_window_size: int = 100
    outlier_threshold: float = 3.0  # Standard deviations

    # Missing data thresholds
    max_time_gap: float = 60.0  # seconds


class TelemetryIngestorAgent(BatteryReactiveAgent):
    """
    Reactive agent for telemetry ingestion and validation.

    Responsibilities:
    - Subscribe to raw telemetry from replay engine or sensors
    - Validate data ranges and detect outliers
    - Clean and normalize telemetry
    - Publish cleaned telemetry for downstream agents
    - Store telemetry in InfluxDB
    - Track ingestion statistics

    This is a Reactive agent because:
    - Fast, stimulus-response behavior
    - No deliberation or planning needed
    - Immediate processing of each telemetry message
    - Stateless validation rules
    """

    def __init__(
        self,
        agent_id: AgentId,
        mqtt_bridge: Optional[MqttBridge] = None,
        storage_manager: Optional[BatteryStorageManager] = None,
        mqtt_config: Optional[MqttConfig] = None,
        validation_rules: Optional[ValidationRules] = None,
        enable_outlier_detection: bool = True,
        enable_storage: bool = True,
        enable_heartbeat: bool = True
    ):
        """
        Initialize Telemetry Ingestor Agent.

        Args:
            agent_id: Agent identifier
            mqtt_bridge: MQTT bridge for communication
            storage_manager: Storage manager for persistence
            mqtt_config: MQTT configuration
            validation_rules: Validation rules for telemetry
            enable_outlier_detection: Enable outlier detection
            enable_storage: Enable storage to InfluxDB
            enable_heartbeat: Enable periodic heartbeat messages
        """
        # Initialize reactive agent
        super().__init__(
            agent_id=agent_id,
            observable_properties={"battery_voltage", "battery_current", "battery_temperature"},
            mqtt_bridge=mqtt_bridge,
            storage_manager=storage_manager,
            mqtt_config=mqtt_config,
            enable_heartbeat=enable_heartbeat
        )

        # Configuration
        self.validation_rules = validation_rules or ValidationRules()
        self.enable_outlier_detection = enable_outlier_detection
        self.enable_storage = enable_storage

        # Statistics
        self.stats = TelemetryStats()

        # Shared telemetry cleaner using hybrid twin sanitisation logic
        self.telemetry_cleaner = TelemetryCleaner(
            window_size=self.validation_rules.outlier_window_size
        )

        # Last seen timestamp per battery
        self.last_timestamp: Dict[str, float] = {}

        # Register action to process raw telemetry using topic manager pattern
        tm = getattr(getattr(self, "transport", None), "topic_manager", None)
        raw_pattern = (
            tm.get_subscription_pattern("raw_telemetry", battery_id=None)
            if tm
            else "battery/+/raw"
        )
        self.register_action(
            action_id="process_raw_telemetry",
            handler=self._on_raw_telemetry,
            topic_pattern=raw_pattern,
            description="Process raw telemetry from replay engine or sensors"
        )

        logger.info(f"TelemetryIngestorAgent initialized: {agent_id}")

    def _on_raw_telemetry(self, topic: str, payload: str):
        """
        Handle incoming raw telemetry message.

        Args:
            topic: MQTT topic (e.g., battery/B0005/raw)
            payload: JSON payload
        """
        try:
            # Update statistics
            self.stats.messages_received += 1
            current_time = time.time()

            if self.stats.first_message_time == 0:
                self.stats.first_message_time = current_time
            self.stats.last_message_time = current_time

            # Parse message
            telemetry = TelemetryMessage.model_validate_json(payload)

            # Clean telemetry using hybrid sanitisation utilities
            cleaned_telemetry, adjustments = self._clean_telemetry(telemetry)

            # Validate telemetry post-cleaning
            is_valid, validation_errors = self._validate_telemetry(
                cleaned_telemetry
            )

            if not is_valid:
                self.stats.messages_rejected += 1
                logger.warning(
                    f"Rejected telemetry for {cleaned_telemetry.battery_id}, "
                    f"cycle {cleaned_telemetry.cycle}: {validation_errors}"
                )
                return

            if self.enable_outlier_detection and adjustments:
                self.stats.outliers_detected += 1
                logger.debug(
                    "Telemetry adjustments detected for %s: %s",
                    cleaned_telemetry.battery_id,
                    adjustments,
                )

            # Check for missing data (time gaps)
            self._check_missing_data(cleaned_telemetry)

            # Telemetry is valid
            self.stats.messages_validated += 1

            # Publish cleaned telemetry
            success = self.publish_message(
                "clean_telemetry",
                cleaned_telemetry,
                battery_id=cleaned_telemetry.battery_id
            )

            if not success:
                logger.error(
                    "Failed to publish cleaned telemetry for %s",
                    cleaned_telemetry.battery_id,
                )

            # Store in InfluxDB
            if self.enable_storage and self.storage_manager:
                self._store_telemetry(cleaned_telemetry)

        except Exception as e:
            logger.error(f"Error processing raw telemetry: {e}")
            self.stats.messages_rejected += 1

    def _clean_telemetry(
        self, telemetry: TelemetryMessage
    ) -> tuple[TelemetryMessage, Dict[str, float]]:
        """
        Clean telemetry using the shared hybrid data loader sanitisation.
        Returns the cleaned message and a dict of adjusted fields.
        """
        sample: Dict[str, Optional[float]] = {
            "voltage": telemetry.voltage,
            "current": telemetry.current,
            "temperature": telemetry.temperature,
        }
        if telemetry.ambient_temperature is not None:
            sample["ambient_temperature"] = telemetry.ambient_temperature

        cleaned_values, adjustments = self.telemetry_cleaner.clean_sample(
            telemetry.battery_id, sample
        )

        update_payload = {
            field: cleaned_values.get(field, getattr(telemetry, field))
            for field in ["voltage", "current", "temperature", "ambient_temperature"]
            if field in cleaned_values
        }

        cleaned_message = telemetry.model_copy(update=update_payload)
        return cleaned_message, adjustments

    def _validate_telemetry(self, telemetry: TelemetryMessage) -> tuple[bool, List[str]]:
        """
        Validate telemetry data.

        Args:
            telemetry: Telemetry message

        Returns:
            (is_valid, list of validation errors)
        """
        errors = []

        # Validate voltage
        if not (self.validation_rules.min_voltage <= telemetry.voltage <= self.validation_rules.max_voltage):
            errors.append(
                f"Voltage {telemetry.voltage}V outside range "
                f"[{self.validation_rules.min_voltage}, {self.validation_rules.max_voltage}]"
            )
            self.stats.voltage_failures += 1

        # Validate current
        if not (self.validation_rules.min_current <= telemetry.current <= self.validation_rules.max_current):
            errors.append(
                f"Current {telemetry.current}A outside range "
                f"[{self.validation_rules.min_current}, {self.validation_rules.max_current}]"
            )
            self.stats.current_failures += 1

        # Validate temperature
        if not (self.validation_rules.min_temperature <= telemetry.temperature <= self.validation_rules.max_temperature):
            errors.append(
                f"Temperature {telemetry.temperature}°C outside range "
                f"[{self.validation_rules.min_temperature}, {self.validation_rules.max_temperature}]"
            )
            self.stats.temperature_failures += 1

        # Validate timestamp
        if telemetry.timestamp <= 0:
            errors.append(f"Invalid timestamp: {telemetry.timestamp}")

        # Validate cycle
        if telemetry.cycle < 0:
            errors.append(f"Invalid cycle number: {telemetry.cycle}")

        return (len(errors) == 0, errors)

    def _check_missing_data(self, telemetry: TelemetryMessage):
        """
        Check for missing data (time gaps).

        Args:
            telemetry: Telemetry message
        """
        battery_id = telemetry.battery_id

        if battery_id in self.last_timestamp:
            time_gap = telemetry.timestamp - self.last_timestamp[battery_id]

            if time_gap > self.validation_rules.max_time_gap:
                self.stats.missing_data_count += 1
                logger.warning(
                    f"Missing data detected for {battery_id}: "
                    f"{time_gap:.1f}s gap (threshold: {self.validation_rules.max_time_gap}s)"
                )

        self.last_timestamp[battery_id] = telemetry.timestamp

    def _store_telemetry(self, telemetry: TelemetryMessage):
        """
        Store telemetry to InfluxDB.

        Args:
            telemetry: Telemetry message
        """
        try:
            self.persist_to_storage(
                operation="telemetry",
                battery_id=telemetry.battery_id,
                timestamp=telemetry.timestamp,
                cycle=telemetry.cycle,
                voltage=telemetry.voltage,
                current=telemetry.current,
                temperature=telemetry.temperature,
                ambient_temperature=telemetry.ambient_temperature
            )
        except Exception as e:
            logger.error(f"Failed to store telemetry: {e}")

    def get_stats(self) -> TelemetryStats:
        """Get telemetry ingestion statistics."""
        return self.stats

    def get_stats_dict(self) -> Dict:
        """Get statistics as dictionary."""
        return self.stats.to_dict()

    def reset_stats(self):
        """Reset statistics."""
        self.stats = TelemetryStats()
        self.telemetry_cleaner = TelemetryCleaner(
            window_size=self.validation_rules.outlier_window_size
        )
        self.last_timestamp.clear()

    def _agent_setup(self) -> bool:
        """Agent-specific setup."""
        logger.info(f"TelemetryIngestorAgent {self.agent_id} ready to process telemetry")
        return True

    def _agent_teardown(self):
        """Agent-specific teardown."""
        logger.info(f"TelemetryIngestorAgent {self.agent_id} shutting down")
        logger.info(f"Final statistics: {self.get_stats_dict()}")


__all__ = [
    'TelemetryIngestorAgent',
    'TelemetryStats',
    'ValidationRules',
]
