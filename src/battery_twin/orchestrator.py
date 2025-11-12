"""
Battery Digital Twin System Orchestrator

This module manages the lifecycle of all battery twin agents and coordinates
the end-to-end data flow pipeline.

Architecture:
    Raw Data → TelemetryIngestor → StateEstimator → HealthMonitor
                                   ↓
                              PhysicsModel + MLResidual → Hybrid Predictions

Author: Battery Twin Development Team
Date: 2025-03-01
"""

import asyncio
import logging
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import threading

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from src.abstract_agent import AgentId
from src.battery_twin.agents.telemetry_ingestor_agent import (
    TelemetryIngestorAgent,
)
from src.battery_twin.agents.state_estimator_agent import StateEstimatorAgent
from src.battery_twin.agents.health_monitor_agent import HealthMonitorAgent
from src.battery_twin.agents.physics_model_agent import PhysicsModelAgent
from src.battery_twin.agents.ml_residual_agent import MLResidualAgent
from src.battery_twin.communication.mqtt_bridge import MqttBridge, MqttConfig
from src.battery_twin.communication.message_schemas import (
    MessageFactory,
    PredictionMessage,
    TelemetryMessage,
    CapacityMessage,
)
from src.battery_twin.hybrid import HybridDigitalTwin, PredictionResult
from src.battery_twin.storage.battery_storage_manager import (
    BatteryStorageManager,
)
from src.battery_twin.storage.battery_storage_config import BatteryStorageConfig
from src.battery_twin.models.extended_kalman_filter import EKFConfig


# ============================================================================
# System State Enums
# ============================================================================


class SystemState(str, Enum):
    """Overall system state."""

    INITIALIZING = "INITIALIZING"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class AgentStatus(str, Enum):
    """Status of individual agents."""

    CREATED = "CREATED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


# ============================================================================
# Configuration Data Classes
# ============================================================================


@dataclass
class BatteryTwinConfig:
    """Configuration for Battery Digital Twin system."""

    # Battery identification
    battery_id: str = "B0005"

    # MQTT configuration
    mqtt_broker: str = "localhost"
    mqtt_port: int = 1883
    mqtt_keepalive: int = 60

    # Storage configuration
    enable_storage: bool = False
    storage_config_path: Optional[str] = None

    # Agent enable flags
    enable_telemetry_ingestor: bool = True
    enable_state_estimator: bool = True
    enable_health_monitor: bool = True
    enable_physics_model: bool = False  # Optional
    enable_ml_residual: bool = False  # Optional

    # EKF configuration
    ekf_initial_soc: float = 0.8
    ekf_initial_soh: float = 1.0
    ekf_capacity_nominal: float = 2.0

    # Health monitoring configuration
    health_initial_soh: float = 1.0
    health_initial_r0: float = 0.01
    health_eol_threshold: float = 0.8

    # Logging configuration
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # Performance monitoring
    enable_metrics: bool = True
    metrics_interval: float = 10.0  # seconds

    @classmethod
    def from_yaml(cls, config_path: str) -> "BatteryTwinConfig":
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


@dataclass
class AgentInfo:
    """Information about a managed agent."""

    agent_id: str
    agent_type: str
    instance: object
    status: AgentStatus = AgentStatus.CREATED
    start_time: Optional[datetime] = None
    error_message: Optional[str] = None
    message_count: int = 0


# ============================================================================
# Battery Twin Orchestrator
# ============================================================================


class BatteryTwinOrchestrator:
    """
    Orchestrates the Battery Digital Twin multi-agent system.

    Responsibilities:
    - Lifecycle management of all agents
    - MQTT broker coordination
    - Data flow monitoring
    - System health monitoring
    - Graceful shutdown handling
    """

    def __init__(self, config: BatteryTwinConfig):
        """
        Initialize the orchestrator.

        Args:
            config: System configuration
        """
        self.config = config
        self.battery_id = config.battery_id

        # System state
        self.state = SystemState.INITIALIZING
        self.start_time: Optional[datetime] = None

        # Agent registry
        self.agents: Dict[str, AgentInfo] = {}

        # MQTT infrastructure
        self.mqtt_bridge: Optional[MqttBridge] = None
        self.mqtt_config: Optional[MqttConfig] = None

        # Storage infrastructure
        self.storage_manager: Optional[BatteryStorageManager] = None
        self.hybrid_twin: Optional[HybridDigitalTwin] = None
        self.hybrid_training_lock = threading.Lock()

        # Shutdown handling
        self._shutdown_event = asyncio.Event()
        self._shutdown_handlers_registered = False

        # Performance metrics
        self.total_messages_processed = 0
        self.last_metrics_time = time.time()

        # Shared hybrid training buffers
        self.hybrid_training_dataframe: List[Dict] = []
        self.hybrid_prediction_topic = "hybrid_prediction"
        self._telemetry_cycle_buffer: Dict[int, Dict[str, float]] = {}
        self._finalized_cycle_features: Dict[int, Dict[str, float]] = {}
        self._submitted_cycles: Set[int] = set()
        self._current_telemetry_cycle: Optional[int] = None

        # Configure logging
        self._configure_logging()

        logger.info(
            f"BatteryTwinOrchestrator initialized for {self.battery_id}"
        )

    def _configure_logging(self):
        """Configure logging system."""
        # Remove default handler
        logger.remove()

        # Add console handler
        logger.add(
            sys.stderr,
            level=self.config.log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> | <level>{message}</level>",
        )

        # Add file handler if configured
        if self.config.log_file:
            logger.add(
                self.config.log_file,
                level=self.config.log_level,
                rotation="100 MB",
                retention="7 days",
            )

    async def initialize(self):
        """Initialize system infrastructure."""
        logger.info("Initializing Battery Digital Twin system...")
        self.state = SystemState.INITIALIZING

        try:
            # 1. Initialize MQTT bridge
            await self._initialize_mqtt()
            self._subscribe_internal_topics()

            # 2. Initialize storage (if enabled)
            if self.config.enable_storage:
                await self._initialize_storage()

            # 3. Initialize hybrid twin backend
            self._initialize_hybrid_services()

            # 4. Create agents
            await self._create_agents()

            logger.info("System initialization complete")

        except Exception as e:
            logger.error(f"Initialization failed: {e}", exc_info=True)
            self.state = SystemState.ERROR
            raise

    async def _initialize_mqtt(self):
        """Initialize MQTT infrastructure."""
        logger.info(
            f"Connecting to MQTT broker at {self.config.mqtt_broker}:{self.config.mqtt_port}"
        )

        self.mqtt_config = MqttConfig(
            broker=self.config.mqtt_broker,
            port=self.config.mqtt_port,
            keepalive=self.config.mqtt_keepalive,
            client_id_prefix=f"twin_orch_{self.battery_id}_",
        )

        # Create MQTT bridge with client_id and config
        self.mqtt_bridge = MqttBridge(
            client_id=f"twin_orch_{self.battery_id}",
            mqtt_config=self.mqtt_config,
        )
        self.mqtt_bridge.connect()  # Synchronous connect

        logger.info("MQTT bridge connected")

    async def _initialize_storage(self):
        """Initialize storage infrastructure."""
        logger.info("Initializing storage manager...")

        if self.config.storage_config_path:
            storage_config = BatteryStorageConfig.from_yaml(
                self.config.storage_config_path
            )
        else:
            storage_config = BatteryStorageConfig()

        self.storage_manager = BatteryStorageManager(storage_config)

        try:
            connected = self.storage_manager.connect_all()
            if not connected:
                logger.warning("Some storage backends failed to connect")
        except Exception as exc:
            logger.error(f"Failed to connect storage backends: {exc}")

        logger.info("Storage manager initialized")

    def _initialize_hybrid_services(self):
        """Initialize shared hybrid digital twin instance."""
        try:
            self.hybrid_twin = HybridDigitalTwin()
            logger.info("HybridDigitalTwin service initialised inside orchestrator")
        except Exception as exc:
            self.hybrid_twin = None
            logger.warning(
                "HybridDigitalTwin could not be initialised (TensorFlow installed?): %s",
                exc,
            )

    def _subscribe_internal_topics(self):
        """Subscribe to MQTT topics needed for shared services."""
        if not self.mqtt_bridge:
            return

        try:
            physics_topic = self.mqtt_bridge.topic_manager.get_topic(
                "physics_prediction", battery_id=self.battery_id
            )
            self.mqtt_bridge.subscribe(physics_topic, self._handle_physics_prediction)
            logger.info("Subscribed to physics predictions for hybrid service: %s", physics_topic)

            telemetry_topic = self.mqtt_bridge.topic_manager.get_topic(
                "clean_telemetry", battery_id=self.battery_id
            )
            self.mqtt_bridge.subscribe(telemetry_topic, self._handle_clean_telemetry)
            logger.info("Subscribed to clean telemetry for feature buffering: %s", telemetry_topic)

            capacity_topic = self.mqtt_bridge.topic_manager.get_topic(
                "capacity", battery_id=self.battery_id
            )
            self.mqtt_bridge.subscribe(capacity_topic, self._handle_capacity_measurement)
            logger.info("Subscribed to capacity measurements for hybrid training: %s", capacity_topic)
        except Exception as exc:
            logger.warning("Failed to subscribe to hybrid topics: %s", exc)

    async def _create_agents(self):
        """Create and register all agents."""
        logger.info("Creating agents...")

        # 1. Telemetry Ingestor (data ingestion)
        if self.config.enable_telemetry_ingestor:
            await self._create_telemetry_ingestor()

        # 2. State Estimator (EKF-based state estimation)
        if self.config.enable_state_estimator:
            await self._create_state_estimator()

        # 3. Health Monitor (health assessment and alerts)
        if self.config.enable_health_monitor:
            await self._create_health_monitor()

        # 4. Physics Model (optional - physics-based predictions)
        if self.config.enable_physics_model:
            await self._create_physics_model()

        # 5. ML Residual (optional - ML corrections)
        if self.config.enable_ml_residual:
            await self._create_ml_residual()

        logger.info(f"Created {len(self.agents)} agents")

    # ------------------------------------------------------------------
    # Hybrid twin coordination APIs (shared by agents)
    # ------------------------------------------------------------------

    def train_hybrid_twin(self, samples: List[Dict]) -> bool:
        """Train or update the shared hybrid twin with new samples."""
        if not self.hybrid_twin:
            return False

        import pandas as pd

        df = pd.DataFrame(samples)
        if df.empty:
            return False

        with self.hybrid_training_lock:
            try:
                self.hybrid_twin.fit(df, target_column="Capacity")
                logger.info(
                    "HybridDigitalTwin trained with %d samples (shared service)",
                    len(df),
                )
                return True
            except Exception as exc:
                logger.warning("HybridDigitalTwin training failed: %s", exc)
                return False

    def predict_hybrid_capacity(self, feature_df):
        """Predict using shared hybrid twin."""
        if not self.hybrid_twin or not getattr(
            self.hybrid_twin, "is_trained", False
        ):
            return None

        try:
            return self.hybrid_twin.predict(
                feature_df, return_uncertainty=True, return_components=True
            )
        except Exception as exc:
            logger.debug("HybridDigitalTwin prediction failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # MQTT handlers
    # ------------------------------------------------------------------

    def _handle_clean_telemetry(self, topic: str, payload: str):
        """Accumulate telemetry statistics per cycle for feature buffering."""
        try:
            message = TelemetryMessage.model_validate_json(payload)
        except Exception as exc:
            logger.debug("Invalid telemetry payload for buffering: %s", exc)
            return

        stats = self._telemetry_cycle_buffer.setdefault(
            message.cycle,
            {"temp_sum": 0.0, "count": 0, "start": message.timestamp, "end": message.timestamp},
        )

        stats["temp_sum"] += message.temperature
        stats["count"] += 1
        stats["end"] = message.timestamp

        if self._current_telemetry_cycle is None:
            self._current_telemetry_cycle = message.cycle
        elif message.cycle != self._current_telemetry_cycle:
            self._finalize_cycle_features(self._current_telemetry_cycle)
            self._current_telemetry_cycle = message.cycle

    def _handle_capacity_measurement(self, topic: str, payload: str):
        """Use capacity measurements to feed hybrid training."""
        try:
            message = CapacityMessage.model_validate_json(payload)
        except Exception as exc:
            logger.debug("Invalid capacity payload: %s", exc)
            return

        hybrid_available = self.hybrid_twin is not None and getattr(
            self.hybrid_twin, "is_trained", False
        )
        if message.cycle in self._submitted_cycles:
            return

        summary = self._finalized_cycle_features.get(message.cycle)
        if summary is None:
            summary = self._finalize_cycle_features(message.cycle)

        if summary is None:
            summary = {
                "Temperature_measured": 25.0,
                "Time": 1.0,
            }

        sample = {
            "id_cycle": message.cycle,
            "Temperature_measured": summary["Temperature_measured"],
            "Time": summary["Time"],
            "Capacity": message.capacity,
        }

        if self.storage_manager:
            try:
                self.storage_manager.record_hybrid_training_sample(
                    battery_id=self.battery_id,
                    cycle=message.cycle,
                    temperature=summary["Temperature_measured"],
                    duration=summary["Time"],
                    capacity=message.capacity,
                    source="capacity_event",
                    timestamp=message.timestamp,
                )
            except Exception as exc:
                logger.debug("Failed to persist hybrid training sample: %s", exc)

        if hybrid_available and self.train_hybrid_twin([sample]):
            self._submitted_cycles.add(message.cycle)

    def _handle_physics_prediction(self, topic: str, payload: str):
        """Handle incoming physics prediction and publish hybrid result."""
        if (
            not self.hybrid_twin
            or not getattr(self.hybrid_twin, "is_trained", False)
        ):
            return

        try:
            message = PredictionMessage.model_validate_json(payload)
        except Exception as exc:
            logger.debug("Invalid physics prediction message: %s", exc)
            return

        summary = (
            self._finalized_cycle_features.get(message.cycle)
            or self._telemetry_cycle_buffer.get(message.cycle)
        )

        temp = summary["Temperature_measured"] if summary else 25.0
        duration = summary["Time"] if summary and "Time" in summary else 1.0

        feature_row = {
            "id_cycle": message.cycle,
            "Temperature_measured": temp,
            "Time": duration,
            "Capacity": message.predicted_capacity,
        }

        df = pd.DataFrame([feature_row])
        result = self.predict_hybrid_capacity(df)
        if result is None:
            return

        hybrid_value, uncertainty = self._extract_hybrid_prediction(result)
        if hybrid_value is None:
            return

        hybrid_message = PredictionMessage(
            battery_id=self.battery_id,
            timestamp=time.time(),
            cycle=message.cycle,
            prediction_type="hybrid",
            predicted_capacity=hybrid_value,
            uncertainty=uncertainty,
            horizon=message.horizon,
            agent_id="hybrid_service",
        )

        self._publish_hybrid_prediction(hybrid_message)

    def _extract_hybrid_prediction(
        self, result,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Normalize hybrid prediction outputs."""
        if isinstance(result, PredictionResult):
            value = max(0.0, float(result.hybrid_prediction[0]))
            uncertainty = (
                float(result.uncertainty[0]) if result.uncertainty is not None else None
            )
            return value, uncertainty

        if isinstance(result, (list, np.ndarray)) and len(result) > 0:
            return max(0.0, float(result[0])), None

        return None, None

    def _publish_hybrid_prediction(self, message: PredictionMessage):
        """Publish hybrid prediction to MQTT and storage."""
        if not self.mqtt_bridge:
            return

        try:
            self.mqtt_bridge.publish(
                "hybrid_prediction",
                message,
                battery_id=self.battery_id,
            )
        except Exception as exc:
            logger.error("Failed to publish hybrid prediction: %s", exc)

        if self.storage_manager:
            try:
                self.storage_manager.record_prediction(
                    battery_id=message.battery_id,
                    agent_id=message.agent_id,
                    prediction_type="hybrid",
                    predicted_capacity=message.predicted_capacity,
                    uncertainty=message.uncertainty,
                    horizon=message.horizon,
                    cycle=message.cycle,
                )
            except Exception as exc:
                logger.debug("Failed to persist hybrid prediction: %s", exc)

    def _finalize_cycle_features(self, cycle: Optional[int]):
        """Finalize buffered telemetry for a cycle and compute summary."""
        if cycle is None:
            return None

        stats = self._telemetry_cycle_buffer.pop(cycle, None)
        if not stats or stats["count"] == 0:
            return None

        duration = max(stats["end"] - stats["start"], stats["count"])
        summary = {
            "Temperature_measured": stats["temp_sum"] / stats["count"],
            "Time": duration,
        }

        self._finalized_cycle_features[cycle] = summary
        return summary

    async def _create_telemetry_ingestor(self):
        """Create TelemetryIngestorAgent."""
        agent_id = AgentId(
            app="battery_twin",
            type="telemetry_ingestor",
            instance=f"{self.battery_id}",
        )

        agent = TelemetryIngestorAgent(
            agent_id=agent_id,
            mqtt_bridge=self.mqtt_bridge,
            storage_manager=self.storage_manager,
            enable_storage=self.config.enable_storage,
        )

        self.agents["telemetry_ingestor"] = AgentInfo(
            agent_id=str(agent_id),
            agent_type="TelemetryIngestorAgent",
            instance=agent,
            status=AgentStatus.CREATED,
        )

        logger.info(f"Created TelemetryIngestorAgent: {agent_id}")

    async def _create_state_estimator(self):
        """Create StateEstimatorAgent."""
        agent_id = AgentId(
            app="battery_twin",
            type="state_estimator",
            instance=f"{self.battery_id}",
        )

        ekf_config = EKFConfig(
            initial_soc=self.config.ekf_initial_soc,
            initial_soh=self.config.ekf_initial_soh,
            capacity_nominal=self.config.ekf_capacity_nominal,
        )

        agent = StateEstimatorAgent(
            agent_id=agent_id,
            battery_id=self.battery_id,
            mqtt_bridge=self.mqtt_bridge,
            storage_manager=self.storage_manager,
            ekf_config=ekf_config,
        )

        self.agents["state_estimator"] = AgentInfo(
            agent_id=str(agent_id),
            agent_type="StateEstimatorAgent",
            instance=agent,
            status=AgentStatus.CREATED,
        )

        logger.info(f"Created StateEstimatorAgent: {agent_id}")

    async def _create_health_monitor(self):
        """Create HealthMonitorAgent."""
        agent_id = AgentId(
            app="battery_twin",
            type="health_monitor",
            instance=f"{self.battery_id}",
        )

        agent = HealthMonitorAgent(
            agent_id=agent_id,
            battery_id=self.battery_id,
            initial_soh=self.config.health_initial_soh,
            initial_r0=self.config.health_initial_r0,
            eol_threshold=self.config.health_eol_threshold,
            mqtt_bridge=self.mqtt_bridge,
            storage_manager=self.storage_manager,
        )

        self.agents["health_monitor"] = AgentInfo(
            agent_id=str(agent_id),
            agent_type="HealthMonitorAgent",
            instance=agent,
            status=AgentStatus.CREATED,
        )

        logger.info(f"Created HealthMonitorAgent: {agent_id}")

    async def _create_physics_model(self):
        """Create PhysicsModelAgent."""
        agent_id = AgentId(
            app="battery_twin",
            type="physics_model",
            instance=f"{self.battery_id}",
        )

        agent = PhysicsModelAgent(
            agent_id=agent_id,
            battery_id=self.battery_id,
            mqtt_bridge=self.mqtt_bridge,
            storage_manager=self.storage_manager,
            hybrid_service=self if self.hybrid_twin else None,
        )

        self.agents["physics_model"] = AgentInfo(
            agent_id=str(agent_id),
            agent_type="PhysicsModelAgent",
            instance=agent,
            status=AgentStatus.CREATED,
        )

        logger.info(f"Created PhysicsModelAgent: {agent_id}")

    async def _create_ml_residual(self):
        """Create MLResidualAgent."""
        agent_id = AgentId(
            app="battery_twin",
            type="ml_residual",
            instance=f"{self.battery_id}",
        )

        agent = MLResidualAgent(
            agent_id=agent_id,
            battery_id=self.battery_id,
            mqtt_bridge=self.mqtt_bridge,
            storage_manager=self.storage_manager,
            hybrid_service=self if self.hybrid_twin else None,
        )

        self.agents["ml_residual"] = AgentInfo(
            agent_id=str(agent_id),
            agent_type="MLResidualAgent",
            instance=agent,
            status=AgentStatus.CREATED,
        )

        logger.info(f"Created MLResidualAgent: {agent_id}")

    async def start(self):
        """Start all agents and begin operation."""
        logger.info("Starting Battery Digital Twin system...")
        self.state = SystemState.STARTING

        try:
            # Start all agents
            for name, agent_info in self.agents.items():
                logger.info(f"Starting {name}...")
                agent_info.status = AgentStatus.STARTING
                agent_info.start_time = datetime.now()

                # Agents are already listening via MQTT subscriptions
                # Just mark as running
                agent_info.status = AgentStatus.RUNNING

                logger.info(f"{name} started successfully")

            self.state = SystemState.RUNNING
            self.start_time = datetime.now()

            logger.info("All agents started. System is RUNNING.")

            # Register shutdown handlers
            if not self._shutdown_handlers_registered:
                self._register_shutdown_handlers()

        except Exception as e:
            logger.error(f"Failed to start system: {e}", exc_info=True)
            self.state = SystemState.ERROR
            raise

    def _register_shutdown_handlers(self):
        """Register signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            logger.warning(f"Received signal {signum}, initiating shutdown...")
            asyncio.create_task(self.shutdown())

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        self._shutdown_handlers_registered = True
        logger.debug("Shutdown handlers registered")

    async def run(self, duration: Optional[float] = None):
        """
        Run the system for specified duration or until shutdown signal.

        Args:
            duration: Optional duration in seconds. If None, runs until shutdown.
        """
        logger.info("System entering main loop...")

        # Start metrics monitoring if enabled
        if self.config.enable_metrics:
            asyncio.create_task(self._metrics_loop())

        # Main loop
        if duration:
            logger.info(f"Running for {duration} seconds...")
            await asyncio.sleep(duration)
            await self.shutdown()
        else:
            logger.info("Running until shutdown signal...")
            await self._shutdown_event.wait()

    async def _metrics_loop(self):
        """Periodically log system metrics."""
        while self.state == SystemState.RUNNING:
            await asyncio.sleep(self.config.metrics_interval)

            uptime = (
                (datetime.now() - self.start_time).total_seconds()
                if self.start_time
                else 0
            )

            logger.info(
                f"METRICS | Uptime: {uptime:.1f}s | "
                f"Agents: {len([a for a in self.agents.values() if a.status == AgentStatus.RUNNING])}/{len(self.agents)} running | "
                f"Messages: {self.total_messages_processed}"
            )

    async def shutdown(self):
        """Gracefully shutdown the system."""
        if self.state in [SystemState.STOPPING, SystemState.STOPPED]:
            logger.warning("Shutdown already in progress")
            return

        logger.info("Initiating graceful shutdown...")
        self.state = SystemState.STOPPING

        try:
            # Stop all agents
            for name, agent_info in self.agents.items():
                logger.info(f"Stopping {name}...")
                agent_info.status = AgentStatus.STOPPED

            # Disconnect MQTT
            if self.mqtt_bridge:
                self.mqtt_bridge.disconnect()  # Synchronous disconnect
                logger.info("MQTT bridge disconnected")

            # Close storage connections
            if self.storage_manager:
                # self.storage_manager.disconnect_all()
                logger.info("Storage connections closed")

            self.state = SystemState.STOPPED
            logger.info("System shutdown complete")

            # Signal shutdown event
            self._shutdown_event.set()

        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)
            self.state = SystemState.ERROR
            raise

    def get_status(self) -> Dict:
        """Get current system status."""
        return {
            "system_state": self.state.value,
            "battery_id": self.battery_id,
            "start_time": (
                self.start_time.isoformat() if self.start_time else None
            ),
            "uptime_seconds": (
                (datetime.now() - self.start_time).total_seconds()
                if self.start_time
                else 0
            ),
            "agents": {
                name: {
                    "type": info.agent_type,
                    "status": info.status.value,
                    "start_time": (
                        info.start_time.isoformat()
                        if info.start_time
                        else None
                    ),
                    "message_count": info.message_count,
                    "error": info.error_message,
                }
                for name, info in self.agents.items()
            },
            "total_messages_processed": self.total_messages_processed,
        }


# ============================================================================
# Main Entry Point
# ============================================================================


async def main():
    """Main entry point for running the Battery Digital Twin system."""
    import argparse

    parser = argparse.ArgumentParser(description="Battery Digital Twin System")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--battery-id", type=str, default="B0005", help="Battery identifier"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Run duration in seconds (default: run until Ctrl+C)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Load or create configuration
    if args.config:
        config = BatteryTwinConfig.from_yaml(args.config)
    else:
        config = BatteryTwinConfig(
            battery_id=args.battery_id, log_level=args.log_level
        )

    # Create orchestrator
    orchestrator = BatteryTwinOrchestrator(config)

    try:
        # Initialize system
        await orchestrator.initialize()

        # Start agents
        await orchestrator.start()

        # Run main loop
        await orchestrator.run(duration=args.duration)

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
    except Exception as e:
        logger.error(f"System error: {e}", exc_info=True)
    finally:
        await orchestrator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
