"""
BatteryAgentBase

Battery-specific base agent built on top of the twin-agnostic CommAgentBase.
Provides:
- MQTT communication via CommAgentBase
- Storage manager integration (Influx/Mongo/etc.)
- Battery heartbeat/status and metrics helpers

All battery twin agents (BDI, Reactive, Hybrid) should inherit from this.
"""

from __future__ import annotations

import logging
import time
import threading
from enum import Enum
from typing import Any, Dict, Optional

from src.communication.agent_base import CommAgentBase, PerformanceMetrics
from src.battery_twin.communication.mqtt_bridge import MqttBridge, MqttConfig
from src.battery_twin.storage.battery_storage_manager import (
    BatteryStorageManager,
)

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    CREATED = "created"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class BatteryAgentBase(CommAgentBase):
    def __init__(
        self,
        mqtt_bridge: Optional[MqttBridge] = None,
        storage_manager: Optional[BatteryStorageManager] = None,
        mqtt_config: Optional[MqttConfig] = None,
        enable_heartbeat: bool = True,
        heartbeat_interval: float = 30.0,
    ):
        # Initialize comm base
        super().__init__(
            mqtt_bridge=mqtt_bridge,
            mqtt_config=mqtt_config or MqttConfig(),
            enable_heartbeat=enable_heartbeat,
            heartbeat_interval=heartbeat_interval,
        )

        # Battery-specific storage
        self.storage_manager = storage_manager

        # Status tracking
        self.status = AgentStatus.CREATED
        self.status_lock = threading.Lock()

    # ---- Lifecycle ----
    def setup(self) -> bool:
        try:
            with self.status_lock:
                self.status = AgentStatus.INITIALIZING

            logger.info(f"Setting up battery agent {self.get_agent_id()}...")

            if not super().setup():
                with self.status_lock:
                    self.status = AgentStatus.ERROR
                return False

            with self.status_lock:
                self.status = AgentStatus.READY
            return True

        except Exception as e:
            logger.error(f"BatteryAgentBase setup failed: {e}")
            with self.status_lock:
                self.status = AgentStatus.ERROR
            return False

    def teardown(self):
        try:
            logger.info(f"Tearing down battery agent {self.get_agent_id()}...")
            with self.status_lock:
                self.status = AgentStatus.STOPPING

            super().teardown()

            with self.status_lock:
                self.status = AgentStatus.STOPPED
        except Exception as e:
            logger.error(f"BatteryAgentBase teardown failed: {e}")

    # ---- Storage ----
    def persist_to_storage(self, operation: str, **kwargs):
        if not self.storage_manager:
            logger.debug("No storage manager configured")
            return
        try:
            if operation == "telemetry":
                self.storage_manager.record_telemetry(**kwargs)
            elif operation == "prediction":
                self.storage_manager.record_prediction(**kwargs)
            elif operation == "state_estimate":
                self.storage_manager.record_state_estimate(**kwargs)
            elif operation == "fault":
                self.storage_manager.record_fault_event(**kwargs)
            elif operation == "parameters":
                self.storage_manager.record_parameters(**kwargs)
            else:
                logger.warning(f"Unknown storage operation: {operation}")
        except Exception as e:
            logger.error(f"Storage operation {operation} failed: {e}")

    # ---- Heartbeat ----
    def _heartbeat_loop(self):
        # Override CommAgentBase to include battery status in heartbeat
        while self.heartbeat_running:
            try:
                from src.battery_twin.communication.message_schemas import (
                    AgentHeartbeatMessage,
                )

                with self.status_lock:
                    status_val = self.status.value

                heartbeat = AgentHeartbeatMessage(
                    agent_id=str(self.get_agent_id()),
                    timestamp=time.time(),
                    status=status_val,
                    uptime=self.metrics.uptime,
                )
                if hasattr(self, "transport") and self.transport and self.transport.is_connected():
                    self.transport.publish(
                        "agent_heartbeat",
                        heartbeat,
                        agent_id=str(self.get_agent_id()),
                    )
                time.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                time.sleep(5.0)

    # ---- Metrics / Status ----
    def get_metrics(self) -> PerformanceMetrics:
        return super().get_metrics()

    def get_metrics_dict(self) -> Dict:
        return super().get_metrics_dict()

    def get_status(self) -> AgentStatus:
        with self.status_lock:
            return self.status

    def is_ready(self) -> bool:
        with self.status_lock:
            return self.status in [AgentStatus.READY, AgentStatus.RUNNING]


__all__ = ["BatteryAgentBase", "AgentStatus", "PerformanceMetrics"]
