"""
CommAgentBase

Twin-agnostic communication base agent that provides:
- MQTT bridge integration (publish/subscribe)
- Action registry with topic routing
- Lifecycle management (setup/teardown)
- Optional heartbeat and basic metrics

This class is intentionally generic so different digital twins can reuse the
same communication behavior by injecting an MQTT bridge configured with their
own topic definitions.
"""

from __future__ import annotations

import logging
import time
import threading
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ActionHandler:
    action_id: str
    handler: Callable
    topic_pattern: Optional[str] = None
    description: str = ""
    enabled: bool = True


@dataclass
class PerformanceMetrics:
    messages_received: int = 0
    messages_sent: int = 0
    actions_executed: int = 0
    errors_count: int = 0
    last_action_time: float = 0.0
    total_processing_time: float = 0.0
    uptime_start: float = field(default_factory=time.time)

    @property
    def uptime(self) -> float:
        return time.time() - self.uptime_start

    @property
    def avg_processing_time(self) -> float:
        if self.actions_executed == 0:
            return 0.0
        return (self.total_processing_time / self.actions_executed) * 1000

    def to_dict(self) -> Dict:
        return {
            "messages_received": self.messages_received,
            "messages_sent": self.messages_sent,
            "actions_executed": self.actions_executed,
            "errors_count": self.errors_count,
            "uptime_seconds": self.uptime,
            "avg_processing_time_ms": self.avg_processing_time,
        }


class CommAgentBase:
    """
    Communication base class (twin agnostic).

    Exposes publish/subscribe and action routing on top of an injected MQTT bridge.
    Concrete twins can subclass this and add storage, schemas, and domain
    specifics without re-implementing comms.
    """

    def __init__(
        self,
        mqtt_bridge: Optional[Any] = None,
        mqtt_config: Optional[Any] = None,
        enable_heartbeat: bool = True,
        heartbeat_interval: float = 30.0,
        transport: Optional[Any] = None,
    ):
        # Transport/bridge (duck-typed). Prefer provided transport, else create/wrap bridge
        self.transport = transport
        self.mqtt_bridge = mqtt_bridge  # kept for backward compatibility
        self.mqtt_config = mqtt_config
        self.own_mqtt = mqtt_bridge is None and transport is None

        # Action registry
        self.action_handlers: Dict[str, ActionHandler] = {}
        self.action_lock = threading.Lock()

        # Metrics
        self.metrics = PerformanceMetrics()

        # Heartbeat
        self.enable_heartbeat = enable_heartbeat
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_thread: Optional[threading.Thread] = None
        self.heartbeat_running = False

        # Lifecycle flags
        self.is_initialized = False
        self.is_running = False

    # ---- Twin must provide -------------------------------------------------
    def get_agent_id(self):  # pragma: no cover - twin subclasses provide id
        raise NotImplementedError("Subclass must provide agent id")

    # ---- Lifecycle ---------------------------------------------------------
    def setup(self) -> bool:
        try:
            # Setup transport/bridge
            if self.transport is None:
                if self.mqtt_bridge is None:
                    # Lazy import to avoid hard dependency for non-MQTT twins
                    from src.battery_twin.communication.mqtt_bridge import (
                        MqttBridge,
                        MqttConfig,
                    )
                    from benchmarks.communication.transport import (
                        MqttBridgeTransport,
                    )

                    client_id = str(self.get_agent_id()).replace(".", "_")
                    cfg = self.mqtt_config or MqttConfig()
                    self.mqtt_bridge = MqttBridge(
                        client_id=client_id, mqtt_config=cfg
                    )
                    self.transport = MqttBridgeTransport(self.mqtt_bridge)
                else:
                    from benchmarks.communication.transport import (
                        MqttBridgeTransport,
                    )
                    self.transport = MqttBridgeTransport(self.mqtt_bridge)

            if self.own_mqtt and hasattr(self.transport, "connect"):
                if not self.transport.is_connected():
                    if not self.transport.connect():
                        logger.error("Failed to connect transport")
                        return False

            # Subscribe actions
            self._subscribe_to_actions()

            # Heartbeat
            if self.enable_heartbeat:
                self._start_heartbeat()

            # Per-agent hook
            if hasattr(self, "_agent_setup") and callable(self._agent_setup):
                if not self._agent_setup():
                    logger.error("Agent-specific setup failed")
                    return False

            self.is_initialized = True
            return True

        except Exception as e:
            logger.error(f"CommAgentBase setup failed: {e}")
            return False

    def teardown(self):
        try:
            # Stop heartbeat
            self._stop_heartbeat()

            # Per-agent hook
            if hasattr(self, "_agent_teardown") and callable(
                self._agent_teardown
            ):
                self._agent_teardown()

            # Disconnect bridge we own
            if self.own_mqtt and self.mqtt_bridge:
                if hasattr(self.mqtt_bridge, "disconnect"):
                    self.mqtt_bridge.disconnect()

        except Exception as e:
            logger.error(f"CommAgentBase teardown failed: {e}")

    # ---- Actions -----------------------------------------------------------
    def register_action(
        self,
        action_id: str,
        handler: Callable,
        topic_pattern: Optional[str] = None,
        description: str = "",
    ):
        with self.action_lock:
            self.action_handlers[action_id] = ActionHandler(
                action_id=action_id,
                handler=handler,
                topic_pattern=topic_pattern,
                description=description,
            )

    def enable_action(self, action_id: str, enabled: bool = True):
        with self.action_lock:
            if action_id in self.action_handlers:
                self.action_handlers[action_id].enabled = enabled

    def _subscribe_to_actions(self):
        if not self.transport:
            return
        with self.action_lock:
            for action_id, handler_info in self.action_handlers.items():
                if handler_info.topic_pattern and handler_info.enabled:

                    def make_wrapper(handler, action_id):
                        def wrapper(topic, payload):
                            self._execute_action_wrapper(
                                action_id, handler, topic, payload
                            )

                        return wrapper

                    self.transport.subscribe_raw(
                        handler_info.topic_pattern,
                        make_wrapper(handler_info.handler, action_id),
                    )

    def _execute_action_wrapper(
        self, action_id: str, handler: Callable, topic: str, payload: str
    ):
        start_time = time.time()
        try:
            self.metrics.messages_received += 1
            handler(topic, payload)
            self.metrics.actions_executed += 1
            self.metrics.last_action_time = time.time()
        except Exception as e:
            self.metrics.errors_count += 1
            logger.error(f"Error in action {action_id}: {e}")
        finally:
            self.metrics.total_processing_time += time.time() - start_time

    # ---- Publish -----------------------------------------------------------
    def publish_message(
        self, topic_name: str, message: Any, **topic_vars
    ) -> bool:
        """
        Publish a validated message via the bridge's topic manager. Twins can
        configure the topic set via the injected bridge.
        """
        if not self.transport:
            logger.warning("Cannot publish: No MQTT bridge")
            return False
        try:
            success = self.transport.publish(
                topic_name, message, **topic_vars
            )
            if success:
                self.metrics.messages_sent += 1
            return success
        except Exception as e:
            logger.error(f"Publish failed: {e}")
            self.metrics.errors_count += 1
            return False

    def publish_raw(self, topic: str, payload: str) -> bool:
        if not self.transport:
            logger.warning("Cannot publish: No MQTT bridge")
            return False
        try:
            success = self.transport.publish_raw(topic, payload)
            if success:
                self.metrics.messages_sent += 1
            return success
        except Exception as e:
            logger.error(f"Publish raw failed: {e}")
            self.metrics.errors_count += 1
            return False

    # ---- Heartbeat ---------------------------------------------------------
    def _start_heartbeat(self):
        if self.heartbeat_running:
            return
        self.heartbeat_running = True
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self.heartbeat_thread.start()

    def _stop_heartbeat(self):
        if not self.heartbeat_running:
            return
        self.heartbeat_running = False
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=5.0)

    def _heartbeat_loop(self):
        while self.heartbeat_running:
            try:
                # Lazy import to avoid hard dependency
                from src.battery_twin.communication.message_schemas import (
                    AgentHeartbeatMessage,
                )

                heartbeat = AgentHeartbeatMessage(
                    agent_id=str(self.get_agent_id()),
                    timestamp=time.time(),
                    status="running" if self.is_initialized else "created",
                    uptime=self.metrics.uptime,
                )
                if self.transport and self.transport.is_connected():
                    self.transport.publish(
                        "agent_heartbeat",
                        heartbeat,
                        agent_id=str(self.get_agent_id()),
                    )
                time.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                time.sleep(5.0)

    # ---- Metrics -----------------------------------------------------------
    def get_metrics(self) -> PerformanceMetrics:
        return self.metrics

    def get_metrics_dict(self) -> Dict:
        return self.metrics.to_dict()


__all__ = ["CommAgentBase", "PerformanceMetrics", "ActionHandler"]
