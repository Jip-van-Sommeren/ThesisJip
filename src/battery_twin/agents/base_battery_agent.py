"""
Base Battery Agent Class

Foundation class for all battery twin agents. Provides common functionality:
- MQTT communication integration
- Storage manager integration
- Action registry for MQTT topic handling
- Lifecycle management
- Performance monitoring
- Common utilities

All battery twin agents (BDI, Reactive, Hybrid) inherit from this.
"""

import logging
import time
import json
import threading
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from enum import Enum

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.abstract_agent import AbstractAgent, AgentId
from src.battery_twin.communication.mqtt_bridge import MqttBridge, MqttConfig
from src.battery_twin.communication.message_schemas import MessageFactory
from src.battery_twin.storage.battery_storage_manager import BatteryStorageManager

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent lifecycle status."""
    CREATED = "created"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ActionHandler:
    """Handler for an action triggered by MQTT or internal events."""
    action_id: str
    handler: Callable
    topic_pattern: Optional[str] = None
    description: str = ""
    enabled: bool = True


@dataclass
class PerformanceMetrics:
    """Performance tracking for agent."""
    messages_received: int = 0
    messages_sent: int = 0
    actions_executed: int = 0
    errors_count: int = 0
    last_action_time: float = 0.0
    total_processing_time: float = 0.0
    uptime_start: float = field(default_factory=time.time)

    @property
    def uptime(self) -> float:
        """Get agent uptime in seconds."""
        return time.time() - self.uptime_start

    @property
    def avg_processing_time(self) -> float:
        """Get average action processing time in milliseconds."""
        if self.actions_executed == 0:
            return 0.0
        return (self.total_processing_time / self.actions_executed) * 1000

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'messages_received': self.messages_received,
            'messages_sent': self.messages_sent,
            'actions_executed': self.actions_executed,
            'errors_count': self.errors_count,
            'uptime_seconds': self.uptime,
            'avg_processing_time_ms': self.avg_processing_time
        }


class BaseBatteryAgent:
    """
    Base class for all battery twin agents.

    Provides common infrastructure:
    - MQTT communication
    - Storage persistence
    - Action registry and routing
    - Lifecycle management
    - Performance monitoring
    - Error handling

    Usage:
        class MyAgent(BDIAgent, BaseBatteryAgent):
            def __init__(self, ...):
                # Initialize parent classes
                BDIAgent.__init__(self, agent_id, observable_properties)
                BaseBatteryAgent.__init__(self, mqtt_bridge, storage_manager)

                # Register actions
                self.register_action("process_data", self._process_data, "battery/+/raw")

            def _process_data(self, topic, payload):
                # Handle incoming data
                pass
    """

    def __init__(
        self,
        mqtt_bridge: Optional[MqttBridge] = None,
        storage_manager: Optional[BatteryStorageManager] = None,
        mqtt_config: Optional[MqttConfig] = None,
        enable_heartbeat: bool = True,
        heartbeat_interval: float = 30.0
    ):
        """
        Initialize base battery agent.

        Args:
            mqtt_bridge: MQTT bridge for communication (will create if None)
            storage_manager: Storage manager for persistence
            mqtt_config: MQTT configuration (used if mqtt_bridge is None)
            enable_heartbeat: Whether to send periodic heartbeats
            heartbeat_interval: Heartbeat interval in seconds
        """
        # MQTT communication
        self.mqtt_bridge = mqtt_bridge
        self.mqtt_config = mqtt_config or MqttConfig()
        self.own_mqtt = mqtt_bridge is None

        # Storage
        self.storage_manager = storage_manager

        # Agent status
        self.status = AgentStatus.CREATED
        self.status_lock = threading.Lock()

        # Action registry
        self.action_handlers: Dict[str, ActionHandler] = {}
        self.action_lock = threading.Lock()

        # Performance metrics
        self.metrics = PerformanceMetrics()

        # Heartbeat
        self.enable_heartbeat = enable_heartbeat
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_thread: Optional[threading.Thread] = None
        self.heartbeat_running = False

        # Lifecycle state
        self.is_initialized = False
        self.is_running = False

        logger.debug(f"BaseBatteryAgent initialized")

    def setup(self) -> bool:
        """
        Setup agent infrastructure.

        Connects to MQTT broker, initializes storage, and starts heartbeat.
        Called once before agent starts processing.

        Returns:
            True if setup successful
        """
        try:
            with self.status_lock:
                self.status = AgentStatus.INITIALIZING

            logger.info(f"Setting up agent {self.get_agent_id()}...")

            # Create/connect MQTT bridge
            if self.mqtt_bridge is None:
                client_id = str(self.get_agent_id()).replace('.', '_')
                self.mqtt_bridge = MqttBridge(
                    client_id=client_id,
                    mqtt_config=self.mqtt_config
                )

            # Connect to MQTT if we own the bridge
            if self.own_mqtt and not self.mqtt_bridge.is_connected():
                if not self.mqtt_bridge.connect():
                    logger.error("Failed to connect to MQTT broker")
                    with self.status_lock:
                        self.status = AgentStatus.ERROR
                    return False

            # Subscribe to registered actions
            self._subscribe_to_actions()

            # Start heartbeat if enabled
            if self.enable_heartbeat:
                self._start_heartbeat()

            # Custom agent setup
            if hasattr(self, '_agent_setup'):
                if not self._agent_setup():
                    logger.error("Agent-specific setup failed")
                    with self.status_lock:
                        self.status = AgentStatus.ERROR
                    return False

            with self.status_lock:
                self.status = AgentStatus.READY
            self.is_initialized = True

            logger.info(f"Agent {self.get_agent_id()} setup complete")
            return True

        except Exception as e:
            logger.error(f"Setup failed: {e}")
            with self.status_lock:
                self.status = AgentStatus.ERROR
            return False

    def teardown(self):
        """
        Teardown agent infrastructure.

        Disconnects from MQTT, stops heartbeat, and cleans up resources.
        """
        try:
            logger.info(f"Tearing down agent {self.get_agent_id()}...")

            with self.status_lock:
                self.status = AgentStatus.STOPPING

            # Stop heartbeat
            self._stop_heartbeat()

            # Custom agent teardown
            if hasattr(self, '_agent_teardown'):
                self._agent_teardown()

            # Disconnect MQTT if we own it
            if self.own_mqtt and self.mqtt_bridge:
                self.mqtt_bridge.disconnect()

            with self.status_lock:
                self.status = AgentStatus.STOPPED

            logger.info(f"Agent {self.get_agent_id()} teardown complete")

        except Exception as e:
            logger.error(f"Teardown failed: {e}")

    def register_action(
        self,
        action_id: str,
        handler: Callable,
        topic_pattern: Optional[str] = None,
        description: str = ""
    ):
        """
        Register an action handler.

        Args:
            action_id: Unique action identifier
            handler: Callback function(topic: str, payload: str) -> None
            topic_pattern: MQTT topic pattern to subscribe to (optional)
            description: Human-readable description
        """
        with self.action_lock:
            action_handler = ActionHandler(
                action_id=action_id,
                handler=handler,
                topic_pattern=topic_pattern,
                description=description
            )
            self.action_handlers[action_id] = action_handler

        logger.debug(f"Registered action: {action_id} -> {topic_pattern}")

    def unregister_action(self, action_id: str):
        """Unregister an action handler."""
        with self.action_lock:
            if action_id in self.action_handlers:
                del self.action_handlers[action_id]
                logger.debug(f"Unregistered action: {action_id}")

    def enable_action(self, action_id: str, enabled: bool = True):
        """Enable or disable an action handler."""
        with self.action_lock:
            if action_id in self.action_handlers:
                self.action_handlers[action_id].enabled = enabled

    def _subscribe_to_actions(self):
        """Subscribe to MQTT topics for registered actions."""
        if not self.mqtt_bridge:
            return

        with self.action_lock:
            for action_id, handler_info in self.action_handlers.items():
                if handler_info.topic_pattern and handler_info.enabled:
                    # Create wrapper that tracks metrics
                    def make_wrapper(handler, action_id):
                        def wrapper(topic, payload):
                            self._execute_action_wrapper(action_id, handler, topic, payload)
                        return wrapper

                    # Subscribe to topic
                    self.mqtt_bridge.subscribe_raw(
                        handler_info.topic_pattern,
                        make_wrapper(handler_info.handler, action_id)
                    )

                    logger.debug(f"Subscribed to {handler_info.topic_pattern} for {action_id}")

    def _execute_action_wrapper(
        self,
        action_id: str,
        handler: Callable,
        topic: str,
        payload: str
    ):
        """Wrapper for action execution with metrics tracking."""
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
            processing_time = time.time() - start_time
            self.metrics.total_processing_time += processing_time

    def publish_message(
        self,
        topic_name: str,
        message: Any,
        **topic_vars
    ) -> bool:
        """
        Publish a message to MQTT.

        Args:
            topic_name: Topic name from mqtt_topics.yaml
            message: Pydantic message instance
            **topic_vars: Variables for topic formatting

        Returns:
            True if publish successful
        """
        if not self.mqtt_bridge:
            logger.warning("Cannot publish: No MQTT bridge")
            return False

        try:
            success = self.mqtt_bridge.publish(topic_name, message, **topic_vars)
            if success:
                self.metrics.messages_sent += 1
            return success

        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            self.metrics.errors_count += 1
            return False

    def persist_to_storage(self, operation: str, **kwargs):
        """
        Persist data to storage manager.

        Args:
            operation: Storage operation (e.g., "telemetry", "prediction", "state")
            **kwargs: Operation-specific arguments
        """
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
            self.metrics.errors_count += 1

    def _start_heartbeat(self):
        """Start heartbeat thread."""
        if self.heartbeat_running:
            return

        self.heartbeat_running = True
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True
        )
        self.heartbeat_thread.start()
        logger.debug("Heartbeat started")

    def _stop_heartbeat(self):
        """Stop heartbeat thread."""
        if not self.heartbeat_running:
            return

        self.heartbeat_running = False
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=5.0)
        logger.debug("Heartbeat stopped")

    def _heartbeat_loop(self):
        """Heartbeat loop that sends periodic status updates."""
        while self.heartbeat_running:
            try:
                # Send heartbeat message
                from src.battery_twin.communication.message_schemas import AgentHeartbeatMessage

                heartbeat = AgentHeartbeatMessage(
                    agent_id=str(self.get_agent_id()),
                    timestamp=time.time(),
                    status=self.status.value,
                    uptime=self.metrics.uptime
                )

                if self.mqtt_bridge and self.mqtt_bridge.is_connected():
                    self.mqtt_bridge.publish(
                        "agent_heartbeat",
                        heartbeat,
                        agent_id=str(self.get_agent_id())
                    )

                # Sleep until next heartbeat
                time.sleep(self.heartbeat_interval)

            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                time.sleep(5.0)  # Retry after short delay on error

    def get_agent_id(self) -> AgentId:
        """
        Get agent ID.

        Must be implemented by subclass or provided via AbstractAgent.
        """
        if hasattr(self, 'id'):
            return self.id
        elif hasattr(self, 'agent_id'):
            return self.agent_id
        else:
            raise NotImplementedError("Subclass must provide agent_id or id")

    def get_status(self) -> AgentStatus:
        """Get current agent status."""
        with self.status_lock:
            return self.status

    def get_metrics(self) -> PerformanceMetrics:
        """Get performance metrics."""
        return self.metrics

    def get_metrics_dict(self) -> Dict:
        """Get performance metrics as dictionary."""
        return self.metrics.to_dict()

    def is_ready(self) -> bool:
        """Check if agent is ready to process."""
        with self.status_lock:
            return self.status in [AgentStatus.READY, AgentStatus.RUNNING]

    def __enter__(self):
        """Context manager entry."""
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.teardown()


__all__ = [
    'BaseBatteryAgent',
    'AgentStatus',
    'ActionHandler',
    'PerformanceMetrics',
]
