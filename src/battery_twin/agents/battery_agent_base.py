"""
Battery Agent Base Classes

Battery-specific agent implementations built on the MAS framework.
Provides three agent types for different use cases:
- BatteryReactiveAgent: Fast stimulus-response (telemetry, fault detection)
- BatteryBDIAgent: Deliberative reasoning (optimization, scheduling)
- BatteryHybridAgent: Combined reactive + deliberative (physics models, state estimation)

All battery agents share:
- Transport injection for MQTT communication
- Storage manager integration for persistence
- Heartbeat and metrics tracking
- Common lifecycle management
"""

from __future__ import annotations

import logging
import time
import threading
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Set

from mas.core import (
    AgentId,
    ReactiveAgent,
    BDIAgent,
    HybridAgent,
)
from mas.communication import (
    Transport,
    AgentHeartbeatMessage,
)

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
class PerformanceMetrics:
    """Performance tracking metrics."""
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


@dataclass
class ActionHandler:
    """Registered action with topic subscription."""
    action_id: str
    handler: Callable
    topic_pattern: Optional[str] = None
    description: str = ""
    enabled: bool = True


class BatteryAgentMixin:
    """
    Mixin providing battery-specific functionality.

    Adds to any agent type:
    - Transport-based communication
    - Storage manager integration
    - Heartbeat management
    - Action registration with topic routing
    - Performance metrics
    """

    def _init_battery_mixin(
        self,
        transport: Transport,
        storage_manager: Optional[Any] = None,
        enable_heartbeat: bool = True,
        heartbeat_interval: float = 30.0,
    ):
        """
        Initialize battery agent mixin.

        Args:
            transport: MQTT transport (required, injected)
            storage_manager: Optional storage manager for persistence
            enable_heartbeat: Enable periodic heartbeat messages
            heartbeat_interval: Heartbeat interval in seconds
        """
        self.transport = transport
        self.storage_manager = storage_manager

        # Action registry
        self.action_handlers: Dict[str, ActionHandler] = {}
        self.action_lock = threading.Lock()

        # Status tracking
        self.status = AgentStatus.CREATED
        self.status_lock = threading.Lock()

        # Metrics
        self.metrics = PerformanceMetrics()

        # Heartbeat configuration
        self.enable_heartbeat = enable_heartbeat
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_thread: Optional[threading.Thread] = None
        self.heartbeat_running = False

        # Lifecycle flags
        self.is_initialized = False

    # ---- Action Registration ----

    def register_action(
        self,
        action_id: str,
        handler: Callable,
        topic_pattern: Optional[str] = None,
        description: str = "",
    ):
        """
        Register an action handler with optional topic subscription.

        Args:
            action_id: Unique action identifier
            handler: Callback function(topic, payload)
            topic_pattern: MQTT topic pattern to subscribe to
            description: Human-readable description
        """
        with self.action_lock:
            self.action_handlers[action_id] = ActionHandler(
                action_id=action_id,
                handler=handler,
                topic_pattern=topic_pattern,
                description=description,
            )

    def enable_action(self, action_id: str, enabled: bool = True):
        """Enable or disable an action."""
        with self.action_lock:
            if action_id in self.action_handlers:
                self.action_handlers[action_id].enabled = enabled

    def _subscribe_to_actions(self):
        """Subscribe to topics for all registered actions."""
        if not self.transport:
            return

        with self.action_lock:
            for action_id, handler_info in self.action_handlers.items():
                if handler_info.topic_pattern and handler_info.enabled:
                    def make_wrapper(handler, aid):
                        def wrapper(topic: str, payload: str):
                            self._execute_action_wrapper(aid, handler, topic, payload)
                        return wrapper

                    self.transport.subscribe(
                        handler_info.topic_pattern,
                        make_wrapper(handler_info.handler, action_id),
                    )
                    logger.debug(f"Subscribed action {action_id} to {handler_info.topic_pattern}")

    def _execute_action_wrapper(
        self,
        action_id: str,
        handler: Callable,
        topic: str,
        payload: str
    ):
        """Execute action with metrics tracking."""
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

    # ---- Publishing ----

    def publish_message(
        self,
        topic_name: str,
        message: Any,
        **topic_vars
    ) -> bool:
        """
        Publish a message using TopicManager for topic resolution.

        Args:
            topic_name: Logical topic name from configuration
            message: Pydantic model to publish
            **topic_vars: Variables for topic formatting

        Returns:
            True if publish successful
        """
        if not self.transport:
            logger.warning("Cannot publish: No transport configured")
            return False

        try:
            success = self.transport.publish_to_topic(
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
        """
        Publish raw payload to a topic.

        Args:
            topic: MQTT topic
            payload: Raw payload string

        Returns:
            True if publish successful
        """
        if not self.transport:
            logger.warning("Cannot publish: No transport configured")
            return False

        try:
            success = self.transport.publish(topic, payload)
            if success:
                self.metrics.messages_sent += 1
            return success
        except Exception as e:
            logger.error(f"Publish raw failed: {e}")
            self.metrics.errors_count += 1
            return False

    # ---- Storage ----

    def persist_to_storage(self, operation: str, **kwargs):
        """
        Persist data to storage manager.

        Args:
            operation: Storage operation type
            **kwargs: Operation-specific data
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

    # ---- Heartbeat ----

    def _start_heartbeat(self):
        """Start heartbeat thread."""
        if self.heartbeat_running:
            return
        self.heartbeat_running = True
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self.heartbeat_thread.start()

    def _stop_heartbeat(self):
        """Stop heartbeat thread."""
        if not self.heartbeat_running:
            return
        self.heartbeat_running = False
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=5.0)

    def _heartbeat_loop(self):
        """Heartbeat loop - publishes periodic status."""
        while self.heartbeat_running:
            try:
                with self.status_lock:
                    status_val = self.status.value

                heartbeat = AgentHeartbeatMessage(
                    agent_id=str(self.agent_id),
                    timestamp=time.time(),
                    status=status_val,
                    uptime=self.metrics.uptime,
                )

                if self.transport and self.transport.is_connected():
                    self.transport.publish_to_topic(
                        "agent_heartbeat",
                        heartbeat,
                        agent_id=str(self.agent_id),
                    )

                time.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                time.sleep(5.0)

    # ---- Lifecycle ----

    def setup(self) -> bool:
        """
        Setup agent: subscribe to actions, start heartbeat.

        Returns:
            True if setup successful
        """
        try:
            with self.status_lock:
                self.status = AgentStatus.INITIALIZING

            logger.info(f"Setting up agent {self.agent_id}...")

            # Subscribe to action topics
            self._subscribe_to_actions()

            # Start heartbeat
            if self.enable_heartbeat:
                self._start_heartbeat()

            # Agent-specific setup
            if hasattr(self, "_agent_setup") and callable(self._agent_setup):
                if not self._agent_setup():
                    logger.error("Agent-specific setup failed")
                    with self.status_lock:
                        self.status = AgentStatus.ERROR
                    return False

            self.is_initialized = True
            with self.status_lock:
                self.status = AgentStatus.READY

            logger.info(f"Agent {self.agent_id} setup complete")
            return True

        except Exception as e:
            logger.error(f"Agent setup failed: {e}")
            with self.status_lock:
                self.status = AgentStatus.ERROR
            return False

    def teardown(self):
        """Teardown agent: stop heartbeat, cleanup."""
        try:
            logger.info(f"Tearing down agent {self.agent_id}...")
            with self.status_lock:
                self.status = AgentStatus.STOPPING

            # Stop heartbeat
            self._stop_heartbeat()

            # Agent-specific teardown
            if hasattr(self, "_agent_teardown") and callable(self._agent_teardown):
                self._agent_teardown()

            with self.status_lock:
                self.status = AgentStatus.STOPPED

            logger.info(f"Agent {self.agent_id} teardown complete")

        except Exception as e:
            logger.error(f"Agent teardown failed: {e}")

    # ---- Status / Metrics ----

    def get_status(self) -> AgentStatus:
        """Get current agent status."""
        with self.status_lock:
            return self.status

    def is_ready(self) -> bool:
        """Check if agent is ready to process."""
        with self.status_lock:
            return self.status in [AgentStatus.READY, AgentStatus.RUNNING]

    def get_metrics(self) -> PerformanceMetrics:
        """Get performance metrics."""
        return self.metrics

    def get_metrics_dict(self) -> Dict:
        """Get metrics as dictionary."""
        return self.metrics.to_dict()


class BatteryReactiveAgent(ReactiveAgent, BatteryAgentMixin):
    """
    Reactive Agent for Battery Digital Twin.

    Combines:
    - ReactiveAgent: Fast stimulus-response behavior
    - BatteryAgentMixin: MQTT communication and storage

    Use for: Telemetry ingestion, fault detection, registry services
    """

    def __init__(
        self,
        agent_id: AgentId,
        transport: Transport,
        observable_properties: Optional[Set[str]] = None,
        storage_manager: Optional[Any] = None,
        enable_heartbeat: bool = True,
        heartbeat_interval: float = 30.0,
    ):
        """
        Initialize Reactive battery agent.

        Args:
            agent_id: Agent identifier
            transport: MQTT transport (required)
            observable_properties: Properties this agent can observe
            storage_manager: Optional storage manager
            enable_heartbeat: Enable periodic heartbeats
            heartbeat_interval: Heartbeat interval in seconds
        """
        # Initialize ReactiveAgent
        ReactiveAgent.__init__(
            self,
            agent_id,
            observable_properties or set(),
        )

        # Initialize battery mixin
        self._init_battery_mixin(
            transport=transport,
            storage_manager=storage_manager,
            enable_heartbeat=enable_heartbeat,
            heartbeat_interval=heartbeat_interval,
        )

        # Store agent_id for mixin access
        self.agent_id = agent_id

        logger.info(f"Initialized BatteryReactiveAgent: {agent_id}")

    def _agent_setup(self) -> bool:
        """Override in subclass for agent-specific setup."""
        return True

    def _agent_teardown(self):
        """Override in subclass for agent-specific teardown."""
        pass


class BatteryBDIAgent(BDIAgent, BatteryAgentMixin):
    """
    BDI Agent for Battery Digital Twin.

    Combines:
    - BDIAgent: Beliefs, Desires, Intentions reasoning
    - BatteryAgentMixin: MQTT communication and storage

    Use for: ML training decisions, parameter identification, orchestration
    """

    def __init__(
        self,
        agent_id: AgentId,
        transport: Transport,
        observable_properties: Optional[Set[str]] = None,
        storage_manager: Optional[Any] = None,
        enable_heartbeat: bool = True,
        heartbeat_interval: float = 30.0,
    ):
        """
        Initialize BDI battery agent.

        Args:
            agent_id: Agent identifier
            transport: MQTT transport (required)
            observable_properties: Properties this agent can observe
            storage_manager: Optional storage manager
            enable_heartbeat: Enable periodic heartbeats
            heartbeat_interval: Heartbeat interval in seconds
        """
        # Initialize BDIAgent
        BDIAgent.__init__(
            self,
            agent_id,
            observable_properties or set(),
        )

        # Initialize battery mixin
        self._init_battery_mixin(
            transport=transport,
            storage_manager=storage_manager,
            enable_heartbeat=enable_heartbeat,
            heartbeat_interval=heartbeat_interval,
        )

        # Store agent_id for mixin access
        self.agent_id = agent_id

        logger.info(f"Initialized BatteryBDIAgent: {agent_id}")

    def _agent_setup(self) -> bool:
        """Override in subclass for agent-specific setup."""
        return True

    def _agent_teardown(self):
        """Override in subclass for agent-specific teardown."""
        pass


class BatteryHybridAgent(HybridAgent, BatteryAgentMixin):
    """
    Hybrid Agent for Battery Digital Twin.

    Combines:
    - HybridAgent: Reactive rules + goal-driven behavior
    - BatteryAgentMixin: MQTT communication and storage

    Use for: Physics models, state estimation, control
    """

    def __init__(
        self,
        agent_id: AgentId,
        transport: Transport,
        observable_properties: Optional[Set[str]] = None,
        storage_manager: Optional[Any] = None,
        enable_heartbeat: bool = True,
        heartbeat_interval: float = 30.0,
    ):
        """
        Initialize Hybrid battery agent.

        Args:
            agent_id: Agent identifier
            transport: MQTT transport (required)
            observable_properties: Properties this agent can observe
            storage_manager: Optional storage manager
            enable_heartbeat: Enable periodic heartbeats
            heartbeat_interval: Heartbeat interval in seconds
        """
        # Initialize HybridAgent
        HybridAgent.__init__(
            self,
            agent_id,
            observable_properties or set(),
        )

        # Initialize battery mixin
        self._init_battery_mixin(
            transport=transport,
            storage_manager=storage_manager,
            enable_heartbeat=enable_heartbeat,
            heartbeat_interval=heartbeat_interval,
        )

        # Store agent_id for mixin access
        self.agent_id = agent_id

        logger.info(f"Initialized BatteryHybridAgent: {agent_id}")

    def _agent_setup(self) -> bool:
        """Override in subclass for agent-specific setup."""
        return True

    def _agent_teardown(self):
        """Override in subclass for agent-specific teardown."""
        pass


__all__ = [
    "BatteryReactiveAgent",
    "BatteryBDIAgent",
    "BatteryHybridAgent",
    "BatteryAgentMixin",
    "AgentStatus",
    "PerformanceMetrics",
    "ActionHandler",
]
