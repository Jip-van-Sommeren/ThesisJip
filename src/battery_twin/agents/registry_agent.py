"""
Registry Agent - Agent Discovery and Health Monitoring

The RegistryAgent is a reactive agent responsible for:
- Maintaining a directory of all active agents in the system
- Monitoring agent health via heartbeats
- Providing agent discovery service
- Tracking agent status and capabilities
- Detecting agent failures (heartbeat timeout)

Formal Definition:
A_registry = ⟨
  Id: "registry.1",
  State: {agent_directory, capability_map, status_map},
  Goals: ∅ (Reactive agent has no goals),
  Perception: {MQTT: agent/{agent_id}/register, agent/{agent_id}/heartbeat},
  Action: {register_agent(), update_status(), respond_to_query()},
  Decision: rule_based
⟩
"""

import logging
import time
import threading
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from src.abstract_agent import AgentId
from src.battery_twin.agents.battery_agent_types import BatteryReactiveAgent
from src.battery_twin.communication.mqtt_bridge import MqttBridge, MqttConfig
from src.battery_twin.communication.message_schemas import (
    AgentDirectoryMessage,
    MessageFactory,
)
from src.battery_twin.storage.battery_storage_manager import (
    BatteryStorageManager,
)

logger = logging.getLogger(__name__)


class AgentHealth(Enum):
    """Agent health status based on heartbeat monitoring."""

    ACTIVE = "active"  # Receiving regular heartbeats
    INACTIVE = "inactive"  # No heartbeats, but not timed out yet
    FAILED = "failed"  # Heartbeat timeout exceeded
    UNKNOWN = "unknown"  # No status information


@dataclass
class AgentRecord:
    """
    Complete record of a registered agent.

    Includes registration metadata, current status, and heartbeat tracking.
    """

    agent_id: str
    agent_type: str  # BDI, Reactive, Hybrid
    capabilities: List[str]
    supervisor: Optional[str] = None
    roles: List[str] = field(default_factory=list)
    groups: List[str] = field(default_factory=list)

    # Status tracking
    health_status: AgentHealth = AgentHealth.UNKNOWN
    registration_time: float = field(default_factory=time.time)
    last_heartbeat_time: float = 0.0
    heartbeat_count: int = 0

    # Additional metadata
    uptime: float = 0.0
    last_status: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "capabilities": self.capabilities,
            "supervisor": self.supervisor,
            "roles": self.roles,
            "groups": self.groups,
            "health_status": self.health_status.value,
            "registration_time": self.registration_time,
            "last_heartbeat_time": self.last_heartbeat_time,
            "heartbeat_count": self.heartbeat_count,
            "uptime": self.uptime,
            "last_status": self.last_status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentRecord":
        """Create AgentRecord from dictionary."""
        # Convert health_status string to enum
        health_status = AgentHealth(data.get("health_status", "unknown"))

        return cls(
            agent_id=data["agent_id"],
            agent_type=data["agent_type"],
            capabilities=data.get("capabilities", []),
            supervisor=data.get("supervisor"),
            roles=data.get("roles", []),
            groups=data.get("groups", []),
            health_status=health_status,
            registration_time=data.get("registration_time", time.time()),
            last_heartbeat_time=data.get("last_heartbeat_time", 0.0),
            heartbeat_count=data.get("heartbeat_count", 0),
            uptime=data.get("uptime", 0.0),
            last_status=data.get("last_status", ""),
        )


class RegistryAgent(BatteryReactiveAgent):
    """
    Registry Agent for Agent Discovery and Health Monitoring.

    This reactive agent maintains a central directory of all agents in the
    battery twin system. It provides:

    1. Registration Service: Agents register with their metadata
    2. Heartbeat Monitoring: Track agent health via periodic heartbeats
    3. Discovery Service: Query agents by ID, type, or capability
    4. Failure Detection: Detect agents that stop sending heartbeats
    5. Directory Publishing: Broadcast directory updates to all agents

    The agent uses:
    - In-memory dictionary for fast access
    - Redis for persistence and distributed access
    - MQTT for communication

    Example:
        >>> registry = RegistryAgent(
        ...     agent_id=AgentId(app="battery_twin", type="registry", instance="1"),
        ...     heartbeat_timeout=30.0
        ... )
        >>> registry.setup()
        >>> # Agents register via MQTT
        >>> # Registry monitors and responds to queries
        >>> registry.teardown()
    """

    def __init__(
        self,
        agent_id: AgentId,
        mqtt_bridge: Optional[MqttBridge] = None,
        storage_manager: Optional[BatteryStorageManager] = None,
        mqtt_config: Optional[MqttConfig] = None,
        heartbeat_timeout: float = 30.0,
        directory_publish_interval: float = 60.0,
        enable_redis_persistence: bool = True,
    ):
        """
        Initialize Registry Agent.

        Args:
            agent_id: Agent identifier (e.g., registry.1)
            mqtt_bridge: MQTT bridge for communication
            storage_manager: Storage manager (for Redis access)
            mqtt_config: MQTT configuration
            heartbeat_timeout: Seconds before marking agent as failed
            directory_publish_interval: Seconds between directory broadcasts
            enable_redis_persistence: Enable Redis persistence
        """
        # Observable properties for reactive agent
        observable_properties = {
            "agent_registration",
            "agent_heartbeat",
            "agent_directory",
            "agent_health",
        }

        # Initialize parent classes
        super().__init__(
            agent_id=agent_id,
            observable_properties=observable_properties,
            mqtt_bridge=mqtt_bridge,
            storage_manager=storage_manager,
            mqtt_config=mqtt_config,
            enable_heartbeat=True,  # Registry agent sends its own heartbeats
        )

        # Configuration
        self.heartbeat_timeout = heartbeat_timeout
        self.directory_publish_interval = directory_publish_interval
        self.enable_redis_persistence = enable_redis_persistence

        # Agent directory (in-memory)
        self.agent_directory: Dict[str, AgentRecord] = {}
        self.directory_lock = threading.Lock()

        # Redis keys
        self.redis_prefix = "registry:"
        self.redis_agent_key = f"{self.redis_prefix}agents"

        # Monitoring thread
        self.monitor_thread: Optional[threading.Thread] = None
        self.monitor_running = False

        # Directory publishing thread
        self.publish_thread: Optional[threading.Thread] = None
        self.publish_running = False

        logger.info(f"RegistryAgent initialized: {agent_id}")

    def _agent_setup(self) -> bool:
        """
        Agent-specific setup.

        Registers MQTT action handlers and starts monitoring threads.
        """
        try:
            # Register action handlers for MQTT topics
            self.register_action(
                action_id="handle_registration",
                handler=self._handle_registration,
                topic_pattern="agent/+/register",
                description="Handle agent registration messages",
            )

            self.register_action(
                action_id="handle_heartbeat",
                handler=self._handle_heartbeat,
                topic_pattern="agent/+/heartbeat",
                description="Handle agent heartbeat messages",
            )

            # Load existing directory from Redis if available
            if self.enable_redis_persistence:
                self._load_directory_from_redis()

            # Start monitoring thread
            self._start_monitoring()

            # Start directory publishing thread
            self._start_directory_publishing()

            logger.info("RegistryAgent setup complete")
            return True

        except Exception as e:
            logger.error(f"RegistryAgent setup failed: {e}")
            return False

    def _agent_teardown(self):
        """Agent-specific teardown."""
        # Stop monitoring threads
        self._stop_monitoring()
        self._stop_directory_publishing()

        # Save directory to Redis
        if self.enable_redis_persistence:
            self._save_directory_to_redis()

        logger.info("RegistryAgent teardown complete")

    # ========================================================================
    # Registration Handling
    # ========================================================================

    def _handle_registration(self, topic: str, payload: str):
        """
        Handle agent registration message.

        Called when an agent publishes to agent/{agent_id}/register

        Args:
            topic: MQTT topic
            payload: JSON message payload
        """
        try:
            # Parse registration message
            message = MessageFactory.parse_message(
                "agent_registration", payload
            )

            logger.info(f"Registration received from {message.agent_id}")

            # Create or update agent record
            with self.directory_lock:
                if message.agent_id in self.agent_directory:
                    # Update existing record
                    record = self.agent_directory[message.agent_id]
                    record.agent_type = message.agent_type
                    record.capabilities = message.capabilities
                    record.supervisor = message.supervisor
                    record.roles = message.roles
                    record.groups = message.groups
                    logger.info(f"Updated registration for {message.agent_id}")
                else:
                    # Create new record
                    record = AgentRecord(
                        agent_id=message.agent_id,
                        agent_type=message.agent_type,
                        capabilities=message.capabilities,
                        supervisor=message.supervisor,
                        roles=message.roles,
                        groups=message.groups,
                        health_status=AgentHealth.INACTIVE,  # Wait for first heartbeat
                        registration_time=message.timestamp,
                    )
                    self.agent_directory[message.agent_id] = record
                    logger.info(
                        f"Registered new agent: {message.agent_id} (type: {message.agent_type})"
                    )

            # Persist to Redis
            if self.enable_redis_persistence:
                self._save_agent_to_redis(record)

            # Update internal belief state
            self.state.update_belief(
                key="agent_registration",
                proposition=f"registered:{message.agent_id}",
                confidence=1.0,
            )

        except Exception as e:
            logger.error(f"Failed to handle registration: {e}")

    # ========================================================================
    # Heartbeat Monitoring
    # ========================================================================

    def _handle_heartbeat(self, topic: str, payload: str):
        """
        Handle agent heartbeat message.

        Called when an agent publishes to agent/{agent_id}/heartbeat

        Args:
            topic: MQTT topic
            payload: JSON message payload
        """
        try:
            # Parse heartbeat message
            message = MessageFactory.parse_message("agent_heartbeat", payload)

            # Update agent status
            with self.directory_lock:
                if message.agent_id in self.agent_directory:
                    record = self.agent_directory[message.agent_id]
                    record.last_heartbeat_time = message.timestamp
                    record.heartbeat_count += 1
                    record.uptime = message.uptime
                    record.last_status = message.status
                    record.health_status = AgentHealth.ACTIVE

                    logger.debug(
                        f"Heartbeat from {message.agent_id} (count: {record.heartbeat_count})"
                    )
                else:
                    # Agent sent heartbeat but not registered - create minimal record
                    logger.warning(
                        f"Heartbeat from unregistered agent: {message.agent_id}"
                    )
                    record = AgentRecord(
                        agent_id=message.agent_id,
                        agent_type="unknown",
                        capabilities=[],
                        health_status=AgentHealth.ACTIVE,
                        last_heartbeat_time=message.timestamp,
                        heartbeat_count=1,
                        uptime=message.uptime,
                        last_status=message.status,
                    )
                    self.agent_directory[message.agent_id] = record

            # Update belief state
            self.state.update_belief(
                key="agent_heartbeat",
                proposition=f"heartbeat_received:{message.agent_id}",
                confidence=1.0,
            )

        except Exception as e:
            logger.error(f"Failed to handle heartbeat: {e}")

    def _start_monitoring(self):
        """Start monitoring thread for heartbeat timeout detection."""
        if self.monitor_running:
            return

        self.monitor_running = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True, name="registry-monitor"
        )
        self.monitor_thread.start()
        logger.info("Started heartbeat monitoring")

    def _stop_monitoring(self):
        """Stop monitoring thread."""
        if not self.monitor_running:
            return

        self.monitor_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Stopped heartbeat monitoring")

    def _monitoring_loop(self):
        """
        Monitoring loop that checks for heartbeat timeouts.

        Runs every 5 seconds and marks agents as FAILED if they haven't
        sent a heartbeat within the timeout period.
        """
        while self.monitor_running:
            try:
                current_time = time.time()

                with self.directory_lock:
                    for agent_id, record in self.agent_directory.items():
                        # Skip if no heartbeat received yet
                        if record.last_heartbeat_time == 0.0:
                            continue

                        # Check timeout
                        time_since_heartbeat = (
                            current_time - record.last_heartbeat_time
                        )

                        if time_since_heartbeat > self.heartbeat_timeout:
                            if record.health_status != AgentHealth.FAILED:
                                record.health_status = AgentHealth.FAILED
                                logger.warning(
                                    f"Agent {agent_id} marked as FAILED "
                                    f"(no heartbeat for {time_since_heartbeat:.1f}s)"
                                )

                                # Update belief state
                                self.state.update_belief(
                                    key="agent_health",
                                    proposition=f"agent_failed:{agent_id}",
                                    confidence=1.0,
                                )

                # Sleep before next check
                time.sleep(5.0)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5.0)

    # ========================================================================
    # Directory Publishing
    # ========================================================================

    def _start_directory_publishing(self):
        """Start directory publishing thread."""
        if self.publish_running:
            return

        self.publish_running = True
        self.publish_thread = threading.Thread(
            target=self._publishing_loop,
            daemon=True,
            name="registry-publisher",
        )
        self.publish_thread.start()
        logger.info("Started directory publishing")

    def _stop_directory_publishing(self):
        """Stop directory publishing thread."""
        if not self.publish_running:
            return

        self.publish_running = False
        if self.publish_thread:
            self.publish_thread.join(timeout=5.0)
        logger.info("Stopped directory publishing")

    def _publishing_loop(self):
        """
        Publishing loop that broadcasts the agent directory.

        Publishes the complete directory to agent/registry/directory
        at regular intervals so all agents can maintain a local copy.
        """
        while self.publish_running:
            try:
                # Publish directory
                self.publish_directory()

                # Sleep until next publish
                time.sleep(self.directory_publish_interval)

            except Exception as e:
                logger.error(f"Error in publishing loop: {e}")
                time.sleep(10.0)

    def publish_directory(self):
        """
        Publish the complete agent directory to MQTT.

        Sends AgentDirectoryMessage to agent/registry/directory topic.
        """
        try:
            with self.directory_lock:
                # Convert all records to dictionaries
                agents_list = [
                    record.to_dict()
                    for record in self.agent_directory.values()
                ]

                # Create directory message
                directory_msg = AgentDirectoryMessage(
                    timestamp=time.time(),
                    agents=agents_list,
                    total_agents=len(agents_list),
                )

                # Publish to MQTT
                success = self.publish_message(
                    topic_name="agent_directory", message=directory_msg
                )

                if success:
                    logger.debug(
                        f"Published directory with {len(agents_list)} agents"
                    )
                else:
                    logger.warning("Failed to publish directory")

        except Exception as e:
            logger.error(f"Failed to publish directory: {e}")

    # ========================================================================
    # Discovery Service
    # ========================================================================

    def get_agent(self, agent_id: str) -> Optional[AgentRecord]:
        """
        Get agent record by ID.

        Args:
            agent_id: Agent identifier

        Returns:
            AgentRecord if found, None otherwise
        """
        with self.directory_lock:
            return self.agent_directory.get(agent_id)

    def get_all_agents(self) -> List[AgentRecord]:
        """
        Get all registered agents.

        Returns:
            List of all agent records
        """
        with self.directory_lock:
            return list(self.agent_directory.values())

    def get_agents_by_type(self, agent_type: str) -> List[AgentRecord]:
        """
        Get all agents of a specific type.

        Args:
            agent_type: Agent type (BDI, Reactive, Hybrid)

        Returns:
            List of matching agent records
        """
        with self.directory_lock:
            return [
                record
                for record in self.agent_directory.values()
                if record.agent_type == agent_type
            ]

    def get_agents_by_capability(self, capability: str) -> List[AgentRecord]:
        """
        Get all agents with a specific capability.

        Args:
            capability: Capability name

        Returns:
            List of matching agent records
        """
        with self.directory_lock:
            return [
                record
                for record in self.agent_directory.values()
                if capability in record.capabilities
            ]

    def get_agents_by_role(self, role: str) -> List[AgentRecord]:
        """
        Get all agents with a specific role.

        Args:
            role: Role name

        Returns:
            List of matching agent records
        """
        with self.directory_lock:
            return [
                record
                for record in self.agent_directory.values()
                if role in record.roles
            ]

    def get_agents_by_health(
        self, health_status: AgentHealth
    ) -> List[AgentRecord]:
        """
        Get all agents with a specific health status.

        Args:
            health_status: Health status filter

        Returns:
            List of matching agent records
        """
        with self.directory_lock:
            return [
                record
                for record in self.agent_directory.values()
                if record.health_status == health_status
            ]

    def get_active_agents(self) -> List[AgentRecord]:
        """Get all active (healthy) agents."""
        return self.get_agents_by_health(AgentHealth.ACTIVE)

    def get_failed_agents(self) -> List[AgentRecord]:
        """Get all failed agents."""
        return self.get_agents_by_health(AgentHealth.FAILED)

    def get_agent_count(self) -> int:
        """Get total number of registered agents."""
        with self.directory_lock:
            return len(self.agent_directory)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Dictionary with counts and status breakdown
        """
        with self.directory_lock:
            total = len(self.agent_directory)
            active = sum(
                1
                for r in self.agent_directory.values()
                if r.health_status == AgentHealth.ACTIVE
            )
            inactive = sum(
                1
                for r in self.agent_directory.values()
                if r.health_status == AgentHealth.INACTIVE
            )
            failed = sum(
                1
                for r in self.agent_directory.values()
                if r.health_status == AgentHealth.FAILED
            )

            # Count by type
            type_counts = {}
            for record in self.agent_directory.values():
                type_counts[record.agent_type] = (
                    type_counts.get(record.agent_type, 0) + 1
                )

            return {
                "total_agents": total,
                "active_agents": active,
                "inactive_agents": inactive,
                "failed_agents": failed,
                "agents_by_type": type_counts,
            }

    # ========================================================================
    # Redis Persistence
    # ========================================================================

    def _get_redis_client(self):
        """Get Redis client from storage manager."""
        if not self.storage_manager:
            return None

        # Storage manager should have a Redis client
        if hasattr(self.storage_manager, "redis_storage"):
            return self.storage_manager.redis_storage

        return None

    def _save_agent_to_redis(self, record: AgentRecord):
        """
        Save single agent record to Redis.

        Args:
            record: Agent record to save
        """
        try:
            redis_client = self._get_redis_client()
            if not redis_client:
                return

            # Store as hash in Redis
            key = f"{self.redis_agent_key}:{record.agent_id}"
            redis_client.hset(
                key, mapping={"data": json.dumps(record.to_dict())}
            )

            # Set expiry (24 hours)
            redis_client.expire(key, 86400)

        except Exception as e:
            logger.error(f"Failed to save agent to Redis: {e}")

    def _save_directory_to_redis(self):
        """Save entire directory to Redis."""
        try:
            redis_client = self._get_redis_client()
            if not redis_client:
                return

            with self.directory_lock:
                for record in self.agent_directory.values():
                    self._save_agent_to_redis(record)

            logger.info(f"Saved {len(self.agent_directory)} agents to Redis")

        except Exception as e:
            logger.error(f"Failed to save directory to Redis: {e}")

    def _load_directory_from_redis(self):
        """Load directory from Redis on startup."""
        try:
            redis_client = self._get_redis_client()
            if not redis_client:
                logger.info(
                    "Redis not available, starting with empty directory"
                )
                return

            # Find all agent keys
            pattern = f"{self.redis_agent_key}:*"
            keys = redis_client.keys(pattern)

            if not keys:
                logger.info("No agents found in Redis")
                return

            # Load each agent
            loaded_count = 0
            with self.directory_lock:
                for key in keys:
                    try:
                        data = redis_client.hget(key, "data")
                        if data:
                            record_dict = json.loads(data)
                            record = AgentRecord.from_dict(record_dict)
                            self.agent_directory[record.agent_id] = record
                            loaded_count += 1
                    except Exception as e:
                        logger.error(f"Failed to load agent from {key}: {e}")

            logger.info(f"Loaded {loaded_count} agents from Redis")

        except Exception as e:
            logger.error(f"Failed to load directory from Redis: {e}")


__all__ = [
    "RegistryAgent",
    "AgentRecord",
    "AgentHealth",
]
