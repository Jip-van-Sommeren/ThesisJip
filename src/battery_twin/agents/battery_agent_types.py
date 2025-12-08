"""
Battery Agent Type Wrappers

Convenient wrapper classes that combine framework agents (BDI, Reactive, Hybrid)
with BaseBatteryAgent functionality.

These classes handle the multiple inheritance and initialization complexity,
providing a clean interface for creating battery twin agents.
"""

import logging
from typing import Set, Optional

import sys
from pathlib import Path


from abstract_agent import AgentId
from bdi_agent import BDIAgent
from reactive_agent import ReactiveAgent
from hybrid_agent import HybridAgent
from src.battery_twin.agents.battery_agent_base import BatteryAgentBase
from src.battery_twin.communication.mqtt_bridge import MqttBridge, MqttConfig
from src.battery_twin.storage.battery_storage_manager import (
    BatteryStorageManager,
)

logger = logging.getLogger(__name__)


class BatteryBDIAgent(BDIAgent, BatteryAgentBase):
    """
    BDI Agent for Battery Twin.

    Combines:
    - BDIAgent: Beliefs, Desires, Intentions reasoning
    - BaseBatteryAgent: MQTT communication and storage

    Use for: ML training decisions, parameter identification, orchestration
    """

    def __init__(
        self,
        agent_id: AgentId,
        observable_properties: Set[str],
        mqtt_bridge: Optional[MqttBridge] = None,
        storage_manager: Optional[BatteryStorageManager] = None,
        mqtt_config: Optional[MqttConfig] = None,
        enable_heartbeat: bool = True,
    ):
        """
        Initialize BDI battery agent.

        Args:
            agent_id: Agent identifier
            observable_properties: Properties this agent can observe
            mqtt_bridge: MQTT bridge for communication
            storage_manager: Storage manager for persistence
            mqtt_config: MQTT configuration
            enable_heartbeat: Enable periodic heartbeats
        """
        # Initialize BDI agent
        BDIAgent.__init__(self, agent_id, observable_properties)

        # Initialize battery agent base
        BatteryAgentBase.__init__(
            self,
            mqtt_bridge=mqtt_bridge,
            storage_manager=storage_manager,
            mqtt_config=mqtt_config,
            enable_heartbeat=enable_heartbeat,
        )

        logger.info(f"Initialized BatteryBDIAgent: {agent_id}")

    def _agent_setup(self) -> bool:
        """Agent-specific setup hook."""
        # Subclasses can override this
        return True

    def _agent_teardown(self):
        """Agent-specific teardown hook."""
        # Subclasses can override this
        pass


class BatteryReactiveAgent(ReactiveAgent, BatteryAgentBase):
    """
    Reactive Agent for Battery Twin.

    Combines:
    - ReactiveAgent: Fast stimulus-response behavior
    - BaseBatteryAgent: MQTT communication and storage

    Use for: Telemetry ingestion, fault detection, registry services
    """

    def __init__(
        self,
        agent_id: AgentId,
        observable_properties: Set[str],
        mqtt_bridge: Optional[MqttBridge] = None,
        storage_manager: Optional[BatteryStorageManager] = None,
        mqtt_config: Optional[MqttConfig] = None,
        enable_heartbeat: bool = True,
    ):
        """
        Initialize Reactive battery agent.

        Args:
            agent_id: Agent identifier
            observable_properties: Properties this agent can observe
            mqtt_bridge: MQTT bridge for communication
            storage_manager: Storage manager for persistence
            mqtt_config: MQTT configuration
            enable_heartbeat: Enable periodic heartbeats
        """
        # Initialize Reactive agent
        ReactiveAgent.__init__(self, agent_id, observable_properties)

        # Initialize battery agent base
        BatteryAgentBase.__init__(
            self,
            mqtt_bridge=mqtt_bridge,
            storage_manager=storage_manager,
            mqtt_config=mqtt_config,
            enable_heartbeat=enable_heartbeat,
        )

        logger.info(f"Initialized BatteryReactiveAgent: {agent_id}")

    def _agent_setup(self) -> bool:
        """Agent-specific setup hook."""
        # Subclasses can override this
        return True

    def _agent_teardown(self):
        """Agent-specific teardown hook."""
        # Subclasses can override this
        pass


class BatteryHybridAgent(HybridAgent, BatteryAgentBase):
    """
    Hybrid Agent for Battery Twin.

    Combines:
    - HybridAgent: Reactive rules + goal-driven behavior
    - BaseBatteryAgent: MQTT communication and storage

    Use for: Physics models, state estimation, control
    """

    def __init__(
        self,
        agent_id: AgentId,
        observable_properties: Set[str],
        mqtt_bridge: Optional[MqttBridge] = None,
        storage_manager: Optional[BatteryStorageManager] = None,
        mqtt_config: Optional[MqttConfig] = None,
        enable_heartbeat: bool = True,
    ):
        """
        Initialize Hybrid battery agent.

        Args:
            agent_id: Agent identifier
            observable_properties: Properties this agent can observe
            mqtt_bridge: MQTT bridge for communication
            storage_manager: Storage manager for persistence
            mqtt_config: MQTT configuration
            enable_heartbeat: Enable periodic heartbeats
        """
        # Initialize Hybrid agent
        HybridAgent.__init__(self, agent_id, observable_properties)

        # Initialize battery agent base
        BatteryAgentBase.__init__(
            self,
            mqtt_bridge=mqtt_bridge,
            storage_manager=storage_manager,
            mqtt_config=mqtt_config,
            enable_heartbeat=enable_heartbeat,
        )

        logger.info(f"Initialized BatteryHybridAgent: {agent_id}")

    def _agent_setup(self) -> bool:
        """Agent-specific setup hook."""
        # Subclasses can override this
        return True

    def _agent_teardown(self):
        """Agent-specific teardown hook."""
        # Subclasses can override this
        pass


__all__ = [
    "BatteryBDIAgent",
    "BatteryReactiveAgent",
    "BatteryHybridAgent",
]
