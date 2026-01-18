"""
Multi-Agent System (MAS) Framework

A framework for building agent-based digital twin applications.

Based on the formal definition: A = (Id, State, Goal, Perception, Action, Decision)

Layers:
- Core: Agent abstractions (Agent, ReactiveAgent, BDIAgent, HybridAgent)
- Communication: MQTT-based messaging (Transport, TopicManager, Mailbox)
- Organization: Roles, Groups, Hierarchy

Example:
    from mas.core import AgentId, HybridAgent
    from mas.communication import MqttTransport, TopicManager, MqttConfig

    # Setup
    config = MqttConfig(broker="localhost", port=1883)
    topic_manager = TopicManager("config/mqtt_topics.yaml")
    transport = MqttTransport(config, topic_manager)
    transport.connect()

    # Create agent
    agent = MyAgent(
        agent_id=AgentId("app", "agent_type", "001"),
        transport=transport
    )
"""

from .core import (
    # Base agent
    Agent,
    AgentId,
    State,
    Belief,
    Goal,
    GoalType,
    Action,
    ActionType,
    Perception,
    Decision,
    ReactiveRule,
    # Agent types
    ReactiveAgent,
    BDIAgent,
    HybridAgent,
    # BDI/Hybrid components
    Intention,
    IntentionStatus,
    LayerPriority,
)

from .communication import (
    # Transport
    Transport,
    MqttConfig,
    MqttTransport,
    MockTransport,
    # Topic management
    TopicManager,
    # Mailbox
    Mailbox,
    Message,
    # Message schemas
    AgentMessage,
    AgentStatusMessage,
    AgentHeartbeatMessage,
    CommandMessage,
    ResponseMessage,
)

from .organization import (
    # Role system
    Role,
    RoleManager,
    Responsibility,
    Permission,
    PermissionType,
    # Group system
    Group,
    GroupManager,
    # Hierarchy system
    HierarchyManager,
    OrganizationalPosition,
)

__version__ = "1.0.0"

__all__ = [
    # Core - Base
    "Agent",
    "AgentId",
    "State",
    "Belief",
    "Goal",
    "GoalType",
    "Action",
    "ActionType",
    "Perception",
    "Decision",
    "ReactiveRule",
    # Core - Agent types
    "ReactiveAgent",
    "BDIAgent",
    "HybridAgent",
    "Intention",
    "IntentionStatus",
    "LayerPriority",
    # Communication
    "Transport",
    "MqttConfig",
    "MqttTransport",
    "MockTransport",
    "TopicManager",
    "Mailbox",
    "Message",
    "AgentMessage",
    "AgentStatusMessage",
    "AgentHeartbeatMessage",
    "CommandMessage",
    "ResponseMessage",
    # Organization
    "Role",
    "RoleManager",
    "Responsibility",
    "Permission",
    "PermissionType",
    "Group",
    "GroupManager",
    "HierarchyManager",
    "OrganizationalPosition",
]
