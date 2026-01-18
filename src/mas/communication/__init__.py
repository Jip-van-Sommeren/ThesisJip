"""
MAS Communication Layer

MQTT-based communication infrastructure for agent messaging.

Components:
- Transport: Abstract transport interface
- MqttTransport: Production MQTT implementation
- MockTransport: In-memory transport for testing
- TopicManager: YAML-based topic template management
- Mailbox: Agent message buffer
- Message schemas: Base message types
"""

from .transport import Transport, TransportConfig, MqttConfig
from .mqtt_transport import MqttTransport
from .mock_transport import MockTransport
from .topic_manager import TopicManager
from .mailbox import Mailbox, Message
from .message import (
    AgentMessage,
    AgentStatusMessage,
    AgentHeartbeatMessage,
    CommandMessage,
    ResponseMessage,
)

__all__ = [
    # Transport
    "Transport",
    "TransportConfig",
    "MqttConfig",
    "MqttTransport",
    "MockTransport",
    # Topic management
    "TopicManager",
    # Mailbox
    "Mailbox",
    "Message",
    # Message schemas
    "AgentMessage",
    "AgentStatusMessage",
    "AgentHeartbeatMessage",
    "CommandMessage",
    "ResponseMessage",
]
