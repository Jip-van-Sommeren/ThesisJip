"""
MQTT Communication Package

MQTT-based implementation of the formal communication model from the thesis.
Provides the same interface as REST, gRPC, and Kafka communication protocols.
"""

from .mqtt_communication import (
    MqttMessageType,
    MqttMessage,
    MqttMailbox,
    MqttCommunicationTopology,
    MqttService,
    MqttServer,
    MqttCommunicatingAgent,
    MqttCommunicationService,
)

from .mqtt_communication_agent import (
    MqttCommunicationAction,
    ExtendedMqttCommunicatingAgent,
    MqttCommunicationEnvironment,
)

__all__ = [
    # Core communication components
    "MqttMessageType",
    "MqttMessage",
    "MqttMailbox",
    "MqttCommunicationTopology",
    "MqttService",
    "MqttServer",
    "MqttCommunicatingAgent",
    "MqttCommunicationService",

    # Agent integration components
    "MqttCommunicationAction",
    "ExtendedMqttCommunicatingAgent",
    "MqttCommunicationEnvironment",
]