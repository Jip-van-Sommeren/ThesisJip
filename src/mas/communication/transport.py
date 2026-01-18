"""
Transport Interface

Protocol-agnostic transport interface for agent communication.
Implementations: MqttTransport (production), MockTransport (testing).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, Any
from pydantic import BaseModel


@dataclass
class TransportConfig:
    """Base configuration for transport implementations."""
    pass


@dataclass
class MqttConfig(TransportConfig):
    """Configuration for MQTT transport."""
    broker: str = "localhost"
    port: int = 1883
    username: Optional[str] = None
    password: Optional[str] = None
    client_id: Optional[str] = None
    use_tls: bool = False
    ca_certs: Optional[str] = None
    certfile: Optional[str] = None
    keyfile: Optional[str] = None
    keepalive: int = 60
    clean_session: bool = True
    reconnect_on_failure: bool = True
    reconnect_delay: float = 1.0
    max_reconnect_delay: float = 30.0
    qos: int = 1
    extra_options: Dict[str, Any] = field(default_factory=dict)


class Transport(ABC):
    """
    Transport interface for MQTT-based agent communication.

    Provides publish/subscribe messaging abstraction used by agents
    for inter-agent communication and environment interaction.
    """

    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the message broker.

        Returns:
            True if connection successful, False otherwise.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the message broker."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if currently connected to the broker.

        Returns:
            True if connected, False otherwise.
        """
        pass

    @abstractmethod
    def publish(
        self,
        topic: str,
        payload: str,
        qos: int = 1,
        retain: bool = False
    ) -> bool:
        """
        Publish a message to a topic.

        Args:
            topic: The MQTT topic to publish to.
            payload: The message payload (JSON string).
            qos: Quality of service level (0, 1, or 2).
            retain: Whether to retain the message on the broker.

        Returns:
            True if publish successful, False otherwise.
        """
        pass

    @abstractmethod
    def subscribe(
        self,
        topic_pattern: str,
        callback: Callable[[str, str], None],
        qos: int = 1
    ) -> bool:
        """
        Subscribe to a topic pattern with a callback.

        Args:
            topic_pattern: MQTT topic pattern (can include + and # wildcards).
            callback: Function(topic, payload) called when message received.
            qos: Quality of service level.

        Returns:
            True if subscription successful, False otherwise.
        """
        pass

    @abstractmethod
    def unsubscribe(self, topic_pattern: str) -> bool:
        """
        Unsubscribe from a topic pattern.

        Args:
            topic_pattern: The topic pattern to unsubscribe from.

        Returns:
            True if unsubscription successful, False otherwise.
        """
        pass

    def publish_message(
        self,
        topic: str,
        message: BaseModel,
        qos: int = 1,
        retain: bool = False
    ) -> bool:
        """
        Publish a Pydantic model as JSON to a topic.

        Convenience method that serializes the message.

        Args:
            topic: The MQTT topic to publish to.
            message: Pydantic model to serialize and publish.
            qos: Quality of service level.
            retain: Whether to retain the message.

        Returns:
            True if publish successful, False otherwise.
        """
        payload = message.model_dump_json()
        return self.publish(topic, payload, qos, retain)


__all__ = ["Transport", "TransportConfig", "MqttConfig"]
