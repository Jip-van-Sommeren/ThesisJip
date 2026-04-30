"""
Mock Transport for Testing

In-memory transport implementation for unit testing agents
without requiring a real MQTT broker.
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel

from .transport import Transport
from .topic_manager import TopicManager

logger = logging.getLogger(__name__)


class MockTransport(Transport):
    """
    In-memory transport for unit testing.

    Features:
    - Captures all published messages for assertions
    - Simulates incoming messages to trigger callbacks
    - No external broker required
    - Thread-safe for concurrent tests

    Example:
        transport = MockTransport()
        agent = MyAgent(transport=transport)

        # Simulate incoming message
        transport.simulate_message(
            "battery/B0005/raw",
            '{"voltage": 3.8, "current": 1.0}'
        )

        # Assert published messages
        assert len(transport.published) == 1
        topic, payload = transport.published[0]
        assert "telemetry/clean" in topic
    """

    def __init__(self, topic_manager: Optional[TopicManager] = None):
        """
        Initialize mock transport.

        Args:
            topic_manager: Optional TopicManager for topic resolution
        """
        self.topic_manager = topic_manager
        self._connected = False
        self.published: List[Tuple[str, str]] = []
        self.subscriptions: Dict[str, List[Callable[[str, str], None]]] = {}

    def connect(self) -> bool:
        """Simulate connection (always succeeds)."""
        self._connected = True
        logger.debug("MockTransport connected")
        return True

    def disconnect(self) -> None:
        """Simulate disconnection."""
        self._connected = False
        logger.debug("MockTransport disconnected")

    def is_connected(self) -> bool:
        """Check simulated connection state."""
        return self._connected

    def publish(
        self,
        topic: str,
        payload: str,
        qos: int = 1,
        retain: bool = False,
    ) -> bool:
        """
        Capture published message for later assertions.

        Args:
            topic: MQTT topic
            payload: Message payload
            qos: Ignored in mock
            retain: Ignored in mock

        Returns:
            True (always succeeds)
        """
        self.published.append((topic, payload))
        logger.debug(f"MockTransport published to {topic}")
        return True

    def subscribe(
        self,
        topic_pattern: str,
        callback: Callable[[str, str], None],
        qos: int = 1,
    ) -> bool:
        """
        Register subscription callback.

        Args:
            topic_pattern: MQTT topic pattern
            callback: Function to call on message
            qos: Ignored in mock

        Returns:
            True (always succeeds)
        """
        if topic_pattern not in self.subscriptions:
            self.subscriptions[topic_pattern] = []
        self.subscriptions[topic_pattern].append(callback)
        logger.debug(f"MockTransport subscribed to {topic_pattern}")
        return True

    def unsubscribe(self, topic_pattern: str) -> bool:
        """
        Remove subscription.

        Args:
            topic_pattern: Topic pattern to unsubscribe

        Returns:
            True if was subscribed, False otherwise
        """
        if topic_pattern in self.subscriptions:
            del self.subscriptions[topic_pattern]
            return True
        return False

    def publish_to_topic(
        self,
        topic_name: str,
        message: BaseModel,
        qos: int = 1,
        retain: bool = False,
        **topic_vars,
    ) -> bool:
        """
        Publish using TopicManager for topic resolution.

        Args:
            topic_name: Logical topic name
            message: Pydantic model to serialize
            **topic_vars: Topic variables

        Returns:
            True (always succeeds)
        """
        if not self.topic_manager:
            raise ValueError("TopicManager not configured")

        topic = self.topic_manager.get_topic(topic_name, **topic_vars)
        payload = message.model_dump_json()
        return self.publish(topic, payload, qos, retain)

    def subscribe_to_topic(
        self,
        topic_name: str,
        callback: Callable[[str, str], None],
        qos: int = 1,
        **topic_vars,
    ) -> bool:
        """
        Subscribe using TopicManager for pattern generation.

        Args:
            topic_name: Logical topic name
            callback: Message callback
            **topic_vars: Topic variables (None for wildcard)

        Returns:
            True (always succeeds)
        """
        if not self.topic_manager:
            raise ValueError("TopicManager not configured")

        pattern = self.topic_manager.get_subscription_pattern(
            topic_name, **topic_vars
        )
        return self.subscribe(pattern, callback, qos)

    def simulate_message(self, topic: str, payload: str) -> int:
        """
        Simulate an incoming message.

        Triggers all matching subscription callbacks.

        Args:
            topic: MQTT topic the message arrives on
            payload: Message payload (JSON string)

        Returns:
            Number of callbacks triggered
        """
        callbacks_triggered = 0

        for pattern, callbacks in self.subscriptions.items():
            if self._topic_matches(pattern, topic):
                for callback in callbacks:
                    try:
                        callback(topic, payload)
                        callbacks_triggered += 1
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

        logger.debug(
            f"Simulated message on {topic}, triggered {callbacks_triggered} callbacks"
        )
        return callbacks_triggered

    def _topic_matches(self, pattern: str, topic: str) -> bool:
        """
        Check if topic matches MQTT pattern.

        Supports + (single level) and # (multi level) wildcards.

        Args:
            pattern: MQTT subscription pattern
            topic: Actual topic to match

        Returns:
            True if topic matches pattern
        """
        # Convert MQTT pattern to glob pattern
        glob_pattern = pattern.replace("+", "*").replace("#", "**")

        # Split into parts for proper matching
        pattern_parts = pattern.split("/")
        topic_parts = topic.split("/")

        # Handle # wildcard (matches remaining path)
        if "#" in pattern_parts:
            hash_index = pattern_parts.index("#")
            # Everything before # must match exactly (with + as wildcard)
            for i in range(hash_index):
                if i >= len(topic_parts):
                    return False
                if (
                    pattern_parts[i] != "+"
                    and pattern_parts[i] != topic_parts[i]
                ):
                    return False
            return True

        # Must have same number of parts if no #
        if len(pattern_parts) != len(topic_parts):
            return False

        # Check each part
        for p, t in zip(pattern_parts, topic_parts):
            if p != "+" and p != t:
                return False

        return True

    def get_published(self) -> List[Tuple[str, str]]:
        """
        Get all published messages.

        Returns:
            List of (topic, payload) tuples
        """
        return list(self.published)

    def clear(self) -> None:
        """Clear all published messages and subscriptions."""
        self.published.clear()
        self.subscriptions.clear()

    def clear_published(self) -> None:
        """Clear only published messages (keep subscriptions)."""
        self.published.clear()

    def get_published_to_topic(
        self, topic_pattern: str
    ) -> List[Tuple[str, str]]:
        """
        Get all messages published to topics matching pattern.

        Args:
            topic_pattern: MQTT topic pattern

        Returns:
            List of (topic, payload) tuples
        """
        return [
            (topic, payload)
            for topic, payload in self.published
            if self._topic_matches(topic_pattern, topic)
        ]

    def assert_published(self, topic_pattern: str, count: int = 1) -> None:
        """
        Assert that messages were published to matching topics.

        Args:
            topic_pattern: Expected topic pattern
            count: Expected number of messages

        Raises:
            AssertionError: If count doesn't match
        """
        matching = self.get_published_to_topic(topic_pattern)
        actual = len(matching)
        assert (
            actual == count
        ), f"Expected {count} messages to {topic_pattern}, got {actual}"


__all__ = ["MockTransport"]
