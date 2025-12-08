"""
Transport abstraction and MQTT bridge adapter.

Defines a light-weight transport interface and an adapter that wraps the
existing Battery Twin MqttBridge so twins can depend on a generic API.
"""

from __future__ import annotations

from typing import Any, Callable, Optional


class Transport:
    """Generic transport interface (publish/subscribe/connect)."""

    def connect(self) -> bool:  # pragma: no cover - simple pass-through
        raise NotImplementedError

    def disconnect(self) -> None:  # pragma: no cover - simple pass-through
        raise NotImplementedError

    def is_connected(self) -> bool:
        raise NotImplementedError

    def publish(self, topic_name: str, message: Any, **topic_vars) -> bool:
        raise NotImplementedError

    def publish_raw(self, topic: str, payload: str, qos: Optional[int] = None) -> bool:
        raise NotImplementedError

    def subscribe(self, topic_name: str, callback: Callable[[str, str], None], **topic_vars) -> bool:
        raise NotImplementedError

    def subscribe_raw(self, topic_pattern: str, callback: Callable[[str, str], None]) -> bool:
        raise NotImplementedError

    @property
    def topic_manager(self) -> Any:  # twin-specific manager
        return None


class MqttBridgeTransport(Transport):
    """Adapter over Battery Twin MqttBridge that implements Transport."""

    def __init__(self, bridge: Any):
        self.bridge = bridge

    def connect(self) -> bool:
        return self.bridge.connect()

    def disconnect(self) -> None:
        return self.bridge.disconnect()

    def is_connected(self) -> bool:
        return self.bridge.is_connected()

    def publish(self, topic_name: str, message: Any, **topic_vars) -> bool:
        return self.bridge.publish(topic_name, message, **topic_vars)

    def publish_raw(self, topic: str, payload: str, qos: Optional[int] = None) -> bool:
        return self.bridge.publish_raw(topic, payload, qos)

    def subscribe(self, topic_name: str, callback: Callable[[str, str], None], **topic_vars) -> bool:
        return self.bridge.subscribe(topic_name, callback, **topic_vars)

    def subscribe_raw(self, topic_pattern: str, callback: Callable[[str, str], None]) -> bool:
        return self.bridge.subscribe_raw(topic_pattern, callback)

    @property
    def topic_manager(self) -> Any:
        return getattr(self.bridge, "topic_manager", None)


__all__ = ["Transport", "MqttBridgeTransport"]

