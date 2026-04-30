"""
MQTT Communication Bridge for Battery Twin

Provides MQTT publish/subscribe functionality with:
- Topic management and templating
- Message validation using Pydantic schemas
- QoS configuration
- Connection management
- Callback routing
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict

import paho.mqtt.client as mqtt
from pydantic import BaseModel

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.battery_twin.communication.message_schemas import MessageFactory
from src.battery_twin.communication.topic_manager import BatteryTopicManager


logger = logging.getLogger(__name__)


@dataclass
class MqttConfig:
    """MQTT broker configuration."""

    broker: str = "localhost"
    port: int = 1883
    qos: int = 1
    keepalive: int = 60
    client_id_prefix: str = "battery_agent_"
    username: Optional[str] = None
    password: Optional[str] = None
    clean_session: bool = True


class MqttBridge:
    """
    MQTT communication bridge for battery twin agents.

    Provides:
    - Topic-based publish/subscribe
    - Message validation with Pydantic schemas
    - Callback routing based on topics
    - Connection management and auto-reconnect
    - QoS support
    """

    def __init__(
        self,
        client_id: str,
        mqtt_config: MqttConfig,
        topic_config_path: str = "src/battery_twin/config/mqtt_topics.yaml",
    ):
        """
        Initialize MQTT bridge.

        Args:
            client_id: Unique client identifier
            mqtt_config: MQTT broker configuration
            topic_config_path: Path to MQTT topics configuration
        """
        self.client_id = client_id
        self.config = mqtt_config
        self.topic_manager = BatteryTopicManager(topic_config_path)

        # MQTT client
        self.client: Optional[mqtt.Client] = None
        self.connected = False
        self.connection_lock = threading.Lock()

        # Callback registry: topic_pattern -> list of callbacks
        self.callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self.callback_lock = threading.Lock()

        # Subscriptions
        self.subscriptions: List[str] = []
        self.subscription_lock = threading.Lock()

        # Statistics
        self.stats = {
            "messages_published": 0,
            "messages_received": 0,
            "publish_errors": 0,
            "validation_errors": 0,
        }

        self._setup_client()

    def _setup_client(self):
        """Setup MQTT client with callbacks."""
        try:
            # Try new API (v2.0+)
            self.client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id=self.client_id,
                clean_session=self.config.clean_session,
            )
        except TypeError:
            # Fall back to old API (v1.x)
            self.client = mqtt.Client(
                client_id=self.client_id,
                clean_session=self.config.clean_session,
            )

        # Set username and password if provided
        if self.config.username and self.config.password:
            self.client.username_pw_set(
                self.config.username, self.config.password
            )

        # Set callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.client.on_publish = self._on_publish

        logger.info(f"MQTT client '{self.client_id}' setup completed")

    def connect(self) -> bool:
        """
        Connect to MQTT broker.

        Returns:
            True if connection successful, False otherwise
        """
        with self.connection_lock:
            if self.connected:
                logger.warning(f"Client '{self.client_id}' already connected")
                return True

            try:
                self.client.connect(
                    self.config.broker, self.config.port, self.config.keepalive
                )
                self.client.loop_start()

                # Wait for connection (timeout 5 seconds)
                timeout = 5.0
                start_time = time.time()
                while (
                    not self.connected and (time.time() - start_time) < timeout
                ):
                    time.sleep(0.1)

                if self.connected:
                    logger.info(
                        f"Connected to MQTT broker at {self.config.broker}:{self.config.port}"
                    )
                    return True
                else:
                    logger.error(f"Connection timeout after {timeout}s")
                    return False

            except Exception as e:
                logger.error(f"Failed to connect to MQTT broker: {e}")
                return False

    def disconnect(self):
        """Disconnect from MQTT broker."""
        with self.connection_lock:
            if not self.connected:
                return

            try:
                self.client.loop_stop()
                self.client.disconnect()
                self.connected = False
                logger.info(f"Disconnected from MQTT broker")
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """Callback when connected to broker."""
        if rc == 0:
            self.connected = True
            logger.info(f"Client '{self.client_id}' connected successfully")

            # Resubscribe to all topics
            self._resubscribe_all()
        else:
            logger.error(f"Connection failed with code {rc}")

    def _on_disconnect(
        self, client, userdata, disconnect_flags, reason_code, properties=None
    ):
        """Callback when disconnected from broker."""
        self.connected = False
        # Handle both v1 and v2 API - reason_code might be rc in v1
        rc = (
            reason_code
            if isinstance(reason_code, int)
            else (reason_code.value if hasattr(reason_code, "value") else 0)
        )
        if rc != 0:
            logger.warning(
                f"Unexpected disconnect (rc={rc}), will auto-reconnect"
            )
        else:
            logger.info("Clean disconnect from broker")

    def _on_message(self, client, userdata, msg):
        """Callback when message received."""
        try:
            self.stats["messages_received"] += 1
            topic = msg.topic
            payload = msg.payload.decode("utf-8")

            logger.debug(
                f"Received message on topic '{topic}': {payload[:100]}..."
            )

            # Route message to registered callbacks
            self._route_message(topic, payload)

        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def _on_publish(
        self, client, userdata, mid, reason_code=None, properties=None
    ):
        """
        Callback when message published.

        Compatible with both paho-mqtt v1.x and v2.x:
        - v1.x: (client, userdata, mid, properties=None)
        - v2.x: (client, userdata, mid, reason_code=None, properties=None)
        """
        logger.debug(f"Message {mid} published")

    def _resubscribe_all(self):
        """Resubscribe to all topics after reconnection."""
        with self.subscription_lock:
            for topic in self.subscriptions:
                self.client.subscribe(topic, qos=self.config.qos)
                logger.debug(f"Resubscribed to '{topic}'")

    def _route_message(self, topic: str, payload: str):
        """
        Route received message to registered callbacks.

        Args:
            topic: MQTT topic
            payload: Message payload (JSON string)
        """
        with self.callback_lock:
            # Find matching callbacks
            matched_callbacks = []

            for pattern, callbacks in self.callbacks.items():
                if mqtt.topic_matches_sub(pattern, topic):
                    matched_callbacks.extend(callbacks)

            if not matched_callbacks:
                logger.debug(f"No callbacks registered for topic '{topic}'")
                return

            # Call all matching callbacks
            for callback in matched_callbacks:
                try:
                    callback(topic, payload)
                except Exception as e:
                    logger.error(f"Error in callback for topic '{topic}': {e}")

    def publish(
        self,
        topic_name: str,
        message: BaseModel,
        qos: Optional[int] = None,
        **topic_vars,
    ) -> bool:
        """
        Publish a message to a topic.

        Args:
            topic_name: Topic name (e.g., "raw_telemetry")
            message: Pydantic message instance
            qos: QoS level (defaults to config qos)
            **topic_vars: Variables for topic formatting (e.g., battery_id="B0005")

        Returns:
            True if publish successful, False otherwise

        Example:
            >>> bridge.publish(
            ...     "raw_telemetry",
            ...     TelemetryMessage(battery_id="B0005", voltage=3.8, ...),
            ...     battery_id="B0005"
            ... )
        """
        if not self.connected:
            logger.error("Cannot publish: Not connected to broker")
            return False

        try:
            # Format topic
            topic = self.topic_manager.get_topic(topic_name, **topic_vars)

            # Convert message to JSON
            payload = MessageFactory.to_json(message)

            # Publish
            qos_level = qos if qos is not None else self.config.qos
            result = self.client.publish(topic, payload, qos=qos_level)

            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                self.stats["messages_published"] += 1
                logger.debug(f"Published to '{topic}': {payload[:100]}...")
                return True
            else:
                self.stats["publish_errors"] += 1
                logger.error(f"Publish failed with rc={result.rc}")
                return False

        except Exception as e:
            self.stats["publish_errors"] += 1
            logger.error(f"Error publishing message: {e}")
            return False

    def publish_raw(
        self, topic: str, payload: str, qos: Optional[int] = None
    ) -> bool:
        """
        Publish raw payload to topic (no validation).

        Args:
            topic: Full MQTT topic string
            payload: Message payload (string)
            qos: QoS level

        Returns:
            True if publish successful, False otherwise
        """
        if not self.connected:
            logger.error("Cannot publish: Not connected to broker")
            return False

        try:
            qos_level = qos if qos is not None else self.config.qos
            result = self.client.publish(topic, payload, qos=qos_level)

            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                self.stats["messages_published"] += 1
                return True
            else:
                self.stats["publish_errors"] += 1
                return False

        except Exception as e:
            self.stats["publish_errors"] += 1
            logger.error(f"Error publishing raw message: {e}")
            return False

    def subscribe(
        self,
        topic_name: str,
        callback: Callable[[str, str], None],
        qos: Optional[int] = None,
        **topic_vars,
    ) -> bool:
        """
        Subscribe to a topic with callback.

        Args:
            topic_name: Topic name or pattern (e.g., "raw_telemetry")
            callback: Callback function(topic: str, payload: str)
            qos: QoS level
            **topic_vars: Variables for topic formatting. Use None for wildcards.

        Returns:
            True if subscribe successful, False otherwise

        Example:
            >>> def on_telemetry(topic, payload):
            ...     print(f"Received: {payload}")
            >>> bridge.subscribe("raw_telemetry", on_telemetry, battery_id=None)
        """
        if not self.connected:
            logger.error("Cannot subscribe: Not connected to broker")
            return False

        try:
            # Get subscription pattern (with wildcards if needed)
            topic_pattern = self.topic_manager.get_subscription_pattern(
                topic_name, **topic_vars
            )

            # Subscribe to topic
            qos_level = qos if qos is not None else self.config.qos
            result, mid = self.client.subscribe(topic_pattern, qos=qos_level)

            if result == mqtt.MQTT_ERR_SUCCESS:
                # Register callback
                with self.callback_lock:
                    self.callbacks[topic_pattern].append(callback)

                # Remember subscription
                with self.subscription_lock:
                    if topic_pattern not in self.subscriptions:
                        self.subscriptions.append(topic_pattern)

                logger.info(f"Subscribed to '{topic_pattern}'")
                return True
            else:
                logger.error(f"Subscribe failed with rc={result}")
                return False

        except Exception as e:
            logger.error(f"Error subscribing to topic: {e}")
            return False

    def subscribe_raw(
        self,
        topic_pattern: str,
        callback: Callable[[str, str], None],
        qos: Optional[int] = None,
    ) -> bool:
        """
        Subscribe to raw topic pattern (no template lookup).

        Args:
            topic_pattern: MQTT topic pattern (e.g., "battery/+/raw")
            callback: Callback function
            qos: QoS level

        Returns:
            True if subscribe successful, False otherwise
        """
        if not self.connected:
            logger.error("Cannot subscribe: Not connected to broker")
            return False

        try:
            qos_level = qos if qos is not None else self.config.qos
            result, mid = self.client.subscribe(topic_pattern, qos=qos_level)

            if result == mqtt.MQTT_ERR_SUCCESS:
                with self.callback_lock:
                    self.callbacks[topic_pattern].append(callback)

                with self.subscription_lock:
                    if topic_pattern not in self.subscriptions:
                        self.subscriptions.append(topic_pattern)

                logger.info(f"Subscribed to '{topic_pattern}'")
                return True
            else:
                logger.error(f"Subscribe failed with rc={result}")
                return False

        except Exception as e:
            logger.error(f"Error subscribing: {e}")
            return False

    def unsubscribe(self, topic_name: str, **topic_vars) -> bool:
        """Unsubscribe from a topic."""
        if not self.connected:
            return False

        try:
            topic_pattern = self.topic_manager.get_subscription_pattern(
                topic_name, **topic_vars
            )

            result, mid = self.client.unsubscribe(topic_pattern)

            if result == mqtt.MQTT_ERR_SUCCESS:
                with self.callback_lock:
                    if topic_pattern in self.callbacks:
                        del self.callbacks[topic_pattern]

                with self.subscription_lock:
                    if topic_pattern in self.subscriptions:
                        self.subscriptions.remove(topic_pattern)

                logger.info(f"Unsubscribed from '{topic_pattern}'")
                return True
            else:
                logger.error(f"Unsubscribe failed with rc={result}")
                return False

        except Exception as e:
            logger.error(f"Error unsubscribing: {e}")
            return False

    def get_stats(self) -> Dict[str, int]:
        """Get communication statistics."""
        return self.stats.copy()

    def is_connected(self) -> bool:
        """Check if connected to broker."""
        return self.connected

    def wait_for_connection(self, timeout: float = 10.0) -> bool:
        """
        Wait for connection to be established.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if connected, False if timeout
        """
        start_time = time.time()
        while not self.connected and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        return self.connected


__all__ = [
    "MqttBridge",
    "MqttConfig",
]
