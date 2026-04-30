"""
MQTT Transport Implementation

Production transport using paho-mqtt for agent communication.
Supports QoS 0/1/2, TLS, authentication, and automatic reconnection.
"""

import logging
import threading
import time
import uuid
from typing import Callable, Dict, List, Optional, Tuple

import paho.mqtt.client as mqtt
from pydantic import BaseModel

from .transport import Transport, MqttConfig
from .topic_manager import TopicManager

logger = logging.getLogger(__name__)


class MqttTransport(Transport):
    """
    MQTT transport implementation using paho-mqtt.

    Features:
    - QoS 0/1/2 support
    - TLS encryption
    - Username/password authentication
    - Automatic reconnection
    - Integration with TopicManager for topic resolution

    Example:
        config = MqttConfig(broker="localhost", port=1883)
        topic_manager = TopicManager("config/mqtt_topics.yaml")
        transport = MqttTransport(config, topic_manager)

        transport.connect()
        transport.subscribe_to_topic("raw_telemetry", callback, battery_id=None)
        transport.publish_to_topic("clean_telemetry", message, battery_id="B0005")
        transport.disconnect()
    """

    def __init__(
        self,
        config: MqttConfig,
        topic_manager: Optional[TopicManager] = None,
    ):
        """
        Initialize MQTT transport.

        Args:
            config: MQTT configuration (broker, port, credentials, etc.)
            topic_manager: Optional TopicManager for topic resolution
        """
        self.config = config
        self.topic_manager = topic_manager

        # Generate client ID if not provided
        client_id = config.client_id or f"mas-{uuid.uuid4().hex[:8]}"

        # Initialize MQTT client
        self.client = mqtt.Client(
            client_id=client_id,
            clean_session=config.clean_session,
            protocol=mqtt.MQTTv311,
        )

        # Setup callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

        # Subscription management
        self._subscriptions: Dict[str, List[Callable[[str, str], None]]] = {}
        self._pending_subscriptions: List[Tuple[str, int]] = []
        self._lock = threading.Lock()

        # Connection state
        self._connected = False
        self._connecting = False

        # Configure authentication
        if config.username:
            self.client.username_pw_set(config.username, config.password)

        # Configure TLS
        if config.use_tls:
            self.client.tls_set(
                ca_certs=config.ca_certs,
                certfile=config.certfile,
                keyfile=config.keyfile,
            )

    def connect(self) -> bool:
        """
        Connect to the MQTT broker.

        Returns:
            True if connection successful or already connected.
        """
        if self._connected:
            return True

        if self._connecting:
            logger.warning("Connection already in progress")
            return False

        try:
            self._connecting = True
            logger.info(
                f"Connecting to MQTT broker {self.config.broker}:{self.config.port}"
            )

            self.client.connect(
                self.config.broker,
                self.config.port,
                keepalive=self.config.keepalive,
            )

            # Start network loop in background thread
            self.client.loop_start()

            # Wait for connection with timeout
            timeout = 10.0
            start_time = time.time()
            while not self._connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)

            if not self._connected:
                logger.error("Connection timeout")
                self._connecting = False
                return False

            logger.info("Connected to MQTT broker")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            self._connecting = False
            return False

    def disconnect(self) -> None:
        """Disconnect from the MQTT broker."""
        if self._connected or self._connecting:
            logger.info("Disconnecting from MQTT broker")
            self.client.loop_stop()
            self.client.disconnect()
            self._connected = False
            self._connecting = False

    def is_connected(self) -> bool:
        """Check if currently connected to the broker."""
        return self._connected

    def publish(
        self,
        topic: str,
        payload: str,
        qos: int = 1,
        retain: bool = False,
    ) -> bool:
        """
        Publish a message to a topic.

        Args:
            topic: MQTT topic to publish to
            payload: Message payload (JSON string)
            qos: Quality of service (0, 1, or 2)
            retain: Whether to retain the message

        Returns:
            True if publish successful
        """
        if not self._connected:
            logger.warning("Not connected to broker, cannot publish")
            return False

        try:
            result = self.client.publish(
                topic, payload, qos=qos, retain=retain
            )
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.debug(f"Published to {topic}")
                return True
            else:
                logger.error(f"Publish failed with rc={result.rc}")
                return False
        except Exception as e:
            logger.error(f"Publish error: {e}")
            return False

    def subscribe(
        self,
        topic_pattern: str,
        callback: Callable[[str, str], None],
        qos: int = 1,
    ) -> bool:
        """
        Subscribe to a topic pattern with a callback.

        Args:
            topic_pattern: MQTT topic pattern (can include + and # wildcards)
            callback: Function(topic, payload) called when message received
            qos: Quality of service level

        Returns:
            True if subscription successful
        """
        with self._lock:
            # Register callback
            if topic_pattern not in self._subscriptions:
                self._subscriptions[topic_pattern] = []
            self._subscriptions[topic_pattern].append(callback)

            # Subscribe if connected, otherwise queue
            if self._connected:
                try:
                    result, _ = self.client.subscribe(topic_pattern, qos)
                    if result == mqtt.MQTT_ERR_SUCCESS:
                        logger.info(f"Subscribed to {topic_pattern}")
                        return True
                    else:
                        logger.error(f"Subscribe failed with rc={result}")
                        return False
                except Exception as e:
                    logger.error(f"Subscribe error: {e}")
                    return False
            else:
                # Queue for later subscription
                self._pending_subscriptions.append((topic_pattern, qos))
                logger.debug(f"Queued subscription to {topic_pattern}")
                return True

    def unsubscribe(self, topic_pattern: str) -> bool:
        """
        Unsubscribe from a topic pattern.

        Args:
            topic_pattern: The topic pattern to unsubscribe from

        Returns:
            True if unsubscription successful
        """
        with self._lock:
            if topic_pattern in self._subscriptions:
                del self._subscriptions[topic_pattern]

            if self._connected:
                try:
                    result, _ = self.client.unsubscribe(topic_pattern)
                    return result == mqtt.MQTT_ERR_SUCCESS
                except Exception as e:
                    logger.error(f"Unsubscribe error: {e}")
                    return False
            return True

    def publish_to_topic(
        self,
        topic_name: str,
        message: BaseModel,
        qos: int = 1,
        retain: bool = False,
        **topic_vars,
    ) -> bool:
        """
        Publish a message using TopicManager for topic resolution.

        Args:
            topic_name: Logical topic name from configuration
            message: Pydantic model to serialize and publish
            qos: Quality of service level
            retain: Whether to retain the message
            **topic_vars: Variables for topic formatting (e.g., battery_id="B0005")

        Returns:
            True if publish successful
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
        Subscribe to a topic using TopicManager for pattern generation.

        Args:
            topic_name: Logical topic name from configuration
            callback: Function(topic, payload) called when message received
            qos: Quality of service level
            **topic_vars: Variables for topic formatting (None for wildcard)

        Returns:
            True if subscription successful
        """
        if not self.topic_manager:
            raise ValueError("TopicManager not configured")

        pattern = self.topic_manager.get_subscription_pattern(
            topic_name, **topic_vars
        )
        return self.subscribe(pattern, callback, qos)

    def _on_connect(self, client, userdata, flags, rc):
        """Handle connection callback."""
        if rc == 0:
            self._connected = True
            self._connecting = False
            logger.info("MQTT connection established")

            # Resubscribe to pending subscriptions
            with self._lock:
                for pattern, qos in self._pending_subscriptions:
                    self.client.subscribe(pattern, qos)
                    logger.debug(f"Resubscribed to {pattern}")
                self._pending_subscriptions.clear()

                # Resubscribe to existing subscriptions (for reconnection)
                for pattern in self._subscriptions.keys():
                    self.client.subscribe(pattern, self.config.qos)
        else:
            logger.error(f"MQTT connection failed with rc={rc}")
            self._connected = False
            self._connecting = False

    def _on_disconnect(self, client, userdata, rc):
        """Handle disconnection callback."""
        self._connected = False
        if rc != 0:
            logger.warning(f"Unexpected disconnect (rc={rc})")
            if self.config.reconnect_on_failure:
                logger.info("Will attempt reconnection...")
        else:
            logger.info("Disconnected from broker")

    def _on_message(self, client, userdata, msg):
        """Handle incoming message callback."""
        topic = msg.topic
        payload = msg.payload.decode("utf-8")

        logger.debug(f"Received message on {topic}")

        # Find matching callbacks
        with self._lock:
            for pattern, callbacks in self._subscriptions.items():
                if mqtt.topic_matches_sub(pattern, topic):
                    for callback in callbacks:
                        try:
                            callback(topic, payload)
                        except Exception as e:
                            logger.error(f"Callback error for {topic}: {e}")


__all__ = ["MqttTransport"]
