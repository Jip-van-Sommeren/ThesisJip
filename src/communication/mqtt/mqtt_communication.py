"""
MQTT-based Communication Model Implementation
Based on the formal communication definition from the thesis.

Implements the same formal model as REST, gRPC, and Kafka but using MQTT:
- Message Space (M): Set of all possible messages
- Communication Topology (Comm): Agent-to-agent communication via MQTT topics
- Mailboxes (MB_i): Message buffers implemented as MQTT subscribers
- Delivery Function: Message transmission via MQTT publish/subscribe
- Communication Actions: send(i,j,m) operations via MQTT messaging
"""

import json
import threading
import time
import uuid
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import logging
import queue

from communication.base_communication import (
    MessageType as BaseMessageType,
    LatencyMode,
)
import paho.mqtt.client as mqtt


class MqttMessageType(Enum):
    """MQTT Message types mapping to thesis definition."""

    INFORM = "inform"
    REQUEST = "request"
    REPLY = "reply"
    BROADCAST = "broadcast"
    ERROR = "error"
    ACK = "ack"  # Acknowledgment for end-to-end latency measurement

    @classmethod
    def from_base_type(cls, base_type: BaseMessageType) -> "MqttMessageType":
        """Convert base message type to MQTT message type."""
        mapping = {
            BaseMessageType.INFORM: cls.INFORM,
            BaseMessageType.REQUEST: cls.REQUEST,
            BaseMessageType.REPLY: cls.REPLY,
            BaseMessageType.BROADCAST: cls.BROADCAST,
            BaseMessageType.ERROR: cls.ERROR,
            BaseMessageType.ACK: cls.ACK,
        }
        return mapping[base_type]

    def to_base_type(self) -> BaseMessageType:
        """Convert MQTT message type to base message type."""
        mapping = {
            self.INFORM: BaseMessageType.INFORM,
            self.REQUEST: BaseMessageType.REQUEST,
            self.REPLY: BaseMessageType.REPLY,
            self.BROADCAST: BaseMessageType.BROADCAST,
            self.ERROR: BaseMessageType.ERROR,
        }
        return mapping[self]


@dataclass
class MqttMessage:
    """
    Message representation for MQTT communication.
    Implements the message space M from the thesis.
    """

    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    sender_id: str = ""
    receiver_id: str = ""
    message_type: MqttMessageType = MqttMessageType.INFORM
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    reply_to: Optional[str] = None

    def to_json(self) -> str:
        """Convert to JSON for MQTT serialization."""
        return json.dumps(
            {
                "message_id": self.message_id,
                "sender_id": self.sender_id,
                "receiver_id": self.receiver_id,
                "message_type": self.message_type.value,
                "content": self.content,
                "timestamp": self.timestamp,
                "reply_to": self.reply_to,
            }
        )

    @classmethod
    def from_json(cls, json_str: str) -> "MqttMessage":
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls(
            message_id=data["message_id"],
            sender_id=data["sender_id"],
            receiver_id=data["receiver_id"],
            message_type=MqttMessageType(data["message_type"]),
            content=data["content"],
            timestamp=data["timestamp"],
            reply_to=data.get("reply_to"),
        )


class MqttMailbox:
    """
    MB_i: Mailbox for agent i using MQTT subscriber.
    Implements asynchronous message buffering via MQTT topics.
    """

    def __init__(
        self,
        agent_id: str,
        mqtt_config: Dict[str, Any],
        max_size: int = 1000,
        broadcast_topic: Optional[str] = None,
    ):
        self.agent_id = agent_id
        self.mqtt_config = mqtt_config
        self.max_size = max_size
        self.messages: deque = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.client = None
        self.running = False

        # Topic for this agent's mailbox
        self.topic = f"agent/mailbox/{agent_id}"
        self.broadcast_topic = broadcast_topic

        self._setup_subscriber()

    def _setup_subscriber(self):
        """Setup MQTT subscriber for this agent's mailbox."""
        if not mqtt:
            logging.warning(
                f"MQTT library not available for {self.agent_id}. \
                    Using fallback mode."
            )
            return

        try:
            # Support both old and new paho-mqtt API versions
            try:
                # Try new API (v2.0+)
                self.client = mqtt.Client(
                    mqtt.CallbackAPIVersion.VERSION1,
                    client_id=f"agent_{self.agent_id}_mailbox",
                    clean_session=True
                )
            except (AttributeError, TypeError):
                # Fall back to old API
                self.client = mqtt.Client(
                    client_id=f"agent_{self.agent_id}_mailbox",
                    clean_session=True
                )

            # Set up callbacks
            self.client.on_connect = self._on_connect
            self.client.on_message = self._on_message
            self.client.on_disconnect = self._on_disconnect

            # Connect to broker
            broker_host = self.mqtt_config.get("broker_host", "localhost")
            broker_port = self.mqtt_config.get("broker_port", 1883)

            self.client.connect(broker_host, broker_port, keepalive=60)
            self.client.loop_start()

            # Wait briefly for connection to establish
            time.sleep(0.2)
            self.running = True

        except Exception as e:
            logging.warning(
                f"Could not setup MQTT subscriber for {self.agent_id}: {e}"
            )

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """Callback for successful MQTT connection."""
        # Handle both v1 and v2 API (v2 adds properties parameter)
        if rc == 0:
            topics = [(self.topic, 1)]
            if self.broadcast_topic:
                topics.append((self.broadcast_topic, 1))
            client.subscribe(topics)
            logging.info(f"Agent {self.agent_id} subscribed to {self.topic}")
        else:
            logging.error(f"Failed to connect agent {self.agent_id}: {rc}")

    def _on_message(self, client, userdata, msg):
        """Callback for receiving MQTT messages."""
        try:
            message_data = msg.payload.decode("utf-8")
            mqtt_message = MqttMessage.from_json(message_data)

            if (
                self.broadcast_topic
                and msg.topic == self.broadcast_topic
                and (not mqtt_message.receiver_id or mqtt_message.receiver_id == "*")
            ):
                mqtt_message.receiver_id = self.agent_id

            with self.lock:
                self.messages.append(mqtt_message)

        except Exception as e:
            logging.warning(
                f"Error processing message for {self.agent_id}: {e}"
            )

    def _on_disconnect(self, client, userdata, rc, properties=None):
        """Callback for MQTT disconnection."""
        # Handle both v1 and v2 API (v2 adds properties parameter)
        # Common disconnect codes:
        # 0 = Normal disconnect
        # 1 = Incorrect protocol version
        # 2 = Invalid client identifier
        # 3 = Server unavailable
        # 4 = Bad username or password
        # 5 = Not authorized
        # 7 = No matching subscribers (in v5)
        # 16 = Connection lost

        if rc != 0:
            reason = {
                1: "Incorrect protocol version",
                2: "Invalid client identifier",
                3: "Server unavailable",
                4: "Bad username or password",
                5: "Not authorized",
                7: "No matching subscribers",
                16: "Connection lost",
            }.get(rc, f"Unknown error code {rc}")

            logging.warning(
                f"Unexpected disconnection for agent {self.agent_id}: {reason}"
            )

            # Only try to reconnect for transient errors
            if rc in [3, 16] and self.running:
                try:
                    logging.info(f"Attempting to reconnect agent {self.agent_id}...")
                    time.sleep(0.5)
                    client.reconnect()
                except Exception as e:
                    logging.error(f"Failed to reconnect agent {self.agent_id}: {e}")

    def add_message(self, message: MqttMessage):
        """Add message to mailbox (for direct delivery when MQTT
        not available)."""
        with self.lock:
            self.messages.append(message)

    def get_messages(self, clear: bool = True) -> List[MqttMessage]:
        """Get all messages from mailbox, optionally clearing it."""
        with self.lock:
            messages = list(self.messages)
            if clear:
                self.messages.clear()
            return messages

    def peek_messages(self) -> List[MqttMessage]:
        """Inspect messages without removing them."""
        with self.lock:
            return list(self.messages)

    def size(self) -> int:
        """Get number of messages in mailbox."""
        with self.lock:
            return len(self.messages)

    def close(self):
        """Close the MQTT subscriber."""
        self.running = False
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()


class MqttCommunicationTopology:
    """
    Comm ⊆ A × A: Communication topology for MQTT.
    Uses MQTT topics to represent communication links.
    """

    def __init__(self):
        self.links: Set[Tuple[str, str]] = set()
        self.lock = threading.Lock()

    def add_link(self, sender_id: str, receiver_id: str):
        """Add communication link (i,j) ∈ Comm."""
        with self.lock:
            self.links.add((sender_id, receiver_id))

    def remove_link(self, sender_id: str, receiver_id: str):
        """Remove communication link."""
        with self.lock:
            self.links.discard((sender_id, receiver_id))

    def can_communicate(self, sender_id: str, receiver_id: str) -> bool:
        """Check if sender can communicate with receiver."""
        with self.lock:
            return (sender_id, receiver_id) in self.links

    def get_reachable_agents(self, sender_id: str) -> Set[str]:
        """Get all agents that sender can communicate with."""
        with self.lock:
            return {
                receiver
                for sender, receiver in self.links
                if sender == sender_id
            }

    def create_fully_connected(self, agent_ids: List[str]):
        """Create fully connected topology: Comm = A × A."""
        with self.lock:
            for sender in agent_ids:
                for receiver in agent_ids:
                    if sender != receiver:
                        self.links.add((sender, receiver))


class MqttService:
    """
    MQTT communication service helper.
    Manages MQTT client connections and topic operations.
    """

    def __init__(self, mqtt_config: Dict[str, Any] = None):
        self.mqtt_config = mqtt_config or {
            "broker_host": "localhost",
            "broker_port": 1883,
            "keepalive": 60,
            "qos": 1,
        }
        self.publisher_client = None
        self.is_running = False

        self._setup_publisher()

    def _setup_publisher(self):
        """Setup MQTT publisher client."""
        if not mqtt:
            logging.warning("MQTT library not available. Using fallback mode.")
            return

        try:
            # Support both old and new paho-mqtt API versions
            try:
                # Try new API (v2.0+)
                self.publisher_client = mqtt.Client(
                    mqtt.CallbackAPIVersion.VERSION1,
                    client_id="mqtt_service_publisher",
                    clean_session=True
                )
            except (AttributeError, TypeError):
                # Fall back to old API
                self.publisher_client = mqtt.Client(
                    client_id="mqtt_service_publisher",
                    clean_session=True
                )

            broker_host = self.mqtt_config["broker_host"]
            broker_port = self.mqtt_config["broker_port"]

            self.publisher_client.connect(
                broker_host,
                broker_port,
                keepalive=self.mqtt_config["keepalive"],
            )
            self.publisher_client.loop_start()

            # Wait briefly for connection to establish
            time.sleep(0.2)
            self.is_running = True

        except Exception as e:
            logging.warning(f"Could not setup MQTT publisher: {e}")
            self.is_running = False

    def publish_message(self, topic: str, message: str) -> bool:
        """Publish message to MQTT topic."""
        if not self.is_running or not self.publisher_client:
            return False

        try:
            result = self.publisher_client.publish(
                topic, message, qos=self.mqtt_config["qos"]
            )
            return result.rc == mqtt.MQTT_ERR_SUCCESS
        except Exception as e:
            logging.warning(f"Failed to publish message: {e}")
            return False

    def close(self):
        """Close MQTT publisher."""
        self.is_running = False
        if self.publisher_client:
            self.publisher_client.loop_stop()
            self.publisher_client.disconnect()


class MqttServer:
    """
    MQTT server interface for communication service.
    Manages broker connection and service discovery.
    """

    def __init__(self, mqtt_config: Dict[str, Any] = None):
        self.mqtt_config = mqtt_config or {
            "broker_host": "localhost",
            "broker_port": 1883,
        }
        self.broker_host = self.mqtt_config["broker_host"]
        self.broker_port = self.mqtt_config["broker_port"]
        self.is_running = False

    def start_server(self):
        """Start MQTT server (broker connection check)."""
        if self.is_running:
            return

        # For MQTT, we don't start a server but verify broker connectivity
        if mqtt:
            try:
                # Support both old and new paho-mqtt API versions
                try:
                    # Try new API (v2.0+)
                    test_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, "mqtt_server_test")
                except (AttributeError, TypeError):
                    # Fall back to old API
                    test_client = mqtt.Client("mqtt_server_test")

                test_client.connect(self.broker_host, self.broker_port, 10)
                test_client.disconnect()
                self.is_running = True
                logging.info(
                    f"MQTT broker available at {self.broker_host}:{self.broker_port}"
                )
            except Exception as e:
                logging.warning(f"Could not connect to MQTT broker: {e}")
                self.is_running = False
        else:
            logging.warning("MQTT library not available")
            self.is_running = False

    def stop_server(self):
        """Stop MQTT server."""
        self.is_running = False
        logging.info("MQTT server stopped")

    def get_service_address(self) -> str:
        """Get the broker address."""
        return f"{self.broker_host}:{self.broker_port}"


class MqttCommunicatingAgent:
    """
    Agent client for MQTT communication.
    Provides the same interface as REST, gRPC, and Kafka agents.
    """

    def __init__(
        self, agent_id: str, mqtt_service, mqtt_config: Dict[str, Any] = None
    ):
        self.agent_id = agent_id
        self.mqtt_service = mqtt_service
        self.mqtt_config = mqtt_config or {
            "broker_host": "localhost",
            "broker_port": 1883,
        }

        # Create mailbox for this agent
        broadcast_topic = getattr(mqtt_service, "broadcast_topic", None)
        self.mailbox = MqttMailbox(
            agent_id,
            self.mqtt_config,
            broadcast_topic=broadcast_topic,
        )

    def send_message(
        self,
        receiver_id: str,
        message_type: MqttMessageType,
        content: Dict[str, Any],
        reply_to: Optional[str] = None,
    ) -> bool:
        """
        Communication action: send(i,j,m)
        Send message from this agent to target agent via MQTT.
        """
        message = MqttMessage(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            reply_to=reply_to,
        )

        return self.mqtt_service.send_message(message)

    def broadcast_message(
        self, message_type: MqttMessageType, content: Dict[str, Any]
    ) -> Dict:
        """
        Broadcast message: send(i,*,m)
        Send message to all reachable agents via MQTT.
        """
        message = MqttMessage(
            sender_id=self.agent_id,
            receiver_id="*",
            message_type=message_type,
            content=content,
        )

        return self.mqtt_service.broadcast_message(message)

    def receive_messages(
        self, clear_mailbox: bool = True
    ) -> List[MqttMessage]:
        """
        Get messages from agent's MQTT mailbox.
        Part of agent's perception input.
        """
        return self.mailbox.get_messages(clear_mailbox)

    def reply_to_message(
        self, original_message: MqttMessage, content: Dict[str, Any]
    ) -> bool:
        """Reply to a received message."""
        return self.send_message(
            receiver_id=original_message.sender_id,
            message_type=MqttMessageType.REPLY,
            content=content,
            reply_to=original_message.message_id,
        )

    def close(self):
        """Close MQTT connections."""
        if self.mailbox:
            self.mailbox.close()


class MqttCommunicationService:
    """
    MQTT-based communication service implementation.
    Manages MQTT connections, topics, and delivery statistics.
    """

    def __init__(
        self,
        mqtt_config: Dict[str, Any] = None,
        latency_mode: LatencyMode = LatencyMode.SEND_ONLY,
    ):
        self.mqtt_config = mqtt_config or {
            "broker_host": "localhost",
            "broker_port": 1883,
            "keepalive": 60,
            "qos": 1,
        }
        # MQTT is naturally async/fire-and-forget, so SEND_ONLY is default
        self.latency_mode = latency_mode

        self.mailboxes: Dict[str, MqttMailbox] = {}
        self.topology = MqttCommunicationTopology()
        self.delivery_stats = {
            "messages_sent": 0,
            "messages_delivered": 0,
            "delivery_failures": 0,
            "avg_delivery_time": 0.0,
        }
        self.lock = threading.Lock()
        self.broadcast_topic = self.mqtt_config.get(
            "broadcast_topic", "agent/broadcast/all"
        )
        self.broadcast_queue: "queue.Queue[Tuple[str, Dict[str, Any], Optional[str]]]" = queue.Queue()
        self.broadcast_worker_stop = threading.Event()
        self.broadcast_worker = threading.Thread(
            target=self._broadcast_worker, daemon=True
        )
        self.broadcast_worker.start()

        self.mqtt_service = MqttService(self.mqtt_config)
        self.mqtt_server = MqttServer(self.mqtt_config)
        self.is_running = False

    def start_service(self):
        """Start the MQTT communication service."""
        if self.is_running:
            return

        self.mqtt_server.start_server()
        self.is_running = self.mqtt_server.is_running

    def stop_service(self):
        """Stop the MQTT communication service."""
        if self.is_running:
            self.mqtt_server.stop_server()
            self.mqtt_service.close()
            self.is_running = False

    def register_agent(
        self, agent_id: str, mailbox: Optional[MqttMailbox] = None
    ):
        """Register an agent and ensure its mailbox is available for delivery."""
        if agent_id not in self.mailboxes:
            self.mailboxes[agent_id] = mailbox or MqttMailbox(
                agent_id,
                self.mqtt_config,
                broadcast_topic=self.broadcast_topic,
            )

    def send_message(self, message: MqttMessage) -> bool:
        """
        Send message via MQTT: implements deliver(i,j,m).
        """
        start_time = time.time()

        with self.lock:
            self.delivery_stats["messages_sent"] += 1

        # Check if communication is allowed
        if not self.topology.can_communicate(
            message.sender_id, message.receiver_id
        ):
            with self.lock:
                self.delivery_stats["delivery_failures"] += 1
            return False

        # Check if receiver exists
        if message.receiver_id not in self.mailboxes:
            with self.lock:
                self.delivery_stats["delivery_failures"] += 1
            return False

        try:
            if self.mqtt_service.is_running:
                # Send via MQTT
                topic = f"agent/mailbox/{message.receiver_id}"
                success = self.mqtt_service.publish_message(
                    topic, message.to_json()
                )

                if not success:
                    # Fallback: direct delivery to mailbox
                    self.mailboxes[message.receiver_id].add_message(message)
            else:
                # Fallback: direct delivery to mailbox
                self.mailboxes[message.receiver_id].add_message(message)

            # Update stats
            delivery_time = time.time() - start_time
            with self.lock:
                self.delivery_stats["messages_delivered"] += 1
                old_avg = self.delivery_stats["avg_delivery_time"]
                count = self.delivery_stats["messages_delivered"]
                self.delivery_stats["avg_delivery_time"] = (
                    old_avg * (count - 1) + delivery_time
                ) / count

            return True

        except Exception as e:
            logging.warning(f"Failed to send message: {e}")
            with self.lock:
                self.delivery_stats["delivery_failures"] += 1
            return False

    def broadcast_message(self, message: MqttMessage) -> Dict[str, Any]:
        """
        Broadcast message to all reachable agents: implements send(i,*,m).
        """
        reachable = self.topology.get_reachable_agents(message.sender_id)
        total_targets = len(reachable)
        if total_targets == 0:
            return {
                "status": "completed",
                "delivered": 0,
                "failed": 0,
                "total_targets": 0,
                "message_ids": [],
            }

        payload_copy = json.loads(json.dumps(message.content))

        delivered = total_targets
        failed = 0
        message_ids: List[str] = []

        broadcast_message = MqttMessage(
            sender_id=message.sender_id,
            receiver_id="*",
            message_type=MqttMessageType.BROADCAST,
            content=payload_copy,
            reply_to=message.reply_to,
        )

        if self.mqtt_service.is_running:
            success = self.mqtt_service.publish_message(
                self.broadcast_topic, broadcast_message.to_json()
            )
            if success:
                message_ids.append(broadcast_message.message_id)
            else:
                self._enqueue_broadcast_task(
                    message.sender_id, payload_copy, message.reply_to
                )
        else:
            self._enqueue_broadcast_task(
                message.sender_id, payload_copy, message.reply_to
            )

        if not message_ids:
            # Direct delivery fallback still counts as delivered
            delivered = total_targets
            failed = 0

        return {
            "status": "completed" if delivered >= failed else "failed",
            "delivered": delivered,
            "failed": failed,
            "total_targets": total_targets,
            "message_ids": message_ids,
        }

    def _enqueue_broadcast_task(
        self, sender_id: str, content: Dict[str, Any], reply_to: Optional[str]
    ):
        payload = json.loads(json.dumps(content))
        self.broadcast_queue.put((sender_id, payload, reply_to))

    def _broadcast_worker(self):
        """Fallback broadcast delivery when MQTT broker is unavailable."""
        while True:
            try:
                task = self.broadcast_queue.get(timeout=0.1)
            except queue.Empty:
                if self.broadcast_worker_stop.is_set():
                    break
                continue

            if task is None:
                self.broadcast_queue.task_done()
                break

            sender_id, content, reply_to = task
            reachable = self.topology.get_reachable_agents(sender_id)
            for receiver_id in reachable:
                mailbox = self.mailboxes.get(receiver_id)
                if not mailbox:
                    continue
                message = MqttMessage(
                    sender_id=sender_id,
                    receiver_id=receiver_id,
                    message_type=MqttMessageType.BROADCAST,
                    content=json.loads(json.dumps(content)),
                    reply_to=reply_to,
                )
                mailbox.add_message(message)

            self.broadcast_queue.task_done()

    def get_messages(
        self, agent_id: str, clear_mailbox: bool = True
    ) -> List[MqttMessage]:
        """Get messages from agent's mailbox."""
        if agent_id not in self.mailboxes:
            return []

        return self.mailboxes[agent_id].get_messages(clear_mailbox)

    def get_statistics(self) -> Dict[str, Any]:
        """Get communication statistics for benchmarking."""
        with self.lock:
            return self.delivery_stats.copy()

    def get_topology_info(self) -> Dict[str, Any]:
        """Get current communication topology information."""
        with self.topology.lock:
            return {
                "links": list(self.topology.links),
                "total_links": len(self.topology.links),
            }

    def get_service_address(self) -> str:
        """Get the service address."""
        return self.mqtt_server.get_service_address()

    def close(self):
        """Close all MQTT connections."""
        self.is_running = False
        self.broadcast_worker_stop.set()
        self.broadcast_queue.put(None)
        if self.broadcast_worker.is_alive():
            self.broadcast_worker.join(timeout=1.0)

        # Close all mailboxes
        for mailbox in self.mailboxes.values():
            mailbox.close()

        # Close MQTT service
        self.mqtt_service.close()
        self.mqtt_server.stop_server()
