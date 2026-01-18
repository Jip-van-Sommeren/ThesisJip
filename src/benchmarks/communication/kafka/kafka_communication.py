"""
Kafka-based Communication Model Implementation
Based on the formal communication definition from the thesis.

Implements the same formal model as REST and gRPC but using Apache Kafka:
- Message Space (M): Set of all possible messages
- Communication Topology (Comm): Agent-to-agent communication via Kafka topics
- Mailboxes (MB_i): Message buffers implemented as Kafka consumers
- Delivery Function: Message transmission via Kafka producers
- Communication Actions: send(i,j,m) operations via Kafka messaging
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
from benchmarks.communication.base_communication import (
    MessageType as BaseMessageType,
    LatencyMode,
)

from kafka import KafkaProducer, KafkaConsumer, KafkaAdminClient
from kafka.admin import NewTopic
from kafka.errors import TopicAlreadyExistsError


class KafkaMessageType(Enum):
    """Kafka Message types mapping to thesis definition."""

    INFORM = "inform"
    REQUEST = "request"
    REPLY = "reply"
    BROADCAST = "broadcast"
    ERROR = "error"
    ACK = "ack"

    @classmethod
    def from_base_type(cls, base_type: BaseMessageType) -> "KafkaMessageType":
        """Convert base message type to Kafka message type."""
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
        """Convert Kafka message type to base message type."""
        mapping = {
            self.INFORM: BaseMessageType.INFORM,
            self.REQUEST: BaseMessageType.REQUEST,
            self.REPLY: BaseMessageType.REPLY,
            self.BROADCAST: BaseMessageType.BROADCAST,
            self.ERROR: BaseMessageType.ERROR,
            self.ACK: BaseMessageType.ACK,
        }
        return mapping[self]


@dataclass
class KafkaMessage:
    """
    Message representation for Kafka communication.
    Implements the message space M from the thesis.
    """

    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    sender_id: str = ""
    receiver_id: str = ""
    message_type: KafkaMessageType = KafkaMessageType.INFORM
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    reply_to: Optional[str] = None

    def to_json(self) -> str:
        """Convert to JSON for Kafka serialization."""
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
    def from_json(cls, json_str: str) -> "KafkaMessage":
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls(
            message_id=data["message_id"],
            sender_id=data["sender_id"],
            receiver_id=data["receiver_id"],
            message_type=KafkaMessageType(data["message_type"]),
            content=data["content"],
            timestamp=data["timestamp"],
            reply_to=data.get("reply_to"),
        )


class KafkaMailbox:
    """
    MB_i: Mailbox for agent i using Kafka consumer.
    Implements asynchronous message buffering via Kafka topics.
    """

    def __init__(
        self,
        agent_id: str,
        kafka_config: Dict[str, Any],
        max_size: int = 1000,
        broadcast_topic: Optional[str] = None,
    ):
        self.agent_id = agent_id
        self.kafka_config = kafka_config
        self.max_size = max_size
        self.messages: deque = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.consumer = None
        self.consumer_thread = None
        self.running = False

        # Topic for this agent's mailbox
        self.topic = f"agent_mailbox_{agent_id}"
        self.broadcast_topic = broadcast_topic

        self._setup_consumer()

    def _setup_consumer(self):
        """Setup Kafka consumer for this agent's mailbox."""
        try:
            topics = [self.topic]
            if self.broadcast_topic:
                topics.append(self.broadcast_topic)

            self.consumer = KafkaConsumer(
                bootstrap_servers=self.kafka_config.get(
                    "bootstrap_servers", ["localhost:9092"]
                ),
                value_deserializer=lambda m: m.decode("utf-8") if m else None,
                consumer_timeout_ms=1000,
                auto_offset_reset="latest",
                group_id=f"agent_{self.agent_id}_consumer",
            )

            if topics:
                self.consumer.subscribe(topics)

            # Start consumer thread
            self.running = True
            self.consumer_thread = threading.Thread(
                target=self._consume_messages, daemon=True
            )
            self.consumer_thread.start()

        except Exception as e:
            logging.warning(
                f"Could not setup Kafka consumer for {self.agent_id}: {e}"
            )

    def _consume_messages(self):
        """Background thread to consume messages from Kafka."""
        while self.running and self.consumer:
            try:
                message_pack = self.consumer.poll(timeout_ms=1000)
                for topic_partition, messages in message_pack.items():
                    for message in messages:
                        if message.value:
                            kafka_message = KafkaMessage.from_json(
                                message.value
                            )
                            with self.lock:
                                self.messages.append(kafka_message)
            except Exception as e:
                if self.running:  # Only log if we're supposed to be running
                    logging.warning(
                        f"Error consuming messages for {self.agent_id}: {e}"
                    )
                time.sleep(0.1)

    def add_message(self, message: KafkaMessage):
        """Add message to mailbox (for direct delivery when Kafka\
            not available)."""
        with self.lock:
            self.messages.append(message)

    def get_messages(self, clear: bool = True) -> List[KafkaMessage]:
        """Get all messages from mailbox, optionally clearing it."""
        with self.lock:
            messages = list(self.messages)
            if clear:
                self.messages.clear()
            return messages

    def peek_messages(self) -> List[KafkaMessage]:
        """Return messages without removing them from the mailbox."""
        return self.get_messages(clear=False)

    def size(self) -> int:
        """Get number of messages in mailbox."""
        with self.lock:
            return len(self.messages)

    def close(self):
        """Close the Kafka consumer."""
        self.running = False
        if self.consumer:
            self.consumer.close()
        if self.consumer_thread and self.consumer_thread.is_alive():
            self.consumer_thread.join(timeout=1.0)


class KafkaCommunicationTopology:
    """
    Comm ⊆ A × A: Communication topology for Kafka.
    Uses Kafka topics to represent communication links.
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


class KafkaCommunicationService:
    """
    Kafka-based communication service implementation.
    Manages Kafka topics, producers, and delivery statistics.
    """

    def __init__(
        self,
        kafka_config: Dict[str, Any] = None,
        latency_mode: LatencyMode = LatencyMode.END_TO_END,
    ):
        self.kafka_config = kafka_config or {
            "bootstrap_servers": ["localhost:9092"],
            "client_id": "agent_communication_service",
        }
        self.latency_mode = latency_mode

        self.mailboxes: Dict[str, KafkaMailbox] = {}
        self.topology = KafkaCommunicationTopology()
        self.delivery_stats = {
            "messages_sent": 0,
            "messages_delivered": 0,
            "delivery_failures": 0,
            "avg_delivery_time": 0.0,
        }
        self.lock = threading.Lock()

        raw_acks = self.kafka_config.get("acks", "all")
        self.producer_acks = self._normalize_acks(raw_acks)
        self.kafka_config["acks"] = self.producer_acks
        self.producer_compression = self.kafka_config.get(
            "compression_type"
        )
        self.broadcast_topic = self.kafka_config.get(
            "broadcast_topic", "agent_broadcast_global"
        )

        self.producer = None
        self.admin_client = None
        self.is_running = False
        self.broadcast_queue: "queue.Queue[Tuple[str, Dict[str, Any], Optional[str]]]" = queue.Queue()
        self.broadcast_worker_stop = threading.Event()
        self.broadcast_worker = threading.Thread(
            target=self._broadcast_worker, daemon=True
        )
        self.broadcast_worker.start()

        self._setup_kafka()

    def _normalize_acks(self, acks_value: Any) -> Any:
        """Normalize producer acks configuration to Kafka client's expected formats."""
        if isinstance(acks_value, str):
            lowered = acks_value.strip().lower()
            if lowered in {"all", "-1"}:
                return "all"
            if lowered in {"0", "1"}:
                return int(lowered)
        elif isinstance(acks_value, int):
            if acks_value == -1:
                return "all"
            if acks_value in {0, 1}:
                return acks_value
        return 1

    def _setup_kafka(self):
        """Setup Kafka producer and admin client."""
        try:
            producer_kwargs = {
                "bootstrap_servers": self.kafka_config["bootstrap_servers"],
                "value_serializer": lambda x: (
                    x.encode("utf-8") if isinstance(x, str) else x
                ),
                "client_id": self.kafka_config.get(
                    "client_id", "agent_communication"
                ),
                "retries": 3,
                "acks": self.producer_acks,
                "linger_ms": 5,
                "batch_size": 32768,
                "max_in_flight_requests_per_connection": 5,
            }

            compression = (
                str(self.producer_compression).lower()
                if self.producer_compression is not None
                else ""
            )
            if compression in {"gzip", "snappy", "lz4"}:
                producer_kwargs["compression_type"] = compression
            elif compression not in {"", "none"}:
                logging.warning(
                    "Kafka compression_type '%s' is not supported; disabling compression",
                    compression,
                )

            self.producer = KafkaProducer(**producer_kwargs)
            client_id_value = self.kafka_config.get(
                "client_id", "agent_communication"
            )
            self.admin_client = KafkaAdminClient(
                bootstrap_servers=self.kafka_config["bootstrap_servers"],
                client_id=f"{client_id_value}_admin",
            )

            if self.admin_client:
                try:
                    broadcast_topic = NewTopic(
                        name=self.broadcast_topic,
                        num_partitions=1,
                        replication_factor=1,
                    )
                    self.admin_client.create_topics([broadcast_topic])
                except TopicAlreadyExistsError:
                    pass
                except Exception as e:
                    logging.warning(
                        f"Could not create broadcast topic '{self.broadcast_topic}': {e}"
                    )

            self.is_running = True

        except Exception as e:
            logging.warning(
                f"Could not setup Kafka: {e}. Using fallback mode."
            )
            self.is_running = False

    def register_agent(self, agent_id: str):
        """Register an agent and create its Kafka topic/mailbox."""
        if agent_id not in self.mailboxes:
            # Create mailbox (which includes Kafka consumer setup)
            self.mailboxes[agent_id] = KafkaMailbox(
                agent_id,
                self.kafka_config,
                broadcast_topic=self.broadcast_topic,
            )

            # Create Kafka topic for this agent if Kafka is available
            if self.admin_client:
                try:
                    topic_name = f"agent_mailbox_{agent_id}"
                    topic = NewTopic(
                        name=topic_name, num_partitions=1, replication_factor=1
                    )
                    self.admin_client.create_topics([topic])
                except TopicAlreadyExistsError:
                    pass  # Topic already exists, which is fine
                except Exception as e:
                    logging.warning(
                        f"Could not create topic for {agent_id}: {e}"
                    )

    def send_message(self, message: KafkaMessage) -> bool:
        """
        Send message via Kafka: implements deliver(i,j,m).
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
            if self.producer and self.is_running:
                # Send via Kafka
                topic = f"agent_mailbox_{message.receiver_id}"
                future = self.producer.send(topic, value=message.to_json())

                # Wait for send to complete (with timeout)
                if self.latency_mode != LatencyMode.SEND_ONLY:
                    future.get(timeout=5)

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

    def broadcast_message(self, message: KafkaMessage) -> Dict[str, Any]:
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

        if self.producer and self.is_running:
            broadcast_message = KafkaMessage(
                sender_id=message.sender_id,
                receiver_id="*",
                message_type=KafkaMessageType.BROADCAST,
                content=payload_copy,
                reply_to=message.reply_to,
            )
            future = self.producer.send(
                self.broadcast_topic, value=broadcast_message.to_json()
            )
            if self.latency_mode != LatencyMode.SEND_ONLY:
                try:
                    future.get(timeout=5)
                except Exception as exc:
                    logging.warning(
                        f"Broadcast publish failed for {message.sender_id}: {exc}"
                    )
                    delivered = 0
                    failed = total_targets
            message_ids.append(broadcast_message.message_id)
        else:
            self._enqueue_broadcast_task(
                message.sender_id, payload_copy, message.reply_to
            )

        return {
            "status": "completed" if delivered >= failed else "failed",
            "delivered": delivered,
            "failed": failed,
            "total_targets": total_targets,
            "message_ids": message_ids,
        }

    def get_messages(
        self, agent_id: str, clear_mailbox: bool = True
    ) -> List[KafkaMessage]:
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

    def close(self):
        """Close all Kafka connections."""
        self.is_running = False
        self.broadcast_worker_stop.set()
        self.broadcast_queue.put(None)
        if self.broadcast_worker.is_alive():
            self.broadcast_worker.join(timeout=1.0)

        # Close all mailboxes
        for mailbox in self.mailboxes.values():
            mailbox.close()

        # Close producer
        if self.producer:
            self.producer.flush()
            self.producer.close()

    def _enqueue_broadcast_task(
        self, sender_id: str, content: Dict[str, Any], reply_to: Optional[str]
    ):
        payload = json.loads(json.dumps(content))
        self.broadcast_queue.put((sender_id, payload, reply_to))

    def _broadcast_worker(self):
        """Asynchronous worker to deliver broadcast payloads when Kafka broker is unavailable."""
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
                message = KafkaMessage(
                    sender_id=sender_id,
                    receiver_id=receiver_id,
                    message_type=KafkaMessageType.BROADCAST,
                    content=json.loads(json.dumps(content)),
                    reply_to=reply_to,
                )
                mailbox.add_message(message)

            self.broadcast_queue.task_done()


class KafkaCommunicatingAgent:
    """
    Agent client for Kafka communication.
    Provides the same interface as REST and gRPC agents.
    """

    def __init__(
        self, agent_id: str, kafka_service: KafkaCommunicationService
    ):
        self.agent_id = agent_id
        self.kafka_service = kafka_service

    def send_message(
        self,
        receiver_id: str,
        message_type: KafkaMessageType,
        content: Dict[str, Any],
        reply_to: Optional[str] = None,
    ) -> bool:
        """
        Communication action: send(i,j,m)
        Send message from this agent to target agent via Kafka.
        """
        message = KafkaMessage(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            reply_to=reply_to,
        )

        return self.kafka_service.send_message(message)

    def broadcast_message(
        self, message_type: KafkaMessageType, content: Dict[str, Any]
    ) -> Dict:
        """
        Broadcast message: send(i,*,m)
        Send message to all reachable agents via Kafka.
        """
        message = KafkaMessage(
            sender_id=self.agent_id,
            receiver_id="*",
            message_type=message_type,
            content=content,
        )

        return self.kafka_service.broadcast_message(message)

    def receive_messages(
        self, clear_mailbox: bool = True
    ) -> List[KafkaMessage]:
        """
        Get messages from agent's Kafka mailbox.
        Part of agent's perception input.
        """
        return self.kafka_service.get_messages(self.agent_id, clear_mailbox)

    def reply_to_message(
        self, original_message: KafkaMessage, content: Dict[str, Any]
    ) -> bool:
        """Reply to a received message."""
        return self.send_message(
            receiver_id=original_message.sender_id,
            message_type=KafkaMessageType.REPLY,
            content=content,
            reply_to=original_message.message_id,
        )
