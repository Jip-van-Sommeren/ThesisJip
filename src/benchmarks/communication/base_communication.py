"""
Base Communication Model Implementation
Core abstractions for the formal communication definition from the thesis.

Implements the fundamental components that are protocol-agnostic:
- Message Space (M): Set of all possible messages
- Message Type definitions
- Communication Topology (Comm): Agent-to-agent communication links
- Mailboxes (MB_i): Message buffers for each agent

These abstractions can be used by REST, gRPC, Kafka, and other communication
implementations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import deque
import time
import uuid
from enum import Enum
import threading


class MessageType(Enum):
    """Types of messages in the message space M."""

    INFORM = "inform"
    REQUEST = "request"
    REPLY = "reply"
    BROADCAST = "broadcast"
    ERROR = "error"
    ACK = "ack"  # Acknowledgment for end-to-end latency measurement


class LatencyMode(Enum):
    """
    Latency measurement modes for benchmarking.

    SEND_ONLY: Measure only time to send/publish message (no delivery confirmation)
    - REST: Time to make HTTP request (async, don't wait for response)
    - gRPC: Time to make RPC call (async, don't wait for response)
    - MQTT: Time to publish to broker
    - Kafka: Time to send to producer (no broker ack)

    END_TO_END: Measure protocol-level round-trip including protocol confirmation
    - REST: HTTP request/response cycle
    - gRPC: RPC call/response cycle
    - MQTT: Publish + broker acknowledgment (QoS 1)
    - Kafka: Send + leader acknowledgment (acks=1)

    APP_ACK: Measure application-level round-trip with explicit receiver acknowledgment
    - Sender sends message and waits for explicit ACK message from receiver
    - Receiver processes message and sends ACK back to sender
    - Measures true end-to-end latency including receiver processing time
    - Most comprehensive latency measurement (includes network + protocol + processing)
    """

    SEND_ONLY = "send_only"
    END_TO_END = "end_to_end"
    APP_ACK = "app_ack"


@dataclass
class Message:
    """
    Message m ∈ M: Represents a communication message.
    Abstract entity with content and intent.
    """

    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    sender_id: str = ""
    receiver_id: str = ""
    message_type: MessageType = MessageType.INFORM
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    reply_to: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert message to dictionary for JSON serialization."""
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "reply_to": self.reply_to,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Message":
        """Create message from dictionary."""
        return cls(
            message_id=data["message_id"],
            sender_id=data["sender_id"],
            receiver_id=data["receiver_id"],
            message_type=MessageType(data["message_type"]),
            content=data["content"],
            timestamp=data["timestamp"],
            reply_to=data.get("reply_to"),
        )


class Mailbox:
    """
    MB_i: Mailbox for agent i.
    Buffer that holds incoming messages as multiset/queue.
    """

    def __init__(self, agent_id: str, max_size: int = 1000):
        self.agent_id = agent_id
        self.messages: deque = deque(maxlen=max_size)
        self.lock = threading.Lock()

    def add_message(self, message: Message):
        """Add message to mailbox (thread-safe)."""
        with self.lock:
            self.messages.append(message)

    def get_messages(self, clear: bool = True) -> List[Message]:
        """Get all messages from mailbox, optionally clearing it."""
        with self.lock:
            messages = list(self.messages)
            if clear:
                self.messages.clear()
            return messages

    def peek_messages(self) -> List[Message]:
        """Get messages without removing them."""
        return self.get_messages(clear=False)

    def size(self) -> int:
        """Get number of messages in mailbox."""
        with self.lock:
            return len(self.messages)


class CommunicationTopology:
    """
    Comm ⊆ A × A: Communication topology relation.
    Defines which agents can communicate with each other.
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
                    if sender != receiver:  # No self-communication by default
                        self.links.add((sender, receiver))
