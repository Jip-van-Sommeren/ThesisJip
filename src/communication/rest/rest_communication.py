"""
REST-based Communication Model Implementation
Based on the formal communication definition from the thesis.

Implements:
- Message Space (M): Set of all possible messages
- Communication Topology (Comm): Agent-to-agent communication links
- Mailboxes (MB_i): Message buffers for each agent
- Delivery Function: Message transmission via REST APIs
- Communication Actions: send(i,j,m) operations
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, deque
import json
import time
import uuid
from abc import ABC, abstractmethod
from enum import Enum
import threading
from queue import Queue
import requests
from flask import Flask, request, jsonify
import logging

from abstract_agent import AgentId


class MessageType(Enum):
    """Types of messages in the message space M."""

    INFORM = "inform"
    REQUEST = "request"
    REPLY = "reply"
    BROADCAST = "broadcast"
    ERROR = "error"


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


class RESTCommunicationService:
    """
    REST-based implementation of the communication framework.
    Provides HTTP endpoints for message delivery and mailbox access.
    """

    def __init__(self, host: str = "localhost", port: int = 5000):
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.app.logger.setLevel(logging.WARNING)  # Reduce Flask logging

        # Communication components
        self.mailboxes: Dict[str, Mailbox] = {}
        self.topology = CommunicationTopology()
        self.delivery_stats = {
            "messages_sent": 0,
            "messages_delivered": 0,
            "delivery_failures": 0,
            "avg_delivery_time": 0.0,
        }

        # Setup routes
        self._setup_routes()

    def _setup_routes(self):
        """Setup REST API routes."""

        @self.app.route("/agent/<agent_id>/mailbox", methods=["GET"])
        def get_mailbox(agent_id: str):
            """Get messages from agent's mailbox."""
            if agent_id not in self.mailboxes:
                return jsonify({"error": "Agent not found"}), 404

            clear = request.args.get("clear", "true").lower() == "true"
            messages = self.mailboxes[agent_id].get_messages(clear=clear)
            return jsonify(
                {
                    "agent_id": agent_id,
                    "messages": [msg.to_dict() for msg in messages],
                    "count": len(messages),
                }
            )

        @self.app.route("/agent/<agent_id>/send", methods=["POST"])
        def send_message(agent_id: str):
            """Send message from agent to another agent."""
            try:
                data = request.get_json()
                message = Message(
                    sender_id=agent_id,
                    receiver_id=data["receiver_id"],
                    message_type=MessageType(data["message_type"]),
                    content=data["content"],
                    reply_to=data.get("reply_to"),
                )

                success = self.deliver_message(message)
                if success:
                    return jsonify(
                        {
                            "status": "delivered",
                            "message_id": message.message_id,
                        }
                    )
                else:
                    return (
                        jsonify(
                            {
                                "status": "failed",
                                "error": "Communication not allowed or receiver not found",
                            }
                        ),
                        400,
                    )

            except Exception as e:
                return jsonify({"error": str(e)}), 400

        @self.app.route("/agent/<agent_id>/broadcast", methods=["POST"])
        def broadcast_message(agent_id: str):
            """Broadcast message to all reachable agents."""
            try:
                data = request.get_json()
                reachable = self.topology.get_reachable_agents(agent_id)

                delivered = 0
                failed = 0

                for receiver_id in reachable:
                    message = Message(
                        sender_id=agent_id,
                        receiver_id=receiver_id,
                        message_type=MessageType.BROADCAST,
                        content=data["content"],
                    )

                    if self.deliver_message(message):
                        delivered += 1
                    else:
                        failed += 1

                return jsonify(
                    {
                        "status": "completed",
                        "delivered": delivered,
                        "failed": failed,
                        "total_targets": len(reachable),
                    }
                )

            except Exception as e:
                return jsonify({"error": str(e)}), 400

        @self.app.route("/topology", methods=["GET"])
        def get_topology():
            """Get current communication topology."""
            return jsonify(
                {
                    "links": list(self.topology.links),
                    "total_links": len(self.topology.links),
                }
            )

        @self.app.route("/topology/link", methods=["POST"])
        def add_topology_link():
            """Add communication link to topology."""
            try:
                data = request.get_json()
                self.topology.add_link(data["sender_id"], data["receiver_id"])
                return jsonify({"status": "link_added"})
            except Exception as e:
                return jsonify({"error": str(e)}), 400

        @self.app.route("/stats", methods=["GET"])
        def get_statistics():
            """Get communication statistics."""
            return jsonify(self.delivery_stats)

    def register_agent(self, agent_id: str):
        """Register an agent and create its mailbox."""
        if agent_id not in self.mailboxes:
            self.mailboxes[agent_id] = Mailbox(agent_id)

    def deliver_message(self, message: Message) -> bool:
        """
        deliver(i,j,m): Message delivery function.
        Delivers message m from agent i to agent j's mailbox.
        """
        start_time = time.time()
        self.delivery_stats["messages_sent"] += 1

        # Check if communication is allowed
        if not self.topology.can_communicate(
            message.sender_id, message.receiver_id
        ):
            self.delivery_stats["delivery_failures"] += 1
            return False

        # Check if receiver exists
        if message.receiver_id not in self.mailboxes:
            self.delivery_stats["delivery_failures"] += 1
            return False

        # Deliver message
        self.mailboxes[message.receiver_id].add_message(message)

        # Update stats
        delivery_time = time.time() - start_time
        self.delivery_stats["messages_delivered"] += 1
        self.delivery_stats["avg_delivery_time"] = (
            self.delivery_stats["avg_delivery_time"]
            * (self.delivery_stats["messages_delivered"] - 1)
            + delivery_time
        ) / self.delivery_stats["messages_delivered"]

        return True

    def start_server(self):
        """Start the REST communication server."""
        self.app.run(
            host=self.host, port=self.port, debug=False, threaded=True
        )


class CommunicatingAgent:
    """
    Agent wrapper that adds communication capabilities to existing agents.
    Extends agents with communication actions and mailbox access.
    """

    def __init__(
        self, agent_id: str, comm_service_url: str = "http://localhost:5000"
    ):
        self.agent_id = agent_id
        self.comm_service_url = comm_service_url

    def send_message(
        self,
        receiver_id: str,
        message_type: MessageType,
        content: Dict[str, Any],
        reply_to: Optional[str] = None,
    ) -> bool:
        """
        Communication action: send(i,j,m)
        Send message from this agent to target agent.
        """
        try:
            response = requests.post(
                f"{self.comm_service_url}/agent/{self.agent_id}/send",
                json={
                    "receiver_id": receiver_id,
                    "message_type": message_type.value,
                    "content": content,
                    "reply_to": reply_to,
                },
                timeout=5.0,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def broadcast_message(
        self, message_type: MessageType, content: Dict[str, Any]
    ) -> Dict:
        """
        Broadcast message: send(i,*,m)
        Send message to all reachable agents.
        """
        try:
            response = requests.post(
                f"{self.comm_service_url}/agent/{self.agent_id}/broadcast",
                json={"message_type": message_type.value, "content": content},
                timeout=5.0,
            )
            if response.status_code == 200:
                return response.json()
            return {"status": "failed"}
        except requests.RequestException:
            return {"status": "failed"}

    def receive_messages(self, clear_mailbox: bool = True) -> List[Message]:
        """
        Get messages from agent's mailbox.
        Part of agent's perception input.
        """
        try:
            response = requests.get(
                f"{self.comm_service_url}/agent/{self.agent_id}/mailbox",
                params={"clear": str(clear_mailbox).lower()},
                timeout=5.0,
            )
            if response.status_code == 200:
                data = response.json()
                return [Message.from_dict(msg) for msg in data["messages"]]
            return []
        except requests.RequestException:
            return []

    def reply_to_message(
        self, original_message: Message, content: Dict[str, Any]
    ) -> bool:
        """Reply to a received message."""
        return self.send_message(
            receiver_id=original_message.sender_id,
            message_type=MessageType.REPLY,
            content=content,
            reply_to=original_message.message_id,
        )
