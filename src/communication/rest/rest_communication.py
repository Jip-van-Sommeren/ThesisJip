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

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
import time
import uuid
from enum import Enum
import threading
import requests
from flask import Flask, request, jsonify
import logging
import json
import queue
from contextlib import nullcontext
try:
    import httpx  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    httpx = None

# Import base communication abstractions
from communication.base_communication import (
    MessageType as BaseMessageType,
    Message,
    CommunicationTopology,
    LatencyMode,
)


class RestMessageType(Enum):
    """REST Message types mapping to thesis definition."""

    INFORM = "inform"
    REQUEST = "request"
    REPLY = "reply"
    BROADCAST = "broadcast"
    ERROR = "error"
    ACK = "ack"  # Acknowledgment for end-to-end latency measurement

    @classmethod
    def from_base_type(cls, base_type: BaseMessageType) -> "RestMessageType":
        """Convert base message type to REST message type."""
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
        """Convert REST message type to base message type."""
        mapping = {
            self.INFORM: BaseMessageType.INFORM,
            self.REQUEST: BaseMessageType.REQUEST,
            self.REPLY: BaseMessageType.REPLY,
            self.BROADCAST: BaseMessageType.BROADCAST,
            self.ERROR: BaseMessageType.ERROR,
        }
        return mapping[self]


@dataclass
class RestMessage:
    """
    Message representation for REST communication.
    Implements the message space M from the thesis.
    """

    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    sender_id: str = ""
    receiver_id: str = ""
    message_type: RestMessageType = RestMessageType.INFORM
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
    def from_dict(cls, data: Dict) -> "RestMessage":
        """Create message from dictionary."""
        return cls(
            message_id=data["message_id"],
            sender_id=data["sender_id"],
            receiver_id=data["receiver_id"],
            message_type=RestMessageType(data["message_type"]),
            content=data["content"],
            timestamp=data["timestamp"],
            reply_to=data.get("reply_to"),
        )

    def to_base_message(self) -> Message:
        """Convert to base message type."""
        return Message(
            message_id=self.message_id,
            sender_id=self.sender_id,
            receiver_id=self.receiver_id,
            message_type=self.message_type.to_base_type(),
            content=self.content,
            timestamp=self.timestamp,
            reply_to=self.reply_to,
        )

    @classmethod
    def from_base_message(cls, base_msg: Message) -> "RestMessage":
        """Create from base message type."""
        return cls(
            message_id=base_msg.message_id,
            sender_id=base_msg.sender_id,
            receiver_id=base_msg.receiver_id,
            message_type=RestMessageType.from_base_type(base_msg.message_type),
            content=base_msg.content,
            timestamp=base_msg.timestamp,
            reply_to=base_msg.reply_to,
        )


class RestMailbox:
    """
    MB_i: Mailbox for agent i using REST communication.
    Thread-safe message buffer for HTTP-based message delivery.
    """

    def __init__(self, agent_id: str, max_size: int = 1000):
        self.agent_id = agent_id
        self.messages: deque = deque(maxlen=max_size)
        self.lock = threading.Lock()

    def add_message(self, message: RestMessage):
        """Add message to mailbox (thread-safe)."""
        with self.lock:
            self.messages.append(message)

    def get_messages(self, clear: bool = True) -> List[RestMessage]:
        """Get all messages from mailbox, optionally clearing it."""
        with self.lock:
            messages = list(self.messages)
            if clear:
                self.messages.clear()
            return messages

    def peek_messages(self) -> List[RestMessage]:
        """Get messages without removing them."""
        return self.get_messages(clear=False)

    def size(self) -> int:
        """Get number of messages in mailbox."""
        with self.lock:
            return len(self.messages)


class RESTCommunicationService:
    """
    REST-based implementation of the communication framework.
    Provides HTTP endpoints for message delivery and mailbox access.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5000,
        latency_mode: LatencyMode = LatencyMode.END_TO_END,
        transport_mode: str = "http1",
    ):
        self.host = host
        self.port = port
        self.latency_mode = latency_mode
        self.transport_mode = transport_mode
        self.app = Flask(__name__)
        self.app.logger.setLevel(logging.WARNING)  # Reduce Flask logging

        # Communication components
        self.mailboxes: Dict[str, RestMailbox] = {}
        self.topology = CommunicationTopology()
        self.delivery_stats = {
            "messages_sent": 0,
            "messages_delivered": 0,
            "delivery_failures": 0,
            "avg_delivery_time": 0.0,
        }
        self.broadcast_queue: "queue.Queue[Tuple[str, Dict[str, Any]]]" = queue.Queue()
        self.broadcast_worker_stop = threading.Event()
        self.broadcast_worker = threading.Thread(
            target=self._broadcast_worker, daemon=True
        )
        self.broadcast_worker.start()

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
                message = RestMessage(
                    sender_id=agent_id,
                    receiver_id=data["receiver_id"],
                    message_type=RestMessageType(data["message_type"]),
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
                                "error": "Communication not allowed\
                                    or receiver not found",
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
                reachable = list(
                    self.topology.get_reachable_agents(agent_id)
                )
                payload = json.loads(json.dumps(data["content"]))
                self._enqueue_broadcast_task(agent_id, payload)
                return jsonify(
                    {
                        "status": "completed",
                        "delivered": len(reachable),
                        "failed": 0,
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

    def _enqueue_broadcast_task(
        self, sender_id: str, content: Dict[str, Any]
    ):
        payload = json.loads(json.dumps(content))
        self.broadcast_queue.put((sender_id, payload))

    def _broadcast_worker(self):
        """Background worker to deliver broadcast messages asynchronously."""
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

            sender_id, content = task
            reachable = self.topology.get_reachable_agents(sender_id)
            for receiver_id in reachable:
                message = RestMessage(
                    sender_id=sender_id,
                    receiver_id=receiver_id,
                    message_type=RestMessageType.BROADCAST,
                    content=json.loads(json.dumps(content)),
                )
                self.deliver_message(message)

            self.broadcast_queue.task_done()

    def register_agent(self, agent_id: str):
        """Register an agent and create its mailbox."""
        if agent_id not in self.mailboxes:
            self.mailboxes[agent_id] = RestMailbox(agent_id)

    def deliver_message(self, message: RestMessage) -> bool:
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

    def shutdown(self):
        """Stop background broadcast worker."""
        self.broadcast_worker_stop.set()
        self.broadcast_queue.put(None)
        if self.broadcast_worker.is_alive():
            self.broadcast_worker.join(timeout=1.0)


class RestCommunicatingAgent:
    """
    Agent wrapper that adds communication capabilities to existing agents.
    Extends agents with communication actions and mailbox access.
    """

    def __init__(
        self,
        agent_id: str,
        comm_service_url: str = "http://localhost:5000",
        transport_mode: str = "http1",
    ):
        self.agent_id = agent_id
        self.comm_service_url = comm_service_url
        self.transport_mode = transport_mode.lower()
        self.session = requests.Session()
        self.http2_client = None

        if self.transport_mode == "http2" and httpx:
            try:
                self.http2_client = httpx.Client(http2=True, timeout=5.0)
            except Exception:
                logging.warning(
                    "HTTP/2 client initialisation failed. Falling back to HTTP/1."
                )
                self.transport_mode = "http1"
        elif self.transport_mode == "http2":
            logging.warning(
                "httpx not available; HTTP/2 variant will use HTTP/1.1 fallback."
            )
            self.transport_mode = "http1"

    def _post(self, path: str, payload: Dict[str, Any]):
        url = f"{self.comm_service_url}{path}"
        try:
            if self.transport_mode == "http2" and self.http2_client:
                return self.http2_client.post(url, json=payload, timeout=5.0)
            return self.session.post(url, json=payload, timeout=5.0)
        except Exception:
            return None

    def _get(self, path: str, params: Dict[str, Any]):
        url = f"{self.comm_service_url}{path}"
        try:
            if self.transport_mode == "http2" and self.http2_client:
                return self.http2_client.get(url, params=params, timeout=5.0)
            return self.session.get(url, params=params, timeout=5.0)
        except Exception:
            return None

    def send_message(
        self,
        receiver_id: str,
        message_type: RestMessageType,
        content: Dict[str, Any],
        reply_to: Optional[str] = None,
    ) -> bool:
        """
        Communication action: send(i,j,m)
        Send message from this agent to target agent.
        """
        response = self._post(
            f"/agent/{self.agent_id}/send",
            {
                "receiver_id": receiver_id,
                "message_type": message_type.value,
                "content": content,
                "reply_to": reply_to,
            },
        )
        return bool(response and response.status_code == 200)

    def broadcast_message(
        self, message_type: RestMessageType, content: Dict[str, Any]
    ) -> Dict:
        """
        Broadcast message: send(i,*,m)
        Send message to all reachable agents.
        """
        response = self._post(
            f"/agent/{self.agent_id}/broadcast",
            {"message_type": message_type.value, "content": content},
        )
        if response and response.status_code == 200:
            return response.json()
        return {"status": "failed"}

    def receive_messages(
        self, clear_mailbox: bool = True
    ) -> List[RestMessage]:
        """
        Get messages from agent's mailbox.
        Part of agent's perception input.
        """
        response = self._get(
            f"/agent/{self.agent_id}/mailbox",
            {"clear": str(clear_mailbox).lower()},
        )
        if response and response.status_code == 200:
            data = response.json()
            return [RestMessage.from_dict(msg) for msg in data["messages"]]
        return []

    def reply_to_message(
        self, original_message: RestMessage, content: Dict[str, Any]
    ) -> bool:
        """Reply to a received message."""
        return self.send_message(
            receiver_id=original_message.sender_id,
            message_type=RestMessageType.REPLY,
            content=content,
            reply_to=original_message.message_id,
        )

    def configure_transport(
        self, comm_service_url: str, transport_mode: str
    ):
        """Update target service endpoint and transport semantics."""
        self.comm_service_url = comm_service_url
        new_mode = transport_mode.lower()

        if new_mode not in {"http1", "http2"}:
            new_mode = "http1"

        # Tear down existing HTTP/2 client if switching modes
        if self.http2_client:
            try:
                self.http2_client.close()
            except Exception:
                pass
            finally:
                self.http2_client = None

        self.transport_mode = new_mode

        if self.transport_mode == "http2":
            if not httpx:
                logging.warning(
                    "httpx not available; HTTP/2 variant will use HTTP/1.1 fallback."
                )
                self.transport_mode = "http1"
            else:
                try:
                    self.http2_client = httpx.Client(http2=True, timeout=5.0)
                except Exception:
                    logging.warning(
                        "HTTP/2 client initialisation failed. Falling back to HTTP/1."
                    )
                    self.transport_mode = "http1"

    def close(self):
        """Close any persistent HTTP clients."""
        if self.http2_client:
            try:
                self.http2_client.close()
            except Exception:
                pass
            finally:
                self.http2_client = None

        if self.session:
            try:
                self.session.close()
            except Exception:
                pass
            finally:
                self.session = None

    def close(self):
        """Close underlying HTTP clients."""
        with nullcontext():
            if self.http2_client:
                self.http2_client.close()
            self.session.close()
