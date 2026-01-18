"""
gRPC-based Communication Model Implementation
Based on the formal communication definition from the thesis.

Implements the same formal model as REST but using gRPC:
- Message Space (M): Set of all possible messages
- Communication Topology (Comm): Agent-to-agent communication links
- Mailboxes (MB_i): Message buffers for each agent
- Delivery Function: Message transmission via gRPC calls
- Communication Actions: send(i,j,m) operations
"""

import grpc
import threading
import time
import uuid
import socket
import json
import queue
from concurrent import futures
from typing import Dict, List, Set, Tuple, Optional, Any, Iterable
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

# Import generated protobuf classes
from . import communication_pb2
from . import communication_pb2_grpc

# Import existing communication models for compatibility
from benchmarks.communication.base_communication import (
    MessageType as BaseMessageType,
    LatencyMode,
)


class GrpcMessageType(Enum):
    """gRPC Message types mapping to protobuf enum."""

    INFORM = communication_pb2.INFORM
    REQUEST = communication_pb2.REQUEST
    REPLY = communication_pb2.REPLY
    BROADCAST = communication_pb2.BROADCAST
    ERROR = communication_pb2.ERROR
    ACK = communication_pb2.ACK

    @classmethod
    def from_base_type(cls, base_type: BaseMessageType) -> "GrpcMessageType":
        """Convert base message type to gRPC message type."""
        mapping = {
            BaseMessageType.INFORM: cls.INFORM,
            BaseMessageType.REQUEST: cls.REQUEST,
            BaseMessageType.REPLY: cls.REPLY,
            BaseMessageType.BROADCAST: cls.BROADCAST,
            BaseMessageType.ACK: cls.ACK,
            BaseMessageType.ERROR: cls.ERROR,
        }
        return mapping[base_type]

    def to_base_type(self) -> BaseMessageType:
        """Convert gRPC message type to base message type."""
        mapping = {
            self.INFORM: BaseMessageType.INFORM,
            self.REQUEST: BaseMessageType.REQUEST,
            self.REPLY: BaseMessageType.REPLY,
            self.BROADCAST: BaseMessageType.BROADCAST,
            self.ERROR: BaseMessageType.ERROR,
        }
        return mapping[self]


@dataclass
class GrpcMessage:
    """
    Message representation for gRPC communication.
    Equivalent to the Message class in REST implementation.
    """

    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    sender_id: str = ""
    receiver_id: str = ""
    message_type: GrpcMessageType = GrpcMessageType.INFORM
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    reply_to: Optional[str] = None

    def to_proto(self) -> communication_pb2.Message:
        """Convert to protobuf message."""
        # Convert content dict to string-string map for protobuf
        content_str = {k: str(v) for k, v in self.content.items()}

        proto_msg = communication_pb2.Message(
            message_id=self.message_id,
            sender_id=self.sender_id,
            receiver_id=self.receiver_id,
            message_type=self.message_type.value,
            content=content_str,
            timestamp=self.timestamp,
        )

        if self.reply_to:
            proto_msg.reply_to = self.reply_to

        return proto_msg

    @classmethod
    def from_proto(cls, proto_msg: communication_pb2.Message) -> "GrpcMessage":
        """Create from protobuf message."""
        # Convert string content back to appropriate types
        content = {}
        for k, v in proto_msg.content.items():
            # Try to convert back to original types
            try:
                if v.lower() in ("true", "false"):
                    content[k] = v.lower() == "true"
                elif v.replace(".", "").replace("-", "").isdigit():
                    content[k] = float(v) if "." in v else int(v)
                else:
                    content[k] = v
            except Exception as e:
                print(f"Error converting content value: {e}")
                content[k] = v

        return cls(
            message_id=proto_msg.message_id,
            sender_id=proto_msg.sender_id,
            receiver_id=proto_msg.receiver_id,
            message_type=GrpcMessageType(proto_msg.message_type),
            content=content,
            timestamp=proto_msg.timestamp,
            reply_to=(
                proto_msg.reply_to if proto_msg.HasField("reply_to") else None
            ),
        )


class GrpcMailbox:
    """
    MB_i: Mailbox for agent i using gRPC.
    Thread-safe message buffer.
    """

    def __init__(self, agent_id: str, max_size: int = 1000):
        self.agent_id = agent_id
        self.messages: deque = deque(maxlen=max_size)
        self.lock = threading.Lock()

    def add_message(self, message: GrpcMessage):
        """Add message to mailbox (thread-safe)."""
        with self.lock:
            self.messages.append(message)

    def get_messages(self, clear: bool = True) -> List[GrpcMessage]:
        """Get all messages from mailbox, optionally clearing it."""
        with self.lock:
            messages = list(self.messages)
            if clear:
                self.messages.clear()
            return messages

    def peek_messages(self) -> List[GrpcMessage]:
        """Inspect messages without removing them."""
        with self.lock:
            return list(self.messages)

    def size(self) -> int:
        """Get number of messages in mailbox."""
        with self.lock:
            return len(self.messages)


class GrpcCommunicationTopology:
    """
    Comm ⊆ A × A: Communication topology for gRPC.
    Same formal definition as REST implementation.
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


class CommunicationServiceImpl(
    communication_pb2_grpc.CommunicationServiceServicer
):
    """
    gRPC service implementation for agent communication.
    Implements the formal communication model via gRPC RPCs.
    """

    def __init__(self, latency_mode=None):
        self.latency_mode = latency_mode or LatencyMode.END_TO_END
        self.mailboxes: Dict[str, GrpcMailbox] = {}
        self.topology = GrpcCommunicationTopology()
        self.delivery_stats = {
            "messages_sent": 0,
            "messages_delivered": 0,
            "delivery_failures": 0,
            "avg_delivery_time": 0.0,
        }
        self.lock = threading.Lock()
        self.broadcast_queue: "queue.Queue[Tuple[str, int, Dict[str, str]]]" = queue.Queue()
        self.broadcast_worker_stop = threading.Event()
        self.broadcast_worker = threading.Thread(
            target=self._broadcast_worker, daemon=True
        )
        self.broadcast_worker.start()

    def RegisterAgent(self, request, context):
        """Register an agent and create its mailbox."""
        with self.lock:
            if request.agent_id not in self.mailboxes:
                self.mailboxes[request.agent_id] = GrpcMailbox(
                    request.agent_id
                )

        return communication_pb2.RegisterAgentResponse(success=True)

    def SendMessage(self, request, context):
        """
        Send message from one agent to another: implements send(i,j,m).
        """
        success, message_id, error_message = self._deliver_request(request)
        response_kwargs = {
            "success": success,
            "message_id": message_id,
        }
        if not success and error_message:
            response_kwargs["error_message"] = error_message

        return communication_pb2.SendMessageResponse(**response_kwargs)

    def BroadcastMessage(self, request, context):
        """
        Broadcast message to all reachable agents: implements send(i,*,m).
        """
        reachable = list(self.topology.get_reachable_agents(request.sender_id))
        payload = dict(request.content)
        self._enqueue_broadcast_task(
            request.sender_id, request.message_type, payload
        )

        return communication_pb2.BroadcastMessageResponse(
            success=True,
            delivered=len(reachable),
            failed=0,
            total_targets=len(reachable),
            message_ids=[],
        )

    def GetMessages(self, request, context):
        """
        Get messages from agent's mailbox: implements MB_i access.
        """
        if request.agent_id not in self.mailboxes:
            return communication_pb2.GetMessagesResponse(messages=[], count=0)

        messages = self.mailboxes[request.agent_id].get_messages(
            request.clear_mailbox
        )
        proto_messages = [msg.to_proto() for msg in messages]

        return communication_pb2.GetMessagesResponse(
            messages=proto_messages, count=len(proto_messages)
        )

    def AddCommunicationLink(self, request, context):
        """Add communication link to topology: manages Comm relation."""
        self.topology.add_link(request.sender_id, request.receiver_id)
        return communication_pb2.AddLinkResponse(success=True)

    def GetStatistics(self, request, context):
        """Get communication statistics for benchmarking."""
        with self.lock:
            stats = self.delivery_stats.copy()

        return communication_pb2.StatisticsResponse(
            messages_sent=stats["messages_sent"],
            messages_delivered=stats["messages_delivered"],
            delivery_failures=stats["delivery_failures"],
            avg_delivery_time=stats["avg_delivery_time"],
        )

    def _enqueue_broadcast_task(
        self, sender_id: str, message_type: int, content: Dict[str, str]
    ):
        payload = json.loads(json.dumps(content))
        self.broadcast_queue.put((sender_id, message_type, payload))

    def _broadcast_worker(self):
        """Asynchronously deliver broadcast messages to subscribers."""
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

            sender_id, message_type, content = task
            reachable = self.topology.get_reachable_agents(sender_id)
            for receiver_id in reachable:
                send_request = communication_pb2.SendMessageRequest(
                    sender_id=sender_id,
                    receiver_id=receiver_id,
                    message_type=message_type,
                )
                for key, value in content.items():
                    send_request.content[key] = value
                self._deliver_request(send_request)

            self.broadcast_queue.task_done()

    def shutdown(self):
        """Stop background workers gracefully."""
        self.broadcast_worker_stop.set()
        self.broadcast_queue.put(None)
        if self.broadcast_worker.is_alive():
            self.broadcast_worker.join(timeout=1.0)

    def GetTopology(self, request, context):
        """Get current communication topology."""
        links = []
        with self.topology.lock:
            for sender, receiver in self.topology.links:
                links.append(
                    communication_pb2.CommunicationLink(
                        sender_id=sender, receiver_id=receiver
                    )
                )

        return communication_pb2.TopologyResponse(
            links=links, total_links=len(links)
        )

    def StreamMessages(self, request_iterator, context):
        """Handle client-streaming message delivery."""
        success = True
        message_ids = []
        first_error: Optional[str] = None

        for request in request_iterator:
            delivered, message_id, error_message = self._deliver_request(
                request
            )
            if message_id:
                message_ids.append(message_id)
            if not delivered:
                success = False
                if first_error is None:
                    first_error = error_message or "Unknown delivery error"

        response_kwargs = {
            "success": success,
            "message_ids": message_ids,
        }
        if not success and first_error:
            response_kwargs["error_message"] = first_error

        return communication_pb2.StreamSummaryResponse(**response_kwargs)

    def _deliver_request(
        self, request: communication_pb2.SendMessageRequest
    ) -> Tuple[bool, str, Optional[str]]:
        """Core delivery routine shared across RPC variants."""
        start_time = time.time()

        with self.lock:
            self.delivery_stats["messages_sent"] += 1

        if not self.topology.can_communicate(
            request.sender_id, request.receiver_id
        ):
            with self.lock:
                self.delivery_stats["delivery_failures"] += 1
            return False, "", "Communication not allowed"

        if request.receiver_id not in self.mailboxes:
            with self.lock:
                self.delivery_stats["delivery_failures"] += 1
            return False, "", "Receiver not found"

        message = GrpcMessage(
            sender_id=request.sender_id,
            receiver_id=request.receiver_id,
            message_type=GrpcMessageType(request.message_type),
            content=dict(request.content),
            reply_to=(
                request.reply_to if request.HasField("reply_to") else None
            ),
        )

        self.mailboxes[request.receiver_id].add_message(message)

        delivery_time = time.time() - start_time
        with self.lock:
            self.delivery_stats["messages_delivered"] += 1
            old_avg = self.delivery_stats["avg_delivery_time"]
            count = self.delivery_stats["messages_delivered"]
            self.delivery_stats["avg_delivery_time"] = (
                old_avg * (count - 1) + delivery_time
            ) / count

        return True, message.message_id, None


class GrpcCommunicationServer:
    """
    gRPC communication server that manages the service.
    """

    def __init__(
        self, host: str = "localhost", port: int = 50051, latency_mode=None
    ):
        self.host = host
        self.port = port
        self.latency_mode = latency_mode or LatencyMode.END_TO_END
        self.server = None
        self.service_impl = CommunicationServiceImpl(
            latency_mode=self.latency_mode
        )
        self.is_running = False

    def start_server(self):
        """Start the gRPC server."""
        if self.is_running:
            return

        # Find available port
        port = self.port
        while port < self.port + 100:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind((self.host, port))
                    break
            except OSError:
                port += 1
        else:
            raise RuntimeError("No available ports found")

        self.port = port

        # Create and start server
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        communication_pb2_grpc.add_CommunicationServiceServicer_to_server(
            self.service_impl, self.server
        )

        listen_addr = f"{self.host}:{self.port}"
        self.server.add_insecure_port(listen_addr)
        self.server.start()
        self.is_running = True

        print(f"gRPC communication server started on {listen_addr}")

    def stop_server(self):
        """Stop the gRPC server."""
        if self.is_running and self.server:
            self.server.stop(grace=1.0)
            self.is_running = False
            print(f"gRPC communication server on port {self.port} stopped")
        self.service_impl.shutdown()

    def get_service_address(self) -> str:
        """Get the service address."""
        return f"{self.host}:{self.port}"


class GrpcCommunicatingAgent:
    """
    Agent client for gRPC communication.
    Provides the same interface as REST CommunicatingAgent.
    """

    def __init__(
        self,
        agent_id: str,
        service_address: str = "localhost:50051",
        communication_mode: str = "unary",
    ):
        self.agent_id = agent_id
        self.service_address = service_address
        self.communication_mode = communication_mode.lower()
        self.channel = None
        self.stub = None
        self.stream_lock = threading.Lock()
        self._connect()

    def _connect(self):
        """Connect to the gRPC service."""
        if self.channel:
            try:
                self.channel.close()
            except Exception:
                pass
        self.channel = grpc.insecure_channel(self.service_address)
        self.stub = communication_pb2_grpc.CommunicationServiceStub(
            self.channel
        )

    def send_message(
        self,
        receiver_id: str,
        message_type: GrpcMessageType,
        content: Dict[str, Any],
        reply_to: Optional[str] = None,
    ) -> bool:
        """
        Communication action: send(i,j,m)
        Send message from this agent to target agent.
        """
        try:
            # Convert content to string dict for protobuf
            content_str = {k: str(v) for k, v in content.items()}

            request = communication_pb2.SendMessageRequest(
                sender_id=self.agent_id,
                receiver_id=receiver_id,
                message_type=message_type.value,
                content=content_str,
            )

            if reply_to:
                request.reply_to = reply_to

            if self.communication_mode == "streaming":
                return self._stream_messages([request])

            response = self.stub.SendMessage(request, timeout=5.0)
            return response.success
        except grpc.RpcError:
            return False

    def broadcast_message(
        self, message_type: GrpcMessageType, content: Dict[str, Any]
    ) -> Dict:
        """
        Broadcast message: send(i,*,m)
        Send message to all reachable agents.
        """
        try:
            content_str = {k: str(v) for k, v in content.items()}

            request = communication_pb2.BroadcastMessageRequest(
                sender_id=self.agent_id,
                message_type=message_type.value,
                content=content_str,
            )

            response = self.stub.BroadcastMessage(request, timeout=5.0)
            return {
                "status": "completed" if response.success else "failed",
                "delivered": response.delivered,
                "failed": response.failed,
                "total_targets": response.total_targets,
            }
        except grpc.RpcError:
            return {"status": "failed"}

    def receive_messages(
        self, clear_mailbox: bool = True
    ) -> List[GrpcMessage]:
        """
        Get messages from agent's mailbox.
        Part of agent's perception input.
        """
        try:
            request = communication_pb2.GetMessagesRequest(
                agent_id=self.agent_id, clear_mailbox=clear_mailbox
            )

            response = self.stub.GetMessages(request, timeout=5.0)
            return [GrpcMessage.from_proto(msg) for msg in response.messages]
        except grpc.RpcError:
            return []

    def reply_to_message(
        self, original_message: GrpcMessage, content: Dict[str, Any]
    ) -> bool:
        """Reply to a received message."""
        return self.send_message(
            receiver_id=original_message.sender_id,
            message_type=GrpcMessageType.REPLY,
            content=content,
            reply_to=original_message.message_id,
        )

    def close(self):
        """Close the gRPC channel."""
        if self.channel:
            self.channel.close()
            self.channel = None
        self.stub = None

    def _stream_messages(
        self, requests: Iterable[communication_pb2.SendMessageRequest]
    ) -> bool:
        """Send one or more messages using the StreamMessages RPC."""

        def request_iter():
            for req in requests:
                yield req

        try:
            with self.stream_lock:
                response = self.stub.StreamMessages(
                    request_iter(), timeout=5.0
                )
            return response.success
        except grpc.RpcError:
            return False
