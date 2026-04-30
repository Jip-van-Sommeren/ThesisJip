"""
gRPC Communicating Agent Implementation
Extends AbstractAgent with gRPC communication capabilities.

Provides the same interface as REST communicating agents but uses gRPC
for message transmission following the formal communication model.
"""

from . import communication_pb2

from typing import Dict, Set, List, Optional, Any
import grpc
from mas.core import (
    Agent as AbstractAgent,
    AgentId,
    Action,
    ActionType,
    ReactiveRule,
)
from .grpc_communication import (
    GrpcCommunicatingAgent,
    GrpcCommunicationServer,
    GrpcMessage,
    GrpcMessageType,
)


class GrpcCommunicationAction(Action):
    """
    gRPC Communication-specific action.
    Implements send(i,j,m) communication actions via gRPC.
    """

    def __init__(self, action_id: str, grpc_agent: GrpcCommunicatingAgent):
        self.grpc_agent = grpc_agent
        super().__init__(
            action_id=action_id,
            action_type=ActionType.TRANSIENT,
            preconditions=lambda _: True,
            effects=self._execute_communication,
        )

    def _execute_communication(self, environment_state: Dict) -> Dict:
        """Execute communication action based on environment state."""
        return environment_state


class ExtendedGrpcCommunicatingAgent(AbstractAgent):
    """
    Agent that combines AbstractAgent capabilities with gRPC communication.

    Extends the formal agent definition A=(Id, State, Goal, Perception,
    Action, Decision)
    with gRPC communication components from the thesis:
    - Mailbox (MB_i) integration with perception
    - gRPC communication actions in Action set
    - Message handling in decision process
    """

    def __init__(
        self,
        agent_id: AgentId,
        observable_properties: Set[str],
        grpc_service_address: str = "localhost:50051",
        communication_mode: str = "unary",
    ):
        super().__init__(agent_id, observable_properties)
        self.communication_mode = communication_mode.lower()

        # Initialize gRPC communication capabilities
        self.grpc_agent = GrpcCommunicatingAgent(
            str(agent_id),
            grpc_service_address,
            communication_mode=self.communication_mode,
        )
        self.mailbox = None
        self.mailbox = None  # populated by environment registration

        # Add communication actions to agent's action set
        self._add_communication_actions()

        # Communication-specific state
        self.pending_replies: Dict[str, GrpcMessage] = {}
        self.message_history: List[GrpcMessage] = []

    def _add_communication_actions(self):
        """Add gRPC communication actions to the agent's action set Act_A."""

        # Basic send action
        send_action = GrpcCommunicationAction("send_message", self.grpc_agent)
        self.add_action(send_action)

        # Broadcast action
        broadcast_action = GrpcCommunicationAction(
            "broadcast_message", self.grpc_agent
        )
        self.add_action(broadcast_action)

        # Reply action
        reply_action = GrpcCommunicationAction(
            "reply_message", self.grpc_agent
        )
        self.add_action(reply_action)

    def perceive(self, environment_state: Dict):
        """
        Extended perception that includes gRPC mailbox messages.
        Implements Ω_A(e) + mailbox message retrieval via gRPC.
        """
        # Standard environment perception
        super().perceive(environment_state)

        # gRPC communication perception: retrieve messages from mailbox
        messages = self.grpc_agent.receive_messages(clear_mailbox=True)

        # Update state with received messages
        for message in messages:
            self.message_history.append(message)

            # Add message content as external beliefs
            belief_key = f"received_message_{message.message_id}"
            belief_content = f"message_from_{message.sender_id}_\
                {message.message_type.name.lower()}"
            self.state.update_belief(
                belief_key, belief_content, confidence=1.0, is_internal=False
            )

            # Handle reply requests
            if message.message_type == GrpcMessageType.REQUEST:
                self.pending_replies[message.message_id] = message

    def send_message(
        self,
        receiver_id: str,
        message_type: GrpcMessageType,
        content: Dict[str, Any],
        reply_to: Optional[str] = None,
    ) -> bool:
        """
        Communication action: send(i,j,m) via gRPC
        High-level interface for sending messages.
        """
        success = self.grpc_agent.send_message(
            receiver_id, message_type, content, reply_to
        )

        # Update internal beliefs about communication
        if success:
            self.state.update_belief(
                f"sent_to_{receiver_id}",
                f"message_sent_{message_type.name.lower()}",
                confidence=1.0,
                is_internal=True,
            )

        return success

    def broadcast_message(self, content: Dict[str, Any]) -> Dict:
        """
        Broadcast action: send(i,*,m) via gRPC
        Send message to all reachable agents.
        """
        result = self.grpc_agent.broadcast_message(
            GrpcMessageType.BROADCAST, content
        )

        # Update internal beliefs about broadcast
        self.state.update_belief(
            "last_broadcast",
            f"broadcast_sent_to_{result.get('delivered', 0)}_agents",
            confidence=1.0,
            is_internal=True,
        )

        return result

    def reply_to_pending_request(
        self, message_id: str, content: Dict[str, Any]
    ) -> bool:
        """Reply to a pending request message."""
        if message_id in self.pending_replies:
            original_message = self.pending_replies[message_id]
            success = self.grpc_agent.reply_to_message(
                original_message, content
            )

            if success:
                del self.pending_replies[message_id]

            return success
        return False

    def receive_messages(
        self, clear_mailbox: bool = True
    ) -> List[GrpcMessage]:
        """
        Get messages from agent's mailbox via gRPC.
        Wrapper around grpc_agent.receive_messages for convenience.
        """
        return self.grpc_agent.receive_messages(clear_mailbox)

    def get_received_messages(
        self,
        message_type: Optional[GrpcMessageType] = None,
        sender_id: Optional[str] = None,
    ) -> List[GrpcMessage]:
        """Get received messages filtered by type and/or sender."""
        filtered_messages = self.message_history

        if message_type:
            filtered_messages = [
                m for m in filtered_messages if m.message_type == message_type
            ]

        if sender_id:
            filtered_messages = [
                m for m in filtered_messages if m.sender_id == sender_id
            ]

        return filtered_messages

    def has_pending_replies(self) -> bool:
        """Check if agent has pending reply requests."""
        return len(self.pending_replies) > 0

    def initialize_agent(self):
        """
        Initialize gRPC communication-enabled agent.
        Subclasses should override this to add specific goals and
        reactive rules.
        """
        # Add basic communication reactive rules

        # Rule: Reply to requests automatically
        reply_rule = ReactiveRule(
            condition=lambda state: any(
                "request" in belief for belief in state["external"].values()
            ),
            action="reply_message",
            priority=0.8,
        )
        self.add_reactive_rule(reply_rule)

    def step(self, environment_state: Dict) -> tuple[str, Dict]:
        """
        Extended agent step that includes gRPC communication processing.
        Implements full sense-think-act cycle with gRPC communication.
        """
        # Standard step with enhanced perception (includes gRPC messages)
        chosen_action, new_environment = super().step(environment_state)

        # Process communication-specific actions
        if chosen_action == "reply_message" and self.pending_replies:
            # Auto-reply to first pending request (simple strategy)
            message_id = next(iter(self.pending_replies.keys()))
            self.reply_to_pending_request(
                message_id, {"status": "acknowledged"}
            )

        return chosen_action, new_environment

    def get_communication_stats(self) -> Dict[str, Any]:
        """Get agent's communication statistics."""
        return {
            "agent_id": str(self.id),
            "messages_received": len(self.message_history),
            "pending_replies": len(self.pending_replies),
            "message_types_received": list(
                set(m.message_type.name for m in self.message_history)
            ),
        }

    def close(self):
        """Close gRPC connections."""
        if self.grpc_agent:
            self.grpc_agent.close()


class GrpcCommunicationEnvironment:
    """
    Environment that manages multiple gRPC communicating agents.
    Handles agent registration and gRPC communication service setup.
    """

    def __init__(
        self,
        service_host: str = "localhost",
        service_port: int = 50051,
        latency_mode=None,
        communication_mode: str = "unary",
    ):
        from benchmarks.communication.base_communication import LatencyMode

        self.service_host = service_host
        self.service_port = service_port
        self.latency_mode = latency_mode or LatencyMode.END_TO_END
        self.communication_mode = communication_mode.lower()
        self.grpc_server = None
        self.agents: Dict[str, ExtendedGrpcCommunicatingAgent] = {}
        self.environment_state: Dict[str, Any] = {}
        self.is_running = False

    def start_service(self):
        """Start the gRPC communication service."""
        if self.is_running:
            return

        self.grpc_server = GrpcCommunicationServer(
            self.service_host,
            self.service_port,
            latency_mode=self.latency_mode,
        )
        self.grpc_server.start_server()
        self.service_port = self.grpc_server.port
        self.is_running = True

        print(
            f"gRPC communication service started on \
                {self.get_service_address()}"
        )

    def stop_service(self):
        """Stop the gRPC communication service."""
        if self.is_running and self.grpc_server:
            self.grpc_server.stop_server()
            self.is_running = False

    def get_service_address(self) -> str:
        """Get the service address."""
        return f"{self.service_host}:{self.service_port}"

    def register_agent(self, agent: ExtendedGrpcCommunicatingAgent):
        """Register an agent with the gRPC communication environment."""
        if not self.is_running:
            raise RuntimeError(
                "Service must be started before registering agents"
            )

        agent_id_str = str(agent.id)
        self.agents[agent_id_str] = agent

        # Register with gRPC service
        try:
            agent.grpc_agent.stub.RegisterAgent(
                communication_pb2.RegisterAgentRequest(agent_id=agent_id_str)
            )
        except grpc.RpcError as e:
            raise RuntimeError(f"Failed to register agent {agent_id_str}: {e}")

        # Update agent's service address
        agent.grpc_agent.service_address = self.get_service_address()
        agent.grpc_agent.communication_mode = self.communication_mode
        agent.communication_mode = self.communication_mode
        agent.grpc_agent._connect()

        # Provide direct access to server-side mailbox for benchmarking hooks
        if self.grpc_server and self.grpc_server.service_impl:
            agent.mailbox = self.grpc_server.service_impl.mailboxes.get(
                agent_id_str
            )

    def create_agent(
        self, agent_id: str, observable_properties: Set[str] = None
    ) -> ExtendedGrpcCommunicatingAgent:
        """Create and register a new gRPC communicating agent."""
        if not self.is_running:
            raise RuntimeError(
                "Environment not started. Call start_service() first."
            )

        if observable_properties is None:
            observable_properties = {"environment", "messages"}

        parts = agent_id.split(".")
        if len(parts) == 3:
            id_obj = AgentId(app=parts[0], type=parts[1], instance=parts[2])
        else:
            id_obj = AgentId(
                app="grpc_benchmark", type="agent", instance=agent_id
            )

        agent = ExtendedGrpcCommunicatingAgent(
            id_obj,
            observable_properties,
            grpc_service_address=self.get_service_address(),
            communication_mode=self.communication_mode,
        )
        agent.initialize_agent()
        self.register_agent(agent)
        return agent

    def setup_fully_connected_topology(self):
        """Setup fully connected communication topology."""
        agent_ids = list(self.agents.keys())

        for sender in agent_ids:
            for receiver in agent_ids:
                if sender != receiver:
                    self.add_communication_link(sender, receiver)

    def add_communication_link(self, sender_id: str, receiver_id: str):
        """Add specific communication link via gRPC."""
        if not self.is_running:
            raise RuntimeError(
                "Service must be running to add communication links"
            )

        try:
            # Use any agent's stub to add the link
            if self.agents:
                agent = next(iter(self.agents.values()))
                agent.grpc_agent.stub.AddCommunicationLink(
                    communication_pb2.AddLinkRequest(
                        sender_id=sender_id, receiver_id=receiver_id
                    )
                )
        except grpc.RpcError as e:
            print(
                f"Failed to add communication link\
                    {sender_id} -> {receiver_id}: {e}"
            )

    def step_all_agents(self) -> Dict[str, str]:
        """Execute one step for all agents."""
        actions = {}

        for agent_id, agent in self.agents.items():
            action, self.environment_state = agent.step(self.environment_state)
            actions[agent_id] = action

        return actions

    def get_system_stats(self) -> Dict[str, Any]:
        """Get communication statistics for the entire system."""
        agent_stats = {
            agent_id: agent.get_communication_stats()
            for agent_id, agent in self.agents.items()
        }

        # Get server statistics
        service_stats = {}
        if self.is_running and self.agents:
            try:
                agent = next(iter(self.agents.values()))
                response = agent.grpc_agent.stub.GetStatistics(
                    communication_pb2.StatisticsRequest()
                )
                service_stats = {
                    "messages_sent": response.messages_sent,
                    "messages_delivered": response.messages_delivered,
                    "delivery_failures": response.delivery_failures,
                    "avg_delivery_time": response.avg_delivery_time,
                }
            except grpc.RpcError:
                service_stats = {"error": "Could not retrieve statistics"}

        # Get topology info
        topology_links = 0
        if self.is_running and self.agents:
            try:
                agent = next(iter(self.agents.values()))
                response = agent.grpc_agent.stub.GetTopology(
                    communication_pb2.TopologyRequest()
                )
                topology_links = response.total_links
            except grpc.RpcError:
                pass

        return {
            "communication_service_stats": service_stats,
            "agent_stats": agent_stats,
            "topology_links": topology_links,
            "total_agents": len(self.agents),
        }

    def close_all_agents(self):
        """Close all agent connections."""
        for agent in self.agents.values():
            agent.close()
