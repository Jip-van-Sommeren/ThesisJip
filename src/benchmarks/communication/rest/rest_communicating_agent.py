"""
REST Communicating Agent Implementation

Integrates the formal communication model (mailboxes, message space, topology)
with the existing BDI-Reactive agent architecture using REST communication.
"""

from typing import Dict, Set, List, Optional, Any
import time
import threading
from mas.core import (
    Agent as AbstractAgent,
    AgentId,
    Action,
    ActionType,
    ReactiveRule,
)
from benchmarks.communication.base_communication import (
    Message,
    MessageType,
)
from .rest_communication import (
    RestCommunicatingAgent,
    RESTCommunicationService,
)


class RestCommunicationAction(Action):
    """
    REST Communication-specific action that extends the base Action class.
    Implements send(i,j,m) communication actions via REST API.
    """

    def __init__(self, action_id: str, comm_agent: RestCommunicatingAgent):
        self.comm_agent = comm_agent
        super().__init__(
            action_id=action_id,
            action_type=ActionType.TRANSIENT,
            preconditions=lambda _: True,  # Communication always available
            effects=self._execute_communication,
        )

    def _execute_communication(self, environment_state: Dict) -> Dict:
        """Execute communication action based on environment state."""
        # Communication actions don't modify environment state directly
        # They operate on the communication layer
        return environment_state


class ExtendedRestCommunicatingAgent(AbstractAgent):
    """
    Agent that combines AbstractAgent capabilities with REST communication.

    Extends the formal agent definition A=(Id, State, Goal, Perception,
    Action, Decision)
    with REST communication components from the thesis:
    - Mailbox (MB_i) integration with perception
    - Communication actions in Action set
    - Message handling in decision process
    """

    def __init__(
        self,
        agent_id: AgentId,
        observable_properties: Set[str],
        comm_service_url: str = "http://localhost:5000",
        transport_mode: str = "http1",
    ):
        super().__init__(agent_id, observable_properties)

        # Initialize communication capabilities
        self.comm_agent = RestCommunicatingAgent(
            str(agent_id), comm_service_url, transport_mode=transport_mode
        )
        self.transport_mode = transport_mode.lower()
        self.mailbox = None

        # Add communication actions to agent's action set
        self._add_communication_actions()

        # Communication-specific state
        self.pending_replies: Dict[str, Message] = {}
        self.message_history: List[Message] = []

    def _add_communication_actions(self):
        """Add communication actions to the agent's action set Act_A."""

        # Basic send action
        send_action = RestCommunicationAction("send_message", self.comm_agent)
        self.add_action(send_action)

        # Broadcast action
        broadcast_action = RestCommunicationAction(
            "broadcast_message", self.comm_agent
        )
        self.add_action(broadcast_action)

        # Reply action
        reply_action = RestCommunicationAction(
            "reply_message", self.comm_agent
        )
        self.add_action(reply_action)

    def perceive(self, environment_state: Dict):
        """
        Extended perception that includes mailbox messages.
        Implements Ω_A(e) + mailbox message retrieval.
        """
        # Standard environment perception
        super().perceive(environment_state)

        # Communication perception: retrieve messages from mailbox
        messages = self.comm_agent.receive_messages(clear_mailbox=True)

        # Update state with received messages
        for message in messages:
            self.message_history.append(message)

            # Add message content as external beliefs
            belief_key = f"received_message_{message.message_id}"
            belief_content = f"message_from\
                _{message.sender_id}_{message.message_type.value}"
            self.state.update_belief(
                belief_key, belief_content, confidence=1.0, is_internal=False
            )

            # Handle reply requests
            if message.message_type == MessageType.REQUEST:
                self.pending_replies[message.message_id] = message

    def send_message(
        self,
        receiver_id: str,
        message_type: MessageType,
        content: Dict[str, Any],
        reply_to: Optional[str] = None,
    ) -> bool:
        """
        Communication action: send(i,j,m)
        High-level interface for sending messages.
        """
        success = self.comm_agent.send_message(
            receiver_id, message_type, content, reply_to
        )

        # Update internal beliefs about communication
        if success:
            self.state.update_belief(
                f"sent_to_{receiver_id}",
                f"message_sent_{message_type.value}",
                confidence=1.0,
                is_internal=True,
            )

        return success

    def broadcast_message(self, content: Dict[str, Any]) -> Dict:
        """
        Broadcast action: send(i,*,m)
        Send message to all reachable agents.
        """
        result = self.comm_agent.broadcast_message(
            MessageType.BROADCAST, content
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
            success = self.comm_agent.reply_to_message(
                original_message, content
            )

            if success:
                del self.pending_replies[message_id]

            return success
        return False

    def receive_messages(self, clear_mailbox: bool = True) -> List[Message]:
        """
        Get messages from agent's mailbox.
        Wrapper around comm_agent.receive_messages for convenience.
        """
        return self.comm_agent.receive_messages(clear_mailbox)

    def get_received_messages(
        self,
        message_type: Optional[MessageType] = None,
        sender_id: Optional[str] = None,
    ) -> List[Message]:
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
        Initialize communication-enabled agent.
        Subclasses should override this to add specific goals
        and reactive rules.
        """
        # Add basic communication reactive rules

        # Rule: Reply to requests automatically (example)
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
        Extended agent step that includes communication processing.
        Implements full sense-think-act cycle with communication.
        """
        # Standard step with enhanced perception (includes messages)
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
                set(m.message_type.value for m in self.message_history)
            ),
        }


class RestCommunicationEnvironment:
    """
    Environment that manages multiple REST communicating agents.
    Handles agent registration and REST communication service setup.
    """

    def __init__(
        self,
        service_host: str = "localhost",
        service_port: int = 5000,
        latency_mode=None,
        transport_mode: str = "http1",
    ):
        from benchmarks.communication.base_communication import LatencyMode

        self.service_host = service_host
        self.service_port = service_port
        self.latency_mode = latency_mode or LatencyMode.END_TO_END
        self.transport_mode = transport_mode
        self.comm_service = None
        self.agents: Dict[str, ExtendedRestCommunicatingAgent] = {}
        self.environment_state: Dict[str, Any] = {}
        self.service_thread = None
        self.is_running = False

    def start_service(self):
        """Start the communication service."""
        if self.is_running:
            return

        # Find available port if default is taken
        import socket

        port = self.service_port
        while port < self.service_port + 100:  # Try up to 100 ports
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind((self.service_host, port))
                    break
            except OSError:
                port += 1
        else:
            raise RuntimeError("No available ports found")

        self.service_port = port
        self.comm_service = RESTCommunicationService(
            self.service_host,
            port,
            latency_mode=self.latency_mode,
            transport_mode=self.transport_mode,
        )

        # Start communication service in background
        self.service_thread = threading.Thread(
            target=self.comm_service.start_server, daemon=True
        )
        self.service_thread.start()
        self.is_running = True
        time.sleep(1)  # Give service time to start

        print(f"Communication service started on {self.service_host}:{port}")

    def stop_service(self):
        """Stop the communication service."""
        if self.is_running and self.comm_service:
            try:
                self.comm_service.shutdown()
            except Exception:
                pass
            # Flask doesn't have a clean shutdown method, so we set a flag
            self.is_running = False
            # In a production environment, you'd use a proper WSGI server
            # that supports graceful shutdown
            print(f"Communication service on port {self.service_port} stopped")
        self.comm_service = None

    def get_service_url(self) -> str:
        """Get the service URL."""
        return f"http://{self.service_host}:{self.service_port}"

    def register_agent(self, agent: ExtendedRestCommunicatingAgent):
        """Register an agent with the communication environment."""
        if not self.is_running:
            raise RuntimeError(
                "Service must be started before registering agents"
            )

        agent_id_str = str(agent.id)
        self.agents[agent_id_str] = agent
        self.comm_service.register_agent(agent_id_str)

        # Update agent's communication service URL
        agent.comm_agent.configure_transport(
            self.get_service_url(), self.transport_mode
        )
        agent.transport_mode = self.transport_mode

        # Provide direct mailbox access for benchmark instrumentation
        agent.mailbox = self.comm_service.mailboxes.get(agent_id_str)

    def setup_fully_connected_topology(self):
        """Setup fully connected communication topology."""
        agent_ids = list(self.agents.keys())
        self.comm_service.topology.create_fully_connected(agent_ids)

    def add_communication_link(self, sender_id: str, receiver_id: str):
        """Add specific communication link."""
        self.comm_service.topology.add_link(sender_id, receiver_id)

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

        return {
            "communication_service_stats": self.comm_service.delivery_stats,
            "agent_stats": agent_stats,
            "topology_links": len(self.comm_service.topology.links),
            "total_agents": len(self.agents),
        }
