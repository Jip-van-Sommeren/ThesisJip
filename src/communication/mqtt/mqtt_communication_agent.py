"""
MQTT Communicating Agent Implementation
Extends AbstractAgent with MQTT communication capabilities.

Provides the same interface as REST, gRPC, and Kafka communicating agents
but uses MQTT for message transmission following the formal communication
model.
"""

from typing import Dict, List, Any, Optional, Set
from .mqtt_communication import (
    MqttCommunicatingAgent,
    MqttMessageType,
    MqttMessage,
    MqttCommunicationService,
)
from abstract_agent import (
    AbstractAgent,
    AgentId,
    Action,
    ActionType,
    ReactiveRule,
)


class MqttCommunicationAction(Action):
    """
    MQTT Communication-specific action that extends the base Action class.
    Implements send(i,j,m) communication actions via MQTT messaging.
    """

    def __init__(self, action_id: str, mqtt_agent: MqttCommunicatingAgent):
        self.mqtt_agent = mqtt_agent
        super().__init__(
            action_id=action_id,
            action_type=ActionType.TRANSIENT,
            preconditions=lambda _: True,  # Communication always available
            effects=self._execute_communication,
        )

    def _execute_communication(self, environment_state: Dict) -> Dict:
        """Execute communication action based on environment state."""
        # Communication actions don't modify environment state directly
        # They operate on the MQTT communication layer
        return environment_state


class ExtendedMqttCommunicatingAgent(AbstractAgent):
    """
    Agent that extends AbstractAgent with MQTT communication capabilities.

    Extends the formal agent definition A=(Id, State, Goal, Perception,
    Action, Decision) with MQTT communication components from the thesis:
    - Mailbox (MB_i) integration with perception
    - MQTT communication actions in Action set
    - Message handling in decision process
    """

    def __init__(
        self,
        agent_id: AgentId,
        observable_properties: Set[str],
        mqtt_service: MqttCommunicationService,
    ):
        super().__init__(agent_id, observable_properties)

        # Initialize MQTT communication capabilities
        self.mqtt_agent = MqttCommunicatingAgent(
            str(agent_id), mqtt_service, mqtt_service.mqtt_config
        )
        self.mqtt_service = mqtt_service
        self.mailbox = self.mqtt_agent.mailbox

        # Register agent with the MQTT service
        self.mqtt_service.register_agent(str(agent_id), self.mailbox)

        # Add communication actions to agent's action set
        self._add_communication_actions()

        # Communication-specific state
        self.pending_replies: Dict[str, MqttMessage] = {}
        self.message_history: List[MqttMessage] = []

    def _add_communication_actions(self):
        """Add MQTT communication actions to the agent's action set Act_A."""

        # Basic send action
        send_action = MqttCommunicationAction("send_message", self.mqtt_agent)
        self.add_action(send_action)

        # Broadcast action
        broadcast_action = MqttCommunicationAction(
            "broadcast_message", self.mqtt_agent
        )
        self.add_action(broadcast_action)

        # Reply action
        reply_action = MqttCommunicationAction(
            "reply_message", self.mqtt_agent
        )
        self.add_action(reply_action)

    def perceive(self, environment_state: Dict):
        """
        Extended perception that includes MQTT mailbox messages.
        Implements Ω_A(e) + mailbox message retrieval via MQTT.
        """
        # Standard environment perception
        super().perceive(environment_state)

        # MQTT communication perception: retrieve messages from mailbox
        messages = self.mqtt_agent.receive_messages(clear_mailbox=True)

        # Update state with received messages
        for message in messages:
            self.message_history.append(message)

            # Add message content as external beliefs
            belief_key = f"received_message_{message.message_id}"
            belief_content = f"message_\
                from_{message.sender_id}_{message.message_type.value}"
            self.state.update_belief(
                belief_key, belief_content, confidence=1.0, is_internal=False
            )

            # Handle reply requests
            if message.message_type == MqttMessageType.REQUEST:
                self.pending_replies[message.message_id] = message

    def send_message(
        self,
        receiver_id: str,
        message_type: MqttMessageType,
        content: Dict[str, Any],
        reply_to: Optional[str] = None,
    ) -> bool:
        """
        Communication action: send(i,j,m) via MQTT
        High-level interface for sending messages.
        """
        success = self.mqtt_agent.send_message(
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
        Broadcast action: send(i,*,m) via MQTT
        Send message to all reachable agents.
        """
        result = self.mqtt_agent.broadcast_message(
            MqttMessageType.BROADCAST, content
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
            success = self.mqtt_agent.reply_to_message(
                original_message, content
            )

            if success:
                del self.pending_replies[message_id]

            return success
        return False

    def receive_messages(
        self, clear_mailbox: bool = True
    ) -> List[MqttMessage]:
        """
        Get messages from agent's mailbox via MQTT.
        Wrapper around mqtt_agent.receive_messages for convenience.
        """
        return self.mqtt_agent.receive_messages(clear_mailbox)

    def get_received_messages(
        self,
        message_type: Optional[MqttMessageType] = None,
        sender_id: Optional[str] = None,
    ) -> List[MqttMessage]:
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
        Initialize MQTT communication-enabled agent.
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
        Extended agent step that includes MQTT communication processing.
        Implements full sense-think-act cycle with MQTT communication.
        """
        # Standard step with enhanced perception (includes MQTT messages)
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

    def close(self):
        """Close MQTT connections."""
        if self.mqtt_agent:
            self.mqtt_agent.close()


class MqttCommunicationEnvironment:
    """
    Environment that manages multiple MQTT communicating agents.
    Handles agent registration and MQTT communication service setup.
    """

    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        mqtt_config: Dict[str, Any] = None,
        latency_mode=None,
    ):
        from communication.base_communication import LatencyMode

        self.broker_host = broker_host
        self.broker_port = broker_port
        self.mqtt_config = mqtt_config or {
            "broker_host": broker_host,
            "broker_port": broker_port,
            "keepalive": 60,
            "qos": 1,
        }
        self.latency_mode = latency_mode or LatencyMode.SEND_ONLY

        self.mqtt_service = None
        self.agents: Dict[str, ExtendedMqttCommunicatingAgent] = {}
        self.environment_state: Dict[str, Any] = {}
        self.is_running = False

    def start_service(self):
        """Start the MQTT communication service."""
        if self.is_running:
            return

        self.mqtt_service = MqttCommunicationService(
            self.mqtt_config, latency_mode=self.latency_mode
        )
        self.mqtt_service.start_service()
        self.is_running = self.mqtt_service.is_running

        print(
            f"MQTT communication service started \
                on {self.get_service_address()}"
        )

    def stop_service(self):
        """Stop the MQTT communication service."""
        if self.is_running and self.mqtt_service:
            self.mqtt_service.stop_service()
            self.is_running = False

    def get_service_address(self) -> str:
        """Get the service address."""
        return f"{self.broker_host}:{self.broker_port}"

    def register_agent(self, agent: ExtendedMqttCommunicatingAgent):
        """Register an agent with the MQTT communication environment."""
        if not self.is_running:
            raise RuntimeError(
                "Service must be started before registering agents"
            )

        agent_id_str = str(agent.id)
        self.agents[agent_id_str] = agent

        # Agent is already registered with MQTT service in its constructor

    def setup_fully_connected_topology(self):
        """Setup fully connected communication topology."""
        agent_ids = list(self.agents.keys())

        if self.mqtt_service:
            self.mqtt_service.topology.create_fully_connected(agent_ids)

    def add_communication_link(self, sender_id: str, receiver_id: str):
        """Add specific communication link via MQTT."""
        if not self.is_running:
            raise RuntimeError(
                "Service must be running to add communication links"
            )

        if self.mqtt_service:
            self.mqtt_service.topology.add_link(sender_id, receiver_id)

    def remove_communication_link(self, sender_id: str, receiver_id: str):
        """Remove specific communication link."""
        if self.mqtt_service:
            self.mqtt_service.topology.remove_link(sender_id, receiver_id)

    def setup_topology(
        self,
        agents: List[ExtendedMqttCommunicatingAgent],
        topology_type: str = "fully_connected",
    ):
        """Setup communication topology for agents."""
        if not self.mqtt_service:
            return

        agent_ids = [str(agent.id) for agent in agents]

        if topology_type == "fully_connected":
            self.mqtt_service.topology.create_fully_connected(agent_ids)
        elif topology_type == "star":
            # Hub agent can communicate with all others
            hub_id = agent_ids[0]
            for other_id in agent_ids[1:]:
                self.mqtt_service.topology.add_link(hub_id, other_id)
                self.mqtt_service.topology.add_link(other_id, hub_id)
        elif topology_type == "chain":
            # Linear chain topology
            for i in range(len(agent_ids) - 1):
                self.mqtt_service.topology.add_link(
                    agent_ids[i], agent_ids[i + 1]
                )
                self.mqtt_service.topology.add_link(
                    agent_ids[i + 1], agent_ids[i]
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

        # Get service statistics
        service_stats = {}
        if self.mqtt_service:
            service_stats = self.mqtt_service.get_statistics()

        # Get topology info
        topology_info = {}
        if self.mqtt_service:
            topology_info = self.mqtt_service.get_topology_info()

        return {
            "mqtt_service_stats": service_stats,
            "agent_stats": agent_stats,
            "topology_info": topology_info,
            "total_agents": len(self.agents),
            "service_address": self.get_service_address(),
        }

    def close_all_agents(self):
        """Close all agent connections."""
        for agent in self.agents.values():
            agent.close()

        if self.mqtt_service:
            self.mqtt_service.close()

    def create_agent(
        self, agent_id: str, observable_properties: Set[str] = None
    ) -> ExtendedMqttCommunicatingAgent:
        """Create a new MQTT communicating agent."""
        if not self.is_running:
            raise RuntimeError(
                "Environment not started. Call start_service() first."
            )

        if observable_properties is None:
            observable_properties = set()

        # Create AgentId
        parts = agent_id.split(".")
        if len(parts) == 3:
            id_obj = AgentId(app=parts[0], type=parts[1], instance=parts[2])
        else:
            id_obj = AgentId(app="mqtt_app", type="agent", instance=agent_id)

        agent = ExtendedMqttCommunicatingAgent(
            id_obj, observable_properties, self.mqtt_service
        )
        agent.mailbox = agent.mqtt_agent.mailbox

        self.agents[str(id_obj)] = agent
        return agent

    def get_agent(
        self, agent_id: str
    ) -> Optional[ExtendedMqttCommunicatingAgent]:
        """Get agent by ID."""
        return self.agents.get(agent_id)

    def get_all_agents(self) -> List[ExtendedMqttCommunicatingAgent]:
        """Get all registered agents."""
        return list(self.agents.values())

    def send_message_between_agents(
        self,
        sender_id: str,
        receiver_id: str,
        message_type: MqttMessageType,
        content: Dict[str, Any],
    ) -> bool:
        """Helper method to send message between agents."""
        sender = self.get_agent(sender_id)
        if sender:
            return sender.send_message(receiver_id, message_type, content)
        return False
