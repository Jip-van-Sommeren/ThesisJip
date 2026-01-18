"""
Kafka Communicating Agent Implementation
Extends AbstractAgent with Kafka communication capabilities.
"""

import time
from typing import Dict, List, Any, Optional
from .kafka_communication import (
    KafkaCommunicatingAgent,
    KafkaMessageType,
    KafkaMessage,
    KafkaCommunicationService,
)
from mas.core import (
    Agent as AbstractAgent,
    Action,
    ActionType,
)


class KafkaCommunicationAction(Action):
    """
    Kafka Communication-specific action that extends the base Action class.
    Implements send(i,j,m) communication actions via Kafka messaging.
    """

    def __init__(self, action_id: str, kafka_agent: KafkaCommunicatingAgent):
        self.kafka_agent = kafka_agent
        super().__init__(
            action_id=action_id,
            action_type=ActionType.TRANSIENT,
            preconditions=lambda _: True,  # Communication always available
            effects=self._execute_communication,
        )

    def _execute_communication(self, environment_state: Dict) -> Dict:
        """Execute communication action based on environment state."""
        # Communication actions don't modify environment state directly
        # They operate on the Kafka communication layer
        return environment_state


class ExtendedKafkaCommunicatingAgent(AbstractAgent):
    """
    Agent that extends AbstractAgent with Kafka communication capabilities.
    Integrates the Kafka communication model with the BDI-Reactive
    architecture.
    """

    def __init__(
        self, agent_id: str, kafka_service: KafkaCommunicationService
    ):
        # Create AgentId from string
        from mas.core import AgentId
        if isinstance(agent_id, str):
            agent_id_obj = AgentId(app="kafka_benchmark", type="communicating", instance=agent_id)
        else:
            agent_id_obj = agent_id

        # Initialize parent with observable properties
        observable_properties = {"messages", "message_count", "communication_status"}
        super().__init__(agent_id_obj, observable_properties)

        # Store string ID for convenience
        self.agent_id = str(agent_id_obj) if not isinstance(agent_id, str) else agent_id

        # Initialize Kafka communication
        self.kafka_agent = KafkaCommunicatingAgent(self.agent_id, kafka_service)
        self.kafka_service = kafka_service

        # Register agent with the Kafka service
        self.kafka_service.register_agent(self.agent_id)
        self.mailbox = None

        # Add communication actions to agent's action set
        self._add_communication_actions()

        # Communication-specific state
        self.message_history: List[KafkaMessage] = []
        self.pending_replies: Dict[str, KafkaMessage] = {}

        # Initialize agent-specific configuration
        self.initialize_agent()

    def initialize_agent(self):
        """Initialize agent-specific goals, actions, and rules (required by AbstractAgent)."""
        from mas.core import Goal, GoalType, ReactiveRule

        # Add communication goal
        comm_goal = Goal(
            condition="maintain_communication",
            goal_type=GoalType.PERFORMANCE,
            priority=1.0
        )
        self.add_goal(comm_goal)

        # Add reactive rule for message processing
        def should_process_messages(state_dict):
            return state_dict.get("external", {}).get("message_count", "0") != "0"

        message_rule = ReactiveRule(
            condition=should_process_messages,
            action="send_message",
            priority=1.0
        )
        self.add_reactive_rule(message_rule)

    def _add_communication_actions(self):
        """Add communication actions to the agent's action set Act_A."""

        # Basic send action
        send_action = KafkaCommunicationAction(
            "send_message", self.kafka_agent
        )
        self.add_action(send_action)

        # Broadcast action
        broadcast_action = KafkaCommunicationAction(
            "broadcast_message", self.kafka_agent
        )
        self.add_action(broadcast_action)

        # Reply action
        reply_action = KafkaCommunicationAction(
            "reply_message", self.kafka_agent
        )
        self.add_action(reply_action)

    def send_message(
        self,
        receiver_id: str,
        message_type: KafkaMessageType,
        content: Dict[str, Any],
        reply_to: Optional[str] = None,
    ) -> bool:
        """
        Send message using Kafka communication.
        Implements send(i,j,m) from thesis definition.
        """
        return self.kafka_agent.send_message(
            receiver_id, message_type, content, reply_to
        )

    def broadcast_message(
        self, message_type: KafkaMessageType, content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Broadcast message using Kafka communication.
        Implements send(i,*,m) from thesis definition.
        """
        return self.kafka_agent.broadcast_message(message_type, content)

    def receive_messages(
        self, clear_mailbox: bool = True
    ) -> List[KafkaMessage]:
        """
        Get messages from Kafka mailbox.
        Part of agent's perception input for BDI reasoning.
        """
        return self.kafka_agent.receive_messages(clear_mailbox)

    def reply_to_message(
        self, original_message: KafkaMessage, content: Dict[str, Any]
    ) -> bool:
        """Reply to a received message using Kafka."""
        return self.kafka_agent.reply_to_message(original_message, content)

    def perceive(self, environment_state: Dict = None) -> Dict[str, Any]:
        """
        Enhanced perception that includes Kafka messages.
        Extends the base perceive method with communication input.
        """
        # Use empty dict if no environment state provided
        if environment_state is None:
            environment_state = {}

        # Call parent perceive (which updates beliefs)
        super().perceive(environment_state)

        # Communication perception: retrieve messages from mailbox
        messages = self.receive_messages(clear_mailbox=True)

        # Update state with received messages
        for message in messages:
            self.message_history.append(message)

            # Handle reply requests
            if message.message_type == KafkaMessageType.REQUEST:
                self.pending_replies[message.message_id] = message

        # Update beliefs with message information
        self.state.update_belief(
            "message_count",
            str(len(messages)),
            confidence=1.0,
            is_internal=False
        )

        # Return perception dict for backward compatibility
        return {
            "messages": messages,
            "message_count": len(messages),
            "environment_state": environment_state
        }

    def deliberate(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced deliberation considering Kafka messages.
        Process received messages in BDI reasoning cycle.
        """
        # Get base deliberation
        decision = super().deliberate(perception)

        # Process messages if any
        if "messages" in perception and perception["messages"]:
            decision["process_messages"] = True
            decision["message_responses"] = []

            # Simple message processing logic
            for message in perception["messages"]:
                if message.message_type == KafkaMessageType.REQUEST:
                    # Prepare response for requests
                    response_content = {
                        "status": "received",
                        "agent_id": self.agent_id,
                        "timestamp": message.timestamp,
                    }
                    decision["message_responses"].append(
                        {
                            "original_message": message,
                            "response_content": response_content,
                        }
                    )

        return decision

    def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced action execution including Kafka communication.
        Execute communication actions based on decisions.
        """
        # Execute base actions
        result = super().act(decision)

        # Execute communication actions
        if decision.get("process_messages", False):
            # Clear messages after processing
            self.receive_messages(clear_mailbox=True)

            # Send replies if any
            if "message_responses" in decision:
                for response_info in decision["message_responses"]:
                    self.reply_to_message(
                        response_info["original_message"],
                        response_info["response_content"],
                    )

        result["communication_actions"] = decision.get("message_responses", [])

        return result

    def get_received_messages(
        self,
        message_type: Optional[KafkaMessageType] = None,
        sender_id: Optional[str] = None,
    ) -> List[KafkaMessage]:
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

    def reply_to_pending_request(
        self, message_id: str, content: Dict[str, Any]
    ) -> bool:
        """Reply to a pending request message."""
        if message_id in self.pending_replies:
            original_message = self.pending_replies[message_id]
            success = self.reply_to_message(original_message, content)

            if success:
                del self.pending_replies[message_id]

            return success
        return False

    def get_communication_stats(self) -> Dict[str, Any]:
        """Get agent's communication statistics."""
        return {
            "agent_id": self.agent_id,
            "messages_received": len(self.message_history),
            "pending_replies": len(self.pending_replies),
            "message_types_received": list(
                set(m.message_type.value for m in self.message_history)
            ),
        }

    def get_kafka_statistics(self) -> Dict[str, Any]:
        """Get Kafka communication statistics."""
        return self.kafka_service.get_statistics()

    def close(self):
        """Close Kafka connections when agent shuts down."""
        if hasattr(self, "kafka_service"):
            # Note: We don't close the service here as it might be shared
            # The service should be closed by the environment
            pass


class KafkaCommunicationEnvironment:
    """
    Kafka-specific communication environment for managing multiple agents.
    Handles Kafka service setup, agent registration, and topology
    configuration.
    """

    def __init__(self, kafka_config: Dict[str, Any] = None, latency_mode=None):
        from benchmarks.communication.base_communication import LatencyMode

        self.kafka_config = kafka_config or {
            "bootstrap_servers": ["localhost:9092"],
            "client_id": "kafka_communication_environment",
        }
        self.latency_mode = latency_mode or LatencyMode.END_TO_END
        self.kafka_service = None
        self.agents: Dict[str, ExtendedKafkaCommunicatingAgent] = {}

    def setup(self, config: Dict[str, Any] = None):
        """Setup Kafka communication environment."""
        # config parameter reserved for future configuration options
        # Create Kafka communication service
        self.kafka_service = KafkaCommunicationService(
            self.kafka_config, latency_mode=self.latency_mode
        )

        # Minimal wait for Kafka service to be ready (optimized for benchmarks)
        time.sleep(0.05)  # 50ms instead of 500ms

    def create_agent(self, agent_id: str) -> ExtendedKafkaCommunicatingAgent:
        """Create a Kafka communicating agent."""
        if not self.kafka_service:
            raise RuntimeError("Environment not set up. Call setup() first.")

        agent = ExtendedKafkaCommunicatingAgent(agent_id, self.kafka_service)
        agent.mailbox = self.kafka_service.mailboxes.get(agent.agent_id)
        self.agents[agent_id] = agent
        return agent

    def register_agent(self, agent: ExtendedKafkaCommunicatingAgent):
        """Register an existing agent with the environment."""
        self.agents[agent.agent_id] = agent
        agent.mailbox = self.kafka_service.mailboxes.get(agent.agent_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Get Kafka communication statistics."""
        if self.kafka_service:
            return self.kafka_service.get_statistics()
        return {}

    def get_topology_info(self) -> Dict[str, Any]:
        """Get communication topology information."""
        if self.kafka_service:
            return self.kafka_service.get_topology_info()
        return {}

    def setup_topology(
        self,
        agents: List[ExtendedKafkaCommunicatingAgent],
        topology_type: str = "fully_connected",
    ):
        """Setup communication topology for agents."""
        if not self.kafka_service:
            return

        agent_ids = [agent.agent_id for agent in agents]

        if topology_type == "fully_connected":
            self.kafka_service.topology.create_fully_connected(agent_ids)
        elif topology_type == "star":
            # Hub agent can communicate with all others
            hub_id = agent_ids[0]
            for other_id in agent_ids[1:]:
                self.kafka_service.topology.add_link(hub_id, other_id)
                self.kafka_service.topology.add_link(other_id, hub_id)
        elif topology_type == "chain":
            # Linear chain topology
            for i in range(len(agent_ids) - 1):
                self.kafka_service.topology.add_link(
                    agent_ids[i], agent_ids[i + 1]
                )
                self.kafka_service.topology.add_link(
                    agent_ids[i + 1], agent_ids[i]
                )

    def add_communication_link(self, sender_id: str, receiver_id: str):
        """Add specific communication link."""
        if self.kafka_service:
            self.kafka_service.topology.add_link(sender_id, receiver_id)

    def remove_communication_link(self, sender_id: str, receiver_id: str):
        """Remove specific communication link."""
        if self.kafka_service:
            self.kafka_service.topology.remove_link(sender_id, receiver_id)

    def step_all_agents(self) -> Dict[str, Dict[str, Any]]:
        """Execute one step for all agents."""
        results = {}

        for agent_id, agent in self.agents.items():
            # Perceive -> Deliberate -> Act cycle
            perception = agent.perceive()
            decision = agent.deliberate(perception)
            action_result = agent.act(decision)

            results[agent_id] = {
                "perception": perception,
                "decision": decision,
                "action_result": action_result,
            }

        return results

    def get_system_stats(self) -> Dict[str, Any]:
        """Get communication statistics for the entire system."""
        agent_stats = {}
        for agent_id, agent in self.agents.items():
            agent_stats[agent_id] = agent.get_kafka_statistics()

        return {
            "kafka_service_stats": self.get_statistics(),
            "agent_stats": agent_stats,
            "topology_info": self.get_topology_info(),
            "total_agents": len(self.agents),
        }

    def teardown(self):
        """Clean up Kafka resources."""
        # Close all agents
        for agent in self.agents.values():
            agent.close()

        # Close Kafka service
        if self.kafka_service:
            self.kafka_service.close()
            self.kafka_service = None

        # Clear agents
        self.agents.clear()
