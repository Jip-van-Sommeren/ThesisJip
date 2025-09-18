"""
Kafka Communicating Agent Implementation
Extends AbstractAgent with Kafka communication capabilities.
"""

from typing import Dict, List, Any, Optional
from communication.kafka.kafka_communication import (
    KafkaCommunicatingAgent, KafkaMessageType, KafkaMessage,
    KafkaCommunicationService
)
from abstract_agent import AbstractAgent


class ExtendedKafkaCommunicatingAgent(AbstractAgent):
    """
    Agent that extends AbstractAgent with Kafka communication capabilities.
    Integrates the Kafka communication model with the BDI-Reactive architecture.
    """

    def __init__(self, agent_id: str, kafka_service: KafkaCommunicationService):
        super().__init__(agent_id)
        
        # Initialize Kafka communication
        self.kafka_agent = KafkaCommunicatingAgent(agent_id, kafka_service)
        self.kafka_service = kafka_service
        
        # Register agent with the Kafka service
        self.kafka_service.register_agent(agent_id)

    def send_message(self, receiver_id: str, message_type: KafkaMessageType,
                    content: Dict[str, Any], reply_to: Optional[str] = None) -> bool:
        """
        Send message using Kafka communication.
        Implements send(i,j,m) from thesis definition.
        """
        return self.kafka_agent.send_message(
            receiver_id, message_type, content, reply_to
        )

    def broadcast_message(self, message_type: KafkaMessageType, 
                         content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Broadcast message using Kafka communication.
        Implements send(i,*,m) from thesis definition.
        """
        return self.kafka_agent.broadcast_message(message_type, content)

    def receive_messages(self, clear_mailbox: bool = True) -> List[KafkaMessage]:
        """
        Get messages from Kafka mailbox.
        Part of agent's perception input for BDI reasoning.
        """
        return self.kafka_agent.receive_messages(clear_mailbox)

    def reply_to_message(self, original_message: KafkaMessage, 
                        content: Dict[str, Any]) -> bool:
        """Reply to a received message using Kafka."""
        return self.kafka_agent.reply_to_message(original_message, content)

    def perceive(self) -> Dict[str, Any]:
        """
        Enhanced perception that includes Kafka messages.
        Extends the base perceive method with communication input.
        """
        perception = super().perceive()
        
        # Add messages to perception
        messages = self.receive_messages(clear_mailbox=False)
        perception['messages'] = messages
        perception['message_count'] = len(messages)
        
        return perception

    def deliberate(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced deliberation considering Kafka messages.
        Process received messages in BDI reasoning cycle.
        """
        # Get base deliberation
        decision = super().deliberate(perception)
        
        # Process messages if any
        if 'messages' in perception and perception['messages']:
            decision['process_messages'] = True
            decision['message_responses'] = []
            
            # Simple message processing logic
            for message in perception['messages']:
                if message.message_type == KafkaMessageType.REQUEST:
                    # Prepare response for requests
                    response_content = {
                        'status': 'received',
                        'agent_id': self.agent_id,
                        'timestamp': message.timestamp
                    }
                    decision['message_responses'].append({
                        'original_message': message,
                        'response_content': response_content
                    })
        
        return decision

    def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced action execution including Kafka communication.
        Execute communication actions based on decisions.
        """
        # Execute base actions
        result = super().act(decision)
        
        # Execute communication actions
        if decision.get('process_messages', False):
            # Clear messages after processing
            self.receive_messages(clear_mailbox=True)
            
            # Send replies if any
            if 'message_responses' in decision:
                for response_info in decision['message_responses']:
                    self.reply_to_message(
                        response_info['original_message'],
                        response_info['response_content']
                    )
                    
        result['communication_actions'] = decision.get('message_responses', [])
        
        return result

    def get_kafka_statistics(self) -> Dict[str, Any]:
        """Get Kafka communication statistics."""
        return self.kafka_service.get_statistics()

    def close(self):
        """Close Kafka connections when agent shuts down."""
        if hasattr(self, 'kafka_service'):
            # Note: We don't close the service here as it might be shared
            # The service should be closed by the environment
            pass