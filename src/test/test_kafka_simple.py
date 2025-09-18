"""
Simple Kafka Communication Tests
Basic test cases for Kafka-based agent communication system.
"""

import unittest
import time
from communication.kafka.kafka_communication import (
    KafkaCommunicationService, KafkaMessage, KafkaMessageType,
    KafkaCommunicatingAgent
)
from communication.kafka.kafka_communication_agent import (
    ExtendedKafkaCommunicatingAgent
)


class TestKafkaSimple(unittest.TestCase):
    """Simple test cases for Kafka communication."""

    def setUp(self):
        """Set up test environment."""
        # Use test Kafka configuration
        self.kafka_config = {
            'bootstrap_servers': ['localhost:9092'],
            'client_id': 'test_kafka_client'
        }
        
        # Create Kafka service
        self.kafka_service = KafkaCommunicationService(self.kafka_config)
        
        # Wait for service to initialize
        time.sleep(0.5)

    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, 'kafka_service'):
            self.kafka_service.close()

    def test_kafka_message_creation(self):
        """Test basic Kafka message creation and serialization."""
        message = KafkaMessage(
            sender_id="test_sender",
            receiver_id="test_receiver", 
            message_type=KafkaMessageType.INFORM,
            content={"test": "data"}
        )
        
        # Test message fields
        self.assertEqual(message.sender_id, "test_sender")
        self.assertEqual(message.receiver_id, "test_receiver")
        self.assertEqual(message.message_type, KafkaMessageType.INFORM)
        self.assertEqual(message.content["test"], "data")
        
        # Test JSON serialization
        json_str = message.to_json()
        self.assertIsInstance(json_str, str)
        
        # Test JSON deserialization
        restored_message = KafkaMessage.from_json(json_str)
        self.assertEqual(restored_message.sender_id, message.sender_id)
        self.assertEqual(restored_message.receiver_id, message.receiver_id)
        self.assertEqual(restored_message.message_type, message.message_type)
        self.assertEqual(restored_message.content, message.content)

    def test_agent_registration(self):
        """Test agent registration with Kafka service."""
        # Register test agents
        self.kafka_service.register_agent("agent_1")
        self.kafka_service.register_agent("agent_2")
        
        # Check that mailboxes were created
        self.assertIn("agent_1", self.kafka_service.mailboxes)
        self.assertIn("agent_2", self.kafka_service.mailboxes)

    def test_communication_topology(self):
        """Test communication topology setup."""
        # Register agents
        self.kafka_service.register_agent("agent_a")
        self.kafka_service.register_agent("agent_b")
        
        # Add communication link
        self.kafka_service.topology.add_link("agent_a", "agent_b")
        
        # Test topology queries
        self.assertTrue(
            self.kafka_service.topology.can_communicate("agent_a", "agent_b")
        )
        self.assertFalse(
            self.kafka_service.topology.can_communicate("agent_b", "agent_a")
        )
        
        # Test fully connected topology
        agents = ["agent_1", "agent_2", "agent_3"]
        self.kafka_service.topology.create_fully_connected(agents)
        
        for sender in agents:
            for receiver in agents:
                if sender != receiver:
                    self.assertTrue(
                        self.kafka_service.topology.can_communicate(sender, receiver)
                    )

    def test_basic_message_sending(self):
        """Test basic message sending between agents."""
        # Setup agents
        self.kafka_service.register_agent("sender")
        self.kafka_service.register_agent("receiver")
        self.kafka_service.topology.add_link("sender", "receiver")
        
        # Create message
        message = KafkaMessage(
            sender_id="sender",
            receiver_id="receiver",
            message_type=KafkaMessageType.INFORM,
            content={"message": "Hello Kafka!"}
        )
        
        # Send message
        success = self.kafka_service.send_message(message)
        self.assertTrue(success, "Message sending should succeed")
        
        # Allow time for delivery (important for Kafka)
        time.sleep(1.0)
        
        # Check message was received
        received_messages = self.kafka_service.get_messages("receiver")
        self.assertGreaterEqual(len(received_messages), 1, 
                               "Should receive at least one message")
        
        # Verify message content
        if received_messages:
            received_msg = received_messages[0]
            self.assertEqual(received_msg.sender_id, "sender")
            self.assertEqual(received_msg.receiver_id, "receiver")
            self.assertEqual(received_msg.content["message"], "Hello Kafka!")

    def test_kafka_communicating_agent(self):
        """Test KafkaCommunicatingAgent functionality."""
        # Setup agents
        self.kafka_service.register_agent("kafka_agent_1")
        self.kafka_service.register_agent("kafka_agent_2")
        self.kafka_service.topology.add_link("kafka_agent_1", "kafka_agent_2")
        
        # Create agents
        agent1 = KafkaCommunicatingAgent("kafka_agent_1", self.kafka_service)
        agent2 = KafkaCommunicatingAgent("kafka_agent_2", self.kafka_service)
        
        # Send message
        success = agent1.send_message(
            "kafka_agent_2",
            KafkaMessageType.INFORM,
            {"greeting": "Hello from agent 1"}
        )
        self.assertTrue(success)
        
        # Allow delivery time
        time.sleep(1.0)
        
        # Receive message
        messages = agent2.receive_messages()
        self.assertGreaterEqual(len(messages), 1)
        
        if messages:
            msg = messages[0]
            self.assertEqual(msg.sender_id, "kafka_agent_1")
            self.assertEqual(msg.content["greeting"], "Hello from agent 1")

    def test_extended_communicating_agent(self):
        """Test ExtendedKafkaCommunicatingAgent with BDI integration."""
        # Setup
        self.kafka_service.register_agent("extended_agent_1")
        self.kafka_service.register_agent("extended_agent_2")
        self.kafka_service.topology.create_fully_connected([
            "extended_agent_1", "extended_agent_2"
        ])
        
        # Create extended agents
        agent1 = ExtendedKafkaCommunicatingAgent("extended_agent_1", self.kafka_service)
        agent2 = ExtendedKafkaCommunicatingAgent("extended_agent_2", self.kafka_service)
        
        # Test perception with messages
        initial_perception = agent1.perceive()
        self.assertIn('messages', initial_perception)
        self.assertEqual(initial_perception['message_count'], 0)
        
        # Send message from agent2 to agent1
        success = agent2.send_message(
            "extended_agent_1",
            KafkaMessageType.REQUEST,
            {"request": "status_update"}
        )
        self.assertTrue(success)
        
        # Allow delivery
        time.sleep(1.0)
        
        # Test perception with messages
        perception_with_msg = agent1.perceive()
        self.assertGreaterEqual(perception_with_msg['message_count'], 1)
        
        # Test deliberation and action cycle
        decision = agent1.deliberate(perception_with_msg)
        self.assertIn('process_messages', decision)
        
        result = agent1.act(decision)
        self.assertIn('communication_actions', result)

    def test_broadcast_functionality(self):
        """Test broadcast message functionality."""
        # Setup multiple agents
        agent_ids = ["broadcast_sender", "receiver_1", "receiver_2", "receiver_3"]
        for agent_id in agent_ids:
            self.kafka_service.register_agent(agent_id)
        
        # Setup star topology (sender can reach all receivers)
        sender_id = agent_ids[0]
        for receiver_id in agent_ids[1:]:
            self.kafka_service.topology.add_link(sender_id, receiver_id)
        
        # Create sender agent
        sender = KafkaCommunicatingAgent(sender_id, self.kafka_service)
        
        # Send broadcast
        result = sender.broadcast_message(
            KafkaMessageType.BROADCAST,
            {"announcement": "System maintenance in 1 hour"}
        )
        
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["delivered"], 3)  # 3 receivers
        self.assertEqual(result["failed"], 0)
        
        # Allow delivery time
        time.sleep(1.5)
        
        # Check all receivers got the message
        for receiver_id in agent_ids[1:]:
            messages = self.kafka_service.get_messages(receiver_id)
            self.assertGreaterEqual(len(messages), 1, 
                                   f"Receiver {receiver_id} should receive broadcast")
            
            if messages:
                msg = messages[0]
                self.assertEqual(msg.message_type, KafkaMessageType.BROADCAST)
                self.assertEqual(msg.content["announcement"], 
                               "System maintenance in 1 hour")

    def test_statistics_collection(self):
        """Test communication statistics collection."""
        # Setup
        self.kafka_service.register_agent("stats_sender")
        self.kafka_service.register_agent("stats_receiver")
        self.kafka_service.topology.add_link("stats_sender", "stats_receiver")
        
        # Get initial stats
        initial_stats = self.kafka_service.get_statistics()
        initial_sent = initial_stats["messages_sent"]
        
        # Send some messages
        message = KafkaMessage(
            sender_id="stats_sender",
            receiver_id="stats_receiver",
            message_type=KafkaMessageType.INFORM,
            content={"test": "statistics"}
        )
        
        for i in range(3):
            self.kafka_service.send_message(message)
            time.sleep(0.1)
        
        # Check updated stats
        updated_stats = self.kafka_service.get_statistics()
        self.assertEqual(updated_stats["messages_sent"], initial_sent + 3)

    def test_fallback_mode_when_kafka_unavailable(self):
        """Test fallback behavior when Kafka is not available."""
        # This test verifies the system works even without Kafka
        # by testing the direct message delivery fallback
        
        # Create service with invalid Kafka config to trigger fallback
        invalid_config = {
            'bootstrap_servers': ['invalid_host:9999'],
            'client_id': 'test_fallback'
        }
        fallback_service = KafkaCommunicationService(invalid_config)
        
        # Register agents (should work in fallback mode)
        fallback_service.register_agent("fallback_sender")
        fallback_service.register_agent("fallback_receiver")
        fallback_service.topology.add_link("fallback_sender", "fallback_receiver")
        
        # Create message
        message = KafkaMessage(
            sender_id="fallback_sender",
            receiver_id="fallback_receiver",
            message_type=KafkaMessageType.INFORM,
            content={"fallback": "test"}
        )
        
        # Send message (should use fallback delivery)
        success = fallback_service.send_message(message)
        self.assertTrue(success, "Fallback message delivery should work")
        
        # Check message was received via fallback
        received_messages = fallback_service.get_messages("fallback_receiver")
        self.assertGreaterEqual(len(received_messages), 1)
        
        # Clean up
        fallback_service.close()


if __name__ == '__main__':
    unittest.main()