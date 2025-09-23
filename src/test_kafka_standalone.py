"""
Standalone Kafka Communication Test
Simple test to verify Kafka implementation works independently.
"""

import time
from communication.kafka.kafka_communication import (
    KafkaCommunicationService,
    KafkaMessage,
    KafkaMessageType,
    KafkaCommunicatingAgent,
)


def test_kafka_standalone():
    """Test Kafka communication in standalone mode."""
    print("Testing Kafka Communication Implementation")
    print("=" * 50)

    # Create Kafka service
    kafka_config = {
        "bootstrap_servers": ["localhost:9092"],
        "client_id": "standalone_test",
    }

    kafka_service = KafkaCommunicationService(kafka_config)

    try:
        # Test 1: Agent Registration
        print("\n1. Testing Agent Registration...")
        kafka_service.register_agent("agent_1")
        kafka_service.register_agent("agent_2")

        assert "agent_1" in kafka_service.mailboxes
        assert "agent_2" in kafka_service.mailboxes
        print("✓ Agent registration successful")

        # Test 2: Communication Topology
        print("\n2. Testing Communication Topology...")
        kafka_service.topology.add_link("agent_1", "agent_2")

        assert kafka_service.topology.can_communicate("agent_1", "agent_2")
        assert not kafka_service.topology.can_communicate("agent_2", "agent_1")
        print("✓ Communication topology setup successful")

        # Test 3: Message Creation and Serialization
        print("\n3. Testing Message Creation...")
        message = KafkaMessage(
            sender_id="agent_1",
            receiver_id="agent_2",
            message_type=KafkaMessageType.INFORM,
            content={"greeting": "Hello Kafka!"},
        )

        json_str = message.to_json()
        restored_message = KafkaMessage.from_json(json_str)

        assert restored_message.sender_id == message.sender_id
        assert restored_message.content["greeting"] == "Hello Kafka!"
        print("✓ Message serialization successful")

        # Test 4: Message Sending (Fallback Mode)
        print("\n4. Testing Message Sending...")
        success = kafka_service.send_message(message)

        assert success, "Message sending should succeed"
        print("✓ Message sending successful")

        # Test 5: Message Reception
        print("\n5. Testing Message Reception...")
        time.sleep(0.5)  # Allow for delivery

        received_messages = kafka_service.get_messages("agent_2")

        assert (
            len(received_messages) >= 1
        ), "Should receive at least one message"
        if received_messages:
            received_msg = received_messages[0]
            assert received_msg.sender_id == "agent_1"
            assert received_msg.content["greeting"] == "Hello Kafka!"
        print("✓ Message reception successful")

        # Test 6: Communicating Agent Interface
        print("\n6. Testing Communicating Agent Interface...")
        agent1 = KafkaCommunicatingAgent("agent_1", kafka_service)
        _ = KafkaCommunicatingAgent("agent_2", kafka_service)

        success = agent1.send_message(
            "agent_2", KafkaMessageType.REQUEST, {"request": "status_update"}
        )

        assert success, "Agent message sending should succeed"
        print("✓ Communicating agent interface successful")

        # Test 7: Statistics Collection
        print("\n7. Testing Statistics Collection...")
        stats = kafka_service.get_statistics()

        assert "messages_sent" in stats
        assert "messages_delivered" in stats
        assert stats["messages_sent"] >= 2  # We sent at least 2 messages
        print("✓ Statistics collection successful")

        print("\n" + "=" * 50)
        print(f"Messages sent: {stats['messages_sent']}")
        print(f"Messages delivered: {stats['messages_delivered']}")
        success = (
            stats["messages_delivered"] / max(stats["messages_sent"], 1) * 100
        )
        print(
            f"Success rate: \
                {success:.1f}%"
        )

        if kafka_service.is_running:
            print("✓ Kafka service is running (connected to Kafka)")
        else:
            print("⚠ Kafka service using fallback mode (Kafka not available)")

    except Exception as e:
        print(f"\nError during testing: {e}")
        return False

    finally:
        # Clean up
        kafka_service.close()
        print("\n✓ Cleanup completed")

    return True


if __name__ == "__main__":
    success = test_kafka_standalone()
    if success:
        print("\nKafka implementation ready for benchmarking!")
    else:
        print("\nKafka implementation needs fixes.")
        exit(1)
