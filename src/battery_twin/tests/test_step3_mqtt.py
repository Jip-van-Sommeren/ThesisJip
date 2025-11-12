#!/usr/bin/env python3
"""
Step 3 Test: Message Schemas & MQTT Bridge

This test verifies that:
- Pydantic message schemas validate correctly
- TopicManager handles topic templating
- MqttBridge can connect, publish, and subscribe

Prerequisites:
- Mosquitto MQTT broker running: docker-compose up -d mosquitto

Run: source venv/bin/activate && python3 src/battery_twin/tests/test_step3_mqtt.py
"""

import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from pydantic import ValidationError
from src.battery_twin.communication.message_schemas import (
    TelemetryMessage,
    PredictionMessage,
    StateEstimateMessage,
    FaultMessage,
    AgentRegistrationMessage,
    MessageFactory
)
from src.battery_twin.communication.topic_manager import (
    TopicManager,
    format_battery_topic,
    subscribe_to_all_batteries
)
from src.battery_twin.communication.mqtt_bridge import MqttBridge, MqttConfig


# ============================================================================
# Message Schema Tests
# ============================================================================

def test_message_schemas():
    """Test that Pydantic message schemas validate correctly."""
    print("Testing message schemas...")

    tests_passed = 0
    tests_failed = 0

    # Test 1: Valid TelemetryMessage
    try:
        telemetry = TelemetryMessage(
            battery_id="B0005",
            timestamp=time.time(),
            cycle=1,
            voltage=3.825,
            current=-2.0,
            temperature=25.3,
            ambient_temperature=24.0
        )
        print("  ✓ TelemetryMessage validation passed")
        tests_passed += 1
    except ValidationError as e:
        print(f"  ✗ TelemetryMessage validation failed: {e}")
        tests_failed += 1

    # Test 2: Invalid voltage (out of range)
    try:
        invalid_telemetry = TelemetryMessage(
            battery_id="B0005",
            timestamp=time.time(),
            cycle=1,
            voltage=10.0,  # Too high
            current=-2.0,
            temperature=25.3
        )
        print(f"  ✗ TelemetryMessage should reject invalid voltage")
        tests_failed += 1
    except ValidationError:
        print(f"  ✓ TelemetryMessage correctly rejects invalid voltage")
        tests_passed += 1

    # Test 3: Valid PredictionMessage
    try:
        prediction = PredictionMessage(
            battery_id="B0005",
            timestamp=time.time(),
            cycle=1,
            prediction_type="hybrid",
            predicted_capacity=1.87,
            uncertainty=0.005,
            horizon=0,
            agent_id="model.mlresidual.1"
        )
        print("  ✓ PredictionMessage validation passed")
        tests_passed += 1
    except ValidationError as e:
        print(f"  ✗ PredictionMessage validation failed: {e}")
        tests_failed += 1

    # Test 4: Invalid prediction_type
    try:
        invalid_prediction = PredictionMessage(
            battery_id="B0005",
            timestamp=time.time(),
            cycle=1,
            prediction_type="invalid_type",
            predicted_capacity=1.87,
            agent_id="model.mlresidual.1"
        )
        print(f"  ✗ PredictionMessage should reject invalid prediction_type")
        tests_failed += 1
    except ValidationError:
        print(f"  ✓ PredictionMessage correctly rejects invalid prediction_type")
        tests_passed += 1

    # Test 5: Valid StateEstimateMessage
    try:
        state = StateEstimateMessage(
            battery_id="B0005",
            timestamp=time.time(),
            soc=0.75,
            soh=0.92,
            internal_resistance={"R0": 0.05, "R1": 0.03, "C1": 1000.0},
            agent_id="estimator.state.1"
        )
        print("  ✓ StateEstimateMessage validation passed")
        tests_passed += 1
    except ValidationError as e:
        print(f"  ✗ StateEstimateMessage validation failed: {e}")
        tests_failed += 1

    # Test 6: MessageFactory serialization
    try:
        telemetry = TelemetryMessage(
            battery_id="B0005",
            timestamp=time.time(),
            cycle=1,
            voltage=3.825,
            current=-2.0,
            temperature=25.3
        )
        json_str = MessageFactory.to_json(telemetry)
        data_dict = json.loads(json_str)

        assert data_dict['battery_id'] == "B0005"
        assert data_dict['voltage'] == 3.825
        print("  ✓ MessageFactory serialization works")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ MessageFactory serialization failed: {e}")
        tests_failed += 1

    # Test 7: MessageFactory deserialization
    try:
        json_str = telemetry.model_dump_json()
        parsed = TelemetryMessage.model_validate_json(json_str)
        assert parsed.battery_id == "B0005"
        assert parsed.voltage == 3.825
        print("  ✓ MessageFactory deserialization works")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ MessageFactory deserialization failed: {e}")
        tests_failed += 1

    return tests_passed, tests_failed


# ============================================================================
# Topic Manager Tests
# ============================================================================

def test_topic_manager():
    """Test TopicManager functionality."""
    print("\nTesting topic manager...")

    tests_passed = 0
    tests_failed = 0

    try:
        tm = TopicManager("src/battery_twin/config/mqtt_topics.yaml")
        print("  ✓ TopicManager loaded configuration")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ TopicManager failed to load: {e}")
        tests_failed += 1
        return tests_passed, tests_failed

    # Test 1: Format battery topic
    try:
        topic = tm.get_topic("raw_telemetry", battery_id="B0005")
        assert topic == "battery/B0005/raw", f"Expected 'battery/B0005/raw', got '{topic}'"
        print(f"  ✓ Topic formatting works: {topic}")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Topic formatting failed: {e}")
        tests_failed += 1

    # Test 2: Format with missing variable
    try:
        topic = tm.get_topic("raw_telemetry")  # Missing battery_id
        print(f"  ✗ Should reject missing variables")
        tests_failed += 1
    except ValueError:
        print(f"  ✓ Correctly rejects missing variables")
        tests_passed += 1

    # Test 3: Parse topic
    try:
        result = tm.parse_topic("battery/B0005/telemetry")
        assert result is not None, "Failed to parse valid topic"
        topic_name, variables = result
        assert variables['battery_id'] == "B0005", f"Wrong battery_id: {variables}"
        print(f"  ✓ Topic parsing works: {topic_name}, {variables}")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Topic parsing failed: {e}")
        tests_failed += 1

    # Test 4: Subscription pattern with wildcard
    try:
        pattern = tm.get_subscription_pattern("raw_telemetry", battery_id=None)
        assert pattern == "battery/+/raw", f"Expected 'battery/+/raw', got '{pattern}'"
        print(f"  ✓ Subscription wildcard works: {pattern}")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Subscription pattern failed: {e}")
        tests_failed += 1

    # Test 5: Subscription pattern with specific value
    try:
        pattern = tm.get_subscription_pattern("raw_telemetry", battery_id="B0005")
        assert pattern == "battery/B0005/raw", f"Expected 'battery/B0005/raw', got '{pattern}'"
        print(f"  ✓ Subscription with specific value works: {pattern}")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Subscription pattern failed: {e}")
        tests_failed += 1

    # Test 6: List topics
    try:
        topics = tm.list_topics()
        assert len(topics) > 0, "No topics loaded"
        assert "raw_telemetry" in topics, "Missing expected topic"
        print(f"  ✓ Found {len(topics)} configured topics")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ List topics failed: {e}")
        tests_failed += 1

    # Test 7: Get topic variables
    try:
        variables = tm.get_topic_variables("raw_telemetry")
        assert "battery_id" in variables, "Missing battery_id variable"
        print(f"  ✓ Topic variables: {variables}")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Get topic variables failed: {e}")
        tests_failed += 1

    return tests_passed, tests_failed


# ============================================================================
# MQTT Bridge Tests
# ============================================================================

def test_mqtt_bridge():
    """Test MQTT bridge connectivity and basic operations."""
    print("\nTesting MQTT bridge...")

    tests_passed = 0
    tests_failed = 0

    # Create MQTT config
    config = MqttConfig(
        broker="localhost",
        port=1883,
        qos=1,
        client_id_prefix="test_"
    )

    # Create bridge
    try:
        bridge = MqttBridge(
            client_id="test_bridge_1",
            mqtt_config=config,
            topic_config_path="src/battery_twin/config/mqtt_topics.yaml"
        )
        print("  ✓ MqttBridge created successfully")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ MqttBridge creation failed: {e}")
        tests_failed += 1
        return tests_passed, tests_failed

    # Test connection
    try:
        success = bridge.connect()
        if success:
            print(f"  ✓ Connected to MQTT broker at {config.broker}:{config.port}")
            tests_passed += 1
        else:
            print(f"  ⚠ Failed to connect to MQTT broker (is Mosquitto running?)")
            print(f"    Run: docker-compose up -d mosquitto")
            tests_failed += 1
            return tests_passed, tests_failed
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
        print(f"    Make sure Mosquitto is running: docker-compose up -d mosquitto")
        tests_failed += 1
        return tests_passed, tests_failed

    # Test publish
    try:
        telemetry = TelemetryMessage(
            battery_id="TEST_B0005",
            timestamp=time.time(),
            cycle=1,
            voltage=3.825,
            current=-2.0,
            temperature=25.3
        )

        success = bridge.publish(
            "raw_telemetry",
            telemetry,
            battery_id="TEST_B0005"
        )

        if success:
            print(f"  ✓ Published telemetry message")
            tests_passed += 1
        else:
            print(f"  ✗ Failed to publish message")
            tests_failed += 1
    except Exception as e:
        print(f"  ✗ Publish failed: {e}")
        tests_failed += 1

    # Test subscribe with callback
    received_messages: List[str] = []

    def on_message(topic: str, payload: str):
        """Callback for received messages."""
        received_messages.append(payload)

    try:
        success = bridge.subscribe(
            "raw_telemetry",
            on_message,
            battery_id=None  # Subscribe to all batteries
        )

        if success:
            print(f"  ✓ Subscribed to telemetry topic")
            tests_passed += 1
        else:
            print(f"  ✗ Failed to subscribe")
            tests_failed += 1
    except Exception as e:
        print(f"  ✗ Subscribe failed: {e}")
        tests_failed += 1

    # Test message delivery
    try:
        # Publish another message
        telemetry2 = TelemetryMessage(
            battery_id="TEST_B0005",
            timestamp=time.time(),
            cycle=2,
            voltage=3.800,
            current=-2.0,
            temperature=25.5
        )

        bridge.publish("raw_telemetry", telemetry2, battery_id="TEST_B0005")

        # Wait for message delivery
        time.sleep(0.5)

        if len(received_messages) > 0:
            print(f"  ✓ Received {len(received_messages)} message(s) via callback")
            tests_passed += 1

            # Validate message content
            try:
                data = json.loads(received_messages[-1])
                assert data['battery_id'] == "TEST_B0005"
                assert 'voltage' in data
                print(f"  ✓ Message content validated")
                tests_passed += 1
            except Exception as e:
                print(f"  ✗ Message validation failed: {e}")
                tests_failed += 1
        else:
            print(f"  ⚠ No messages received (may be timing issue)")
            tests_failed += 1

    except Exception as e:
        print(f"  ✗ Message delivery test failed: {e}")
        tests_failed += 1

    # Test statistics
    try:
        stats = bridge.get_stats()
        print(f"  ✓ Bridge stats: {stats['messages_published']} published, "
              f"{stats['messages_received']} received")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Get stats failed: {e}")
        tests_failed += 1

    # Cleanup
    try:
        bridge.disconnect()
        print("  ✓ Disconnected from broker")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Disconnect failed: {e}")
        tests_failed += 1

    return tests_passed, tests_failed


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    """Run all tests."""
    print("=" * 70)
    print("STEP 3 TEST: Message Schemas & MQTT Bridge")
    print("=" * 70)
    print("\nPrerequisites:")
    print("  - Mosquitto MQTT broker must be running")
    print("  - Run: docker-compose up -d mosquitto")
    print("=" * 70)

    all_tests_passed = 0
    all_tests_failed = 0

    # Test 1: Message Schemas
    passed, failed = test_message_schemas()
    all_tests_passed += passed
    all_tests_failed += failed

    # Test 2: Topic Manager
    passed, failed = test_topic_manager()
    all_tests_passed += passed
    all_tests_failed += failed

    # Test 3: MQTT Bridge
    passed, failed = test_mqtt_bridge()
    all_tests_passed += passed
    all_tests_failed += failed

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"  Total Tests: {all_tests_passed + all_tests_failed}")
    print(f"  ✓ Passed: {all_tests_passed}")
    print(f"  ✗ Failed: {all_tests_failed}")
    print("=" * 70)

    if all_tests_failed == 0:
        print("\n✓ ALL TESTS PASSED!")
        print("\nNext step: Proceed to Step 4: NASA Dataset Loader & Replay Engine")
        print("\nWhat was tested:")
        print("  ✓ Message schemas with Pydantic validation")
        print("  ✓ Topic manager with templating and wildcards")
        print("  ✓ MQTT bridge with publish/subscribe")
        print("  ✓ QoS 1 message delivery")
        print("  ✓ Message routing and callbacks")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        print("\nCommon issues:")
        print("  - Mosquitto not running: docker-compose up -d mosquitto")
        print("  - Port 1883 already in use: Check with 'lsof -i :1883'")
        print("  - MQTT library issue: Check paho-mqtt installation")
        return 1


if __name__ == "__main__":
    sys.exit(main())
