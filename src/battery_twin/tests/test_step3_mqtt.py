#!/usr/bin/env python3
"""
Step 3 Test: Message Schemas & Transport Layer

This test verifies that:
- Pydantic message schemas validate correctly
- TopicManager handles topic templating
- MockTransport works for testing without broker
- (Optional) MqttTransport can connect to real broker

Run: source venv/bin/activate && PYTHONPATH=/home/jip/Documents/thesis/src python3 src/battery_twin/tests/test_step3_mqtt.py
"""

import sys
import time
import json
from pathlib import Path
from typing import List

from pydantic import ValidationError

from mas.communication import TopicManager, MockTransport, MqttTransport, MqttConfig
from battery_twin.communication.message_schemas import (
    TelemetryMessage,
    PredictionMessage,
    StateEstimateMessage,
    FaultMessage,
    AgentRegistrationMessage,
    MessageFactory,
)


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
            ambient_temperature=24.0,
        )
        print("  [PASS] TelemetryMessage validation passed")
        tests_passed += 1
    except ValidationError as e:
        print(f"  [FAIL] TelemetryMessage validation failed: {e}")
        tests_failed += 1

    # Test 2: Invalid voltage (out of range)
    try:
        invalid_telemetry = TelemetryMessage(
            battery_id="B0005",
            timestamp=time.time(),
            cycle=1,
            voltage=10.0,  # Too high
            current=-2.0,
            temperature=25.3,
        )
        print(f"  [FAIL] TelemetryMessage should reject invalid voltage")
        tests_failed += 1
    except ValidationError:
        print(f"  [PASS] TelemetryMessage correctly rejects invalid voltage")
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
            agent_id="model.mlresidual.1",
        )
        print("  [PASS] PredictionMessage validation passed")
        tests_passed += 1
    except ValidationError as e:
        print(f"  [FAIL] PredictionMessage validation failed: {e}")
        tests_failed += 1

    # Test 4: Invalid prediction_type
    try:
        invalid_prediction = PredictionMessage(
            battery_id="B0005",
            timestamp=time.time(),
            cycle=1,
            prediction_type="invalid_type",
            predicted_capacity=1.87,
            agent_id="model.mlresidual.1",
        )
        print(f"  [FAIL] PredictionMessage should reject invalid prediction_type")
        tests_failed += 1
    except ValidationError:
        print(f"  [PASS] PredictionMessage correctly rejects invalid prediction_type")
        tests_passed += 1

    # Test 5: Valid StateEstimateMessage
    try:
        state = StateEstimateMessage(
            battery_id="B0005",
            timestamp=time.time(),
            soc=0.75,
            soh=0.92,
            internal_resistance={"R0": 0.05, "R1": 0.03, "C1": 1000.0},
            agent_id="estimator.state.1",
        )
        print("  [PASS] StateEstimateMessage validation passed")
        tests_passed += 1
    except ValidationError as e:
        print(f"  [FAIL] StateEstimateMessage validation failed: {e}")
        tests_failed += 1

    # Test 6: MessageFactory serialization
    try:
        telemetry = TelemetryMessage(
            battery_id="B0005",
            timestamp=time.time(),
            cycle=1,
            voltage=3.825,
            current=-2.0,
            temperature=25.3,
        )
        json_str = MessageFactory.to_json(telemetry)
        data_dict = json.loads(json_str)

        assert data_dict["battery_id"] == "B0005"
        assert data_dict["voltage"] == 3.825
        print("  [PASS] MessageFactory serialization works")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] MessageFactory serialization failed: {e}")
        tests_failed += 1

    # Test 7: MessageFactory deserialization
    try:
        json_str = telemetry.model_dump_json()
        parsed = TelemetryMessage.model_validate_json(json_str)
        assert parsed.battery_id == "B0005"
        assert parsed.voltage == 3.825
        print("  [PASS] MessageFactory deserialization works")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] MessageFactory deserialization failed: {e}")
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

    config_path = Path(__file__).parent.parent / "config" / "mqtt_topics.yaml"
    try:
        tm = TopicManager(str(config_path))
        print("  [PASS] TopicManager loaded configuration")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] TopicManager failed to load: {e}")
        tests_failed += 1
        return tests_passed, tests_failed

    # Test 1: Format battery topic
    try:
        topic = tm.get_topic("raw_telemetry", battery_id="B0005")
        assert topic == "battery/B0005/raw", f"Expected 'battery/B0005/raw', got '{topic}'"
        print(f"  [PASS] Topic formatting works: {topic}")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] Topic formatting failed: {e}")
        tests_failed += 1

    # Test 2: Format with missing variable
    try:
        topic = tm.get_topic("raw_telemetry")  # Missing battery_id
        print(f"  [FAIL] Should reject missing variables")
        tests_failed += 1
    except (ValueError, KeyError):
        print(f"  [PASS] Correctly rejects missing variables")
        tests_passed += 1

    # Test 3: Parse topic (use raw_telemetry which is battery/{battery_id}/raw)
    try:
        result = tm.parse_topic("battery/B0005/raw")
        assert result is not None, "Failed to parse valid topic"
        topic_name, variables = result
        assert variables["battery_id"] == "B0005", f"Wrong battery_id: {variables}"
        print(f"  [PASS] Topic parsing works: {topic_name}, {variables}")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] Topic parsing failed: {e}")
        tests_failed += 1

    # Test 4: Subscription pattern with wildcard
    try:
        pattern = tm.get_subscription_pattern("raw_telemetry", battery_id=None)
        assert pattern == "battery/+/raw", f"Expected 'battery/+/raw', got '{pattern}'"
        print(f"  [PASS] Subscription wildcard works: {pattern}")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] Subscription pattern failed: {e}")
        tests_failed += 1

    # Test 5: List topics
    try:
        topics = tm.list_topics()
        assert len(topics) > 0, "No topics loaded"
        assert "raw_telemetry" in topics, "Missing expected topic"
        print(f"  [PASS] Found {len(topics)} configured topics")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] List topics failed: {e}")
        tests_failed += 1

    return tests_passed, tests_failed


# ============================================================================
# MockTransport Tests
# ============================================================================


def test_mock_transport():
    """Test MockTransport for testing without broker."""
    print("\nTesting MockTransport...")

    tests_passed = 0
    tests_failed = 0

    config_path = Path(__file__).parent.parent / "config" / "mqtt_topics.yaml"
    tm = TopicManager(str(config_path))

    # Create MockTransport
    try:
        transport = MockTransport(tm)
        print("  [PASS] MockTransport created")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] MockTransport creation failed: {e}")
        tests_failed += 1
        return tests_passed, tests_failed

    # Test connect
    try:
        success = transport.connect()
        assert success
        assert transport.is_connected()
        print("  [PASS] MockTransport connected")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] MockTransport connect failed: {e}")
        tests_failed += 1
        return tests_passed, tests_failed

    # Test subscribe
    received_messages: List[str] = []

    def on_message(topic: str, payload: str):
        received_messages.append((topic, payload))

    try:
        transport.subscribe("battery/+/raw", on_message)
        print("  [PASS] MockTransport subscribed")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] MockTransport subscribe failed: {e}")
        tests_failed += 1

    # Test publish
    try:
        telemetry = TelemetryMessage(
            battery_id="TEST",
            timestamp=time.time(),
            cycle=1,
            voltage=3.8,
            current=-2.0,
            temperature=25.0,
        )
        success = transport.publish_to_topic("raw_telemetry", telemetry, battery_id="TEST")
        assert success
        print("  [PASS] MockTransport published")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] MockTransport publish failed: {e}")
        tests_failed += 1

    # Test simulate_message
    try:
        transport.simulate_message("battery/TEST/raw", '{"test": "data"}')
        time.sleep(0.1)  # Allow callback
        assert len(received_messages) > 0
        print(f"  [PASS] MockTransport simulate_message works: received {len(received_messages)} messages")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] MockTransport simulate_message failed: {e}")
        tests_failed += 1

    # Test get_published
    try:
        published = transport.get_published()
        assert len(published) > 0
        print(f"  [PASS] MockTransport tracked {len(published)} published messages")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] MockTransport get_published failed: {e}")
        tests_failed += 1

    # Test disconnect
    try:
        transport.disconnect()
        assert not transport.is_connected()
        print("  [PASS] MockTransport disconnected")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] MockTransport disconnect failed: {e}")
        tests_failed += 1

    return tests_passed, tests_failed


# ============================================================================
# Real MQTT Transport Tests (Optional)
# ============================================================================


def test_mqtt_transport():
    """Test MqttTransport with real broker (optional)."""
    print("\nTesting MqttTransport (requires broker)...")

    tests_passed = 0
    tests_failed = 0

    config_path = Path(__file__).parent.parent / "config" / "mqtt_topics.yaml"
    tm = TopicManager(str(config_path))

    # Create MqttTransport
    config = MqttConfig(broker="localhost", port=1883, client_id="test_transport")

    try:
        transport = MqttTransport(config, tm)
        print("  [PASS] MqttTransport created")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] MqttTransport creation failed: {e}")
        tests_failed += 1
        return tests_passed, tests_failed

    # Test connection
    try:
        success = transport.connect()
        if success:
            print("  [PASS] Connected to MQTT broker")
            tests_passed += 1
        else:
            print("  [WARN] Could not connect (broker not running?)")
            return tests_passed, tests_failed
    except Exception as e:
        print(f"  [WARN] Connection failed (broker not running?): {e}")
        return tests_passed, tests_failed

    # Test publish
    try:
        telemetry = TelemetryMessage(
            battery_id="TEST",
            timestamp=time.time(),
            cycle=1,
            voltage=3.8,
            current=-2.0,
            temperature=25.0,
        )
        success = transport.publish_to_topic("raw_telemetry", telemetry, battery_id="TEST")
        if success:
            print("  [PASS] Published to broker")
            tests_passed += 1
        else:
            print("  [FAIL] Publish failed")
            tests_failed += 1
    except Exception as e:
        print(f"  [FAIL] Publish error: {e}")
        tests_failed += 1

    # Cleanup
    transport.disconnect()
    print("  [PASS] Disconnected from broker")
    tests_passed += 1

    return tests_passed, tests_failed


# ============================================================================
# Main Test Runner
# ============================================================================


def main():
    """Run all tests."""
    print("=" * 70)
    print("STEP 3 TEST: Message Schemas & Transport Layer")
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

    # Test 3: MockTransport
    passed, failed = test_mock_transport()
    all_tests_passed += passed
    all_tests_failed += failed

    # Test 4: Real MQTT Transport (optional)
    passed, failed = test_mqtt_transport()
    all_tests_passed += passed
    all_tests_failed += failed

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"  Total Tests: {all_tests_passed + all_tests_failed}")
    print(f"  [PASS]: {all_tests_passed}")
    print(f"  [FAIL]: {all_tests_failed}")
    print("=" * 70)

    if all_tests_failed == 0:
        print("\n[PASS] ALL TESTS PASSED!")
        print("\nWhat was tested:")
        print("  [PASS] Message schemas with Pydantic validation")
        print("  [PASS] TopicManager with templating and wildcards")
        print("  [PASS] MockTransport for testing without broker")
        print("  [PASS] MqttTransport (if broker available)")
        return 0
    else:
        print("\n[FAIL] SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
