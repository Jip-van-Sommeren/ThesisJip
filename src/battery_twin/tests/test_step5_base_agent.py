#!/usr/bin/env python3
"""
Step 5 Test: Base Battery Agent Class

This test verifies that:
- Base agent infrastructure initializes correctly
- MockTransport for testing works
- Action registry and routing works
- Performance metrics tracking works
- Lifecycle management (setup/teardown) works

Run: source venv/bin/activate && PYTHONPATH=/home/jip/Documents/thesis/src python3 src/battery_twin/tests/test_step5_base_agent.py
"""

import sys
import time
from pathlib import Path

from mas.core import AgentId
from mas.communication import MockTransport, TopicManager
from battery_twin.agents.battery_agent_base import (
    BatteryReactiveAgent,
    BatteryBDIAgent,
    BatteryHybridAgent,
    AgentStatus,
)
from battery_twin.communication.message_schemas import TelemetryMessage


# ============================================================================
# Helper Test Agent
# ============================================================================


class TestBatteryAgent(BatteryReactiveAgent):
    """Simple test agent for validation."""

    def __init__(self, agent_id: AgentId, transport: MockTransport):
        super().__init__(
            agent_id=agent_id,
            transport=transport,
            observable_properties={"battery_voltage", "battery_current"},
            enable_heartbeat=False,  # Disable for testing
        )

        # Track received messages
        self.received_messages = []

        # Register test action
        self.register_action(
            "process_telemetry",
            self._on_telemetry,
            "battery/+/raw",
            "Process raw telemetry",
        )

    def _on_telemetry(self, topic: str, payload: str):
        """Handle telemetry message."""
        self.received_messages.append((topic, payload))


# ============================================================================
# Test Fixtures
# ============================================================================


def create_mock_transport():
    """Create a MockTransport for testing."""
    # Use the actual mqtt_topics.yaml if available, otherwise create inline config
    config_path = Path(__file__).parent.parent / "config" / "mqtt_topics.yaml"
    if config_path.exists():
        topic_manager = TopicManager(str(config_path))
    else:
        topic_manager = TopicManager()

    transport = MockTransport(topic_manager)
    transport.connect()
    return transport


# ============================================================================
# Base Agent Infrastructure Tests
# ============================================================================


def test_agent_creation():
    """Test agent creation and initialization."""
    print("Testing agent creation...")

    tests_passed = 0
    tests_failed = 0

    # Create mock transport
    transport = create_mock_transport()

    # Test 1: Create agent ID
    try:
        agent_id = AgentId(app="battery_twin", type="test", instance="1")
        assert str(agent_id) == "battery_twin.test.1"
        print("  [PASS] AgentId created successfully")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] AgentId creation failed: {e}")
        tests_failed += 1
        return tests_passed, tests_failed

    # Test 2: Create Reactive agent
    try:
        agent = BatteryReactiveAgent(
            agent_id=agent_id,
            transport=transport,
            observable_properties={"test_property"},
            enable_heartbeat=False,
        )
        print("  [PASS] BatteryReactiveAgent created")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] BatteryReactiveAgent creation failed: {e}")
        tests_failed += 1
        return tests_passed, tests_failed

    # Test 3: Check initial status
    try:
        status = agent.get_status()
        assert status == AgentStatus.CREATED
        print(f"  [PASS] Initial status: {status.value}")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] Status check failed: {e}")
        tests_failed += 1

    # Test 4: Create BDI agent
    try:
        bdi_agent = BatteryBDIAgent(
            agent_id=AgentId(app="battery_twin", type="bdi_test", instance="1"),
            transport=transport,
            observable_properties={"test_property"},
            enable_heartbeat=False,
        )
        print("  [PASS] BatteryBDIAgent created")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] BatteryBDIAgent creation failed: {e}")
        tests_failed += 1

    # Test 5: Create Hybrid agent
    try:
        hybrid_agent = BatteryHybridAgent(
            agent_id=AgentId(app="battery_twin", type="hybrid_test", instance="1"),
            transport=transport,
            observable_properties={"test_property"},
            enable_heartbeat=False,
        )
        print("  [PASS] BatteryHybridAgent created")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] BatteryHybridAgent creation failed: {e}")
        tests_failed += 1

    return tests_passed, tests_failed


def test_action_registry():
    """Test action registration and management."""
    print("\nTesting action registry...")

    tests_passed = 0
    tests_failed = 0

    # Create transport and agent
    transport = create_mock_transport()
    agent_id = AgentId(app="battery_twin", type="test", instance="1")
    agent = TestBatteryAgent(agent_id, transport)

    # Test 1: Action registered during init
    try:
        assert "process_telemetry" in agent.action_handlers
        handler = agent.action_handlers["process_telemetry"]
        assert handler.topic_pattern == "battery/+/raw"
        print("  [PASS] Action registered successfully")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] Action registration check failed: {e}")
        tests_failed += 1

    # Test 2: Register additional action
    try:

        def dummy_handler(topic, payload):
            pass

        agent.register_action("test_action", dummy_handler, "test/topic", "Test action")
        assert "test_action" in agent.action_handlers
        print("  [PASS] Additional action registered")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] Additional action registration failed: {e}")
        tests_failed += 1

    # Test 3: Disable action
    try:
        agent.enable_action("test_action", enabled=False)
        assert not agent.action_handlers["test_action"].enabled
        print("  [PASS] Action disabled successfully")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] Action disable failed: {e}")
        tests_failed += 1

    # Test 4: Re-enable action
    try:
        agent.enable_action("test_action", enabled=True)
        assert agent.action_handlers["test_action"].enabled
        print("  [PASS] Action re-enabled successfully")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] Action re-enable failed: {e}")
        tests_failed += 1

    return tests_passed, tests_failed


def test_metrics_tracking():
    """Test performance metrics tracking."""
    print("\nTesting metrics tracking...")

    tests_passed = 0
    tests_failed = 0

    # Create transport and agent
    transport = create_mock_transport()
    agent_id = AgentId(app="battery_twin", type="test", instance="1")
    agent = TestBatteryAgent(agent_id, transport)

    # Test 1: Get initial metrics
    try:
        metrics = agent.get_metrics()
        assert metrics.messages_received == 0
        assert metrics.messages_sent == 0
        assert metrics.actions_executed == 0
        print("  [PASS] Initial metrics retrieved")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] Get metrics failed: {e}")
        tests_failed += 1

    # Test 2: Metrics as dictionary
    try:
        metrics_dict = agent.get_metrics_dict()
        assert isinstance(metrics_dict, dict)
        assert "messages_received" in metrics_dict
        assert "uptime_seconds" in metrics_dict
        print("  [PASS] Metrics converted to dictionary")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] Metrics to dict failed: {e}")
        tests_failed += 1

    # Test 3: Check uptime
    try:
        time.sleep(0.1)  # Small delay
        uptime = agent.metrics.uptime
        assert uptime > 0
        print(f"  [PASS] Uptime tracking works: {uptime:.3f}s")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] Uptime check failed: {e}")
        tests_failed += 1

    return tests_passed, tests_failed


def test_lifecycle():
    """Test agent lifecycle (setup/teardown)."""
    print("\nTesting lifecycle management...")

    tests_passed = 0
    tests_failed = 0

    # Create transport and agent
    transport = create_mock_transport()
    agent_id = AgentId(app="battery_twin", type="test", instance="1")
    agent = TestBatteryAgent(agent_id, transport)

    # Test 1: Check initial status
    try:
        assert agent.get_status() == AgentStatus.CREATED
        print("  [PASS] Initial status is CREATED")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] Initial status check failed: {e}")
        tests_failed += 1

    # Test 2: Setup agent (should work with MockTransport)
    try:
        success = agent.setup()
        assert success, "Setup should succeed with MockTransport"
        assert agent.get_status() == AgentStatus.READY
        print("  [PASS] Agent setup successful")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] Setup failed: {e}")
        tests_failed += 1

    # Test 3: Teardown agent
    try:
        agent.teardown()
        status = agent.get_status()
        assert status == AgentStatus.STOPPED
        print("  [PASS] Agent teardown successful")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] Teardown failed: {e}")
        tests_failed += 1

    return tests_passed, tests_failed


def test_mock_transport_messaging():
    """Test messaging with MockTransport."""
    print("\nTesting MockTransport messaging...")

    tests_passed = 0
    tests_failed = 0

    # Create transport
    transport = create_mock_transport()

    # Create agent
    agent_id = AgentId(app="battery_twin", type="test", instance="1")
    agent = TestBatteryAgent(agent_id, transport)
    agent.setup()

    # Test 1: Publish message
    try:
        telemetry = TelemetryMessage(
            battery_id="TEST_B0005",
            timestamp=time.time(),
            cycle=1,
            voltage=3.8,
            current=-2.0,
            temperature=25.0,
        )

        success = agent.publish_message(
            "raw_telemetry", telemetry, battery_id="TEST_B0005"
        )
        assert success, "Publish should succeed with MockTransport"
        print("  [PASS] Message published successfully")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] Publish failed: {e}")
        tests_failed += 1

    # Test 2: Check metrics updated
    try:
        assert agent.metrics.messages_sent > 0
        print(f"  [PASS] Messages sent count: {agent.metrics.messages_sent}")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] Metrics not updated: {e}")
        tests_failed += 1

    # Test 3: Simulate receiving a message
    try:
        # Simulate incoming message via MockTransport
        transport.simulate_message("battery/TEST/raw", '{"test": "data"}')
        # Allow time for callback
        time.sleep(0.1)
        assert len(agent.received_messages) > 0
        print(f"  [PASS] Message received via MockTransport")
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] Message reception failed: {e}")
        tests_failed += 1

    agent.teardown()
    return tests_passed, tests_failed


# ============================================================================
# Main Test Runner
# ============================================================================


def main():
    """Run all tests."""
    print("=" * 70)
    print("STEP 5 TEST: Base Battery Agent Class (with MockTransport)")
    print("=" * 70)

    all_tests_passed = 0
    all_tests_failed = 0

    # Test 1: Agent Creation
    passed, failed = test_agent_creation()
    all_tests_passed += passed
    all_tests_failed += failed

    # Test 2: Action Registry
    passed, failed = test_action_registry()
    all_tests_passed += passed
    all_tests_failed += failed

    # Test 3: Metrics Tracking
    passed, failed = test_metrics_tracking()
    all_tests_passed += passed
    all_tests_failed += failed

    # Test 4: Lifecycle
    passed, failed = test_lifecycle()
    all_tests_passed += passed
    all_tests_failed += failed

    # Test 5: MockTransport Messaging
    passed, failed = test_mock_transport_messaging()
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
        print("  [PASS] Agent creation (BDI, Reactive, Hybrid)")
        print("  [PASS] Action registry and management")
        print("  [PASS] Performance metrics tracking")
        print("  [PASS] Lifecycle management (setup/teardown)")
        print("  [PASS] MockTransport messaging")
        return 0
    else:
        print("\n[FAIL] SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
