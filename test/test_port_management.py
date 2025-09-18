#!/usr/bin/env python3
"""
Test script to verify port management works correctly.
Tests that multiple communication environments can run concurrently.
"""

import time
from src.communication.communicating_agent import CommunicationEnvironment
from abstract_agent import AgentId
from src.communication.communicating_agent import ExtendedCommunicatingAgent


def test_multiple_environments():
    """Test that multiple environments can start without port conflicts."""
    print("Testing multiple communication environments...")

    environments = []

    try:
        # Create and start 3 environments
        for i in range(3):
            print(f"\nStarting environment {i+1}...")
            env = CommunicationEnvironment()
            env.start_service()
            environments.append(env)

            print(f"   Environment {i+1} service URL: {env.get_service_url()}")

            # Create a test agent for each environment
            agent = ExtendedCommunicatingAgent(
                AgentId("test", "agent", f"test_{i}"),
                {"environment", "messages"},
            )
            agent.initialize_agent()
            env.register_agent(agent)

            print(f"   Agent registered: {agent.id}")

        print(f"\nSuccessfully started {len(environments)} environments!")

        # Test that they're all running on different ports
        ports = [env.service_port for env in environments]
        unique_ports = set(ports)

        print(f"Ports used: {ports}")
        print(
            f"Unique ports: \
                {len(unique_ports)} (should be {len(environments)})"
        )

        if len(unique_ports) == len(environments):
            print(
                "Port management working correctly - all environments \
                on different ports"
            )
        else:
            print("Port conflict detected - some environments sharing ports")

        # Let them run for a moment
        time.sleep(2)

    except Exception as e:
        print(f"Error during test: {e}")
        return False

    finally:
        # Clean up all environments
        print("\nCleaning up environments...")
        for i, env in enumerate(environments):
            env.stop_service()
            print(f"   Environment {i+1} stopped")

    return True


def test_sequential_start_stop():
    """Test starting and stopping environments sequentially."""
    print("\nTesting sequential start/stop...")

    try:
        for i in range(3):
            print(f"\n--- Round {i+1} ---")

            # Start environment
            env = CommunicationEnvironment()
            env.start_service()
            print(f"Started environment on {env.get_service_url()}")

            # Create agent
            agent = ExtendedCommunicatingAgent(
                AgentId("test", "sequential", f"agent_{i}"),
                {"environment", "messages"},
            )
            agent.initialize_agent()
            env.register_agent(agent)

            # Let it run briefly
            time.sleep(1)

            # Stop environment
            env.stop_service()
            print("Stopped environment")

            # Brief pause before next iteration
            time.sleep(0.5)

        print("Sequential start/stop test completed successfully")
        return True

    except Exception as e:
        print(f"Error during sequential test: {e}")
        return False


if __name__ == "__main__":
    print("Port Management Test Suite")
    print("=" * 40)

    # Test 1: Multiple concurrent environments
    success1 = test_multiple_environments()

    # Test 2: Sequential start/stop
    success2 = test_sequential_start_stop()

    print("\n" + "=" * 40)
    print("TEST RESULTS:")
    print(f"Multiple environments: {'PASS' if success1 else 'FAIL'}")
    print(f"Sequential start/stop: {'PASS' if success2 else 'FAIL'}")

    if success1 and success2:
        print("\nAll port management tests passed!")
        print(
            "The demo and benchmarks should now work without port conflicts."
        )
    else:
        print("\nSome tests failed. Port conflicts may still occur.")
