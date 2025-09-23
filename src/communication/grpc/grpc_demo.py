#!/usr/bin/env python3
"""
gRPC Communication Demo
Simple demonstration of the gRPC-based communication implementation.

Shows basic usage of the gRPC communication framework based on the thesis
definition
and allows for comparison with the REST implementation.
"""

from abstract_agent import AgentId
from communication.grpc.grpc_communication_agent import (
    ExtendedGrpcCommunicatingAgent,
    GrpcCommunicationEnvironment,
)
from communication.communication_config import (
    CommunicationConfiguration,
    TopologyPattern,
)
from grpc_communication import GrpcMessageType
from benchmarks.grpc_benchmark_scenarios import create_grpc_benchmark_scenarios
import time


def simple_grpc_demo():
    """Simple demonstration of gRPC communication."""
    print("gRPC Communication Framework Demo")
    print("=" * 50)

    # 1. Create gRPC communication environment
    print("\n1. Setting up gRPC communication environment...")
    env = GrpcCommunicationEnvironment()
    env.start_service()

    service_address = env.get_service_address()
    print(f"   gRPC Service Address: {service_address}")

    # 2. Create agents
    print("2. Creating gRPC agents...")
    agent1 = ExtendedGrpcCommunicatingAgent(
        AgentId("grpc_demo", "sender", "alice"), {"environment", "messages"}
    )
    agent2 = ExtendedGrpcCommunicatingAgent(
        AgentId("grpc_demo", "receiver", "bob"), {"environment", "messages"}
    )
    agent3 = ExtendedGrpcCommunicatingAgent(
        AgentId("grpc_demo", "coordinator", "charlie"),
        {"environment", "messages"},
    )

    # Initialize agents
    agent1.initialize_agent()
    agent2.initialize_agent()
    agent3.initialize_agent()

    # Register with environment
    env.register_agent(agent1)
    env.register_agent(agent2)
    env.register_agent(agent3)

    # 3. Setup communication topology
    print("3. Setting up gRPC communication topology...")
    config = CommunicationConfiguration()
    config.set_agents([str(agent1.id), str(agent2.id), str(agent3.id)])
    topology = config.set_topology(TopologyPattern.FULLY_CONNECTED)

    # Apply topology
    for sender, receiver in topology.links:
        env.add_communication_link(sender, receiver)

    print(f"   Created {len(topology.links)} gRPC communication links")

    # 4. Demonstrate gRPC messaging
    print("4. Demonstrating gRPC messaging...")

    # Point-to-point message
    print("   Alice -> Bob: Hello message (gRPC)")
    success = agent1.send_message(
        str(agent2.id),
        GrpcMessageType.INFORM,
        {"greeting": "Hello Bob!", "from": "Alice", "protocol": "gRPC"},
    )
    print(f"   gRPC message sent: {success}")

    # Request-reply pattern
    print("   Bob -> Charlie: Status request (gRPC)")
    success = agent2.send_message(
        str(agent3.id),
        GrpcMessageType.REQUEST,
        {
            "request_type": "status",
            "parameters": {"include_details": True},
            "protocol": "gRPC",
        },
    )
    print(f"   gRPC request sent: {success}")

    # Broadcast message
    print("   Charlie -> All: Broadcast announcement (gRPC)")
    result = agent3.broadcast_message(
        {
            "announcement": "gRPC system update scheduled",
            "details": "Maintenance window: 3AM-5AM",
            "protocol": "gRPC",
        }
    )
    print(f"   gRPC broadcast result: {result}")

    # 5. Check message reception
    print("5. Checking gRPC message reception...")
    time.sleep(0.5)  # Allow message processing

    # Check Bob's messages
    bob_messages = agent2.receive_messages(clear_mailbox=False)
    print(f"   Bob received {len(bob_messages)} gRPC messages")
    for msg in bob_messages:
        print(f"     From {msg.sender_id}: {msg.content}")

    # Check Charlie's messages
    charlie_messages = agent3.receive_messages(clear_mailbox=False)
    print(f"   Charlie received {len(charlie_messages)} gRPC messages")
    for msg in charlie_messages:
        print(f"     From {msg.sender_id}: {msg.content}")

    # 6. Show statistics
    print("6. gRPC communication statistics...")
    stats = env.get_system_stats()
    print(f"   Total agents: {stats['total_agents']}")
    print(f"   Topology links: {stats['topology_links']}")
    print(f"   gRPC service stats: {stats['communication_service_stats']}")

    # 7. Cleanup
    print("7. Cleaning up gRPC resources...")
    env.close_all_agents()
    env.stop_service()

    print("\ngRPC demo completed successfully!")


def grpc_benchmark_demo():
    """Demonstrate gRPC benchmarking capabilities."""
    print("\ngRPC Benchmark Demo")
    print("=" * 30)

    # Create gRPC benchmark suite
    benchmark = create_grpc_benchmark_scenarios()

    # Run a simple gRPC latency test
    print("Running gRPC point-to-point latency test...")
    result = benchmark.run_scenario(
        "grpc_point_to_point_latency", agent_count=2, message_count=20
    )

    benchmark.print_summary(result)

    # Export results
    benchmark.export_results("grpc_demo_benchmark_results.json")
    print("gRPC benchmark results saved to grpc_demo_benchmark_results.json")


def performance_comparison_overview():
    """Show comparison between REST and gRPC implementations."""
    print("\nREST vs gRPC Performance Comparison")
    print("=" * 45)

    print("Both implementations follow the same formal communication model:")
    print("• Message Space (𝓜): Same message types and structure")
    print("• Communication Topology (Comm ⊆ A × A): Same topology patterns")
    print("• Mailboxes (MB_i): Same mailbox implementation")
    print("• Delivery Function: Different transport (HTTP vs gRPC)")
    print("• Communication Actions: Same send(i,j,m) semantics")

    print("\nKey Differences:")
    print("REST Implementation:")
    print("  • HTTP/JSON-based communication")
    print("  • Request-response model")
    print("  • Stateless protocol")
    print("  • Text-based serialization")

    print("\ngRPC Implementation:")
    print("  • Binary protocol buffers")
    print("  • Persistent connections")
    print("  • Strongly typed interfaces")
    print("  • Efficient binary serialization")

    print("\nExpected Performance Characteristics:")
    print("  • gRPC: Lower latency, higher throughput")
    print("  • gRPC: Better CPU efficiency")
    print("  • gRPC: Lower network overhead")
    print("  • REST: Better debugging, wider compatibility")


if __name__ == "__main__":
    try:
        # Run gRPC demos
        simple_grpc_demo()

        # Ask user for additional demos
        if input("\nRun gRPC benchmark demo? (y/n): ").lower() == "y":
            grpc_benchmark_demo()

        if (
            input("\nShow REST vs gRPC comparison overview? (y/n): ").lower()
            == "y"
        ):
            performance_comparison_overview()

    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nError during gRPC demo: {e}")
        print(
            "Make sure to install requirements: \
                pip install -r requirements.txt"
        )

    print("\ngRPC demo finished.")
