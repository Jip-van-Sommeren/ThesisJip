#!/usr/bin/env python3
"""
REST Communication Demo
Simple demonstration of the REST-based communication implementation.

Shows basic usage of the communication framework based on the thesis
definition.
"""


# Add src to path for imports
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from abstract_agent import AgentId
from rest_communicating_agent import (
    ExtendedRestCommunicatingAgent,
    RestCommunicationEnvironment,
)
from communication_config import CommunicationConfiguration, TopologyPattern
from communication.base_communication import MessageType
from benchmarks.rest_benchmark_scenarios import create_benchmark_scenarios
import time


def simple_demo():
    """Simple demonstration of REST communication."""
    print("REST Communication Framework Demo")
    print("=" * 50)

    # 1. Create communication environment
    print("\n1. Setting up communication environment...")
    env = RestCommunicationEnvironment()
    env.start_service()

    service_url = env.get_service_url()
    print(f"   Service URL: {service_url}")

    # 2. Create agents
    print("2. Creating agents...")
    agent1 = ExtendedRestCommunicatingAgent(
        AgentId("demo", "sender", "alice"), {"environment", "messages"}
    )
    agent2 = ExtendedRestCommunicatingAgent(
        AgentId("demo", "receiver", "bob"), {"environment", "messages"}
    )
    agent3 = ExtendedRestCommunicatingAgent(
        AgentId("demo", "coordinator", "charlie"), {"environment", "messages"}
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
    print("3. Setting up communication topology...")
    config = CommunicationConfiguration()
    config.set_agents([str(agent1.id), str(agent2.id), str(agent3.id)])
    topology = config.set_topology(TopologyPattern.FULLY_CONNECTED)

    # Apply topology
    for sender, receiver in topology.links:
        env.add_communication_link(sender, receiver)

    print(f"   Created {len(topology.links)} communication links")

    # 4. Demonstrate messaging
    print("4. Demonstrating messaging...")

    # Point-to-point message
    print("   Alice -> Bob: Hello message")
    success = agent1.send_message(
        str(agent2.id),
        MessageType.INFORM,
        {"greeting": "Hello Bob!", "from": "Alice"},
    )
    print(f"   Message sent: {success}")

    # Request-reply pattern
    print("   Bob -> Charlie: Status request")
    success = agent2.send_message(
        str(agent3.id),
        MessageType.REQUEST,
        {"request_type": "status", "parameters": {"include_details": True}},
    )
    print(f"   Request sent: {success}")

    # Broadcast message
    print("   Charlie -> All: Broadcast announcement")
    result = agent3.broadcast_message(
        {
            "announcement": "System update scheduled",
            "details": "Maintenance window: 2AM-4AM",
        }
    )
    print(f"   Broadcast result: {result}")

    # 5. Check message reception
    print("5. Checking message reception...")
    time.sleep(0.5)  # Allow message processing

    # Check Bob's messages
    bob_messages = agent2.receive_messages(clear_mailbox=False)
    print(f"   Bob received {len(bob_messages)} messages")
    for msg in bob_messages:
        print(f"     From {msg.sender_id}: {msg.content}")

    # Check Charlie's messages
    charlie_messages = agent3.receive_messages(clear_mailbox=False)
    print(f"   Charlie received {len(charlie_messages)} messages")
    for msg in charlie_messages:
        print(f"     From {msg.sender_id}: {msg.content}")

    # 6. Show statistics
    print("6. Communication statistics...")
    stats = env.get_system_stats()
    print(f"   Total agents: {stats['total_agents']}")
    print(f"   Topology links: {stats['topology_links']}")
    print(f"   Service stats: {stats['communication_service_stats']}")

    # 7. Cleanup
    print("7. Cleaning up...")
    env.stop_service()

    print("\nDemo completed successfully!")


def benchmark_demo():
    """Demonstrate benchmarking capabilities."""
    print("\nBenchmark Demo")
    print("=" * 30)

    # Create benchmark suite
    benchmark = create_benchmark_scenarios()

    # Run a simple latency test
    print("Running point-to-point latency test...")
    result = benchmark.run_scenario(
        "point_to_point_latency", agent_count=2, message_count=20
    )

    benchmark.print_summary(result)

    # Export results
    benchmark.export_results("demo_benchmark_results.json")
    print("Benchmark results saved to demo_benchmark_results.json")


def performance_metrics_overview():
    """Show the performance metrics that can be measured."""
    print("\nPerformance Metrics Overview")
    print("=" * 40)

    metrics = [
        (
            "Latency Metrics",
            [
                "Average message latency (ms)",
                "95th percentile latency (ms)",
                "99th percentile latency (ms)",
                "Maximum latency (ms)",
            ],
        ),
        (
            "Throughput Metrics",
            [
                "Messages per second (avg)",
                "Peak throughput (msg/s)",
                "Sustained throughput over time",
            ],
        ),
        (
            "Reliability Metrics",
            [
                "Message delivery success rate (%)",
                "Number of delivery failures",
                "Number of timeout failures",
            ],
        ),
        (
            "Resource Metrics",
            ["CPU usage (%)", "Memory usage (MB)", "Network utilization"],
        ),
        (
            "Scalability Metrics",
            [
                "Performance vs agent count",
                "Performance vs topology density",
                "Performance vs message load",
            ],
        ),
    ]

    for category, metric_list in metrics:
        print(f"\n{category}:")
        for metric in metric_list:
            print(f"  • {metric}")

    print("\nTopology Patterns Supported:")
    patterns = [
        "Fully Connected",
        "Star",
        "Ring",
        "Chain",
        "Hierarchical",
        "Small World",
        "Scale Free",
    ]
    for pattern in patterns:
        print(f"  • {pattern}")


if __name__ == "__main__":
    try:
        # Run demos
        simple_demo()

        # Ask user for additional demos
        if input("\nRun benchmark demo? (y/n): ").lower() == "y":
            benchmark_demo()

        if (
            input("\nShow performance metrics overview? (y/n): ").lower()
            == "y"
        ):
            performance_metrics_overview()

    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nError during demo: {e}")
        print(
            "Make sure to install requirements: pip \
                install -r requirements.txt"
        )

    print("\nDemo finished.")
