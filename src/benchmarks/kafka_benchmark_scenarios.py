"""
Kafka Benchmark Test Scenarios
Provides the same benchmark scenarios as REST, gRPC, and MQTT but using Kafka communication.

Enables direct performance comparison between Kafka and other protocol implementations
of the same formal communication model from the thesis.
"""

import time
import random
from typing import Dict, Any
import concurrent.futures
from abstract_agent import AgentId
from communication.kafka.kafka_communication_agent import (
    ExtendedKafkaCommunicatingAgent,
    KafkaCommunicationEnvironment,
)
from communication.communication_config import (
    CommunicationConfiguration,
    TopologyPattern,
)
from benchmarks.communication_benchmark import (
    CommunicationBenchmark,
    BenchmarkScenario,
)
from communication.kafka.kafka_communication import KafkaMessageType


def create_kafka_test_agent(
    agent_id: str, observable_props: set = None
) -> ExtendedKafkaCommunicatingAgent:
    """Create a Kafka test agent with basic configuration."""
    if observable_props is None:
        observable_props = {"environment", "messages"}

    agent_id_obj = AgentId("kafka_benchmark", "test", agent_id)
    agent = ExtendedKafkaCommunicatingAgent(agent_id_obj, observable_props)
    agent.initialize_agent()
    return agent


def setup_kafka_basic_scenario(params: Dict[str, Any]) -> Dict[str, Any]:
    """Setup for basic Kafka latency/throughput scenarios."""
    agent_count = params.get("agent_count", 5)
    topology_pattern = params.get(
        "topology_pattern", TopologyPattern.FULLY_CONNECTED
    )

    # Create Kafka communication environment
    kafka_config = {
        "bootstrap_servers": ["localhost:9092"],
        "client_id": "kafka_benchmark_service",
    }
    env = KafkaCommunicationEnvironment(kafka_config)
    env.start_service()

    # Create agents
    agents = []
    for i in range(agent_count):
        agent = create_kafka_test_agent(f"agent_{i}")
        agents.append(agent)
        env.register_agent(agent)

    # Setup topology
    config = CommunicationConfiguration()
    config.set_agents([str(agent.id) for agent in agents])
    topology = config.set_topology(topology_pattern)

    # Apply topology to environment
    for sender, receiver in topology.links:
        env.add_communication_link(sender, receiver)

    return {
        "environment": env,
        "agents": agents,
        "config": config,
        "agent_count": agent_count,
        "topology_density": (
            len(topology.links) / (agent_count * (agent_count - 1))
            if agent_count > 1
            else 0
        ),
    }


def teardown_kafka_basic_scenario(params: Dict[str, Any]):
    """Cleanup for Kafka basic scenarios."""
    # Stop the Kafka communication service and close connections
    if "environment" in params:
        env = params["environment"]
        env.close_all_agents()
        env.stop_service()

    # Clear references
    params.clear()


def test_kafka_point_to_point_latency(
    params: Dict[str, Any], benchmark: CommunicationBenchmark
) -> Dict[str, Any]:
    """Test basic point-to-point message latency via Kafka."""
    agents = params["agents"]
    message_count = params.get("message_count", 100)

    if len(agents) < 2:
        return {"delivery_failures": message_count}

    sender = agents[0]
    receiver = agents[1]

    delivery_failures = 0

    for i in range(message_count):
        message_id = f"kafka_latency_test_{i}"

        # Start timing
        benchmark.latency_tracker.start_message_timing(message_id)
        benchmark.throughput_tracker.record_message()

        # Send message via Kafka
        content = {"test_id": message_id, "data": f"kafka_test_data_{i}"}
        success = sender.send_message(
            str(receiver.id), KafkaMessageType.INFORM, content
        )

        if success:
            # Simulate processing time and end timing
            time.sleep(0.001)  # 1ms processing simulation
            benchmark.latency_tracker.end_message_timing(message_id)
        else:
            delivery_failures += 1

        # Small delay between messages
        time.sleep(0.01)

    return {"delivery_failures": delivery_failures}


def test_kafka_broadcast_throughput(
    params: Dict[str, Any], benchmark: CommunicationBenchmark
) -> Dict[str, Any]:
    """Test Kafka broadcast message throughput."""
    agents = params["agents"]
    message_count = params.get("message_count", 50)

    if len(agents) < 1:
        return {"delivery_failures": message_count}

    sender = agents[0]
    delivery_failures = 0

    for i in range(message_count):
        message_id = f"kafka_broadcast_test_{i}"

        # Start timing
        benchmark.latency_tracker.start_message_timing(message_id)
        benchmark.throughput_tracker.record_message()

        # Send broadcast via Kafka
        content = {
            "broadcast_id": message_id,
            "announcement": f"kafka_broadcast_{i}",
        }
        result = sender.broadcast_message(content)

        if result.get("status") == "completed":
            # End timing
            benchmark.latency_tracker.end_message_timing(message_id)
        else:
            delivery_failures += 1

        time.sleep(0.02)  # Slight delay between broadcasts

    return {"delivery_failures": delivery_failures}


def test_kafka_concurrent_messaging(
    params: Dict[str, Any], benchmark: CommunicationBenchmark
) -> Dict[str, Any]:
    """Test concurrent Kafka messaging between multiple agents."""
    agents = params["agents"]
    messages_per_agent = params.get("messages_per_agent", 20)

    if len(agents) < 2:
        return {"delivery_failures": messages_per_agent * len(agents)}

    delivery_failures = 0
    timeout_failures = 0

    def kafka_agent_messaging_task(agent, agent_index):
        nonlocal delivery_failures, timeout_failures
        targets = [a for a in agents if a != agent]

        for i in range(messages_per_agent):
            target = random.choice(targets)
            message_id = f"kafka_concurrent_{agent_index}_{i}"

            # Start timing
            benchmark.latency_tracker.start_message_timing(message_id)
            benchmark.throughput_tracker.record_message()

            # Send message via Kafka
            content = {
                "sender": agent_index,
                "message_num": i,
                "timestamp": time.time(),
            }
            success = agent.send_message(
                str(target.id), KafkaMessageType.INFORM, content
            )

            if success:
                # End timing
                benchmark.latency_tracker.end_message_timing(message_id)
            else:
                delivery_failures += 1

            # Random delay to simulate realistic messaging patterns
            time.sleep(random.uniform(0.005, 0.02))

    # Run concurrent Kafka messaging
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(agents)
    ) as executor:
        futures = []
        for i, agent in enumerate(agents):
            future = executor.submit(kafka_agent_messaging_task, agent, i)
            futures.append(future)

        # Wait for all tasks to complete
        concurrent.futures.wait(futures, timeout=30)

        # Check for timeout failures
        for future in futures:
            if not future.done():
                timeout_failures += 1

    return {
        "delivery_failures": delivery_failures,
        "timeout_failures": timeout_failures,
    }


def test_kafka_scalability_stress(
    params: Dict[str, Any], benchmark: CommunicationBenchmark
) -> Dict[str, Any]:
    """Test Kafka system under high message load for scalability."""
    agents = params["agents"]
    stress_duration = params.get("stress_duration", 5.0)  # seconds

    if len(agents) < 2:
        return {"delivery_failures": 1000}

    delivery_failures = 0
    message_count = 0

    def kafka_stress_messaging_task(agent):
        nonlocal delivery_failures, message_count
        end_time = time.time() + stress_duration
        targets = [a for a in agents if a != agent]

        while time.time() < end_time:
            target = random.choice(targets)
            message_id = f"kafka_stress_{agent.id}_{message_count}"

            # Track timing and throughput
            benchmark.latency_tracker.start_message_timing(message_id)
            benchmark.throughput_tracker.record_message()

            # Send message via Kafka
            content = {"stress_test": True, "count": message_count}
            success = agent.send_message(
                str(target.id),
                random.choice(
                    [KafkaMessageType.INFORM, KafkaMessageType.REQUEST]
                ),
                content,
            )

            if success:
                benchmark.latency_tracker.end_message_timing(message_id)
            else:
                delivery_failures += 1

            message_count += 1

            # Minimal delay for high throughput
            time.sleep(0.001)

    # Run Kafka stress test
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(agents)
    ) as executor:
        futures = [
            executor.submit(kafka_stress_messaging_task, agent)
            for agent in agents
        ]
        concurrent.futures.wait(futures, timeout=stress_duration + 5)

    return {
        "delivery_failures": delivery_failures,
        "total_stress_messages": message_count,
    }


def create_kafka_benchmark_scenarios() -> CommunicationBenchmark:
    """Create and configure all Kafka benchmark scenarios."""
    benchmark = CommunicationBenchmark()

    # Scenario 1: Kafka Point-to-Point Latency
    kafka_latency_scenario = BenchmarkScenario(
        name="point_to_point_latency",
        description="Measures basic Kafka message latency between two agents",
    )
    kafka_latency_scenario.set_setup(setup_kafka_basic_scenario)
    kafka_latency_scenario.set_test(test_kafka_point_to_point_latency)
    kafka_latency_scenario.set_teardown(teardown_kafka_basic_scenario)
    benchmark.add_scenario(kafka_latency_scenario)

    # Scenario 2: Kafka Broadcast Throughput
    kafka_broadcast_scenario = BenchmarkScenario(
        name="broadcast_throughput",
        description="Tests Kafka broadcast message throughput and delivery",
    )
    kafka_broadcast_scenario.set_setup(setup_kafka_basic_scenario)
    kafka_broadcast_scenario.set_test(test_kafka_broadcast_throughput)
    kafka_broadcast_scenario.set_teardown(teardown_kafka_basic_scenario)
    benchmark.add_scenario(kafka_broadcast_scenario)

    # Scenario 3: Kafka Concurrent Messaging
    kafka_concurrent_scenario = BenchmarkScenario(
        name="concurrent_messaging",
        description="Tests Kafka concurrent messaging between multiple agents",
    )
    kafka_concurrent_scenario.set_setup(setup_kafka_basic_scenario)
    kafka_concurrent_scenario.set_test(test_kafka_concurrent_messaging)
    kafka_concurrent_scenario.set_teardown(teardown_kafka_basic_scenario)
    benchmark.add_scenario(kafka_concurrent_scenario)

    # Scenario 4: Kafka Scalability Stress Test
    kafka_stress_scenario = BenchmarkScenario(
        name="scalability_stress",
        description="High-load Kafka stress test for scalability analysis",
    )
    kafka_stress_scenario.set_setup(setup_kafka_basic_scenario)
    kafka_stress_scenario.set_test(test_kafka_scalability_stress)
    kafka_stress_scenario.set_teardown(teardown_kafka_basic_scenario)
    benchmark.add_scenario(kafka_stress_scenario)

    return benchmark


def run_kafka_topology_comparison():
    """Run Kafka performance comparison across different topology patterns."""
    benchmark = create_kafka_benchmark_scenarios()
    topologies = [
        TopologyPattern.FULLY_CONNECTED,
        TopologyPattern.STAR,
        TopologyPattern.RING,
        TopologyPattern.CHAIN,
    ]

    results = {}

    for topology in topologies:
        print(f"\nTesting Kafka topology: {topology.value}")
        result = benchmark.run_scenario(
            "concurrent_messaging",
            agent_count=6,
            topology_pattern=topology,
            messages_per_agent=15,
        )
        results[topology.value] = result
        benchmark.print_summary(result)

    # Compare results
    scenario_names = [
        f"concurrent_messaging_{t.value}" for t in topologies
    ]
    comparison = benchmark.compare_scenarios(scenario_names)

    print("\n" + "=" * 60)
    print("KAFKA TOPOLOGY COMPARISON SUMMARY")
    print("=" * 60)
    for topology, metrics in comparison.items():
        print(f"\n{topology}:")
        print(f"  Avg Latency: {metrics['avg_latency_ms']:.2f}ms")
        print(f"  Throughput: {metrics['throughput_msg_per_sec']:.1f} msg/s")
        print(f"  Success Rate: {metrics['success_rate_percent']:.1f}%")

    return results


def run_kafka_scalability_analysis():
    """Run Kafka scalability analysis with increasing agent counts."""
    benchmark = create_kafka_benchmark_scenarios()
    agent_counts = [3, 5, 8, 12]

    results = {}

    for count in agent_counts:
        print(f"\nTesting Kafka with {count} agents")
        result = benchmark.run_scenario(
            "scalability_stress",
            agent_count=count,
            topology_pattern=TopologyPattern.FULLY_CONNECTED,
            stress_duration=3.0,
        )
        results[count] = result
        benchmark.print_summary(result)

    print("\n" + "=" * 60)
    print("KAFKA SCALABILITY ANALYSIS SUMMARY")
    print("=" * 60)
    for count, result in results.items():
        print(f"\n{count} Agents:")
        print(f"  Avg Latency: {result.message_latency_avg*1000:.2f}ms")
        print(f"  Throughput: {result.throughput_avg:.1f} msg/s")
        print(f"  Success Rate: {result.success_rate*100:.1f}%")
        print(f"  CPU Usage: {result.cpu_usage_avg:.1f}%")
        print(f"  Memory Usage: {result.memory_usage_avg:.1f}MB")

    return results


if __name__ == "__main__":
    print("Kafka Communication Benchmark Suite")
    print("=" * 60)

    # Create Kafka benchmark suite
    benchmark = create_kafka_benchmark_scenarios()

    # Run individual Kafka scenarios
    print("\n1. Kafka Point-to-Point Latency Test")
    result1 = benchmark.run_scenario(
        "point_to_point_latency", agent_count=2, message_count=50
    )
    benchmark.print_summary(result1)

    print("\n2. Kafka Broadcast Throughput Test")
    result2 = benchmark.run_scenario(
        "broadcast_throughput", agent_count=5, message_count=30
    )
    benchmark.print_summary(result2)

    print("\n3. Kafka Concurrent Messaging Test")
    result3 = benchmark.run_scenario(
        "concurrent_messaging", agent_count=4, messages_per_agent=10
    )
    benchmark.print_summary(result3)

    # Export results
    benchmark.export_results("kafka_benchmark_results.json")
    print("\nResults exported to kafka_benchmark_results.json")

    # Optional: Run extended analysis
    extended_tests = input(
        "\nRun extended Kafka topology and scalability analysis? (y/n): "
    )
    if extended_tests.lower() == "y":
        print("\nRunning Kafka topology comparison...")
        run_kafka_topology_comparison()

        print("\nRunning Kafka scalability analysis...")
        run_kafka_scalability_analysis()
