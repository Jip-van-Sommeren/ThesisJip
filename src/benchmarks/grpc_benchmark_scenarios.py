"""
gRPC Benchmark Test Scenarios
Provides the same benchmark scenarios as REST but using gRPC communication.

Enables direct performance comparison between REST and gRPC implementations
of the same formal communication model from the thesis.
"""

import time
import random
from typing import Dict, Any, List, Optional
import concurrent.futures
from abstract_agent import AgentId
from communication.grpc.grpc_communication_agent import (
    ExtendedGrpcCommunicatingAgent,
    GrpcCommunicationEnvironment,
)
from communication.communication_config import (
    CommunicationConfiguration,
    TopologyPattern,
)
from benchmarks.communication_benchmark import (
    CommunicationBenchmark,
    BenchmarkScenario,
)
from communication.grpc.grpc_communication import GrpcMessageType


def create_grpc_test_agent(
    agent_id: str, observable_props: set = None
) -> ExtendedGrpcCommunicatingAgent:
    """Create a gRPC test agent with basic configuration."""
    if observable_props is None:
        observable_props = {"environment", "messages"}

    agent_id_obj = AgentId("grpc_benchmark", "test", agent_id)
    agent = ExtendedGrpcCommunicatingAgent(agent_id_obj, observable_props)
    agent.initialize_agent()
    return agent


def setup_grpc_basic_scenario(params: Dict[str, Any]) -> Dict[str, Any]:
    """Setup for basic gRPC latency/throughput scenarios."""
    agent_count = params.get("agent_count", 5)
    topology_pattern = params.get(
        "topology_pattern", TopologyPattern.FULLY_CONNECTED
    )

    # Create gRPC communication environment with dynamic port allocation
    env = GrpcCommunicationEnvironment()
    env.start_service()

    # Create agents
    agents = []
    for i in range(agent_count):
        agent = create_grpc_test_agent(f"agent_{i}")
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


def teardown_grpc_basic_scenario(params: Dict[str, Any]):
    """Cleanup for gRPC basic scenarios."""
    # Stop the gRPC communication service and close connections
    if "environment" in params:
        env = params["environment"]
        env.close_all_agents()
        env.stop_service()

    # Clear references
    params.clear()


def test_grpc_point_to_point_latency(
    params: Dict[str, Any], benchmark: CommunicationBenchmark
) -> Dict[str, Any]:
    """Test basic point-to-point message latency via gRPC."""
    agents = params["agents"]
    message_count = params.get("message_count", 100)

    if len(agents) < 2:
        return {"delivery_failures": message_count}

    sender = agents[0]
    receiver = agents[1]

    delivery_failures = 0

    for i in range(message_count):
        message_id = f"grpc_latency_test_{i}"

        # Start timing
        benchmark.latency_tracker.start_message_timing(message_id)
        benchmark.throughput_tracker.record_message()

        # Send message via gRPC
        content = {"test_id": message_id, "data": f"grpc_test_data_{i}"}
        success = sender.send_message(
            str(receiver.id), GrpcMessageType.INFORM, content
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


def test_grpc_broadcast_throughput(
    params: Dict[str, Any], benchmark: CommunicationBenchmark
) -> Dict[str, Any]:
    """Test gRPC broadcast message throughput."""
    agents = params["agents"]
    message_count = params.get("message_count", 50)

    if len(agents) < 1:
        return {"delivery_failures": message_count}

    sender = agents[0]
    delivery_failures = 0

    for i in range(message_count):
        message_id = f"grpc_broadcast_test_{i}"

        # Start timing
        benchmark.latency_tracker.start_message_timing(message_id)
        benchmark.throughput_tracker.record_message()

        # Send broadcast via gRPC
        content = {
            "broadcast_id": message_id,
            "announcement": f"grpc_broadcast_{i}",
        }
        result = sender.broadcast_message(content)

        if result.get("status") == "completed":
            # End timing
            benchmark.latency_tracker.end_message_timing(message_id)
        else:
            delivery_failures += 1

        time.sleep(0.02)  # Slight delay between broadcasts

    return {"delivery_failures": delivery_failures}


def test_grpc_concurrent_messaging(
    params: Dict[str, Any], benchmark: CommunicationBenchmark
) -> Dict[str, Any]:
    """Test concurrent gRPC messaging between multiple agents."""
    agents = params["agents"]
    messages_per_agent = params.get("messages_per_agent", 20)

    if len(agents) < 2:
        return {"delivery_failures": messages_per_agent * len(agents)}

    delivery_failures = 0
    timeout_failures = 0

    def grpc_agent_messaging_task(agent, agent_index):
        nonlocal delivery_failures, timeout_failures

        # Get valid targets based on topology from configuration
        config = params.get("config")
        valid_targets = []

        if config and hasattr(config, "topology"):
            agent_id_str = str(agent.id)
            for other_agent in agents:
                if other_agent != agent:
                    other_id_str = str(other_agent.id)
                    if (agent_id_str, other_id_str) in config.topology.links:
                        valid_targets.append(other_agent)

        if not valid_targets:
            valid_targets = [a for a in agents if a != agent]

        if not valid_targets:
            return

        for i in range(messages_per_agent):
            target = random.choice(valid_targets)
            message_id = f"grpc_concurrent_{agent_index}_{i}"

            # Start timing
            benchmark.latency_tracker.start_message_timing(message_id)
            benchmark.throughput_tracker.record_message()

            # Send message via gRPC
            content = {
                "sender": agent_index,
                "message_num": i,
                "timestamp": time.time(),
            }
            success = agent.send_message(
                str(target.id), GrpcMessageType.INFORM, content
            )

            if success:
                # End timing
                benchmark.latency_tracker.end_message_timing(message_id)
            else:
                delivery_failures += 1

            # Random delay to simulate realistic messaging patterns
            time.sleep(random.uniform(0.005, 0.02))

    # Run concurrent gRPC messaging
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(agents)
    ) as executor:
        futures = []
        for i, agent in enumerate(agents):
            future = executor.submit(grpc_agent_messaging_task, agent, i)
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


def test_grpc_scalability_stress(
    params: Dict[str, Any], benchmark: CommunicationBenchmark
) -> Dict[str, Any]:
    """Test gRPC system under high message load for scalability."""
    agents = params["agents"]
    stress_duration = params.get("stress_duration", 5.0)  # seconds

    if len(agents) < 2:
        return {"delivery_failures": 1000}

    delivery_failures = 0
    message_count = 0

    def grpc_stress_messaging_task(agent):
        nonlocal delivery_failures, message_count
        end_time = time.time() + stress_duration
        targets = [a for a in agents if a != agent]

        while time.time() < end_time:
            target = random.choice(targets)
            message_id = f"grpc_stress_{agent.id}_{message_count}"

            # Track timing and throughput
            benchmark.latency_tracker.start_message_timing(message_id)
            benchmark.throughput_tracker.record_message()

            # Send message via gRPC
            content = {"stress_test": True, "count": message_count}
            success = agent.send_message(
                str(target.id),
                random.choice(
                    [GrpcMessageType.INFORM, GrpcMessageType.REQUEST]
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

    # Run gRPC stress test
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(agents)
    ) as executor:
        futures = [
            executor.submit(grpc_stress_messaging_task, agent)
            for agent in agents
        ]
        concurrent.futures.wait(futures, timeout=stress_duration + 5)

    return {
        "delivery_failures": delivery_failures,
        "total_stress_messages": message_count,
    }


def create_grpc_benchmark_scenarios() -> CommunicationBenchmark:
    """Create and configure all gRPC benchmark scenarios."""
    benchmark = CommunicationBenchmark()

    # Scenario 1: gRPC Point-to-Point Latency
    grpc_latency_scenario = BenchmarkScenario(
        name="point_to_point_latency",
        description="Measures basic gRPC message latency between two agents",
    )
    grpc_latency_scenario.set_setup(setup_grpc_basic_scenario)
    grpc_latency_scenario.set_test(test_grpc_point_to_point_latency)
    grpc_latency_scenario.set_teardown(teardown_grpc_basic_scenario)
    benchmark.add_scenario(grpc_latency_scenario)

    # Scenario 2: gRPC Broadcast Throughput
    grpc_broadcast_scenario = BenchmarkScenario(
        name="broadcast_throughput",
        description="Tests gRPC broadcast message throughput and delivery",
    )
    grpc_broadcast_scenario.set_setup(setup_grpc_basic_scenario)
    grpc_broadcast_scenario.set_test(test_grpc_broadcast_throughput)
    grpc_broadcast_scenario.set_teardown(teardown_grpc_basic_scenario)
    benchmark.add_scenario(grpc_broadcast_scenario)

    # Scenario 3: gRPC Concurrent Messaging
    grpc_concurrent_scenario = BenchmarkScenario(
        name="concurrent_messaging",
        description="Tests gRPC concurrent messaging between multiple agents",
    )
    grpc_concurrent_scenario.set_setup(setup_grpc_basic_scenario)
    grpc_concurrent_scenario.set_test(test_grpc_concurrent_messaging)
    grpc_concurrent_scenario.set_teardown(teardown_grpc_basic_scenario)
    benchmark.add_scenario(grpc_concurrent_scenario)

    # Scenario 4: gRPC Scalability Stress Test
    grpc_stress_scenario = BenchmarkScenario(
        name="scalability_stress",
        description="High-load gRPC stress test for scalability analysis",
    )
    grpc_stress_scenario.set_setup(setup_grpc_basic_scenario)
    grpc_stress_scenario.set_test(test_grpc_scalability_stress)
    grpc_stress_scenario.set_teardown(teardown_grpc_basic_scenario)
    benchmark.add_scenario(grpc_stress_scenario)

    return benchmark


def run_grpc_topology_comparison(
    benchmark: CommunicationBenchmark,
):
    """Run gRPC performance comparison across different topology patterns."""
    topologies = [
        TopologyPattern.FULLY_CONNECTED,
        TopologyPattern.STAR,
        TopologyPattern.RING,
        TopologyPattern.CHAIN,
    ]

    results = {}

    for topology in topologies:
        print(f"\nTesting gRPC topology: {topology.value}")
        result = benchmark.run_scenario(
            "concurrent_messaging",
            agent_count=20,
            topology_pattern=topology,
            messages_per_agent=500,
        )
        results[topology.value] = result
        benchmark.print_summary(result)

    # Compare results
    scenario_names = [f"concurrent_messaging_{t.value}" for t in topologies]
    comparison = benchmark.compare_scenarios(scenario_names)

    print("\n" + "=" * 60)
    print("gRPC TOPOLOGY COMPARISON SUMMARY")
    print("=" * 60)
    for topology, metrics in comparison.items():
        print(f"\n{topology}:")
        print(f"  Avg Latency: {metrics['avg_latency_ms']:.2f}ms")
        print(f"  Throughput: {metrics['throughput_msg_per_sec']:.1f} msg/s")
        print(f"  Success Rate: {metrics['success_rate_percent']:.1f}%")

    return results


def run_grpc_scalability_analysis(
    benchmark: CommunicationBenchmark, agent_counts: Optional[List[int]] = None
):
    """Run gRPC scalability analysis with increasing agent counts."""
    if agent_counts is None:
        agent_counts = [3, 5, 8, 12]

    results = {}

    for count in agent_counts:
        print(f"\nTesting gRPC with {count} agents")
        result = benchmark.run_scenario(
            "scalability_stress",
            agent_count=count,
            topology_pattern=TopologyPattern.FULLY_CONNECTED,
            stress_duration=3.0,
        )
        results[count] = result
        benchmark.print_summary(result)

    print("\n" + "=" * 60)
    print("gRPC SCALABILITY ANALYSIS SUMMARY")
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
    print("gRPC Communication Benchmark Suite")
    print("=" * 60)

    # Create gRPC benchmark suite
    benchmark = create_grpc_benchmark_scenarios()

    # Run individual gRPC scenarios
    print("\n1. gRPC Point-to-Point Latency Test")
    result1 = benchmark.run_scenario(
        "point_to_point_latency", agent_count=2, message_count=50
    )
    benchmark.print_summary(result1)

    print("\n2. gRPC Broadcast Throughput Test")
    result2 = benchmark.run_scenario(
        "broadcast_throughput", agent_count=5, message_count=30
    )
    benchmark.print_summary(result2)

    print("\n3. gRPC Concurrent Messaging Test")
    result3 = benchmark.run_scenario(
        "concurrent_messaging", agent_count=4, messages_per_agent=10
    )
    benchmark.print_summary(result3)

    # Optional: Run extended analysis

    print("\nRunning gRPC topology comparison...")
    run_grpc_topology_comparison(benchmark)

    print("\nRunning gRPC scalability analysis...")
    run_grpc_scalability_analysis(benchmark, [5, 10, 15, 20])

    # Export results (after all benchmarks are complete)
    benchmark.export_results("grpc_benchmark_results.json")
    print("\nResults exported to grpc_benchmark_results.json")
