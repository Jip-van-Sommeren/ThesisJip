"""
gRPC Benchmark Test Scenarios
Provides the same benchmark scenarios as REST but using gRPC communication.

Enables direct performance comparison between REST and gRPC implementations
of the same formal communication model from the thesis.
"""

import time
import random
import threading
from typing import Dict, Any, List, Optional
import concurrent.futures
from mas.core import AgentId
from benchmarks.communication_benchmarks.communication_benchmark import generate_payload
from benchmarks.communication.grpc.grpc_communication_agent import (
    ExtendedGrpcCommunicatingAgent,
    GrpcCommunicationEnvironment,
)
from benchmarks.communication.communication_config import (
    CommunicationConfiguration,
    TopologyPattern,
)
from benchmarks.communication_benchmarks.communication_benchmark import (
    CommunicationBenchmark,
    BenchmarkScenario,
)
from benchmarks.communication.grpc.grpc_communication import GrpcMessageType


def _is_ack_message(message) -> bool:
    msg_type = getattr(message, "message_type", None)
    if isinstance(msg_type, GrpcMessageType):
        return msg_type == GrpcMessageType.ACK
    if msg_type is not None:
        try:
            return GrpcMessageType(msg_type) == GrpcMessageType.ACK
        except ValueError:
            return False
    return False


def _drain_non_ack_messages(mailbox) -> List[Any]:
    """Drain non-ACK messages while preserving ACKs in the mailbox."""
    non_ack = []
    with mailbox.lock:
        if not mailbox.messages:
            return non_ack
        keep = []
        for msg in mailbox.messages:
            if _is_ack_message(msg):
                keep.append(msg)
            else:
                non_ack.append(msg)
        if non_ack:
            mailbox.messages.clear()
            mailbox.messages.extend(keep)
    return non_ack


def _consume_ack_for(mailbox, message_id: str) -> int:
    """Remove ACKs for a message and return how many were consumed."""
    removed = 0
    with mailbox.lock:
        if not mailbox.messages:
            return 0
        keep = []
        for msg in mailbox.messages:
            if _is_ack_message(msg) and msg.content.get("ack_for") == message_id:
                removed += 1
                continue
            keep.append(msg)
        if removed:
            mailbox.messages.clear()
            mailbox.messages.extend(keep)
    return removed


def create_grpc_test_agent(
    agent_id: str,
    observable_props: set = None,
    communication_mode: str = "unary",
) -> ExtendedGrpcCommunicatingAgent:
    """Create a gRPC test agent with basic configuration."""
    if observable_props is None:
        observable_props = {"environment", "messages"}

    agent_id_obj = AgentId("grpc_benchmark", "test", agent_id)
    agent = ExtendedGrpcCommunicatingAgent(
        agent_id_obj,
        observable_props,
        communication_mode=communication_mode,
    )
    agent.initialize_agent()
    return agent


def setup_grpc_basic_scenario(params: Dict[str, Any]) -> Dict[str, Any]:
    """Setup for basic gRPC latency/throughput scenarios.

    Default latency_mode is 'end_to_end' for fair comparison across all protocols.
    This measures complete message delivery time including gRPC response acknowledgments.
    """
    agent_count = params.get("agent_count", 5)
    topology_pattern = params.get(
        "topology_pattern", TopologyPattern.FULLY_CONNECTED
    )
    latency_mode = params.get("latency_mode", "end_to_end")
    communication_mode = params.get("grpc_mode", "unary")

    # Import LatencyMode enum
    from benchmarks.communication.base_communication import LatencyMode

    # Convert string to enum
    if latency_mode == "send_only":
        latency_mode_enum = LatencyMode.SEND_ONLY
    elif latency_mode == "app_ack":
        latency_mode_enum = LatencyMode.APP_ACK
    else:
        latency_mode_enum = LatencyMode.END_TO_END

    # Create gRPC communication environment with dynamic port allocation
    env = GrpcCommunicationEnvironment(
        latency_mode=latency_mode_enum, communication_mode=communication_mode
    )
    env.start_service()

    # Create agents
    agents = []
    for i in range(agent_count):
        agent = create_grpc_test_agent(
            f"agent_{i}", communication_mode=communication_mode
        )
        agents.append(agent)
        env.register_agent(agent)
        if getattr(agent, "mailbox", None) is None and env.grpc_server:
            agent.mailbox = env.grpc_server.service_impl.mailboxes.get(
                str(agent.id)
            )

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


def _wait_for_ack(agent, message_id: str, timeout: float = 0.5) -> bool:
    """Wait for ACK message with matching message_id.

    Args:
        agent: Agent to check for ACK
        message_id: Expected message ID in ACK
        timeout: Maximum time to wait in seconds

    Returns:
        True if ACK received, False if timeout
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        if _consume_ack_for(agent.mailbox, message_id):
            return True
        time.sleep(0.001)  # Small polling interval
    return False


def test_grpc_point_to_point_latency(
    params: Dict[str, Any], benchmark: CommunicationBenchmark
) -> Dict[str, Any]:
    """Test basic point-to-point message latency via gRPC."""
    from benchmarks.communication.base_communication import MessageType

    agents = params["agents"]
    message_count = params.get("message_count", 100)
    payload_size = params.get("payload_size_bytes", 100)
    latency_mode = params.get("latency_mode", "end_to_end")

    if len(agents) < 2:
        return {"delivery_failures": message_count}

    sender = agents[0]
    receiver = agents[1]

    delivery_failures = 0
    ack_timeouts = 0

    # In app_ack mode, start a receiver thread that sends ACKs
    stop_ack_thread = threading.Event()
    ack_thread = None

    def receiver_ack_loop():
        """Background thread to send ACKs for received messages."""
        while not stop_ack_thread.is_set():
            messages = _drain_non_ack_messages(receiver.mailbox)
            if not messages:
                time.sleep(0.001)
                continue
            for msg in messages:
                ack_content = {"ack_for": msg.content.get("test_id")}
                receiver.send_message(
                    str(sender.id), GrpcMessageType.ACK, ack_content
                )
            time.sleep(0.001)

    if latency_mode == "app_ack":
        ack_thread = threading.Thread(target=receiver_ack_loop, daemon=True)
        ack_thread.start()

    for i in range(message_count):
        message_id = f"grpc_latency_test_{i}"

        # Start timing
        benchmark.latency_tracker.start_message_timing(message_id)
        benchmark.throughput_tracker.record_message()

        # Send message via gRPC with specified payload size
        payload_data = generate_payload(payload_size)
        content = {"test_id": message_id, "data": payload_data}
        success = sender.send_message(
            str(receiver.id), GrpcMessageType.INFORM, content
        )

        if success:
            if latency_mode == "app_ack":
                # Wait for ACK from receiver
                if _wait_for_ack(sender, message_id):
                    benchmark.latency_tracker.end_message_timing(message_id)
                else:
                    ack_timeouts += 1
                    delivery_failures += 1
            else:
                # End timing immediately after send (send_only or end_to_end)
                benchmark.latency_tracker.end_message_timing(message_id)
        else:
            delivery_failures += 1

        # Small delay between messages (simulates realistic pacing)
        time.sleep(0.01)

    # Stop ACK thread
    if ack_thread:
        stop_ack_thread.set()
        ack_thread.join(timeout=1.0)

    return {"delivery_failures": delivery_failures, "ack_timeouts": ack_timeouts}


def test_grpc_broadcast_throughput(
    params: Dict[str, Any], benchmark: CommunicationBenchmark
) -> Dict[str, Any]:
    """Test gRPC broadcast message throughput."""
    from benchmarks.communication.base_communication import MessageType

    agents = params["agents"]
    message_count = params.get("message_count", 50)
    payload_size = params.get("payload_size_bytes", 100)
    latency_mode = params.get("latency_mode", "end_to_end")

    if len(agents) < 1:
        return {"delivery_failures": message_count}

    sender = agents[0]
    receivers = agents[1:]  # All other agents are receivers
    delivery_failures = 0
    ack_timeouts = 0

    # In app_ack mode, start receiver threads that send ACKs
    stop_ack_threads = threading.Event()
    ack_threads = []

    def receiver_ack_loop(receiver):
        """ACK loop for broadcast; drain non-ACK messages to preserve ACKs."""
        while not stop_ack_threads.is_set():
            inbox = _drain_non_ack_messages(receiver.mailbox)
            if not inbox:
                time.sleep(0.001)
                continue

            for msg in inbox:
                ack_content = {"ack_for": msg.content.get("broadcast_id")}
                receiver.send_message(
                    str(sender.id), GrpcMessageType.ACK, ack_content
                )

    if latency_mode == "app_ack" and receivers:
        for receiver in receivers:
            thread = threading.Thread(target=receiver_ack_loop, args=(receiver,), daemon=True)
            thread.start()
            ack_threads.append(thread)

    for i in range(message_count):
        message_id = f"grpc_broadcast_test_{i}"

        # Track throughput
        benchmark.throughput_tracker.record_message()

        # Start timing
        benchmark.latency_tracker.start_message_timing(message_id)

        # Send broadcast via gRPC with specified payload size
        payload_data = generate_payload(payload_size)
        content = {
            "broadcast_id": message_id,
            "announcement": payload_data,
        }
        result = sender.broadcast_message(content)

        if result.get("status") == "completed":
            if latency_mode == "app_ack" and receivers:
                # Wait for ACKs from all receivers
                expected_acks = len(receivers)
                received_acks = 0
                timeout_start = time.time()
                ack_timeout = float(params.get("ack_timeout", params.get("ack_timeout_ms", 0.5)))
                timeout = ack_timeout

                while received_acks < expected_acks and time.time() - timeout_start < timeout:
                    received_acks += _consume_ack_for(
                        sender.mailbox, message_id
                    )
                    if received_acks < expected_acks:
                        time.sleep(0.001)

                if received_acks >= expected_acks:
                    benchmark.latency_tracker.end_message_timing(message_id)
                else:
                    ack_timeouts += 1
                    delivery_failures += 1
            else:
                # End timing (send-only or end_to_end semantics)
                benchmark.latency_tracker.end_message_timing(message_id)
        else:
            delivery_failures += 1

        time.sleep(0.02)  # Slight delay between broadcasts

    # Stop ACK threads
    if ack_threads:
        stop_ack_threads.set()
        for thread in ack_threads:
            thread.join(timeout=1.0)

    return {"delivery_failures": delivery_failures, "ack_timeouts": ack_timeouts}


def test_grpc_concurrent_messaging(
    params: Dict[str, Any], benchmark: CommunicationBenchmark
) -> Dict[str, Any]:
    """Test concurrent gRPC messaging between multiple agents."""
    from benchmarks.communication.base_communication import MessageType

    agents = params["agents"]
    messages_per_agent = params.get("messages_per_agent", 20)
    payload_size = params.get("payload_size_bytes", 100)
    latency_mode = params.get("latency_mode", "end_to_end")

    if len(agents) < 2:
        return {"delivery_failures": messages_per_agent * len(agents)}

    delivery_failures = 0
    timeout_failures = 0
    ack_timeouts = 0

    # In app_ack mode, start receiver threads for all agents
    stop_ack_threads = threading.Event()
    ack_threads = []

    def receiver_ack_loop(receiver):
        """ACK loop for concurrent messaging; drain non-ACK messages."""
        while not stop_ack_threads.is_set():
            inbox = _drain_non_ack_messages(receiver.mailbox)
            if not inbox:
                time.sleep(0.001)
                continue

            for msg in inbox:
                sender_index = msg.content.get("sender")
                message_num = msg.content.get("message_num")
                if sender_index is None or message_num is None:
                    continue
                full_msg_id = f"grpc_concurrent_{sender_index}_{message_num}"
                ack_content = {"ack_for": full_msg_id}
                receiver.send_message(
                    msg.sender_id, GrpcMessageType.ACK, ack_content
                )

    if latency_mode == "app_ack":
        for agent in agents:
            thread = threading.Thread(target=receiver_ack_loop, args=(agent,), daemon=True)
            thread.start()
            ack_threads.append(thread)

    concurrency_limit = params.get("concurrent_senders", len(agents))
    active_agents = agents[: max(1, min(concurrency_limit, len(agents)))]

    def grpc_agent_messaging_task(agent, agent_index):
        nonlocal delivery_failures, timeout_failures, ack_timeouts

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

            # Send message via gRPC with specified payload size
            payload_data = generate_payload(payload_size)
            content = {
                "sender": agent_index,
                "message_num": i,
                "data": payload_data,
                "timestamp": time.time(),
            }
            success = agent.send_message(
                str(target.id), GrpcMessageType.INFORM, content
            )

            if success:
                if latency_mode == "app_ack":
                    # Wait for ACK from receiver (configurable)
                    ack_timeout = float(params.get("ack_timeout", params.get("ack_timeout_ms", 0.5)))
                    if _wait_for_ack(agent, message_id, timeout=ack_timeout):
                        benchmark.latency_tracker.end_message_timing(message_id)
                    else:
                        ack_timeouts += 1
                        delivery_failures += 1
                else:
                    # End timing (send_only or end_to_end)
                    benchmark.latency_tracker.end_message_timing(message_id)
            else:
                delivery_failures += 1

            # Random delay to simulate realistic messaging patterns
            time.sleep(random.uniform(0.005, 0.02))

    # Run concurrent gRPC messaging
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(active_agents)
    ) as executor:
        futures = []
        for i, agent in enumerate(active_agents):
            future = executor.submit(grpc_agent_messaging_task, agent, i)
            futures.append(future)

        # Wait for all tasks to complete
        concurrent.futures.wait(futures, timeout=30)

        # Check for timeout failures
        for future in futures:
            if not future.done():
                timeout_failures += 1

    # Stop ACK threads
    if ack_threads:
        stop_ack_threads.set()
        for thread in ack_threads:
            thread.join(timeout=1.0)

    return {
        "delivery_failures": delivery_failures,
        "timeout_failures": timeout_failures,
        "timeout_failures_are_messages": False,
        "ack_timeouts": ack_timeouts,
    }


def test_grpc_scalability_stress(
    params: Dict[str, Any], benchmark: CommunicationBenchmark
) -> Dict[str, Any]:
    """Test gRPC system under high message load for scalability."""
    from benchmarks.communication.base_communication import MessageType

    agents = params["agents"]
    stress_duration = params.get("stress_duration", 5.0)  # seconds
    payload_size = params.get("payload_size_bytes", 100)
    latency_mode = params.get("latency_mode", "end_to_end")

    if len(agents) < 2:
        return {"delivery_failures": 1000}

    delivery_failures = 0
    message_count = 0
    ack_timeouts = 0

    # In app_ack mode, start receiver threads for all agents
    stop_ack_threads = threading.Event()
    ack_threads = []

    def receiver_ack_loop(receiver):
        """ACK loop for stress; drain non-ACK messages."""
        while not stop_ack_threads.is_set():
            inbox = _drain_non_ack_messages(receiver.mailbox)
            if not inbox:
                time.sleep(0.001)
                continue

            for msg in inbox:
                msg_id = msg.content.get("count", "")
                full_msg_id = f"grpc_stress_{msg.sender_id}_{msg_id}"
                ack_content = {"ack_for": full_msg_id}
                receiver.send_message(
                    msg.sender_id, GrpcMessageType.ACK, ack_content
                )

    if latency_mode == "app_ack":
        for agent in agents:
            thread = threading.Thread(target=receiver_ack_loop, args=(agent,), daemon=True)
            thread.start()
            ack_threads.append(thread)

    def grpc_stress_messaging_task(agent):
        nonlocal delivery_failures, message_count, ack_timeouts
        end_time = time.time() + stress_duration
        targets = [a for a in agents if a != agent]
        local_count = 0

        while time.time() < end_time:
            target = random.choice(targets)
            message_id = f"grpc_stress_{agent.id}_{local_count}"

            # Track timing and throughput
            benchmark.latency_tracker.start_message_timing(message_id)
            benchmark.throughput_tracker.record_message()

            # Send message via gRPC with specified payload size
            payload_data = generate_payload(payload_size)
            content = {
                "stress_test": True,
                "count": local_count,
                "data": payload_data,
            }
            success = agent.send_message(
                str(target.id),
                random.choice(
                    [GrpcMessageType.INFORM, GrpcMessageType.REQUEST]
                ),
                content,
            )

            if success:
                if latency_mode == "app_ack":
                    # Wait for ACK from receiver
                    ack_timeout = float(params.get("ack_timeout", params.get("ack_timeout_ms", 0.5)))
                    if _wait_for_ack(agent, message_id, timeout=ack_timeout):
                        benchmark.latency_tracker.end_message_timing(message_id)
                    else:
                        ack_timeouts += 1
                        delivery_failures += 1
                else:
                    benchmark.latency_tracker.end_message_timing(message_id)
            else:
                delivery_failures += 1

            local_count += 1
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

    # Stop ACK threads
    if ack_threads:
        stop_ack_threads.set()
        for thread in ack_threads:
            thread.join(timeout=1.0)

    return {
        "delivery_failures": delivery_failures,
        "total_stress_messages": message_count,
        "ack_timeouts": ack_timeouts,
    }


def create_grpc_benchmark_scenarios(
    latency_mode: str = "end_to_end",
) -> CommunicationBenchmark:
    """Create and configure all gRPC benchmark scenarios."""
    benchmark = CommunicationBenchmark()
    benchmark.latency_mode = latency_mode  # Store for use in scenarios

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
