"""
Benchmark Test Scenarios for REST Communication Implementation
Provides pre-defined scenarios to test different aspects of the communication
system.

Scenarios include:
- Basic latency and throughput testing
- Scalability with different agent counts
- Topology impact analysis
- Message type performance comparison
- Stress testing under high load
"""

import time
import random
import threading
from typing import Dict, Any, List, Optional
import concurrent.futures
from abstract_agent import AgentId
from benchmarks.communication_benchmark import generate_payload
from communication.rest.rest_communicating_agent import (
    ExtendedRestCommunicatingAgent,
    RestCommunicationEnvironment,
)
from communication.communication_config import (
    CommunicationConfiguration,
    TopologyPattern,
)
from benchmarks.communication_benchmark import (
    CommunicationBenchmark,
    BenchmarkScenario,
)
from communication.base_communication import MessageType, LatencyMode


def create_test_agent(
    agent_id: str,
    observable_props: set = None,
    transport_mode: str = "http1",
) -> ExtendedRestCommunicatingAgent:
    """Create a test agent with basic configuration."""
    if observable_props is None:
        observable_props = {"environment", "messages"}

    agent_id_obj = AgentId("benchmark", "test", agent_id)
    agent = ExtendedRestCommunicatingAgent(
        agent_id_obj, observable_props, transport_mode=transport_mode
    )
    agent.initialize_agent()
    return agent


def setup_basic_scenario(params: Dict[str, Any]) -> Dict[str, Any]:
    """Setup for basic latency/throughput scenarios.

    Default latency_mode is 'end_to_end' for fair comparison across all protocols.
    This measures complete message delivery time including acknowledgments.
    """
    agent_count = params.get("agent_count", 5)
    topology_pattern = params.get(
        "topology_pattern", TopologyPattern.FULLY_CONNECTED
    )
    latency_mode = params.get("latency_mode", "end_to_end")
    transport_mode = params.get("transport_mode", "http1")

    # Convert string to enum
    if latency_mode == "send_only":
        latency_mode_enum = LatencyMode.SEND_ONLY
    elif latency_mode == "app_ack":
        latency_mode_enum = LatencyMode.APP_ACK
    else:
        latency_mode_enum = LatencyMode.END_TO_END

    # Create communication environment with dynamic port allocation
    env = RestCommunicationEnvironment(
        latency_mode=latency_mode_enum, transport_mode=transport_mode
    )
    env.start_service()

    # Create agents
    agents = []
    for i in range(agent_count):
        agent = create_test_agent(f"agent_{i}", transport_mode=transport_mode)
        agents.append(agent)
        env.register_agent(agent)
        if getattr(agent, "mailbox", None) is None and env.comm_service:
            agent.mailbox = env.comm_service.mailboxes.get(str(agent.id))
            print(f"Agent {agent.id} mailbox: {agent.mailbox}")

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
        "agent_count": agent_count,  # Add this line to fix the issue
        "topology_density": (
            len(topology.links) / (agent_count * (agent_count - 1))
            if agent_count > 1
            else 0
        ),
    }


def teardown_basic_scenario(params: Dict[str, Any]):
    """Cleanup for basic scenarios."""
    # Stop the communication service to free up ports
    for agent in params.get("agents", []):
        if hasattr(agent, "comm_agent"):
            try:
                agent.comm_agent.close()
            except AttributeError:
                pass

    if "environment" in params:
        env = params["environment"]
        env.stop_service()

    # Clear references
    params.clear()


def _is_ack_message(message) -> bool:
    """Check whether the message is an ACK (protocol-agnostic)."""
    msg_type = getattr(message, "message_type", None)
    if hasattr(msg_type, "value"):
        msg_type = msg_type.value
    return msg_type == MessageType.ACK.value


def _wait_for_ack(agent, message_id: str, timeout: float = 5.0) -> bool:
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
        # Check mailbox for ACK messages
        messages = agent.mailbox.peek_messages()
        for msg in messages:
            if (
                _is_ack_message(msg)
                and msg.content.get("ack_for") == message_id
            ):
                agent.mailbox.get_messages(clear=True)
                return True
        time.sleep(0.005)  # Optimized polling interval (was 0.001, reduced CPU usage by 5x)
    return False


def test_point_to_point_latency(
    params: Dict[str, Any], benchmark: CommunicationBenchmark
) -> Dict[str, Any]:
    """Test basic point-to-point message latency."""
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
            messages = receiver.mailbox.peek_messages()
            for msg in messages:
                if _is_ack_message(msg):
                    continue

                ack_content = {"ack_for": msg.content.get("test_id")}
                receiver.send_message(
                    str(sender.id), MessageType.ACK, ack_content
                )

            if messages:
                receiver.mailbox.get_messages(clear=True)
            time.sleep(0.005)  # Optimized: reduced from 0.001

    if latency_mode == "app_ack":
        ack_thread = threading.Thread(target=receiver_ack_loop, daemon=True)
        ack_thread.start()

    for i in range(message_count):
        message_id = f"latency_test_{i}"

        # Start timing
        benchmark.latency_tracker.start_message_timing(message_id)
        benchmark.throughput_tracker.record_message()

        # Send message with specified payload size
        payload_data = generate_payload(payload_size)
        content = {"test_id": message_id, "data": payload_data}
        success = sender.send_message(
            str(receiver.id), MessageType.INFORM, content
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

    return {
        "delivery_failures": delivery_failures,
        "ack_timeouts": ack_timeouts,
    }


def test_broadcast_throughput(
    params: Dict[str, Any], benchmark: CommunicationBenchmark
) -> Dict[str, Any]:
    """Test broadcast message throughput."""
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
        """Background thread to send ACKs for received broadcast messages.

        Avoid dropping ACKs addressed to this receiver by re-queuing them
        after clearing the inbox.
        """
        while not stop_ack_threads.is_set():
            inbox = receiver.mailbox.get_messages(clear=True)
            if not inbox:
                time.sleep(0.005)  # Optimized: reduced from 0.001
                continue

            for msg in inbox:
                if _is_ack_message(msg):
                    # Keep ACKs available for local sender-side waiters
                    receiver.mailbox.add_message(msg)
                    continue

                ack_content = {"ack_for": msg.content.get("broadcast_id")}
                receiver.send_message(
                    str(sender.id), MessageType.ACK, ack_content
                )

    if latency_mode == "app_ack" and receivers:
        for receiver in receivers:
            thread = threading.Thread(
                target=receiver_ack_loop, args=(receiver,), daemon=True
            )
            thread.start()
            ack_threads.append(thread)

    for i in range(message_count):
        message_id = f"broadcast_test_{i}"

        # Track throughput
        benchmark.throughput_tracker.record_message()

        # Start timing
        benchmark.latency_tracker.start_message_timing(message_id)

        # Send broadcast with specified payload size
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
                timeout = 5.0

                while (
                    received_acks < expected_acks
                    and time.time() - timeout_start < timeout
                ):
                    messages = sender.mailbox.peek_messages()
                    for msg in messages:
                        if (
                            _is_ack_message(msg)
                            and msg.content.get("ack_for") == message_id
                        ):
                            received_acks += 1
                    if received_acks < expected_acks:
                        time.sleep(0.005)  # Optimized: reduced from 0.001

                # Clear ACK messages from mailbox
                sender.mailbox.get_messages(clear=True)

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

    return {
        "delivery_failures": delivery_failures,
        "ack_timeouts": ack_timeouts,
    }


def test_concurrent_messaging(
    params: Dict[str, Any], benchmark: CommunicationBenchmark
) -> Dict[str, Any]:
    """Test concurrent messaging between multiple agents."""
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
        """Background thread to send ACKs for received messages.

        Clear the inbox safely and re-queue ACKs so we don't starve local
        sender waiters in app_ack mode.
        """
        while not stop_ack_threads.is_set():
            inbox = receiver.mailbox.get_messages(clear=True)
            if not inbox:
                time.sleep(0.005)
                continue

            for msg in inbox:
                if _is_ack_message(msg):
                    receiver.mailbox.add_message(msg)
                    continue

                sender_id = msg.content.get("sender")
                message_index = msg.content.get("message_num")
                if sender_id is None or message_index is None:
                    continue
                full_msg_id = f"concurrent_{sender_id}_{message_index}"
                ack_content = {"ack_for": full_msg_id}
                receiver.send_message(
                    msg.sender_id, MessageType.ACK, ack_content
                )

    if latency_mode == "app_ack":
        for agent in agents:
            thread = threading.Thread(
                target=receiver_ack_loop, args=(agent,), daemon=True
            )
            thread.start()
            ack_threads.append(thread)

    concurrency_limit = params.get("concurrent_senders", len(agents))
    active_agents = agents[: max(1, min(concurrency_limit, len(agents)))]

    def agent_messaging_task(agent, agent_index):
        nonlocal delivery_failures, timeout_failures, ack_timeouts

        # Get valid targets based on topology from configuration
        config = params.get("config")
        valid_targets = []

        if config and hasattr(config, "topology"):
            # Get agents this agent can send to based on topology links
            agent_id_str = str(agent.id)
            for other_agent in agents:
                if other_agent != agent:
                    other_id_str = str(other_agent.id)
                    # Check if there's a link from this agent to the other
                    if (agent_id_str, other_id_str) in config.topology.links:
                        valid_targets.append(other_agent)

        # Fall back to all agents if no topology or no valid targets
        if not valid_targets:
            valid_targets = [a for a in agents if a != agent]

        if not valid_targets:
            # No valid targets, skip this agent
            return

        for i in range(messages_per_agent):
            target = random.choice(valid_targets)
            message_id = f"concurrent_{agent_index}_{i}"

            # Start timing
            benchmark.latency_tracker.start_message_timing(message_id)
            benchmark.throughput_tracker.record_message()

            # Send message with specified payload size
            payload_data = generate_payload(payload_size)
            content = {
                "sender": agent_index,
                "message_num": i,
                "data": payload_data,
                "timestamp": time.time(),
            }
            success = agent.send_message(
                str(target.id), MessageType.INFORM, content
            )

            if success:
                if latency_mode == "app_ack":
                    # Wait for ACK from receiver
                    if _wait_for_ack(agent, message_id):
                        benchmark.latency_tracker.end_message_timing(
                            message_id
                        )
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

    # Run concurrent messaging
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(active_agents)
    ) as executor:
        futures = []
        for i, agent in enumerate(active_agents):
            future = executor.submit(agent_messaging_task, agent, i)
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
        "ack_timeouts": ack_timeouts,
    }


def test_scalability_stress(
    params: Dict[str, Any], benchmark: CommunicationBenchmark
) -> Dict[str, Any]:
    """Test system under high message load for scalability."""
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
        """Background thread to send ACKs for received messages.

        Re-queue ACKs to avoid losing them due to mailbox clears while the
        same agent is also sending messages.
        """
        while not stop_ack_threads.is_set():
            inbox = receiver.mailbox.get_messages(clear=True)
            if not inbox:
                time.sleep(0.005)
                continue

            for msg in inbox:
                if _is_ack_message(msg):
                    receiver.mailbox.add_message(msg)
                    continue

                msg_id = msg.content.get("count", "")
                full_msg_id = f"stress_{msg.sender_id}_{msg_id}"
                ack_content = {"ack_for": full_msg_id}
                receiver.send_message(
                    msg.sender_id, MessageType.ACK, ack_content
                )

    if latency_mode == "app_ack":
        for agent in agents:
            thread = threading.Thread(
                target=receiver_ack_loop, args=(agent,), daemon=True
            )
            thread.start()
            ack_threads.append(thread)

    def stress_messaging_task(agent):
        nonlocal delivery_failures, message_count, ack_timeouts
        end_time = time.time() + stress_duration
        targets = [a for a in agents if a != agent]
        local_count = 0

        while time.time() < end_time:
            target = random.choice(targets)
            message_id = f"stress_{agent.id}_{local_count}"

            # Track timing and throughput
            benchmark.latency_tracker.start_message_timing(message_id)
            benchmark.throughput_tracker.record_message()

            # Send message with specified payload size
            payload_data = generate_payload(payload_size)
            content = {
                "stress_test": True,
                "count": local_count,
                "data": payload_data,
            }
            success = agent.send_message(
                str(target.id),
                random.choice([MessageType.INFORM, MessageType.REQUEST]),
                content,
            )

            if success:
                if latency_mode == "app_ack":
                    # Wait for ACK from receiver
                    if _wait_for_ack(agent, message_id, timeout=2.0):
                        benchmark.latency_tracker.end_message_timing(
                            message_id
                        )
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

    # Run stress test
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(agents)
    ) as executor:
        futures = [
            executor.submit(stress_messaging_task, agent) for agent in agents
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


def create_rest_benchmark_scenarios(
    latency_mode: str = "end_to_end",
) -> CommunicationBenchmark:
    """Create and configure all benchmark scenarios."""
    benchmark = CommunicationBenchmark()
    benchmark.latency_mode = latency_mode  # Store for use in scenarios

    # Scenario 1: Basic Point-to-Point Latency
    latency_scenario = BenchmarkScenario(
        name="point_to_point_latency",
        description="Measures basic message latency between two agents",
    )
    latency_scenario.set_setup(setup_basic_scenario)
    latency_scenario.set_test(test_point_to_point_latency)
    latency_scenario.set_teardown(teardown_basic_scenario)
    benchmark.add_scenario(latency_scenario)

    # Scenario 2: Broadcast Throughput
    broadcast_scenario = BenchmarkScenario(
        name="broadcast_throughput",
        description="Tests broadcast message throughput and delivery",
    )
    broadcast_scenario.set_setup(setup_basic_scenario)
    broadcast_scenario.set_test(test_broadcast_throughput)
    broadcast_scenario.set_teardown(teardown_basic_scenario)
    benchmark.add_scenario(broadcast_scenario)

    # Scenario 3: Concurrent Messaging
    concurrent_scenario = BenchmarkScenario(
        name="concurrent_messaging",
        description="Tests concurrent messaging between multiple agents",
    )
    concurrent_scenario.set_setup(setup_basic_scenario)
    concurrent_scenario.set_test(test_concurrent_messaging)
    concurrent_scenario.set_teardown(teardown_basic_scenario)
    benchmark.add_scenario(concurrent_scenario)

    # Scenario 4: Scalability Stress Test
    stress_scenario = BenchmarkScenario(
        name="scalability_stress",
        description="High-load stress test for scalability analysis",
    )
    stress_scenario.set_setup(setup_basic_scenario)
    stress_scenario.set_test(test_scalability_stress)
    stress_scenario.set_teardown(teardown_basic_scenario)
    benchmark.add_scenario(stress_scenario)

    return benchmark


def run_topology_comparison(benchmark: CommunicationBenchmark):
    """Run comparison across different topology patterns."""
    topologies = [
        TopologyPattern.FULLY_CONNECTED,
        TopologyPattern.STAR,
        TopologyPattern.RING,
        TopologyPattern.CHAIN,
    ]

    results = {}

    for topology in topologies:
        print(f"\nTesting topology: {topology.value}")
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
    print("TOPOLOGY COMPARISON SUMMARY")
    print("=" * 60)
    for topology, metrics in comparison.items():
        print(f"\n{topology}:")
        print(f"  Avg Latency: {metrics['avg_latency_ms']:.2f}ms")
        print(f"  Throughput: {metrics['throughput_msg_per_sec']:.1f} msg/s")
        print(f"  Success Rate: {metrics['success_rate_percent']:.1f}%")

    return results


def run_scalability_analysis(
    benchmark: CommunicationBenchmark, agent_counts: Optional[List[int]] = None
):
    """Run scalability analysis with increasing agent counts."""
    if agent_counts is None:
        agent_counts = [3, 5, 8, 12]

    results = {}

    for count in agent_counts:
        print(f"\nTesting with {count} agents")
        result = benchmark.run_scenario(
            "scalability_stress",
            agent_count=count,
            topology_pattern=TopologyPattern.FULLY_CONNECTED,
            stress_duration=3.0,
        )
        results[count] = result
        benchmark.print_summary(result)

    print("\n" + "=" * 60)
    print("SCALABILITY ANALYSIS SUMMARY")
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
    print("REST Communication Benchmark Suite")
    print("=" * 60)

    # Create benchmark suite
    benchmark = create_rest_benchmark_scenarios()

    # Run individual scenarios
    print("\n1. Point-to-Point Latency Test")
    result1 = benchmark.run_scenario(
        "point_to_point_latency", agent_count=2, message_count=50
    )
    benchmark.print_summary(result1)

    print("\n2. Broadcast Throughput Test")
    result2 = benchmark.run_scenario(
        "broadcast_throughput", agent_count=5, message_count=30
    )
    benchmark.print_summary(result2)

    print("\n3. Concurrent Messaging Test")
    result3 = benchmark.run_scenario(
        "concurrent_messaging", agent_count=4, messages_per_agent=10
    )
    benchmark.print_summary(result3)

    # Optional: Run extended analysis

    print("\nRunning topology comparison...")
    run_topology_comparison(benchmark)

    print("\nRunning scalability analysis...")
    run_scalability_analysis(benchmark, [5, 10, 15, 20])

    # Export results (after all benchmarks are complete)
    benchmark.export_results("rest_benchmark_results.json")
    print("\nResults exported to rest_benchmark_results.json")
