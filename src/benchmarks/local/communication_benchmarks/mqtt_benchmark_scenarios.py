"""
MQTT Benchmark Test Scenarios
Provides the same benchmark scenarios as REST and gRPC but using MQTT
communication.

Enables direct performance comparison between MQTT and other protocol
implementations
of the same formal communication model from the thesis.
"""

import time
import random
import threading
import subprocess
import socket
from typing import Dict, Any, List, Optional
import concurrent.futures
from benchmarks.local.communication_benchmarks.communication_benchmark import generate_payload
from benchmarks.communication.mqtt.mqtt_communication_agent import (
    MqttCommunicationEnvironment,
)
from benchmarks.communication.communication_config import (
    CommunicationConfiguration,
    TopologyPattern,
)
from benchmarks.local.communication_benchmarks.communication_benchmark import (
    CommunicationBenchmark,
    BenchmarkScenario,
)
from benchmarks.communication.mqtt.mqtt_communication import MqttMessageType
from benchmarks.communication.base_communication import MessageType


# Global variable to track broker container
_mqtt_broker_container: Optional[str] = None


def _is_ack_message(message) -> bool:
    msg_type = getattr(message, "message_type", None)
    if hasattr(msg_type, "value"):
        msg_type = msg_type.value
    return msg_type == MessageType.ACK.value


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


def is_mqtt_running(host="localhost", port=1883, timeout=2) -> bool:
    """Check if MQTT broker is accessible."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def start_mqtt_docker(wait_time=3) -> bool:
    """Start MQTT broker (Mosquitto) using Docker if not already running."""
    global _mqtt_broker_container

    print("[DEBUG] Checking if MQTT is running...")
    if is_mqtt_running():
        print("✓ MQTT broker is already running on localhost:1883")
        return True

    print("Starting MQTT broker (Mosquitto) using Docker...")
    import sys

    sys.stdout.flush()  # Ensure output is visible immediately

    try:
        # Check if container exists but is stopped
        result = subprocess.run(
            [
                "docker",
                "ps",
                "-a",
                "--filter",
                "name=mqtt-benchmark",
                "--format",
                "{{.Names}}",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        if "mqtt-benchmark" in result.stdout:
            # Start existing container
            print("[DEBUG] Found existing container, starting it...")
            start_result = subprocess.run(
                ["docker", "start", "mqtt-benchmark"],
                check=True,
                capture_output=True,
                text=True,
            )
            print(f"[DEBUG] Start output: {start_result.stdout}")
            _mqtt_broker_container = "mqtt-benchmark"
        else:
            # Create and start new container
            print("[DEBUG] Creating new container...")
            run_result = subprocess.run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    "mqtt-benchmark",
                    "-p",
                    "1883:1883",
                    "-p",
                    "9001:9001",
                    "eclipse-mosquitto:latest",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            print(f"[DEBUG] Container ID: {run_result.stdout.strip()}")
            _mqtt_broker_container = "mqtt-benchmark"

        print(f"Waiting {wait_time}s for MQTT broker to be ready...")
        time.sleep(wait_time)

        max_retries = 5
        for i in range(max_retries):
            if is_mqtt_running():
                print("✓ mqtt broker is ready")
                return True
            if i < max_retries - 1:
                print(f"  Still waiting... ({i+1}/{max_retries})")
                time.sleep(3)
        print("✗ mqtt broker failed to start properly")
        return False

    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to start MQTT broker: {e}")
        print(f"[DEBUG] stderr: {e.stderr if hasattr(e, 'stderr') else 'N/A'}")
        print("Please start MQTT broker manually or install Docker")
        return False
    except FileNotFoundError:
        print(
            "Docker not found. Please install Docker or start MQTT broker\
                manually"
        )
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return False


def stop_mqtt_docker():
    """Stop the MQTT broker Docker container."""
    global _mqtt_broker_container

    if _mqtt_broker_container:
        print("\nStopping MQTT broker...")
        try:
            subprocess.run(
                ["docker", "stop", _mqtt_broker_container],
                capture_output=True,
                check=True,
            )
            print("✓ MQTT broker stopped")
        except subprocess.CalledProcessError:
            pass  # Container might already be stopped
        finally:
            _mqtt_broker_container = None


def ensure_mqtt_running() -> bool:
    """Ensure MQTT broker is running, start it if needed."""
    if is_mqtt_running():
        return True

    print("\n MQTT broker not detected on localhost:1883")

    return start_mqtt_docker()


def setup_mqtt_basic_scenario(params: Dict[str, Any]) -> Dict[str, Any]:
    """Setup for basic MQTT latency/throughput scenarios.

    Default latency_mode is 'end_to_end' for fair comparison across all protocols.
    MQTT uses QoS 1 (at-least-once delivery) for comparable durability to Kafka acks=1.
    """
    agent_count = params.get("agent_count", 5)
    topology_pattern = params.get(
        "topology_pattern", TopologyPattern.FULLY_CONNECTED
    )
    latency_mode = params.get("latency_mode", "end_to_end")
    mqtt_qos = int(params.get("mqtt_qos", 1))

    # Ensure MQTT broker is running before creating environment
    if not ensure_mqtt_running():
        raise RuntimeError("Failed to start MQTT broker")

    # Import LatencyMode enum
    from benchmarks.communication.base_communication import LatencyMode

    # Convert string to enum
    if latency_mode == "send_only":
        latency_mode_enum = LatencyMode.SEND_ONLY
    elif latency_mode == "app_ack":
        latency_mode_enum = LatencyMode.APP_ACK
    else:
        latency_mode_enum = LatencyMode.END_TO_END

    # Create MQTT communication environment
    # QoS 1 configuration for fair benchmark comparison:
    # - QoS 1: At-least-once delivery with broker acknowledgment
    # - This provides similar durability guarantees to Kafka acks=1
    mqtt_config = {
        "broker_host": "localhost",
        "broker_port": 1883,
        "keepalive": 60,
        "qos": mqtt_qos,
    }
    env = MqttCommunicationEnvironment(
        broker_host="localhost",
        broker_port=1883,
        mqtt_config=mqtt_config,
        latency_mode=latency_mode_enum,
    )
    env.start_service()

    # Check if service started successfully
    if not env.is_running:
        raise RuntimeError(
            "MQTT service failed to start.\
                Ensure MQTT broker is running on localhost:1883"
        )

    # Create agents
    agents = []
    for i in range(agent_count):
        agent = env.create_agent(f"agent_{i}")
        if getattr(agent, "mailbox", None) is None and hasattr(
            agent, "mqtt_agent"
        ):
            agent.mailbox = agent.mqtt_agent.mailbox
        agents.append(agent)

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


def teardown_mqtt_basic_scenario(params: Dict[str, Any]):
    """Cleanup for MQTT basic scenarios."""
    # Stop the MQTT communication service and close connections
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


def test_mqtt_point_to_point_latency(
    params: Dict[str, Any], benchmark: CommunicationBenchmark
) -> Dict[str, Any]:
    """Test basic point-to-point message latency via MQTT."""
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
                    str(sender.id), MqttMessageType.ACK, ack_content
                )
            time.sleep(0.001)

    if latency_mode == "app_ack":
        ack_thread = threading.Thread(target=receiver_ack_loop, daemon=True)
        ack_thread.start()

    for i in range(message_count):
        message_id = f"mqtt_latency_test_{i}"

        # Start timing
        benchmark.latency_tracker.start_message_timing(message_id)
        benchmark.throughput_tracker.record_message()

        # Send message via MQTT with specified payload size
        payload_data = generate_payload(payload_size)
        content = {"test_id": message_id, "data": payload_data}
        success = sender.send_message(
            str(receiver.id), MqttMessageType.INFORM, content
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


def test_mqtt_broadcast_throughput(
    params: Dict[str, Any], benchmark: CommunicationBenchmark
) -> Dict[str, Any]:
    """Test MQTT broadcast message throughput."""
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
        """ACK loop for broadcast; drain non-ACK messages."""
        while not stop_ack_threads.is_set():
            inbox = _drain_non_ack_messages(receiver.mailbox)
            if not inbox:
                time.sleep(0.001)
                continue

            for msg in inbox:
                ack_content = {"ack_for": msg.content.get("broadcast_id")}
                receiver.send_message(
                    str(sender.id), MqttMessageType.ACK, ack_content
                )

    if latency_mode == "app_ack" and receivers:
        for receiver in receivers:
            thread = threading.Thread(
                target=receiver_ack_loop, args=(receiver,), daemon=True
            )
            thread.start()
            ack_threads.append(thread)

    for i in range(message_count):
        message_id = f"mqtt_broadcast_test_{i}"

        # Track throughput
        benchmark.throughput_tracker.record_message()

        # Start timing
        benchmark.latency_tracker.start_message_timing(message_id)

        # Send broadcast via MQTT with specified payload size
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

    return {
        "delivery_failures": delivery_failures,
        "ack_timeouts": ack_timeouts,
    }


def test_mqtt_concurrent_messaging(
    params: Dict[str, Any], benchmark: CommunicationBenchmark
) -> Dict[str, Any]:
    """Test concurrent MQTT messaging between multiple agents."""
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
        """ACK loop for concurrent messaging; keep ACKs available locally."""
        while not stop_ack_threads.is_set():
            inbox = _drain_non_ack_messages(receiver.mailbox)
            if not inbox:
                time.sleep(0.001)
                continue

            for msg in inbox:
                msg_id = (
                    str(msg.content.get("sender", ""))
                    + "_"
                    + str(msg.content.get("message_num", ""))
                )
                full_msg_id = f"mqtt_concurrent_{msg_id}"
                ack_content = {"ack_for": full_msg_id}
                receiver.send_message(
                    msg.sender_id, MqttMessageType.ACK, ack_content
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

    def mqtt_agent_messaging_task(agent, agent_index):
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
            message_id = f"mqtt_concurrent_{agent_index}_{i}"

            # Start timing
            benchmark.latency_tracker.start_message_timing(message_id)
            benchmark.throughput_tracker.record_message()

            # Send message via MQTT with specified payload size
            payload_data = generate_payload(payload_size)
            content = {
                "sender": agent_index,
                "message_num": i,
                "data": payload_data,
                "timestamp": time.time(),
            }
            success = agent.send_message(
                str(target.id), MqttMessageType.INFORM, content
            )

            if success:
                if latency_mode == "app_ack":
                    # Wait for ACK from receiver (configurable, default 0.5s)
                    ack_timeout = float(params.get("ack_timeout", params.get("ack_timeout_ms", 0.5)))
                    if _wait_for_ack(agent, message_id, timeout=ack_timeout):
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

    # Run concurrent MQTT messaging
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(active_agents)
    ) as executor:
        futures = []
        for i, agent in enumerate(active_agents):
            future = executor.submit(mqtt_agent_messaging_task, agent, i)
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


def test_mqtt_scalability_stress(
    params: Dict[str, Any], benchmark: CommunicationBenchmark
) -> Dict[str, Any]:
    """Test MQTT system under high message load for scalability."""
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
                msg_count = msg.content.get("count")
                if msg_count is None:
                    continue
                full_msg_id = f"mqtt_stress_{msg.sender_id}_{msg_count}"
                ack_content = {"ack_for": full_msg_id}
                receiver.send_message(
                    msg.sender_id, MqttMessageType.ACK, ack_content
                )

    if latency_mode == "app_ack":
        for agent in agents:
            thread = threading.Thread(
                target=receiver_ack_loop, args=(agent,), daemon=True
            )
            thread.start()
            ack_threads.append(thread)

    def mqtt_stress_messaging_task(agent):
        nonlocal delivery_failures, message_count, ack_timeouts
        end_time = time.time() + stress_duration
        targets = [a for a in agents if a != agent]
        local_count = 0

        while time.time() < end_time:
            target = random.choice(targets)
            message_id = f"mqtt_stress_{agent.id}_{local_count}"

            # Track timing and throughput
            benchmark.latency_tracker.start_message_timing(message_id)
            benchmark.throughput_tracker.record_message()

            # Send message via MQTT with specified payload size
            payload_data = generate_payload(payload_size)
            content = {
                "stress_test": True,
                "count": local_count,
                "data": payload_data,
            }
            success = agent.send_message(
                str(target.id),
                random.choice(
                    [MqttMessageType.INFORM, MqttMessageType.REQUEST]
                ),
                content,
            )

            if success:
                if latency_mode == "app_ack":
                    # Wait for ACK from receiver
                    ack_timeout = float(params.get("ack_timeout", params.get("ack_timeout_ms", 0.5)))
                    if _wait_for_ack(agent, message_id, timeout=ack_timeout):
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

    # Run MQTT stress test
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(agents)
    ) as executor:
        futures = [
            executor.submit(mqtt_stress_messaging_task, agent)
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


def create_mqtt_benchmark_scenarios(
    latency_mode: str = "end_to_end",
) -> CommunicationBenchmark:
    """Create and configure all MQTT benchmark scenarios."""
    benchmark = CommunicationBenchmark()
    benchmark.latency_mode = latency_mode  # Store for use in scenarios

    # Scenario 1: MQTT Point-to-Point Latency
    mqtt_latency_scenario = BenchmarkScenario(
        name="point_to_point_latency",
        description="Measures basic MQTT message latency between two agents",
    )
    mqtt_latency_scenario.set_setup(setup_mqtt_basic_scenario)
    mqtt_latency_scenario.set_test(test_mqtt_point_to_point_latency)
    mqtt_latency_scenario.set_teardown(teardown_mqtt_basic_scenario)
    benchmark.add_scenario(mqtt_latency_scenario)

    # Scenario 2: MQTT Broadcast Throughput
    mqtt_broadcast_scenario = BenchmarkScenario(
        name="broadcast_throughput",
        description="Tests MQTT broadcast message throughput and delivery",
    )
    mqtt_broadcast_scenario.set_setup(setup_mqtt_basic_scenario)
    mqtt_broadcast_scenario.set_test(test_mqtt_broadcast_throughput)
    mqtt_broadcast_scenario.set_teardown(teardown_mqtt_basic_scenario)
    benchmark.add_scenario(mqtt_broadcast_scenario)

    # Scenario 3: MQTT Concurrent Messaging
    mqtt_concurrent_scenario = BenchmarkScenario(
        name="concurrent_messaging",
        description="Tests MQTT concurrent messaging between multiple agents",
    )
    mqtt_concurrent_scenario.set_setup(setup_mqtt_basic_scenario)
    mqtt_concurrent_scenario.set_test(test_mqtt_concurrent_messaging)
    mqtt_concurrent_scenario.set_teardown(teardown_mqtt_basic_scenario)
    benchmark.add_scenario(mqtt_concurrent_scenario)

    # Scenario 4: MQTT Scalability Stress Test
    mqtt_stress_scenario = BenchmarkScenario(
        name="scalability_stress",
        description="High-load MQTT stress test for scalability analysis",
    )
    mqtt_stress_scenario.set_setup(setup_mqtt_basic_scenario)
    mqtt_stress_scenario.set_test(test_mqtt_scalability_stress)
    mqtt_stress_scenario.set_teardown(teardown_mqtt_basic_scenario)
    benchmark.add_scenario(mqtt_stress_scenario)

    return benchmark


def run_mqtt_topology_comparison(benchmark: CommunicationBenchmark):
    """Run MQTT performance comparison across different topology patterns."""
    topologies = [
        TopologyPattern.FULLY_CONNECTED,
        TopologyPattern.STAR,
        TopologyPattern.RING,
        TopologyPattern.CHAIN,
    ]

    results = {}

    for topology in topologies:
        print(f"\nTesting MQTT topology: {topology.value}")
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
    print("MQTT TOPOLOGY COMPARISON SUMMARY")
    print("=" * 60)
    for topology, metrics in comparison.items():
        print(f"\n{topology}:")
        print(f"  Avg Latency: {metrics['avg_latency_ms']:.2f}ms")
        print(f"  Throughput: {metrics['throughput_msg_per_sec']:.1f} msg/s")
        print(f"  Success Rate: {metrics['success_rate_percent']:.1f}%")

    return results


def run_mqtt_scalability_analysis(
    benchmark: CommunicationBenchmark, agent_counts: Optional[List[int]] = None
):
    """Run MQTT scalability analysis with increasing agent counts."""
    if agent_counts is None:
        agent_counts = [3, 5, 8, 12]

    results = {}

    for count in agent_counts:
        print(f"\nTesting MQTT with {count} agents")
        result = benchmark.run_scenario(
            "scalability_stress",
            agent_count=count,
            topology_pattern=TopologyPattern.FULLY_CONNECTED,
            stress_duration=3.0,
        )
        results[count] = result
        benchmark.print_summary(result)

    print("\n" + "=" * 60)
    print("MQTT SCALABILITY ANALYSIS SUMMARY")
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
    print("MQTT Communication Benchmark Suite")
    print("=" * 60)

    # Ensure MQTT broker is running
    start_mqtt_docker()
    if not is_mqtt_running():
        print("\nCannot run benchmarks without MQTT broker")
        exit(1)

    try:
        # Create MQTT benchmark suite
        benchmark = create_mqtt_benchmark_scenarios()

        # Run individual MQTT scenarios
        print("\n1. MQTT Point-to-Point Latency Test")
        result1 = benchmark.run_scenario(
            "point_to_point_latency", agent_count=2, message_count=50
        )
        benchmark.print_summary(result1)

        print("\n2. MQTT Broadcast Throughput Test")
        result2 = benchmark.run_scenario(
            "broadcast_throughput", agent_count=5, message_count=30
        )
        benchmark.print_summary(result2)

        print("\n3. MQTT Concurrent Messaging Test")
        result3 = benchmark.run_scenario(
            "concurrent_messaging", agent_count=4, messages_per_agent=10
        )
        benchmark.print_summary(result3)

        # Optional: Run extended analysis

        print("\nRunning MQTT topology comparison...")
        run_mqtt_topology_comparison(benchmark)

        print("\nRunning MQTT scalability analysis...")
        run_mqtt_scalability_analysis(benchmark, [5, 10, 15, 20])

        # Export results (after all benchmarks are complete)
        benchmark.export_results("mqtt_benchmark_results.json")
        print("\nResults exported to mqtt_benchmark_results.json")

    finally:
        # Clean up: stop broker if we started it
        stop_mqtt_docker()
