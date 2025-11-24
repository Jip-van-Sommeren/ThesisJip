"""
Kafka Benchmark Test Scenarios
Provides the same benchmark scenarios as REST, gRPC, and MQTT but using Kafka
communication.

Enables direct performance comparison between Kafka and other protocol
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
from benchmarks.communication_benchmark import generate_payload

from communication.kafka.kafka_communication_agent import (
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

_kafka_broker_container: Optional[str] = None


def _is_ack_message(message) -> bool:
    from communication.base_communication import MessageType

    msg_type = getattr(message, "message_type", None)
    if hasattr(msg_type, "value"):
        msg_type = msg_type.value
    return msg_type == MessageType.ACK.value


def is_kafka_running(host="localhost", port=9092, timeout=2) -> bool:
    """Check if Kafka broker is accessible."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def stop_kafka_docker():
    """Stop the kafka broker Docker container."""
    global _kafka_broker_container

    if _kafka_broker_container:
        print("\nStopping kafka broker...")
        try:
            subprocess.run(
                ["docker", "stop", _kafka_broker_container],
                capture_output=True,
                check=True,
            )
            print("✓ kafka broker stopped")
        except subprocess.CalledProcessError:
            pass  # Container might already be stopped
        finally:
            _kafka_broker_container = None


def start_kafka_docker(wait_time=15) -> bool:
    """Start Kafka using Docker if not already running."""
    global _kafka_broker_container
    if is_kafka_running():
        print("Kafka is already running on localhost:9092")
        return True

    print("Starting Kafka broker using Docker...")
    try:
        # Check if container exists but is stopped
        result = subprocess.run(
            [
                "docker",
                "ps",
                "-a",
                "--filter",
                "name=kafka-benchmark",
                "--format",
                "{{.Names}}",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        container_exists = "kafka-benchmark" in result.stdout
        should_create_new = True

        if container_exists:
            # Check if container is running or stopped
            status_result = subprocess.run(
                [
                    "docker",
                    "ps",
                    "-a",
                    "--filter",
                    "name=kafka-benchmark",
                    "--format",
                    "{{.Status}}",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            if "Exited" in status_result.stdout:
                print("[DEBUG] Removing old Kafka container with errors...")
                subprocess.run(
                    ["docker", "rm", "kafka-benchmark"],
                    capture_output=True,
                    check=True,
                )
                should_create_new = True
            elif "Up" in status_result.stdout:
                # Container is already running
                _kafka_broker_container = "kafka-benchmark"
                print("✓ Kafka broker is already running")
                return True
            else:
                # Container exists but not running, try to start it
                print("[DEBUG] Starting existing Kafka container...")
                start_result = subprocess.run(
                    ["docker", "start", "kafka-benchmark"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                print(f"[DEBUG] Start output: {start_result.stdout}")
                _kafka_broker_container = "kafka-benchmark"
                should_create_new = False

                print(f"Waiting {wait_time}s for Kafka to be fully ready...")
                time.sleep(wait_time)

                # Verify and return early
                max_retries = 5
                for i in range(max_retries):
                    if is_kafka_running():
                        print("✓ Kafka broker is ready")
                        return True
                    if i < max_retries - 1:
                        print(f"  Still waiting... ({i+1}/{max_retries})")
                        time.sleep(3)

                print("✗ Kafka broker failed to start properly")
                return False

        # Create new container if needed
        if should_create_new:
            # Create and start new container with proper configuration
            run_result = subprocess.run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    "kafka-benchmark",
                    "-p",
                    "9092:9092",
                    "-e",
                    "KAFKA_NODE_ID=1",
                    "-e",
                    "KAFKA_PROCESS_ROLES=broker,controller",
                    "-e",
                    "KAFKA_LISTENERS=PLAINTEXT://0.0.0.0:9092,CONTROLLER://0.0.0.0:9093",
                    "-e",
                    "KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092",
                    "-e",
                    "KAFKA_CONTROLLER_LISTENER_NAMES=CONTROLLER",
                    "-e",
                    "KAFKA_LISTENER_SECURITY_PROTOCOL_MAP=CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT",
                    "-e",
                    "KAFKA_CONTROLLER_QUORUM_VOTERS=1@localhost:9093",
                    "-e",
                    "KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1",
                    "-e",
                    "KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR=1",
                    "-e",
                    "KAFKA_TRANSACTION_STATE_LOG_MIN_ISR=1",
                    "-e",
                    "KAFKA_GROUP_INITIAL_REBALANCE_DELAY_MS=0",
                    "-e",
                    "KAFKA_AUTO_CREATE_TOPICS_ENABLE=true",
                    "-e",
                    "KAFKA_NUM_PARTITIONS=4",
                    "apache/kafka:latest",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            print(f"[DEBUG] Container ID: {run_result.stdout.strip()}")
            _kafka_broker_container = "kafka-benchmark"

        print(f"Waiting {wait_time}s for Kafka to be fully ready...")
        time.sleep(wait_time)

        # Verify Kafka is responding
        max_retries = 5
        for i in range(max_retries):
            if is_kafka_running():
                print("✓ Kafka broker is ready")
                return True
            if i < max_retries - 1:
                print(f"  Still waiting... ({i+1}/{max_retries})")
                time.sleep(3)

        print("✗ Kafka broker failed to start properly")
        return False

    except subprocess.CalledProcessError as e:
        print(f"Failed to start Kafka: {e}")
        print("Please start Kafka manually or install Docker")
        return False
    except FileNotFoundError:
        print(
            "Docker not found. Please install Docker or start Kafka manually"
        )
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return False


def ensure_kafka_running() -> bool:
    """Ensure Kafka is running, start it if needed."""
    if is_kafka_running():
        return True
    print("\nKafka broker not detected on localhost:9092")
    return start_kafka_docker()


def setup_kafka_basic_scenario(params: Dict[str, Any]) -> Dict[str, Any]:
    """Setup for basic Kafka latency/throughput scenarios.

    Default latency_mode is 'end_to_end' for fair comparison across all protocols.
    Kafka producer uses acks=1 (leader acknowledgment) for comparable durability to MQTT QoS 1.
    """
    agent_count = params.get("agent_count", 5)
    topology_pattern = params.get(
        "topology_pattern", TopologyPattern.FULLY_CONNECTED
    )
    latency_mode = params.get("latency_mode", "end_to_end")

    kafka_acks_param = params.get("kafka_acks", 1)
    if isinstance(kafka_acks_param, str):
        lowered = kafka_acks_param.strip().lower()
        if lowered in {"all", "-1"}:
            kafka_acks = "all"
        elif lowered in {"0", "1"}:
            kafka_acks = int(lowered)
        else:
            kafka_acks = 1
    elif isinstance(kafka_acks_param, int):
        if kafka_acks_param == -1:
            kafka_acks = "all"
        elif kafka_acks_param in {0, 1}:
            kafka_acks = kafka_acks_param
        else:
            kafka_acks = 1
    else:
        kafka_acks = 1

    # Ensure Kafka broker is running before creating environment
    if not ensure_kafka_running():
        raise RuntimeError("Failed to start Kafka broker")

    # Import LatencyMode enum
    from communication.base_communication import LatencyMode

    # Convert string to enum
    if latency_mode == "send_only":
        latency_mode_enum = LatencyMode.SEND_ONLY
    elif latency_mode == "app_ack":
        latency_mode_enum = LatencyMode.APP_ACK
    else:
        latency_mode_enum = LatencyMode.END_TO_END

    # Create Kafka communication environment
    # Configuration optimized for fair benchmark comparison:
    # - Producer uses acks=1 (configured in KafkaCommunicationService)
    # - This matches MQTT QoS 1 durability guarantees
    kafka_config = {
        "bootstrap_servers": ["localhost:9092"],
        "client_id": "kafka_benchmark_service",
        "acks": kafka_acks,
    }
    env = KafkaCommunicationEnvironment(
        kafka_config, latency_mode=latency_mode_enum
    )
    env.setup()

    # Create agents
    agents = []
    for i in range(agent_count):
        agent = env.create_agent(f"agent_{i}")
        # Expose underlying Kafka mailbox for low-latency ACK checks
        kafka_service = getattr(env, "kafka_service", None)
        if kafka_service:
            mailbox = kafka_service.mailboxes.get(agent.agent_id)
            if mailbox is None:
                # Register again as a safeguard in case the mailbox wasn't ready yet
                kafka_service.register_agent(agent.agent_id)
                mailbox = kafka_service.mailboxes.get(agent.agent_id)
            if mailbox is None:
                raise RuntimeError(
                    f"Kafka mailbox not available for agent {agent.agent_id}"
                )
            agent.mailbox = mailbox
        agents.append(agent)

    # Setup topology
    config = CommunicationConfiguration()
    config.set_agents([agent.agent_id for agent in agents])
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
        env.teardown()

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
        # Check mailbox for ACK messages
        messages = agent.mailbox.peek_messages()
        for msg in messages:
            if (
                _is_ack_message(msg)
                and msg.content.get("ack_for") == message_id
            ):
                agent.mailbox.get_messages(clear=True)
                return True
        time.sleep(0.001)  # Small polling interval
    return False


def test_kafka_point_to_point_latency(
    params: Dict[str, Any], benchmark: CommunicationBenchmark
) -> Dict[str, Any]:
    """Test basic point-to-point message latency via Kafka."""
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
                    sender.agent_id, KafkaMessageType.ACK, ack_content
                )

            if messages:
                receiver.mailbox.get_messages(clear=True)
            time.sleep(0.001)

    if latency_mode == "app_ack":
        ack_thread = threading.Thread(target=receiver_ack_loop, daemon=True)
        ack_thread.start()

    for i in range(message_count):
        message_id = f"kafka_latency_test_{i}"

        # Start timing
        benchmark.latency_tracker.start_message_timing(message_id)
        benchmark.throughput_tracker.record_message()

        # Send message via Kafka with specified payload size
        payload_data = generate_payload(payload_size)
        content = {"test_id": message_id, "data": payload_data}
        success = sender.send_message(
            receiver.agent_id, KafkaMessageType.INFORM, content
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


def test_kafka_broadcast_throughput(
    params: Dict[str, Any], benchmark: CommunicationBenchmark
) -> Dict[str, Any]:
    """Test Kafka broadcast message throughput."""
    agents = params["agents"]
    message_count = params.get("message_count", 50)
    payload_size = params.get("payload_size_bytes", 100)
    latency_mode = params.get("latency_mode", "end_to_end")

    if len(agents) < 1:
        return {"delivery_failures": message_count}

    sender = agents[0]
    receivers = agents[1:]
    delivery_failures = 0
    ack_timeouts = 0

    # In app_ack mode, start receiver threads that send ACKs
    stop_ack_threads = threading.Event()
    ack_threads: List[threading.Thread] = []

    def receiver_ack_loop(receiver):
        """ACK loop for broadcast; re-queue local ACKs when clearing inbox."""
        while not stop_ack_threads.is_set():
            inbox = receiver.mailbox.get_messages(clear=True)
            if not inbox:
                time.sleep(0.001)
                continue

            for msg in inbox:
                if _is_ack_message(msg):
                    receiver.mailbox.add_message(msg)
                    continue

                ack_id = msg.content.get("broadcast_id")
                if ack_id is None:
                    continue
                ack_content = {"ack_for": ack_id}
                receiver.send_message(
                    sender.agent_id, KafkaMessageType.ACK, ack_content
                )

    if latency_mode == "app_ack" and receivers:
        for receiver in receivers:
            thread = threading.Thread(
                target=receiver_ack_loop, args=(receiver,), daemon=True
            )
            thread.start()
            ack_threads.append(thread)

    for i in range(message_count):
        message_id = f"kafka_broadcast_test_{i}"

        # Track throughput
        benchmark.throughput_tracker.record_message()

        # Track latency
        benchmark.latency_tracker.start_message_timing(message_id)

        # Send broadcast via Kafka with specified payload size
        payload_data = generate_payload(payload_size)
        content = {
            "broadcast_id": message_id,
            "announcement": payload_data,
        }
        result = sender.broadcast_message(KafkaMessageType.INFORM, content)

        if result.get("status") == "completed":
            if latency_mode == "app_ack" and receivers:
                expected_acks = len(receivers)
                ack_senders = set()
                # Allow override via parameters; default 0.5s for local app_ack
                ack_timeout = float(params.get("ack_timeout", params.get("ack_timeout_ms", 0.5)))
                timeout = ack_timeout
                start_time = time.time()

                while (
                    len(ack_senders) < expected_acks
                    and time.time() - start_time < timeout
                ):
                    messages = sender.mailbox.peek_messages()
                    for msg in messages:
                        if (
                            _is_ack_message(msg)
                            and msg.content.get("ack_for") == message_id
                        ):
                            ack_senders.add(msg.sender_id)
                    if len(ack_senders) < expected_acks:
                        time.sleep(0.001)

                sender.mailbox.get_messages(clear=True)

                if len(ack_senders) >= expected_acks:
                    benchmark.latency_tracker.end_message_timing(message_id)
                else:
                    ack_timeouts += 1
                    delivery_failures += 1
            else:
                # End timing (send_only or end_to_end semantics)
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


def test_kafka_concurrent_messaging(
    params: Dict[str, Any], benchmark: CommunicationBenchmark
) -> Dict[str, Any]:
    """Test concurrent Kafka messaging between multiple agents."""
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
    ack_threads: List[threading.Thread] = []

    def receiver_ack_loop(receiver):
        """ACK loop for concurrent messaging; keep ACKs available locally."""
        while not stop_ack_threads.is_set():
            inbox = receiver.mailbox.get_messages(clear=True)
            if not inbox:
                time.sleep(0.001)
                continue

            for msg in inbox:
                if _is_ack_message(msg):
                    receiver.mailbox.add_message(msg)
                    continue

                sender_index = msg.content.get("sender")
                message_num = msg.content.get("message_num")
                if sender_index is None or message_num is None:
                    continue
                full_msg_id = f"kafka_concurrent_{sender_index}_{message_num}"
                ack_content = {"ack_for": full_msg_id}
                receiver.send_message(
                    msg.sender_id, KafkaMessageType.ACK, ack_content
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

    def kafka_agent_messaging_task(agent, agent_index):
        nonlocal delivery_failures, timeout_failures, ack_timeouts

        # Get valid targets based on topology from configuration
        config = params.get("config")
        valid_targets = []

        if config and hasattr(config, "topology"):
            agent_id_str = agent.agent_id
            for other_agent in agents:
                if other_agent != agent:
                    other_id_str = other_agent.agent_id
                    if (agent_id_str, other_id_str) in config.topology.links:
                        valid_targets.append(other_agent)

        if not valid_targets:
            valid_targets = [a for a in agents if a != agent]

        if not valid_targets:
            return

        for i in range(messages_per_agent):
            target = random.choice(valid_targets)
            message_id = f"kafka_concurrent_{agent_index}_{i}"

            # Start timing
            benchmark.latency_tracker.start_message_timing(message_id)
            benchmark.throughput_tracker.record_message()

            # Send message via Kafka with specified payload size
            payload_data = generate_payload(payload_size)
            content = {
                "sender": agent_index,
                "message_num": i,
                "data": payload_data,
                "timestamp": time.time(),
            }
            success = agent.send_message(
                target.agent_id, KafkaMessageType.INFORM, content
            )

            if success:
                if latency_mode == "app_ack":
                    ack_timeout = float(params.get("ack_timeout", params.get("ack_timeout_ms", 0.5)))
                    if _wait_for_ack(agent, message_id, timeout=ack_timeout):
                        benchmark.latency_tracker.end_message_timing(
                            message_id
                        )
                    else:
                        ack_timeouts += 1
                        delivery_failures += 1
                else:
                    # End timing
                    benchmark.latency_tracker.end_message_timing(message_id)
            else:
                delivery_failures += 1

            # Random delay to simulate realistic messaging patterns
            time.sleep(random.uniform(0.005, 0.02))

    # Run concurrent Kafka messaging
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(active_agents)
    ) as executor:
        futures = []
        for i, agent in enumerate(active_agents):
            future = executor.submit(kafka_agent_messaging_task, agent, i)
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


def test_kafka_scalability_stress(
    params: Dict[str, Any], benchmark: CommunicationBenchmark
) -> Dict[str, Any]:
    """Test Kafka system under high message load for scalability."""
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
    ack_threads: List[threading.Thread] = []

    def receiver_ack_loop(receiver):
        """ACK loop for stress; re-queue ACKs to avoid sender starvation."""
        while not stop_ack_threads.is_set():
            inbox = receiver.mailbox.get_messages(clear=True)
            if not inbox:
                time.sleep(0.001)
                continue

            for msg in inbox:
                if _is_ack_message(msg):
                    receiver.mailbox.add_message(msg)
                    continue

                msg_count = msg.content.get("count")
                if msg_count is None:
                    continue
                full_msg_id = f"kafka_stress_{msg.sender_id}_{msg_count}"
                ack_content = {"ack_for": full_msg_id}
                receiver.send_message(
                    msg.sender_id, KafkaMessageType.ACK, ack_content
                )

    if latency_mode == "app_ack":
        for agent in agents:
            thread = threading.Thread(
                target=receiver_ack_loop, args=(agent,), daemon=True
            )
            thread.start()
            ack_threads.append(thread)

    def kafka_stress_messaging_task(agent):
        nonlocal delivery_failures, message_count, ack_timeouts
        end_time = time.time() + stress_duration
        targets = [a for a in agents if a != agent]
        local_count = 0

        while time.time() < end_time:
            target = random.choice(targets)
            message_id = f"kafka_stress_{agent.agent_id}_{local_count}"

            # Track timing and throughput
            benchmark.latency_tracker.start_message_timing(message_id)
            benchmark.throughput_tracker.record_message()

            # Send message via Kafka with specified payload size
            payload_data = generate_payload(payload_size)
            content = {
                "stress_test": True,
                "count": local_count,
                "data": payload_data,
            }
            success = agent.send_message(
                target.agent_id,
                random.choice(
                    [KafkaMessageType.INFORM, KafkaMessageType.REQUEST]
                ),
                content,
            )

            if success:
                if latency_mode == "app_ack":
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

            message_count += 1
            local_count += 1

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


def create_kafka_benchmark_scenarios(
    latency_mode: str = "end_to_end",
) -> CommunicationBenchmark:
    """Create and configure all Kafka benchmark scenarios."""
    benchmark = CommunicationBenchmark()
    benchmark.latency_mode = latency_mode  # Store for use in scenarios

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


def run_kafka_topology_comparison(benchmark: CommunicationBenchmark):
    """Run Kafka performance comparison across different topology patterns."""
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
    print("KAFKA TOPOLOGY COMPARISON SUMMARY")
    print("=" * 60)
    for topology, metrics in comparison.items():
        print(f"\n{topology}:")
        print(f"  Avg Latency: {metrics['avg_latency_ms']:.2f}ms")
        print(f"  Throughput: {metrics['throughput_msg_per_sec']:.1f} msg/s")
        print(f"  Success Rate: {metrics['success_rate_percent']:.1f}%")

    return results


def run_kafka_scalability_analysis(
    benchmark: CommunicationBenchmark, agent_counts: Optional[List[int]] = None
):
    """Run Kafka scalability analysis with increasing agent counts."""
    if agent_counts is None:
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

    # Ensure Kafka is running
    start_kafka_docker()
    if not is_kafka_running():
        print("\n✗ Cannot run benchmarks without Kafka broker")
        exit(1)

    # Create Kafka benchmark suite
    try:
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

        print("\nRunning Kafka topology comparison...")
        run_kafka_topology_comparison(benchmark)

        print("\nRunning Kafka scalability analysis...")
        run_kafka_scalability_analysis(benchmark, [5, 10, 15, 20])

        # Export results (after all benchmarks are complete)
        benchmark.export_results("kafka_benchmark_results.json")
        print("\nResults exported to kafka_benchmark_results.json")
    finally:
        stop_kafka_docker()
