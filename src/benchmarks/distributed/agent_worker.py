"""
Agent Worker - Flask control server for distributed benchmarks.

Runs on each remote host. The orchestrator sends commands to set up
environments, run benchmark scenarios, and collect metrics.

Usage:
    python3 -m benchmarks.distributed.agent_worker --port 8080
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from threading import Thread
from typing import Any, Dict

from flask import Flask, jsonify, request

# Add project root to path so benchmark imports work
_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


app = Flask(__name__)

# Worker state
_state: Dict[str, Any] = {
    "status": "idle",  # idle, setting_up, ready, running, completed, error
    "config": None,
    "environment": None,
    "agents": [],
    "benchmark": None,
    "metrics": None,
    "error": None,
    "run_thread": None,
}


def _agent_instance_names(params: Dict[str, Any]) -> list[str]:
    """Resolve the worker-local agent instance names for a benchmark run."""
    agent_count = int(params.get("agent_count", 2))
    configured_names = params.get("agent_names")

    if configured_names is not None:
        names = [str(name) for name in configured_names]
        if len(names) != agent_count:
            raise ValueError("agent_names length does not match agent_count")
        return names

    offset = int(params.get("agent_offset", 0))
    return [f"agent_{offset + i}" for i in range(agent_count)]


def _create_rest_env(params: Dict[str, Any]):
    """Create REST benchmark environment.

    When ``remote_service_address`` is present in *params* the worker
    connects to a service running on a remote host instead of starting
    its own.  This ensures messages traverse the network, making the
    benchmark comparable with broker-based protocols (MQTT/Kafka).
    """
    from benchmarks.communication.base_communication import LatencyMode
    from benchmarks.communication.rest.rest_communicating_agent import (
        ExtendedRestCommunicatingAgent,
        RestCommunicationEnvironment,
    )
    from benchmarks.communication.communication_config import (
        CommunicationConfiguration,
        TopologyPattern,
    )
    from mas.core import AgentId

    latency_mode = params.get("latency_mode", "app_ack")
    transport_mode = params.get("transport_mode", "http1")
    service_host = params.get("service_host", "0.0.0.0")
    agent_names = _agent_instance_names(params)
    topology_pattern = params.get(
        "topology_pattern", TopologyPattern.FULLY_CONNECTED
    )

    mode_map = {
        "send_only": LatencyMode.SEND_ONLY,
        "app_ack": LatencyMode.APP_ACK,
        "end_to_end": LatencyMode.END_TO_END,
    }
    latency_mode_enum = mode_map.get(latency_mode, LatencyMode.APP_ACK)

    remote_addr = params.get("remote_service_address")

    if remote_addr:
        # --- Remote mode: connect to service on another host ---
        import requests as http_requests
        import threading as _threading
        from benchmarks.communication.rest.rest_communication import (
            RestMailbox,
        )

        r_host, r_port = remote_addr.rsplit(":", 1)
        service_url = f"http://{remote_addr}"

        # Create environment shell (no local service)
        env = RestCommunicationEnvironment(
            service_host=r_host,
            latency_mode=latency_mode_enum,
            transport_mode=transport_mode,
        )
        env.service_port = int(r_port)
        env.is_running = True  # service lives on the remote host

        agents = []
        for agent_name in agent_names:
            id_obj = AgentId(
                app="rest_benchmark", type="agent", instance=agent_name
            )
            agent = ExtendedRestCommunicatingAgent(
                id_obj,
                {"environment", "messages"},
                transport_mode=transport_mode,
            )
            agent.initialize_agent()

            agent_id_str = str(id_obj)
            # Register with the remote service via HTTP.
            # Retry in case the service is still starting.
            for _attempt in range(10):
                try:
                    http_requests.post(
                        f"{service_url}/register/{agent_id_str}", timeout=10
                    )
                    break
                except http_requests.ConnectionError:
                    if _attempt == 9:
                        raise
                    time.sleep(1)
            agent.comm_agent.configure_transport(service_url, transport_mode)
            agent.transport_mode = transport_mode

            # Create a local mailbox so benchmark scenarios can access
            # agent.mailbox directly (matches local-mode behaviour).
            agent.mailbox = RestMailbox(agent_id_str)

            env.agents[agent_id_str] = agent
            agents.append(agent)

        # Start a background poller per agent that pulls messages from
        # the remote service into the local mailbox.
        _stop_pollers = _threading.Event()

        def _mailbox_poller(ag):
            while not _stop_pollers.is_set():
                try:
                    msgs = ag.comm_agent.receive_messages(clear_mailbox=True)
                    for msg in msgs:
                        ag.mailbox.add_message(msg)
                except Exception:
                    pass
                _stop_pollers.wait(0.002)

        poller_threads = []
        for ag in agents:
            t = _threading.Thread(
                target=_mailbox_poller, args=(ag,), daemon=True
            )
            t.start()
            poller_threads.append(t)

        # Store the stop event so cleanup can halt the pollers.
        env._stop_pollers = _stop_pollers
        env._poller_threads = poller_threads

        config = CommunicationConfiguration()
        config.set_agents([str(a.id) for a in agents])
        if isinstance(topology_pattern, str):
            topology_pattern = TopologyPattern(topology_pattern)
        topology = config.set_topology(topology_pattern)
        for sender, receiver in topology.links:
            http_requests.post(
                f"{service_url}/topology/link",
                json={"sender_id": sender, "receiver_id": receiver},
                timeout=10,
            )

        return env, agents, config

    # --- Local mode (original behaviour) ---
    env = RestCommunicationEnvironment(
        service_host=service_host,
        latency_mode=latency_mode_enum,
        transport_mode=transport_mode,
    )
    env.start_service()

    agents = []
    for agent_name in agent_names:
        agent = env.create_agent(agent_name)
        agents.append(agent)

    config = CommunicationConfiguration()
    config.set_agents([str(a.id) for a in agents])
    if isinstance(topology_pattern, str):
        topology_pattern = TopologyPattern(topology_pattern)
    topology = config.set_topology(topology_pattern)
    for sender, receiver in topology.links:
        env.add_communication_link(sender, receiver)

    return env, agents, config


def _create_grpc_env(params: Dict[str, Any]):
    """Create gRPC benchmark environment.

    When ``remote_service_address`` is present in *params* the worker
    connects to a gRPC service running on a remote host instead of
    starting its own.
    """
    from benchmarks.communication.base_communication import LatencyMode
    from benchmarks.communication.grpc.grpc_communication_agent import (
        ExtendedGrpcCommunicatingAgent,
        GrpcCommunicationEnvironment,
    )
    from benchmarks.communication.grpc import communication_pb2
    from benchmarks.communication.communication_config import (
        CommunicationConfiguration,
        TopologyPattern,
    )
    from mas.core import AgentId

    latency_mode = params.get("latency_mode", "app_ack")
    communication_mode = params.get("grpc_mode", "unary")
    service_host = params.get("service_host", "0.0.0.0")
    agent_names = _agent_instance_names(params)
    topology_pattern = params.get(
        "topology_pattern", TopologyPattern.FULLY_CONNECTED
    )

    mode_map = {
        "send_only": LatencyMode.SEND_ONLY,
        "app_ack": LatencyMode.APP_ACK,
        "end_to_end": LatencyMode.END_TO_END,
    }
    latency_mode_enum = mode_map.get(latency_mode, LatencyMode.APP_ACK)

    remote_addr = params.get("remote_service_address")

    if remote_addr:
        # --- Remote mode: connect to gRPC service on another host ---
        import threading as _threading
        from benchmarks.communication.grpc.grpc_communication import (
            GrpcMailbox,
        )

        r_host, r_port = remote_addr.rsplit(":", 1)

        env = GrpcCommunicationEnvironment(
            service_host=r_host,
            service_port=int(r_port),
            latency_mode=latency_mode_enum,
            communication_mode=communication_mode,
        )
        env.is_running = True  # service lives on the remote host

        agents = []
        for agent_name in agent_names:
            id_obj = AgentId(
                app="grpc_benchmark", type="agent", instance=agent_name
            )
            agent = ExtendedGrpcCommunicatingAgent(
                id_obj,
                {"environment", "messages"},
                grpc_service_address=remote_addr,
                communication_mode=communication_mode,
            )
            agent.initialize_agent()

            # Register via gRPC RPC (works over the network).
            # Retry a few times in case the service is still starting.
            agent_id_str = str(id_obj)
            for _attempt in range(10):
                try:
                    agent.grpc_agent.stub.RegisterAgent(
                        communication_pb2.RegisterAgentRequest(
                            agent_id=agent_id_str
                        )
                    )
                    break
                except Exception:
                    if _attempt == 9:
                        raise
                    time.sleep(1)

            # Create a local mailbox so benchmark scenarios can access
            # agent.mailbox directly (matches local-mode behaviour).
            agent.mailbox = GrpcMailbox(agent_id_str)

            env.agents[agent_id_str] = agent
            agents.append(agent)

        # Start a background poller per agent that pulls messages from
        # the remote service into the local mailbox.
        _stop_pollers = _threading.Event()

        def _mailbox_poller(ag):
            while not _stop_pollers.is_set():
                try:
                    msgs = ag.grpc_agent.receive_messages(clear_mailbox=True)
                    for msg in msgs:
                        ag.mailbox.add_message(msg)
                except Exception:
                    pass
                _stop_pollers.wait(0.002)

        poller_threads = []
        for ag in agents:
            t = _threading.Thread(
                target=_mailbox_poller, args=(ag,), daemon=True
            )
            t.start()
            poller_threads.append(t)

        env._stop_pollers = _stop_pollers
        env._poller_threads = poller_threads

        config = CommunicationConfiguration()
        config.set_agents([str(a.id) for a in agents])
        if isinstance(topology_pattern, str):
            topology_pattern = TopologyPattern(topology_pattern)
        topology = config.set_topology(topology_pattern)

        # Add topology links via gRPC RPC
        stub = agents[0].grpc_agent.stub
        for sender, receiver in topology.links:
            stub.AddCommunicationLink(
                communication_pb2.AddLinkRequest(
                    sender_id=sender, receiver_id=receiver
                )
            )

        return env, agents, config

    # --- Local mode (original behaviour) ---
    env = GrpcCommunicationEnvironment(
        service_host=service_host,
        latency_mode=latency_mode_enum,
        communication_mode=communication_mode,
    )
    env.start_service()

    agents = []
    for agent_name in agent_names:
        agent = env.create_agent(agent_name)
        agents.append(agent)

    config = CommunicationConfiguration()
    config.set_agents([str(a.id) for a in agents])
    if isinstance(topology_pattern, str):
        topology_pattern = TopologyPattern(topology_pattern)
    topology = config.set_topology(topology_pattern)
    for sender, receiver in topology.links:
        env.add_communication_link(sender, receiver)

    return env, agents, config


def _create_mqtt_env(params: Dict[str, Any]):
    """Create MQTT benchmark environment."""
    from benchmarks.communication.base_communication import LatencyMode
    from benchmarks.communication.mqtt.mqtt_communication_agent import (
        MqttCommunicationEnvironment,
    )
    from benchmarks.communication.communication_config import (
        CommunicationConfiguration,
        TopologyPattern,
    )

    latency_mode = params.get("latency_mode", "app_ack")
    broker_host = params.get("broker_host", "localhost")
    broker_port = int(params.get("broker_port", 1883))
    mqtt_qos = int(params.get("mqtt_qos", 1))
    agent_count = params.get("agent_count", 2)
    topology_pattern = params.get(
        "topology_pattern", TopologyPattern.FULLY_CONNECTED
    )

    mode_map = {
        "send_only": LatencyMode.SEND_ONLY,
        "app_ack": LatencyMode.APP_ACK,
        "end_to_end": LatencyMode.END_TO_END,
    }
    latency_mode_enum = mode_map.get(latency_mode, LatencyMode.APP_ACK)

    mqtt_config = {
        "broker_host": broker_host,
        "broker_port": broker_port,
        "keepalive": 60,
        "qos": mqtt_qos,
    }
    env = MqttCommunicationEnvironment(
        broker_host=broker_host,
        broker_port=broker_port,
        mqtt_config=mqtt_config,
        latency_mode=latency_mode_enum,
    )
    env.start_service()

    agents = []
    for i in range(agent_count):
        agent = env.create_agent(f"agent_{i}")
        agents.append(agent)

    config = CommunicationConfiguration()
    config.set_agents([str(getattr(a, "agent_id", a.id)) for a in agents])
    if isinstance(topology_pattern, str):
        topology_pattern = TopologyPattern(topology_pattern)
    topology = config.set_topology(topology_pattern)
    for sender, receiver in topology.links:
        env.add_communication_link(sender, receiver)

    return env, agents, config


def _create_kafka_env(params: Dict[str, Any]):
    """Create Kafka benchmark environment."""
    from benchmarks.communication.base_communication import LatencyMode
    from benchmarks.communication.kafka.kafka_communication_agent import (
        KafkaCommunicationEnvironment,
    )
    from benchmarks.communication.communication_config import (
        CommunicationConfiguration,
        TopologyPattern,
    )

    latency_mode = params.get("latency_mode", "app_ack")
    broker_host = params.get("broker_host", "localhost")
    broker_port = int(params.get("broker_port", 9092))
    kafka_acks = params.get("kafka_acks", 1)
    agent_count = params.get("agent_count", 2)
    topology_pattern = params.get(
        "topology_pattern", TopologyPattern.FULLY_CONNECTED
    )

    mode_map = {
        "send_only": LatencyMode.SEND_ONLY,
        "app_ack": LatencyMode.APP_ACK,
        "end_to_end": LatencyMode.END_TO_END,
    }
    latency_mode_enum = mode_map.get(latency_mode, LatencyMode.APP_ACK)

    kafka_config = {
        "bootstrap_servers": [f"{broker_host}:{broker_port}"],
        "client_id": "distributed_benchmark_worker",
        "acks": kafka_acks,
    }
    compression_type = params.get("compression_type")
    if compression_type:
        kafka_config["compression_type"] = compression_type

    env = KafkaCommunicationEnvironment(
        kafka_config, latency_mode=latency_mode_enum
    )
    env.setup()

    agents = []
    for i in range(agent_count):
        agent = env.create_agent(f"agent_{i}")
        kafka_service = getattr(env, "kafka_service", None)
        if kafka_service:
            mailbox = kafka_service.mailboxes.get(agent.agent_id)
            if mailbox is None:
                kafka_service.register_agent(agent.agent_id)
                mailbox = kafka_service.mailboxes.get(agent.agent_id)
            if mailbox:
                agent.mailbox = mailbox
        agents.append(agent)

    config = CommunicationConfiguration()
    config.set_agents([str(getattr(a, "agent_id", a.id)) for a in agents])
    if isinstance(topology_pattern, str):
        topology_pattern = TopologyPattern(topology_pattern)
    topology = config.set_topology(topology_pattern)
    for sender, receiver in topology.links:
        env.add_communication_link(sender, receiver)

    return env, agents, config


_ENV_FACTORIES = {
    "rest": _create_rest_env,
    "grpc": _create_grpc_env,
    "mqtt": _create_mqtt_env,
    "kafka": _create_kafka_env,
}


def _get_scenario_functions(protocol: str):
    """Import and return (setup, test_funcs, teardown) for a protocol."""
    if protocol == "rest":
        from benchmarks.local.communication_benchmarks import (
            rest_benchmark_scenarios as mod,
        )

        return {
            "point_to_point_latency": mod.test_point_to_point_latency,
            "broadcast_throughput": mod.test_broadcast_throughput,
            "concurrent_messaging": mod.test_concurrent_messaging,
            "scalability_stress": mod.test_scalability_stress,
        }
    elif protocol == "grpc":
        from benchmarks.local.communication_benchmarks import (
            grpc_benchmark_scenarios as mod,
        )

        return {
            "point_to_point_latency": mod.test_grpc_point_to_point_latency,
            "broadcast_throughput": mod.test_grpc_broadcast_throughput,
            "concurrent_messaging": mod.test_grpc_concurrent_messaging,
            "scalability_stress": mod.test_grpc_scalability_stress,
        }
    elif protocol == "mqtt":
        from benchmarks.local.communication_benchmarks import (
            mqtt_benchmark_scenarios as mod,
        )

        return {
            "point_to_point_latency": mod.test_mqtt_point_to_point_latency,
            "broadcast_throughput": mod.test_mqtt_broadcast_throughput,
            "concurrent_messaging": mod.test_mqtt_concurrent_messaging,
            "scalability_stress": mod.test_mqtt_scalability_stress,
        }
    elif protocol == "kafka":
        from benchmarks.local.communication_benchmarks import (
            kafka_benchmark_scenarios as mod,
        )

        return {
            "point_to_point_latency": mod.test_kafka_point_to_point_latency,
            "broadcast_throughput": mod.test_kafka_broadcast_throughput,
            "concurrent_messaging": mod.test_kafka_concurrent_messaging,
            "scalability_stress": mod.test_kafka_scalability_stress,
        }
    else:
        raise ValueError(f"Unknown protocol: {protocol}")


def _run_benchmark(config: Dict[str, Any]):
    """Run benchmark in background thread."""
    from benchmarks.local.communication_benchmarks.communication_benchmark import (
        CommunicationBenchmark,
    )

    try:
        _state["status"] = "running"
        protocol = config["protocol"]
        scenario_name = config["scenario"]
        params = config.get("params", {})

        # Create environment using our factory (with remote addresses)
        factory = _ENV_FACTORIES.get(protocol)
        if factory is None:
            raise ValueError(f"Unsupported protocol: {protocol}")

        env, agents, comm_config = factory(params)
        _state["environment"] = env
        _state["agents"] = agents

        # Create benchmark tracker
        benchmark = CommunicationBenchmark()
        _state["benchmark"] = benchmark

        # Get the test function
        test_funcs = _get_scenario_functions(protocol)
        test_func = test_funcs.get(scenario_name)
        if test_func is None:
            raise ValueError(
                f"Unknown scenario '{scenario_name}' for {protocol}"
            )

        # Build params dict for the test function
        test_params = {
            "agents": agents,
            "config": comm_config,
            "environment": env,
            "agent_count": len(agents),
            **params,
        }

        # Start resource monitoring
        benchmark.resource_monitor.start_monitoring()
        start_time = time.time()

        # Run the test
        test_result = test_func(test_params, benchmark)

        # Stop monitoring
        duration = time.time() - start_time
        benchmark.resource_monitor.stop_monitoring()

        # Collect metrics
        latency_stats = benchmark.latency_tracker.get_latency_stats()
        resource_stats = benchmark.resource_monitor.get_resource_stats()
        throughput_samples = (
            benchmark.throughput_tracker.get_throughput_history(20)
        )

        metrics = {
            "latency": latency_stats,
            "throughput": {
                "total_messages": benchmark.throughput_tracker.total_messages,
                "samples": throughput_samples,
                "avg": (
                    benchmark.throughput_tracker.total_messages / duration
                    if duration > 0
                    else 0
                ),
            },
            "resources": resource_stats,
            "test_result": test_result,
            "duration": duration,
            "latency_samples": (
                benchmark.latency_tracker.completed_latencies[:2000]
            ),
        }

        _state["metrics"] = metrics
        _state["status"] = "completed"

    except Exception as e:
        _state["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        _state["status"] = "error"
    finally:
        # Teardown environment
        _teardown_env()


def _teardown_env():
    """Clean up the current environment."""
    env = _state.get("environment")
    if env is None:
        return

    # Stop remote-mode mailbox pollers if running.
    if hasattr(env, "_stop_pollers"):
        env._stop_pollers.set()
    if hasattr(env, "_poller_threads"):
        for thread in env._poller_threads:
            try:
                thread.join(timeout=0.5)
            except Exception:
                pass

    try:
        if hasattr(env, "stop_service"):
            env.stop_service()
        elif hasattr(env, "teardown"):
            env.teardown()
    except Exception:
        pass

    for agent in _state.get("agents", []):
        try:
            if hasattr(agent, "close"):
                agent.close()
            if hasattr(agent, "comm_agent"):
                agent.comm_agent.close()
            if hasattr(agent, "grpc_agent"):
                agent.grpc_agent.close()
        except Exception:
            pass

    _state["environment"] = None
    _state["agents"] = []


# --- Flask endpoints ---


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "worker_status": _state["status"]})


@app.route("/setup", methods=["POST"])
def setup():
    """Receive benchmark configuration.

    Expects JSON body with:
    {
        "protocol": "rest|grpc|mqtt|kafka",
        "scenario": "point_to_point_latency|...",
        "params": { ... scenario parameters ... }
    }
    """
    if _state["status"] not in ("idle", "completed", "error"):
        return (
            jsonify({"error": "Worker busy", "status": _state["status"]}),
            409,
        )

    config = request.get_json()
    if not config:
        return jsonify({"error": "No JSON body"}), 400

    _state["config"] = config
    _state["metrics"] = None
    _state["error"] = None
    _state["status"] = "ready"
    return jsonify({"status": "ready"})


@app.route("/start", methods=["POST"])
def start():
    """Start the benchmark run."""
    if _state["status"] != "ready":
        return (
            jsonify(
                {
                    "error": "Not ready. Call /setup first.",
                    "status": _state["status"],
                }
            ),
            409,
        )

    config = _state["config"]
    thread = Thread(target=_run_benchmark, args=(config,), daemon=True)
    _state["run_thread"] = thread
    thread.start()
    return jsonify({"status": "running"})


@app.route("/status", methods=["GET"])
def status():
    return jsonify(
        {
            "status": _state["status"],
            "error": _state.get("error"),
        }
    )


@app.route("/metrics", methods=["GET"])
def metrics():
    """Return collected metrics."""
    if _state["status"] == "completed" and _state["metrics"]:
        return jsonify(
            {
                "status": "completed",
                "metrics": _state["metrics"],
            }
        )
    elif _state["status"] == "error":
        return (
            jsonify(
                {
                    "status": "error",
                    "error": _state["error"],
                }
            ),
            500,
        )
    else:
        return (
            jsonify(
                {
                    "status": _state["status"],
                    "message": "Benchmark not yet completed",
                }
            ),
            202,
        )


@app.route("/teardown", methods=["POST"])
def teardown():
    """Clean up and reset worker."""
    _teardown_env()
    _state["status"] = "idle"
    _state["config"] = None
    _state["metrics"] = None
    _state["error"] = None
    return jsonify({"status": "idle"})


def main():
    parser = argparse.ArgumentParser(
        description="Distributed benchmark agent worker"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the worker control server (default: 8080)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    args = parser.parse_args()

    print(f"Starting agent worker on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
