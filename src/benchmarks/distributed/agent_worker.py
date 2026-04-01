"""
Agent Worker - Flask control server for distributed benchmarks.

Runs on each remote host. The orchestrator sends commands to set up
environments, run benchmark scenarios, and collect metrics.

Usage:
    python3 -m benchmarks.distributed.agent_worker --port 8080
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from threading import Thread
from typing import Any, Dict, Optional

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


def _create_rest_env(params: Dict[str, Any]):
    """Create REST benchmark environment."""
    from benchmarks.communication.base_communication import LatencyMode
    from benchmarks.communication.rest.rest_communicating_agent import (
        RestCommunicationEnvironment,
    )
    from benchmarks.communication.communication_config import (
        CommunicationConfiguration,
        TopologyPattern,
    )
    from mas.core import AgentId
    from benchmarks.communication.rest.rest_communicating_agent import (
        ExtendedRestCommunicatingAgent,
    )

    latency_mode = params.get("latency_mode", "app_ack")
    transport_mode = params.get("transport_mode", "http1")
    service_host = params.get("service_host", "0.0.0.0")
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

    env = RestCommunicationEnvironment(
        service_host=service_host,
        latency_mode=latency_mode_enum,
        transport_mode=transport_mode,
    )
    env.start_service()

    agents = []
    for i in range(agent_count):
        agent_id = AgentId("benchmark", "distributed", f"agent_{i}")
        agent = ExtendedRestCommunicatingAgent(
            agent_id,
            {"environment", "messages"},
            transport_mode=transport_mode,
        )
        agent.initialize_agent()
        agents.append(agent)
        env.register_agent(agent)
        if getattr(agent, "mailbox", None) is None and env.comm_service:
            agent.mailbox = env.comm_service.mailboxes.get(str(agent.id))

    config = CommunicationConfiguration()
    config.set_agents([str(a.id) for a in agents])
    if isinstance(topology_pattern, str):
        topology_pattern = TopologyPattern(topology_pattern)
    topology = config.set_topology(topology_pattern)
    for sender, receiver in topology.links:
        env.add_communication_link(sender, receiver)

    return env, agents, config


def _create_grpc_env(params: Dict[str, Any]):
    """Create gRPC benchmark environment."""
    from benchmarks.communication.base_communication import LatencyMode
    from benchmarks.communication.grpc.grpc_communication_agent import (
        GrpcCommunicationEnvironment,
    )
    from benchmarks.communication.communication_config import (
        CommunicationConfiguration,
        TopologyPattern,
    )

    latency_mode = params.get("latency_mode", "app_ack")
    communication_mode = params.get("grpc_mode", "unary")
    service_host = params.get("service_host", "0.0.0.0")
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

    env = GrpcCommunicationEnvironment(
        service_host=service_host,
        latency_mode=latency_mode_enum,
        communication_mode=communication_mode,
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

    try:
        if hasattr(env, "stop_service"):
            env.stop_service()
        elif hasattr(env, "teardown"):
            env.teardown()
    except Exception:
        pass

    for agent in _state.get("agents", []):
        try:
            if hasattr(agent, "comm_agent"):
                agent.comm_agent.close()
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
        return jsonify({"error": "Worker busy", "status": _state["status"]}), 409

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
        return jsonify({
            "error": "Not ready. Call /setup first.",
            "status": _state["status"],
        }), 409

    config = _state["config"]
    thread = Thread(target=_run_benchmark, args=(config,), daemon=True)
    _state["run_thread"] = thread
    thread.start()
    return jsonify({"status": "running"})


@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "status": _state["status"],
        "error": _state.get("error"),
    })


@app.route("/metrics", methods=["GET"])
def metrics():
    """Return collected metrics."""
    if _state["status"] == "completed" and _state["metrics"]:
        return jsonify({
            "status": "completed",
            "metrics": _state["metrics"],
        })
    elif _state["status"] == "error":
        return jsonify({
            "status": "error",
            "error": _state["error"],
        }), 500
    else:
        return jsonify({
            "status": _state["status"],
            "message": "Benchmark not yet completed",
        }), 202


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
        "--port", type=int, default=8080,
        help="Port for the worker control server (default: 8080)",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    args = parser.parse_args()

    print(f"Starting agent worker on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
