"""
Distributed Benchmark Runner.

Orchestrates benchmark execution across multiple remote hosts.
Deploys code, starts brokers, coordinates agent workers, and
aggregates results.

Usage:
    python3 -m benchmarks.distributed.distributed_runner config.yml
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import sys
import time
from enum import Enum
from typing import Any, Dict, List, Optional

import requests

from benchmarks.distributed.distributed_config import (
    DistributedConfig,
    HostConfig,
    load_distributed_config,
)
from benchmarks.distributed.ssh_utils import (
    SSHConfig,
    deploy_to_all_hosts,
    ssh_run,
    ssh_run_background,
    ssh_wait_until_ready,
)
from benchmarks.distributed.clock_sync import (
    all_hosts_synced,
    get_clock_offsets,
    verify_clock_sync,
)


WORKER_PORT = 8080


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


def _worker_url(host: HostConfig, path: str) -> str:
    return f"http://{host.ip}:{WORKER_PORT}{path}"


def _wait_for_worker(host: HostConfig, timeout: int = 60) -> bool:
    """Wait until a worker's /health endpoint responds."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(
                _worker_url(host, "/health"), timeout=5
            )
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(2)
    return False


def _poll_worker_status(
    host: HostConfig, timeout: int = 300, poll_interval: float = 2.0
) -> Dict[str, Any]:
    """Poll worker until benchmark completes or errors."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(
                _worker_url(host, "/status"), timeout=10
            )
            data = r.json()
            if data["status"] in ("completed", "error"):
                return data
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(poll_interval)
    return {"status": "timeout", "error": "Worker poll timed out"}


class DistributedBenchmarkRunner:
    """Orchestrates distributed benchmark execution."""

    def __init__(self, config: DistributedConfig):
        self.config = config
        self.results: Dict[str, Any] = {}
        self.clock_offsets: Dict[str, float] = {}

    def run(self):
        """Execute full distributed benchmark pipeline."""
        print("=" * 60)
        print("DISTRIBUTED BENCHMARK RUNNER")
        print("=" * 60)

        start_time = time.time()

        # 1. Verify SSH connectivity
        print("\n[1/6] Verifying host connectivity...")
        self._verify_hosts()

        # 2. Deploy code
        print("\n[2/6] Deploying code to remote hosts...")
        self._deploy_code()

        # 3. Verify clock sync
        if self.config.time_sync.enabled:
            print("\n[3/6] Verifying clock synchronization...")
            self._verify_clocks()
        else:
            print("\n[3/6] Clock sync verification skipped")

        # 4. Start brokers (if needed)
        print("\n[4/6] Starting message brokers...")
        self._start_brokers()

        # 5. Start workers and run benchmarks
        print("\n[5/6] Running benchmarks...")
        self._run_benchmarks()

        # 6. Export results
        total_time = time.time() - start_time
        print(f"\n[6/6] Exporting results (total time: {total_time:.1f}s)...")
        self._export_results(total_time)

        print("\nDistributed benchmark complete!")

    def _verify_hosts(self):
        for host in self.config.all_remote_hosts:
            ssh_cfg = self._ssh_config(host)
            print(f"  Checking {host.name} ({host.ip})...")
            if not ssh_wait_until_ready(ssh_cfg, max_retries=3, retry_interval=5):
                raise RuntimeError(
                    f"Cannot reach host {host.name} ({host.ip}) via SSH"
                )
            print(f"    {host.name} OK")

    def _deploy_code(self):
        project_root = os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
        )
        results = deploy_to_all_hosts(
            hosts=self.config.all_remote_hosts,
            ssh_user=self.config.ssh_user,
            ssh_key=self.config.ssh_key,
            local_project_path=project_root,
            remote_project_path=self.config.code_path,
        )
        failed = [name for name, ok in results.items() if not ok]
        if failed:
            raise RuntimeError(f"Deploy failed on hosts: {failed}")

    def _verify_clocks(self):
        statuses = verify_clock_sync(
            hosts=self.config.all_remote_hosts,
            ssh_user=self.config.ssh_user,
            ssh_key=self.config.ssh_key,
            max_offset_ms=self.config.time_sync.max_offset_ms,
        )
        self.clock_offsets = get_clock_offsets(statuses)
        if not all_hosts_synced(
            statuses, self.config.time_sync.max_offset_ms
        ):
            print(
                "  WARNING: Not all hosts meet clock sync threshold. "
                "Latency measurements may be less accurate."
            )

    def _start_brokers(self):
        """Start MQTT/Kafka brokers on the broker host."""
        broker_host = self.config.broker_host
        if broker_host is None:
            broker_protocols = {"mqtt", "kafka"}
            if broker_protocols & set(self.config.protocols):
                print(
                    "  WARNING: No broker host configured but "
                    "broker-based protocols requested"
                )
            else:
                print("  No broker needed for configured protocols")
            return

        ssh_cfg = self._ssh_config(broker_host)

        if "mqtt" in self.config.protocols:
            print(f"  Starting MQTT broker on {broker_host.name}...")
            ssh_run(
                ssh_cfg,
                "docker rm -f mqtt-benchmark 2>/dev/null; "
                "docker run -d --name mqtt-benchmark "
                "-p 1883:1883 -p 9001:9001 "
                "eclipse-mosquitto:latest",
                check=False,
            )
            time.sleep(3)
            print("    MQTT broker started")

        if "kafka" in self.config.protocols:
            print(f"  Starting Kafka broker on {broker_host.name}...")
            broker_ip = broker_host.ip
            ssh_run(
                ssh_cfg,
                "docker rm -f kafka-benchmark 2>/dev/null; "
                "docker run -d --name kafka-benchmark "
                "-p 9092:9092 "
                "-e KAFKA_NODE_ID=1 "
                "-e KAFKA_PROCESS_ROLES=broker,controller "
                "-e 'KAFKA_LISTENERS=PLAINTEXT://0.0.0.0:9092,"
                "CONTROLLER://0.0.0.0:9093' "
                f"-e 'KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://"
                f"{broker_ip}:9092' "
                "-e KAFKA_CONTROLLER_LISTENER_NAMES=CONTROLLER "
                "-e 'KAFKA_LISTENER_SECURITY_PROTOCOL_MAP="
                "CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT' "
                "-e 'KAFKA_CONTROLLER_QUORUM_VOTERS=1@localhost:9093' "
                "-e KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1 "
                "-e KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR=1 "
                "-e KAFKA_TRANSACTION_STATE_LOG_MIN_ISR=1 "
                "-e KAFKA_GROUP_INITIAL_REBALANCE_DELAY_MS=0 "
                "-e KAFKA_AUTO_CREATE_TOPICS_ENABLE=true "
                "-e KAFKA_NUM_PARTITIONS=4 "
                "apache/kafka:latest",
                check=False,
            )
            print("    Waiting for Kafka to initialize...")
            time.sleep(15)
            print("    Kafka broker started")

    def _start_worker(self, host: HostConfig):
        """Start the agent worker on a remote host."""
        ssh_cfg = self._ssh_config(host)
        code_path = self.config.code_path

        # Kill any existing worker
        ssh_run(
            ssh_cfg,
            "pkill -f 'agent_worker' 2>/dev/null || true",
            check=False,
        )
        time.sleep(1)

        # Start worker in background
        ssh_run_background(
            ssh_cfg,
            f"cd {code_path}/src && "
            f"python3 -m benchmarks.distributed.agent_worker "
            f"--port {WORKER_PORT}",
        )

        if not _wait_for_worker(host, timeout=30):
            raise RuntimeError(
                f"Worker on {host.name} ({host.ip}) failed to start"
            )
        print(f"    Worker on {host.name} ready")

    def _stop_worker(self, host: HostConfig):
        """Stop the agent worker on a remote host."""
        ssh_cfg = self._ssh_config(host)
        ssh_run(
            ssh_cfg,
            "pkill -f 'agent_worker' 2>/dev/null || true",
            check=False,
        )

    def _run_benchmarks(self):
        """Run all protocol/scenario/variant combinations."""
        agent_hosts = self.config.agent_hosts

        # Start workers on all agent hosts
        for host in agent_hosts:
            print(f"  Starting worker on {host.name}...")
            self._start_worker(host)

        try:
            for protocol in self.config.protocols:
                variants = self.config.protocol_variants.get(
                    protocol, ["default"]
                )
                for variant in variants:
                    for scenario in self.config.scenarios:
                        for agent_count in self.config.agent_counts:
                            self._run_single_benchmark(
                                protocol=protocol,
                                variant=variant,
                                scenario=scenario,
                                agent_count=agent_count,
                                agent_hosts=agent_hosts,
                            )
        finally:
            # Stop all workers
            for host in agent_hosts:
                self._stop_worker(host)

    def _run_single_benchmark(
        self,
        protocol: str,
        variant: str,
        scenario: str,
        agent_count: int,
        agent_hosts: List[HostConfig],
    ):
        """Run a single benchmark across distributed workers."""
        label = f"{protocol}::{variant}"
        print(
            f"\n  Running: {label} / {scenario} "
            f"/ {agent_count} agents"
        )

        # Determine broker address for broker-based protocols
        broker_params = {}
        if protocol in ("mqtt", "kafka") and self.config.broker_host:
            broker_ip = self.config.broker_host.ip
            if protocol == "mqtt":
                broker_params = {
                    "broker_host": broker_ip,
                    "broker_port": 1883,
                }
            elif protocol == "kafka":
                broker_params = {
                    "broker_host": broker_ip,
                    "broker_port": 9092,
                }

        # Build variant-specific params
        variant_params = {}
        vs = self.config.variant_settings.get(protocol, {})
        variant_cfg = vs.get(variant, {})
        if isinstance(variant_cfg, dict):
            variant_params = variant_cfg.get("parameters", {})
            # Also merge scenario-specific overrides
            scenario_overrides = variant_cfg.get("scenarios", {})
            if scenario in scenario_overrides:
                variant_params.update(scenario_overrides[scenario])

        # Map variant names to protocol-specific params
        variant_params.update(
            _variant_to_params(protocol, variant)
        )

        # Distribute agents across hosts (round-robin)
        agents_per_host = _distribute_agents(
            agent_count, len(agent_hosts), self.config.agent_placement
        )

        trial_results = []
        for trial in range(self.config.num_trials):
            print(f"    Trial {trial + 1}/{self.config.num_trials}...")

            # Setup and run on each worker
            host_metrics = []
            for i, host in enumerate(agent_hosts):
                host_agent_count = agents_per_host[i]
                if host_agent_count == 0:
                    continue

                params = {
                    "agent_count": host_agent_count,
                    "latency_mode": self.config.latency_mode,
                    **broker_params,
                    **variant_params,
                }

                # Setup worker
                setup_payload = {
                    "protocol": protocol,
                    "scenario": scenario,
                    "params": params,
                }

                try:
                    r = requests.post(
                        _worker_url(host, "/setup"),
                        json=setup_payload,
                        timeout=30,
                    )
                    if r.status_code != 200:
                        print(
                            f"      Setup failed on {host.name}: "
                            f"{r.text}"
                        )
                        continue
                except requests.RequestException as e:
                    print(f"      Setup error on {host.name}: {e}")
                    continue

                # Start benchmark
                try:
                    r = requests.post(
                        _worker_url(host, "/start"), timeout=10
                    )
                except requests.RequestException as e:
                    print(f"      Start error on {host.name}: {e}")
                    continue

            # Poll all workers for completion
            for i, host in enumerate(agent_hosts):
                if agents_per_host[i] == 0:
                    continue

                result = _poll_worker_status(host, timeout=300)
                if result["status"] == "completed":
                    try:
                        r = requests.get(
                            _worker_url(host, "/metrics"), timeout=30
                        )
                        metrics = r.json().get("metrics", {})
                        host_metrics.append({
                            "host": host.name,
                            "metrics": metrics,
                        })
                    except requests.RequestException:
                        pass
                else:
                    print(
                        f"      {host.name}: {result.get('error', 'unknown')}"
                    )

                # Teardown worker for next trial
                try:
                    requests.post(
                        _worker_url(host, "/teardown"), timeout=10
                    )
                except requests.RequestException:
                    pass

            if host_metrics:
                aggregated = _aggregate_host_metrics(host_metrics)
                trial_results.append(aggregated)

        if trial_results:
            key = f"{label}/{scenario}/{agent_count}_agents"
            self.results[key] = {
                "protocol": protocol,
                "variant": variant,
                "scenario": scenario,
                "agent_count": agent_count,
                "trials": trial_results,
                "aggregated": _aggregate_trials(trial_results),
            }
            latency = self.results[key]["aggregated"].get(
                "latency_avg_ms", 0
            )
            throughput = self.results[key]["aggregated"].get(
                "throughput_avg", 0
            )
            print(
                f"    Result: avg_latency={latency:.2f}ms, "
                f"throughput={throughput:.1f} msg/s"
            )

    def _export_results(self, total_time: float):
        """Export results to JSON."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        export_data = {
            "benchmark_metadata": {
                "timestamp": timestamp,
                "execution_mode": "distributed",
                "total_duration_sec": total_time,
                "latency_mode": self.config.latency_mode,
                "protocols_tested": self.config.protocols,
                "scenarios_tested": self.config.scenarios,
                "num_trials": self.config.num_trials,
                "hosts": {
                    name: {"ip": h.ip, "role": h.role}
                    for name, h in self.config.hosts.items()
                },
                "clock_offsets_ms": self.clock_offsets,
                "hardware": _get_hardware_metadata(),
            },
            "results": self.results,
        }

        output_file = os.path.join(
            self.config.output_dir,
            f"distributed_results_{timestamp}.json",
        )
        with open(output_file, "w") as f:
            json.dump(export_data, f, indent=2, cls=EnumEncoder)

        print(f"  Results exported to: {output_file}")

    def _ssh_config(self, host: HostConfig) -> SSHConfig:
        return SSHConfig(
            host=host.ip,
            user=self.config.ssh_user,
            key_path=self.config.ssh_key,
        )


# --- Helper functions ---


def _variant_to_params(protocol: str, variant: str) -> Dict[str, Any]:
    """Map variant names to protocol-specific parameters."""
    if variant == "default":
        return {}

    mappings = {
        "rest": {
            "http1": {"transport_mode": "http1"},
            "http2": {"transport_mode": "http2"},
        },
        "grpc": {
            "unary": {"grpc_mode": "unary"},
            "streaming": {"grpc_mode": "streaming"},
        },
        "mqtt": {
            "qos0": {"mqtt_qos": 0},
            "qos1": {"mqtt_qos": 1},
            "qos2": {"mqtt_qos": 2},
        },
        "kafka": {
            "acks0": {"kafka_acks": 0},
            "acks1": {"kafka_acks": 1},
            "acksall": {"kafka_acks": "all"},
        },
    }
    return mappings.get(protocol, {}).get(variant, {})


def _distribute_agents(
    total: int, num_hosts: int, placement: str
) -> List[int]:
    """Distribute agents across hosts.

    Returns list of agent counts per host.
    """
    if num_hosts == 0:
        return []
    if placement == "all_on_separate" and total <= num_hosts:
        # One agent per host (for point-to-point scenarios)
        counts = [1] * total + [0] * (num_hosts - total)
        return counts

    # Round-robin distribution
    base = total // num_hosts
    remainder = total % num_hosts
    counts = [base] * num_hosts
    for i in range(remainder):
        counts[i] += 1
    return counts


def _aggregate_host_metrics(
    host_metrics: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate metrics from multiple hosts into one result."""
    all_latency_samples = []
    total_messages = 0
    total_duration = 0
    total_failures = 0
    cpu_samples = []
    memory_samples = []

    for hm in host_metrics:
        m = hm["metrics"]
        latency = m.get("latency", {})
        throughput = m.get("throughput", {})
        resources = m.get("resources", {})
        test_result = m.get("test_result", {})

        all_latency_samples.extend(m.get("latency_samples", []))
        total_messages += throughput.get("total_messages", 0)
        total_duration = max(total_duration, m.get("duration", 0))
        total_failures += test_result.get("delivery_failures", 0)

        if resources.get("cpu_avg"):
            cpu_samples.append(resources["cpu_avg"])
        if resources.get("memory_avg"):
            memory_samples.append(resources["memory_avg"])

    # Calculate aggregated latency stats
    latency_stats = {}
    if all_latency_samples:
        sorted_samples = sorted(all_latency_samples)
        n = len(sorted_samples)
        latency_stats = {
            "avg_ms": statistics.mean(sorted_samples) * 1000,
            "min_ms": min(sorted_samples) * 1000,
            "max_ms": max(sorted_samples) * 1000,
            "p50_ms": _percentile(sorted_samples, 0.50) * 1000,
            "p95_ms": _percentile(sorted_samples, 0.95) * 1000,
            "p99_ms": _percentile(sorted_samples, 0.99) * 1000,
            "std_ms": (
                statistics.stdev(sorted_samples) * 1000 if n > 1 else 0
            ),
            "sample_count": n,
        }

    total_success = total_messages - total_failures
    success_rate = (
        total_success / total_messages * 100 if total_messages > 0 else 0
    )

    return {
        "latency": latency_stats,
        "throughput_avg": (
            total_messages / total_duration if total_duration > 0 else 0
        ),
        "total_messages": total_messages,
        "delivery_failures": total_failures,
        "success_rate": success_rate,
        "cpu_avg": statistics.mean(cpu_samples) if cpu_samples else 0,
        "memory_avg_mb": (
            statistics.mean(memory_samples) if memory_samples else 0
        ),
        "duration": total_duration,
        "hosts_reporting": len(host_metrics),
        "latency_samples": all_latency_samples[:2000],
    }


def _aggregate_trials(
    trials: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate multiple trial results with confidence intervals."""
    if not trials:
        return {}

    def _ci(values):
        if not values:
            return {"mean": 0, "ci_lower": 0, "ci_upper": 0}
        mean = statistics.mean(values)
        if len(values) < 2:
            return {"mean": mean, "ci_lower": mean, "ci_upper": mean}
        std = statistics.stdev(values)
        n = len(values)
        margin = 1.96 * std / (n ** 0.5)
        return {
            "mean": mean,
            "ci_lower": mean - margin,
            "ci_upper": mean + margin,
        }

    latency_avgs = [
        t["latency"].get("avg_ms", 0) for t in trials if t.get("latency")
    ]
    latency_p95s = [
        t["latency"].get("p95_ms", 0) for t in trials if t.get("latency")
    ]
    latency_p99s = [
        t["latency"].get("p99_ms", 0) for t in trials if t.get("latency")
    ]
    throughputs = [t.get("throughput_avg", 0) for t in trials]
    success_rates = [t.get("success_rate", 0) for t in trials]

    return {
        "latency_avg_ms": _ci(latency_avgs)["mean"],
        "latency_avg": _ci(latency_avgs),
        "latency_p95": _ci(latency_p95s),
        "latency_p99": _ci(latency_p99s),
        "throughput_avg": _ci(throughputs)["mean"],
        "throughput": _ci(throughputs),
        "success_rate": _ci(success_rates),
        "num_trials": len(trials),
    }


def _percentile(sorted_data: List[float], p: float) -> float:
    if not sorted_data:
        return 0.0
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]
    idx = p * (n - 1)
    lower = int(idx)
    upper = min(lower + 1, n - 1)
    frac = idx - lower
    return sorted_data[lower] * (1 - frac) + sorted_data[upper] * frac


def _get_hardware_metadata() -> Dict[str, Any]:
    """Collect orchestrator hardware metadata."""
    try:
        import psutil
        return {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "node": platform.node(),
            "cpu_count": psutil.cpu_count(logical=True),
            "memory_total_gb": round(
                psutil.virtual_memory().total / (1024**3), 1
            ),
        }
    except ImportError:
        return {
            "system": platform.system(),
            "machine": platform.machine(),
            "node": platform.node(),
        }


def main():
    """Entry point for distributed benchmark runner."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run distributed benchmarks across AWS hosts"
    )
    parser.add_argument(
        "config",
        help="Path to distributed benchmark YAML config",
    )
    parser.add_argument(
        "--skip-deploy",
        action="store_true",
        help="Skip code deployment (use if already deployed)",
    )
    parser.add_argument(
        "--skip-brokers",
        action="store_true",
        help="Skip broker startup (use if already running)",
    )
    args = parser.parse_args()

    config = load_distributed_config(args.config)
    config.validate()

    runner = DistributedBenchmarkRunner(config)

    if args.skip_deploy:
        runner._deploy_code = lambda: print("  Skipped (--skip-deploy)")
    if args.skip_brokers:
        runner._start_brokers = lambda: print("  Skipped (--skip-brokers)")

    runner.run()


if __name__ == "__main__":
    main()
