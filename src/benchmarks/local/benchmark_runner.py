#!/usr/bin/env python3
"""
Unified Communication Protocol Benchmark Runner
Runs standardized benchmarks across all 4 communication protocols (REST, gRPC,
MQTT, Kafka).

Provides both simple and extensive benchmark modes with structured output
for cross-protocol performance comparison and analysis.
"""

import argparse
import time
import json
import os
import copy
import platform
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from enum import Enum
import psutil
from benchmarks.communication.communication_config import TopologyPattern


# Import protocol-specific benchmark scenarios
from benchmarks.local.config_loader import load_config_from_yaml
from benchmarks.local.communication_benchmarks.communication_benchmark import (
    CommunicationBenchmark,
)
from benchmarks.local.hierarchy_benchmarks.hierarchy_benchmark_scenarios import (
    HierarchyComparisonBenchmark,
)
from benchmarks.local.hierarchy_benchmarks.hierarchy_strategies import HierarchyType


class EnumEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles Enum types."""

    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


def _read_cpu_model() -> str:
    """Best-effort CPU model string."""
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    cpu_model = platform.processor()
    if cpu_model:
        return cpu_model
    try:
        return platform.uname().processor
    except Exception:
        return ""


def _get_hardware_metadata() -> Dict[str, Any]:
    """Collect basic hardware and OS metadata for reproducibility."""
    os_info = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "node": platform.node(),
    }

    cpu_info = {
        "model": _read_cpu_model() or "unknown",
        "physical_cores": psutil.cpu_count(logical=False) or 0,
        "logical_cores": psutil.cpu_count(logical=True) or 0,
    }
    freq = psutil.cpu_freq()
    if freq:
        if freq.min:
            cpu_info["min_mhz"] = round(freq.min, 2)
        if freq.max:
            cpu_info["max_mhz"] = round(freq.max, 2)
        if freq.current:
            cpu_info["current_mhz"] = round(freq.current, 2)

    memory_info = {}
    try:
        total_gb = psutil.virtual_memory().total / (1024**3)
        memory_info["total_gb"] = round(total_gb, 2)
    except Exception:
        pass

    return {
        "os": os_info,
        "python_version": platform.python_version(),
        "cpu": cpu_info,
        "memory": memory_info,
    }


def _create_rest_benchmark(latency_mode: str):
    from benchmarks.local.communication_benchmarks.rest_benchmark_scenarios import (
        create_rest_benchmark_scenarios,
    )

    return create_rest_benchmark_scenarios(latency_mode=latency_mode)


def _create_grpc_benchmark(latency_mode: str):
    from benchmarks.local.communication_benchmarks.grpc_benchmark_scenarios import (
        create_grpc_benchmark_scenarios,
    )

    return create_grpc_benchmark_scenarios(latency_mode=latency_mode)


def _create_mqtt_benchmark(latency_mode: str):
    from benchmarks.local.communication_benchmarks.mqtt_benchmark_scenarios import (
        create_mqtt_benchmark_scenarios,
    )

    return create_mqtt_benchmark_scenarios(latency_mode=latency_mode)


def _create_kafka_benchmark(latency_mode: str):
    from benchmarks.local.communication_benchmarks.kafka_benchmark_scenarios import (
        create_kafka_benchmark_scenarios,
    )

    return create_kafka_benchmark_scenarios(latency_mode=latency_mode)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""

    protocols: List[str]
    scenarios: List[str]
    simple_mode: bool = True
    extensive_mode: bool = False
    agent_counts: List[int] = None
    topologies: List[TopologyPattern] = None
    output_dir: str = "results"
    export_csv: bool = True
    export_json: bool = True
    latency_mode: str = "end_to_end"  # "send_only", "end_to_end", or "app_ack"
    protocol_variants: Dict[str, List[str]] = None
    variant_settings: Dict[str, Dict[str, Any]] = None
    num_trials: int = 1
    warm_up_operations: int = 0
    # Hierarchy benchmark options
    hierarchy_mode: bool = False
    hierarchy_types: List[str] = None
    hierarchy_environments: List[str] = None
    hierarchy_episodes: int = 10
    run_ablation: bool = False


class ProtocolBenchmarkRunner:
    """Unified runner for all communication protocol benchmarks."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: Dict[str, Dict[str, Any]] = {}
        self.comparison_data: Dict[str, Dict[str, float]] = {}
        self.variant_settings = (
            copy.deepcopy(config.variant_settings)
            if config.variant_settings
            else {}
        )

        # Default configurations
        if not config.agent_counts:
            config.agent_counts = [3, 5, 8] if config.extensive_mode else [5]
        if not config.topologies:
            config.topologies = (
                [
                    TopologyPattern.FULLY_CONNECTED,
                    TopologyPattern.STAR,
                    TopologyPattern.RING,
                    TopologyPattern.CHAIN,
                ]
                if config.extensive_mode
                else [TopologyPattern.FULLY_CONNECTED]
            )

        default_variant_map = {
            "rest": ["http1", "http2"],
            "grpc": ["unary", "streaming"],
            "mqtt": ["qos0", "qos1", "qos2"],
            "kafka": ["acks0", "acks1", "acksall"],
        }

        if not config.protocol_variants:
            config.protocol_variants = {}

        for protocol in config.protocols:
            if protocol not in config.protocol_variants:
                config.protocol_variants[protocol] = default_variant_map.get(
                    protocol, ["default"]
                )
            else:
                # Ensure variants list is not empty
                if not config.protocol_variants[protocol]:
                    config.protocol_variants[protocol] = (
                        default_variant_map.get(protocol, ["default"])
                    )

            self.variant_settings.setdefault(protocol, {})
            for variant in config.protocol_variants[protocol]:
                self.variant_settings[protocol].setdefault(variant, {})

        # Protocol factory mapping
        self.protocol_factories = {
            "rest": _create_rest_benchmark,
            "grpc": _create_grpc_benchmark,
            "mqtt": _create_mqtt_benchmark,
            "kafka": _create_kafka_benchmark,
        }

        # Ensure output directory exists
        os.makedirs(self.config.output_dir, exist_ok=True)

    def _ensure_brokers_running(self):
        """Start required message brokers before benchmarks begin."""
        print("\nEnsuring message brokers are running...")

        if "mqtt" in self.config.protocols:
            print("  Starting MQTT broker...")
            from benchmarks.local.communication_benchmarks.mqtt_benchmark_scenarios import (
                start_mqtt_docker,
                is_mqtt_running,
            )

            if not is_mqtt_running():
                start_mqtt_docker()
            else:
                print("  ✓ MQTT broker already running")

        if "kafka" in self.config.protocols:
            print("  Starting Kafka broker...")
            from benchmarks.local.communication_benchmarks.kafka_benchmark_scenarios import (
                start_kafka_docker,
                is_kafka_running,
            )

            if not is_kafka_running():
                start_kafka_docker()
            else:
                print("  ✓ Kafka broker already running")

        print("✓ All required brokers are ready\n")

    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run benchmarks for all specified protocols."""
        print("=" * 80)
        print("UNIFIED COMMUNICATION PROTOCOL BENCHMARK SUITE")
        print("=" * 80)
        print(
            f"Mode: {'Extensive' if self.config.extensive_mode else 'Simple'}"
        )
        print(f"Protocols: {', '.join(self.config.protocols)}")
        print(f"Scenarios: {', '.join(self.config.scenarios)}")
        print(f"Agent counts: {self.config.agent_counts}")
        print(f"Topologies: {[t.value for t in self.config.topologies]}")
        print("=" * 80)

        # Check and start required brokers BEFORE timing begins
        self._ensure_brokers_running()

        start_time = time.time()

        for protocol in self.config.protocols:
            if protocol not in self.protocol_factories:
                print(f"Warning: Unknown protocol '{protocol}', skipping...")
                continue

            print(f"\nRunning {protocol.upper()} benchmarks...")
            try:
                protocol_results = self._run_protocol_benchmarks(protocol)
                self.results[protocol] = protocol_results
                print(f"{protocol.upper()} benchmarks completed")
            except Exception as e:
                print(f"{protocol.upper()} benchmarks failed: {e}")
                self.results[protocol] = {"error": str(e)}

        total_time = time.time() - start_time

        self._generate_comparison_data()

        if self.config.export_json:
            self._export_json_results(total_time)
        if self.config.export_csv:
            self._export_csv_results()

        self._print_final_summary(total_time)

        return self.results

    def _run_protocol_benchmarks(self, protocol: str) -> Dict[str, Any]:
        """Run benchmarks for a specific protocol."""
        variants = self.config.protocol_variants.get(protocol, ["default"])

        # Exclude incompatible variants for certain modes
        # In app_ack mode, Kafka acks=0 is incompatible with reliable app-level ACKs
        if (
            protocol == "kafka"
            and str(self.config.latency_mode).lower() == "app_ack"
        ):
            variants = [v for v in variants if v.lower() != "acks0"]

        protocol_results = {
            "variants": {},
            "metadata": {
                "protocol": protocol,
                "timestamp": time.time(),
                "config": asdict(self.config),
            },
        }

        for variant in variants:
            print(f"\n  • Variant: {protocol.upper()} [{variant}]")
            variant_results = self._execute_protocol_variant(protocol, variant)
            protocol_results["variants"][variant] = variant_results

        return protocol_results

    def _execute_protocol_variant(
        self, protocol: str, variant: str
    ) -> Dict[str, Any]:
        """Execute all scenarios for a specific protocol variant."""
        benchmark = self.protocol_factories[protocol](
            latency_mode=self.config.latency_mode
        )

        variant_config: Dict[str, Any] = {}
        base_variant_config = (
            self.variant_settings.get(protocol, {}).get("__default__", {})
            or {}
        )
        specific_variant_config = (
            self.variant_settings.get(protocol, {}).get(variant, {}) or {}
        )

        # Parameters
        parameters: Dict[str, Any] = {}
        parameters.update(base_variant_config.get("parameters", {}) or {})
        parameters.update(specific_variant_config.get("parameters", {}) or {})
        if parameters:
            variant_config["parameters"] = parameters

        # Concurrency levels
        concurrency_levels = specific_variant_config.get("concurrency_levels")
        if concurrency_levels is None:
            concurrency_levels = base_variant_config.get("concurrency_levels")
        if concurrency_levels:
            variant_config["concurrency_levels"] = concurrency_levels

        # Scenario-specific overrides
        scenario_overrides: Dict[str, Dict[str, Any]] = {}
        for scenario_name, overrides in (
            base_variant_config.get("scenarios", {}) or {}
        ).items():
            scenario_overrides[scenario_name] = dict(overrides)
        for scenario_name, overrides in (
            specific_variant_config.get("scenarios", {}) or {}
        ).items():
            merged = scenario_overrides.get(scenario_name, {}).copy()
            merged.update(overrides or {})
            scenario_overrides[scenario_name] = merged
        if scenario_overrides:
            variant_config["scenarios"] = scenario_overrides

        variant_params = self._get_variant_params(protocol, variant)
        variant_specific_params = variant_config.get("parameters", {})
        if variant_specific_params:
            variant_params.update(variant_specific_params)

        variant_results: Dict[str, Any] = {
            "scenarios": {},
            "topology_comparison": {},
            "metadata": {
                "variant": variant,
                "protocol": protocol,
                "parameters": variant_params.copy(),
                "variant_config": variant_config,
            },
        }

        for scenario in self.config.scenarios:
            if (
                scenario == "broadcast_throughput"
                and self.config.latency_mode == "end_to_end"
            ):
                print(
                    f"Skipping {scenario} scenario (not applicable\
                        in end_to_end mode)"
                )
                continue

            print(f"    - Scenario: {scenario}")

            scenario_overrides = variant_config.get("scenarios", {}).get(
                scenario, {}
            )

            if self.config.simple_mode:
                params = self._get_simple_params(scenario)
                params.update(variant_params)
                params.update(scenario_overrides)
                params["protocol_variant"] = variant
                params["protocol_name"] = protocol

                result = benchmark.run_scenario(
                    scenario,
                    num_trials=self.config.num_trials,
                    warm_up_operations=self.config.warm_up_operations,
                    **params,
                )
                variant_results["scenarios"][scenario] = self._extract_metrics(
                    result
                )

            elif self.config.extensive_mode:
                scenario_results = {}

                for agent_count in self.config.agent_counts:
                    print(f"      • Agents: {agent_count}")
                    params = self._get_extensive_params(
                        scenario, agent_count=agent_count
                    )
                    params.update(variant_params)
                    params.update(scenario_overrides)
                    params["protocol_variant"] = variant
                    params["protocol_name"] = protocol

                    result = benchmark.run_scenario(
                        scenario,
                        num_trials=self.config.num_trials,
                        warm_up_operations=self.config.warm_up_operations,
                        **params,
                    )
                    scenario_results[f"{agent_count}_agents"] = (
                        self._extract_metrics(result)
                    )

                variant_results["scenarios"][scenario] = scenario_results

        if self.config.extensive_mode:
            print("    - Topology comparison sweep")
            topology_results = {}
            for topology in self.config.topologies:
                print(f"      • Topology: {topology.value}")
                params = {
                    "agent_count": 20,
                    "topology_pattern": topology,
                    "messages_per_agent": 50,
                    "protocol_variant": variant,
                    "protocol_name": protocol,
                }
                params.update(variant_params)
                topology_overrides = variant_config.get("scenarios", {}).get(
                    "concurrent_messaging", {}
                )
                params.update(topology_overrides)

                result = benchmark.run_scenario(
                    "concurrent_messaging",
                    num_trials=self.config.num_trials,
                    warm_up_operations=self.config.warm_up_operations,
                    **params,
                )
                topology_results[topology.value] = self._extract_metrics(
                    result
                )
            variant_results["topology_comparison"] = topology_results

        variant_results["concurrency_matrix"] = self._run_concurrency_matrix(
            benchmark, protocol, variant, variant_params, variant_config
        )

        return variant_results

    def _get_simple_params(self, scenario: str) -> Dict[str, Any]:
        """Get simple benchmark parameters."""
        base_params = {
            "agent_count": 5,
            "topology_pattern": TopologyPattern.FULLY_CONNECTED,
            "latency_mode": self.config.latency_mode,
        }

        scenario_params = {
            "point_to_point_latency": {"agent_count": 2, "message_count": 50},
            "broadcast_throughput": {"agent_count": 5, "message_count": 30},
            "concurrent_messaging": {
                "agent_count": 4,
                "messages_per_agent": 100,
            },
            "scalability_stress": {"agent_count": 5, "stress_duration": 3.0},
        }

        base_params.update(scenario_params.get(scenario, {}))
        return base_params

    def _get_extensive_params(
        self, scenario: str, **overrides
    ) -> Dict[str, Any]:
        """Get extensive benchmark parameters."""
        base_params = {
            "agent_count": 8,
            "topology_pattern": TopologyPattern.FULLY_CONNECTED,
            "latency_mode": self.config.latency_mode,
        }

        scenario_params = {
            "point_to_point_latency": {"message_count": 100},
            "broadcast_throughput": {"message_count": 50},
            "concurrent_messaging": {"messages_per_agent": 100},
            "scalability_stress": {"stress_duration": 5.0},
        }

        base_params.update(scenario_params.get(scenario, {}))
        base_params.update(overrides)
        return base_params

    def _get_variant_params(
        self, protocol: str, variant: str
    ) -> Dict[str, Any]:
        """Map protocol variant identifiers to scenario parameters."""
        if protocol == "rest":
            if variant.lower() == "http2":
                return {"transport_mode": "http2"}
            return {"transport_mode": "http1"}

        if protocol == "grpc":
            return {
                "grpc_mode": (
                    "streaming" if variant.lower() == "streaming" else "unary"
                )
            }

        if protocol == "mqtt":
            qos_map = {
                "qos0": 0,
                "qos1": 1,
                "qos2": 2,
            }
            qos_value = qos_map.get(variant.lower(), 1)
            return {"mqtt_qos": qos_value}

        if protocol == "kafka":
            ack_map = {
                "acks0": 0,  # Integer for acks=0
                "acks1": 1,  # Integer for acks=1
                "acksall": "all",  # String "all" is valid
                "acks-1": -1,  # Integer for acks=-1 (same as "all")
            }
            ack_value = ack_map.get(variant.lower(), 1)  # Default to 1 (int)
            return {"kafka_acks": ack_value}

        return {}

    def _run_concurrency_matrix(
        self,
        benchmark: CommunicationBenchmark,
        protocol: str,
        variant: str,
        variant_params: Dict[str, Any],
        variant_config: Dict[str, Any],
    ) -> Dict[str, Dict[str, float]]:
        """Execute concurrency sweep for concurrent messaging scenario."""
        concurrency_levels = variant_config.get("concurrency_levels")
        if concurrency_levels is None:
            concurrency_levels = [1, 4, 16, 64]
        matrix_results: Dict[str, Dict[str, float]] = {}

        if not concurrency_levels:
            return matrix_results

        if (
            "concurrent_messaging" not in benchmark.scenarios
            or "concurrent_messaging" not in self.config.scenarios
        ):
            return matrix_results

        for level in concurrency_levels:
            agent_count = max(level, 2)
            print(f"    - Concurrency Matrix: {level} sender(s)")

            params = self._get_simple_params("concurrent_messaging")
            params.update(variant_params)
            scenario_overrides = variant_config.get("scenarios", {}).get(
                "concurrent_messaging", {}
            )
            params.update(scenario_overrides)
            params.update(
                {
                    "agent_count": max(
                        agent_count, params.get("agent_count", agent_count)
                    ),
                    "protocol_variant": variant,
                    "protocol_name": protocol,
                    "concurrent_senders": level,
                }
            )

            result = benchmark.run_scenario(
                "concurrent_messaging",
                num_trials=self.config.num_trials,
                warm_up_operations=self.config.warm_up_operations,
                **params,
            )
            matrix_results[str(level)] = self._extract_metrics(result)

        return matrix_results

    def _extract_metrics(self, result) -> Dict[str, float]:
        """Extract key metrics from benchmark result."""
        if result is None:
            return {}

        is_multi_trial = hasattr(result, "trial_count") and hasattr(
            result, "latency_avg_mean"
        )

        if is_multi_trial:
            trials = list(getattr(result, "trials", []) or [])
            latency_samples_sec = []
            peak_throughputs = []
            durations = []
            total_messages_all_trials = 0
            total_delivery_failures = 0
            total_timeout_failures = 0
            weighted_successes = 0.0
            weighted_total_messages = 0.0
            trial_count = int(getattr(result, "trial_count", 0) or 0)

            for trial in trials:
                metrics = getattr(trial, "metrics", None)
                if metrics is None:
                    continue
                latency_samples_sec.extend(
                    getattr(metrics, "latency_samples", []) or []
                )
                peak_throughputs.append(
                    getattr(metrics, "throughput_peak", 0.0)
                )
                durations.append(getattr(metrics, "test_duration", 0.0))
                trial_total_messages = int(
                    getattr(metrics, "total_messages", 0) or 0
                )
                total_messages_all_trials += trial_total_messages
                total_delivery_failures += int(
                    getattr(metrics, "delivery_failures", 0) or 0
                )
                total_timeout_failures += int(
                    getattr(metrics, "timeout_failures", 0) or 0
                )
                trial_success_rate = float(
                    getattr(metrics, "success_rate", 0.0) or 0.0
                )
                weighted_successes += trial_success_rate * trial_total_messages
                weighted_total_messages += trial_total_messages

            # Use effective count from collected trials when possible.
            if not trials:
                trial_count = max(trial_count, 1)
            else:
                trial_count = len(trials)

            latency_samples_ms = [
                sample * 1000 for sample in latency_samples_sec
            ]
            sample_cap = 2000
            if len(latency_samples_ms) > sample_cap:
                step = max(len(latency_samples_ms) // sample_cap, 1)
                latency_samples_ms = latency_samples_ms[::step][:sample_cap]

            peak_throughput_mean = (
                sum(peak_throughputs) / len(peak_throughputs)
                if peak_throughputs
                else 0.0
            )
            test_duration_mean = (
                sum(durations) / len(durations) if durations else 0.0
            )
            total_messages_per_trial = (
                total_messages_all_trials / trial_count
                if trial_count > 0 and total_messages_all_trials > 0
                else float(getattr(result, "total_messages", 0) or 0)
            )
            delivery_failures_per_trial = (
                total_delivery_failures / trial_count
                if trial_count > 0
                else 0.0
            )
            timeout_failures_per_trial = (
                total_timeout_failures / trial_count
                if trial_count > 0
                else 0.0
            )
            success_rate_weighted_percent = (
                (weighted_successes / weighted_total_messages) * 100
                if weighted_total_messages > 0
                else result.success_rate_mean * 100
            )

            return {
                "avg_latency_ms": result.latency_avg_mean * 1000,
                "avg_latency_ci_lower_ms": result.latency_avg_ci_lower * 1000,
                "avg_latency_ci_upper_ms": result.latency_avg_ci_upper * 1000,
                "p95_latency_ms": result.latency_p95_mean * 1000,
                "p95_latency_ci_lower_ms": result.latency_p95_ci_lower * 1000,
                "p95_latency_ci_upper_ms": result.latency_p95_ci_upper * 1000,
                "p99_latency_ms": result.latency_p99_mean * 1000,
                "p99_latency_ci_lower_ms": result.latency_p99_ci_lower * 1000,
                "p99_latency_ci_upper_ms": result.latency_p99_ci_upper * 1000,
                "throughput_msg_per_sec": result.throughput_mean,
                "throughput_ci_lower": result.throughput_ci_lower,
                "throughput_ci_upper": result.throughput_ci_upper,
                "peak_throughput_msg_per_sec": peak_throughput_mean,
                "success_rate_percent": result.success_rate_mean * 100,
                "success_rate_ci_lower_percent": result.success_rate_ci_lower
                * 100,
                "success_rate_ci_upper_percent": result.success_rate_ci_upper
                * 100,
                "cpu_usage_percent": result.cpu_usage_mean,
                "cpu_usage_ci_lower": result.cpu_usage_ci_lower,
                "cpu_usage_ci_upper": result.cpu_usage_ci_upper,
                "memory_usage_mb": result.memory_usage_mean,
                "memory_usage_ci_lower_mb": result.memory_usage_ci_lower,
                "memory_usage_ci_upper_mb": result.memory_usage_ci_upper,
                # Keep primary counters per-trial to align with mean-based metrics.
                "total_messages": total_messages_per_trial,
                "test_duration_sec": test_duration_mean,
                "delivery_failures": delivery_failures_per_trial,
                "timeout_failures": timeout_failures_per_trial,
                # Explicit count scales for analysis/export.
                "total_messages_per_trial": total_messages_per_trial,
                "total_messages_all_trials": total_messages_all_trials,
                "delivery_failures_per_trial": delivery_failures_per_trial,
                "delivery_failures_all_trials": total_delivery_failures,
                "timeout_failures_per_trial": timeout_failures_per_trial,
                "timeout_failures_all_trials": total_timeout_failures,
                "success_rate_weighted_percent": success_rate_weighted_percent,
                "latency_samples_ms": latency_samples_ms,
                "latency_sample_count": len(latency_samples_ms),
                "payload_size_bytes": result.payload_size_bytes,
                "num_trials": trial_count,
                "warm_up_operations": result.warm_up_operations,
            }

        latency_samples_ms = [
            sample * 1000 for sample in getattr(result, "latency_samples", [])
        ]

        return {
            "avg_latency_ms": result.message_latency_avg * 1000,
            "p95_latency_ms": result.message_latency_p95 * 1000,
            "p99_latency_ms": result.message_latency_p99 * 1000,
            "throughput_msg_per_sec": result.throughput_avg,
            "peak_throughput_msg_per_sec": result.throughput_peak,
            "success_rate_percent": result.success_rate * 100,
            "cpu_usage_percent": result.cpu_usage_avg,
            "memory_usage_mb": result.memory_usage_avg,
            "total_messages": result.total_messages,
            "test_duration_sec": result.test_duration,
            "delivery_failures": result.delivery_failures,
            "timeout_failures": result.timeout_failures,
            "total_messages_per_trial": result.total_messages,
            "total_messages_all_trials": result.total_messages,
            "delivery_failures_per_trial": result.delivery_failures,
            "delivery_failures_all_trials": result.delivery_failures,
            "timeout_failures_per_trial": result.timeout_failures,
            "timeout_failures_all_trials": result.timeout_failures,
            "success_rate_weighted_percent": result.success_rate * 100,
            "latency_samples_ms": latency_samples_ms,
            "latency_sample_count": len(latency_samples_ms),
            "payload_size_bytes": result.payload_size_bytes,
            "num_trials": 1,
            "warm_up_operations": 0,
        }

    def _generate_comparison_data(self):
        """Generate cross-protocol comparison data."""
        for scenario in self.config.scenarios:
            scenario_comparison = {}

            for protocol, protocol_results in self.results.items():
                if "error" in protocol_results:
                    continue

                variant_map = protocol_results.get("variants", {})
                if not variant_map:
                    continue

                for variant_name, variant_data in variant_map.items():
                    display_key = (
                        f"{protocol}::{variant_name}"
                        if variant_name != "default"
                        else protocol
                    )

                    if self.config.simple_mode:
                        metrics = variant_data.get("scenarios", {}).get(
                            scenario, {}
                        )
                        scenario_comparison[display_key] = metrics
                    else:
                        scenario_data = variant_data.get("scenarios", {}).get(
                            scenario, {}
                        )
                        if scenario_data:
                            avg_metrics = self._average_metrics(
                                scenario_data.values()
                            )
                            scenario_comparison[display_key] = avg_metrics

            self.comparison_data[scenario] = scenario_comparison

    def _average_metrics(
        self, metrics_list: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate average metrics across multiple runs."""
        if not metrics_list:
            return {}

        avg_metrics = {}
        all_keys = set()
        for metrics in metrics_list:
            all_keys.update(metrics.keys())

        for key in all_keys:
            values: List[float] = []
            for metrics in metrics_list:
                if key not in metrics:
                    continue
                value = metrics[key]
                if isinstance(value, (int, float)):
                    values.append(float(value))
            if values:
                avg_metrics[key] = sum(values) / len(values)

        return avg_metrics

    def _export_json_results(self, total_time: float):
        """Export results to JSON format."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        latency_mode_suffix = f"_{self.config.latency_mode}"

        export_data = {
            "benchmark_metadata": {
                "timestamp": timestamp,
                "total_duration_sec": total_time,
                "mode": (
                    "extensive" if self.config.extensive_mode else "simple"
                ),
                "latency_mode": self.config.latency_mode,
                "protocols_tested": self.config.protocols,
                "scenarios_tested": self.config.scenarios,
                "hardware": _get_hardware_metadata(),
            },
            "protocol_results": self.results,
            "cross_protocol_comparison": self.comparison_data,
            "configuration": asdict(self.config),
        }

        main_file = os.path.join(
            self.config.output_dir,
            f"benchmark_results{latency_mode_suffix}_{timestamp}.json",
        )
        with open(main_file, "w") as f:
            json.dump(export_data, f, indent=2, cls=EnumEncoder)

        summary_file = os.path.join(
            self.config.output_dir,
            f"comparison_summary{latency_mode_suffix}_{timestamp}.json",
        )
        with open(summary_file, "w") as f:
            json.dump(
                {
                    "timestamp": timestamp,
                    "latency_mode": self.config.latency_mode,
                    "comparison_data": self.comparison_data,
                    "protocol_ranking": self._rank_protocols(),
                    "hardware": _get_hardware_metadata(),
                },
                f,
                cls=EnumEncoder,
                indent=2,
            )

        print("\nResults exported to:")
        print(f"  {main_file}")
        print(f"  {summary_file}")

    def _export_csv_results(self):
        """Export results to CSV format for plotting tools."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        latency_mode_suffix = f"_{self.config.latency_mode}"

        latency_file = os.path.join(
            self.config.output_dir,
            f"latency_comparison{latency_mode_suffix}_{timestamp}.csv",
        )
        with open(latency_file, "w") as f:
            f.write(
                "Protocol,Scenario,LatencyMode,AvgLatency_ms,P95Latency_ms,\
                    P99Latency_ms\n"
            )
            for scenario, protocol_data in self.comparison_data.items():
                for protocol, metrics in protocol_data.items():
                    variant_label = protocol.replace("::", "/")
                    f.write(
                        f"{variant_label},{scenario},{self.config.latency_mode},"
                        f"{metrics.get('avg_latency_ms', 0):.2f},"
                        f"{metrics.get('p95_latency_ms', 0):.2f},"
                        f"{metrics.get('p99_latency_ms', 0):.2f}\n"
                    )

        # Throughput comparison CSV
        throughput_file = os.path.join(
            self.config.output_dir,
            f"throughput_comparison{latency_mode_suffix}_{timestamp}.csv",
        )
        with open(throughput_file, "w") as f:
            f.write(
                "Protocol,Scenario,LatencyMode,AvgThroughput_msg_per_sec,"
                "PeakThroughput_msg_per_sec,SuccessRate_percent\n"
            )
            for scenario, protocol_data in self.comparison_data.items():
                for protocol, metrics in protocol_data.items():
                    variant_label = protocol.replace("::", "/")
                    f.write(
                        f"{variant_label},{scenario},{self.config.latency_mode},"
                        f"{metrics.get('throughput_msg_per_sec', 0):.1f},"
                        f"{metrics.get('peak_throughput_msg_per_sec', 0):.1f},"
                        f"{metrics.get('success_rate_percent', 0):.1f}\n"
                    )

        resource_file = os.path.join(
            self.config.output_dir,
            f"resource_usage{latency_mode_suffix}_{timestamp}.csv",
        )
        with open(resource_file, "w") as f:
            f.write("Protocol,Scenario,LatencyMode,CPU_percent,Memory_mb\n")
            for scenario, protocol_data in self.comparison_data.items():
                for protocol, metrics in protocol_data.items():
                    variant_label = protocol.replace("::", "/")
                    f.write(
                        f"{variant_label},{scenario},{self.config.latency_mode},"
                        f"{metrics.get('cpu_usage_percent', 0):.1f},"
                        f"{metrics.get('memory_usage_mb', 0):.1f}\n"
                    )

        print(f"  {latency_file}")
        print(f"  {throughput_file}")
        print(f"  {resource_file}")

    def _rank_protocols(self) -> Dict[str, Dict[str, int]]:
        """Rank protocols by performance metrics."""
        rankings = {}

        for scenario, protocol_data in self.comparison_data.items():
            if not protocol_data:
                continue

            scenario_rankings = {}

            latency_ranking = sorted(
                protocol_data.items(),
                key=lambda x: x[1].get("avg_latency_ms", float("inf")),
            )
            for i, (protocol, _) in enumerate(latency_ranking):
                scenario_rankings[f"{protocol}_latency_rank"] = i + 1

            throughput_ranking = sorted(
                protocol_data.items(),
                key=lambda x: x[1].get("throughput_msg_per_sec", 0),
                reverse=True,
            )
            for i, (protocol, _) in enumerate(throughput_ranking):
                scenario_rankings[f"{protocol}_throughput_rank"] = i + 1

            success_ranking = sorted(
                protocol_data.items(),
                key=lambda x: x[1].get("success_rate_percent", 0),
                reverse=True,
            )
            for i, (protocol, _) in enumerate(success_ranking):
                scenario_rankings[f"{protocol}_reliability_rank"] = i + 1

            rankings[scenario] = scenario_rankings

        return rankings

    def _print_final_summary(self, total_time: float):
        """Print final benchmark summary."""
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        print(f"Total execution time: {total_time:.1f} seconds")
        print(f"Protocols tested: {len(self.config.protocols)}")
        print(f"Scenarios per protocol: {len(self.config.scenarios)}")

        if self.comparison_data:
            print("\nCROSS-PROTOCOL PERFORMANCE COMPARISON")
            print("-" * 50)

            for scenario, protocol_data in self.comparison_data.items():
                print(f"\n{scenario.replace('_', ' ').title()}:")

                if not protocol_data:
                    print("   No data available")
                    continue

                sorted_protocols = sorted(
                    protocol_data.items(),
                    key=lambda x: x[1].get("avg_latency_ms", float("inf")),
                )

                for protocol, metrics in sorted_protocols:
                    display_name = protocol.replace("::", " / ").upper()
                    print(
                        f"   {display_name:>12}: "
                        f"Latency {metrics.get('avg_latency_ms', 0):6.1f}ms | "
                        f"Throughput \
                            {metrics.get('throughput_msg_per_sec', 0):6.1f}\
                                msg/s | "
                        f"Success \
                            {metrics.get('success_rate_percent', 0):5.1f}%"
                    )

        print("\n" + "=" * 80)
        print("Benchmark completed successfully!")
        print("=" * 80)


def main():
    """Main entry point for the benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Unified Benchmark Runner - Protocols & Hierarchies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run both hierarchy and communication benchmarks (default)
  python benchmark_runner.py --simple

  # Run only communication protocol benchmarks
  python benchmark_runner.py --communication-only --simple

  # Run only hierarchy strategy benchmarks
  python benchmark_runner.py --hierarchy-only

  # Extensive protocol benchmarks for specific protocols
  python benchmark_runner.py --communication-only --extensive\
      --protocols rest grpc

  # Hierarchy benchmarks with specific strategies
  python benchmark_runner.py --hierarchy-only --hierarchy-types tree hybrid

  # Full hierarchy comparison with ablation study
  python benchmark_runner.py --hierarchy-only --extensive --hierarchy-ablation

  # Hierarchy-only extensive run via YAML (multi-trial + ablation)
  python benchmark_runner.py --config-file \
      src/benchmarks/benchmark_configs/hierarchy_extensive_ablation.yml

  # Both benchmarks with extensive mode
  python benchmark_runner.py --extensive
        """,
    )

    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=["rest", "grpc", "mqtt", "kafka"],
        default=["rest", "grpc", "mqtt", "kafka"],
        help="Protocols to benchmark (default: all)",
    )

    parser.add_argument(
        "--config-file",
        help="Path to YAML configuration describing benchmark matrices",
    )

    parser.add_argument(
        "--protocol-variants",
        nargs="+",
        help=(
            "Override protocol variants per protocol. "
            "Format: rest=http1,http2 grpc=unary,streaming mqtt=qos0,qos1 "
            "kafka=acks0,acks1,acksall. Overrides YAML definitions if set."
        ),
    )

    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=[
            "point_to_point_latency",
            "broadcast_throughput",
            "concurrent_messaging",
            "scalability_stress",
        ],
        default=[
            "point_to_point_latency",
            "broadcast_throughput",
            "concurrent_messaging",
            "scalability_stress",
        ],
        help="Scenarios to run (default: all)",
    )

    parser.add_argument(
        "--simple",
        action="store_true",
        help="Run simple benchmarks (quick, basic parameters)",
    )

    parser.add_argument(
        "--extensive",
        action="store_true",
        help="Run extensive benchmarks (comprehensive, multiple configs)",
    )

    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for results (default: results)",
    )

    parser.add_argument(
        "--no-csv", action="store_true", help="Skip CSV export"
    )

    parser.add_argument(
        "--no-json", action="store_true", help="Skip JSON export"
    )

    parser.add_argument(
        "--latency-mode",
        choices=["send_only", "end_to_end", "app_ack"],
        default="end_to_end",
        help=(
            "Latency measurement mode: "
            "'send_only' measures publish/send time only (fair for\
                async protocols), "
            "'end_to_end' waits for delivery confirmation (fair comparison\
                across all protocols), "
            "'app_ack' measures full application-level ACK round-trip including\
                receiver processing. "
            "Default: end_to_end"
        ),
    )

    parser.add_argument(
        "--num-trials",
        type=int,
        default=1,
        help="Number of trials per configuration point (default: 1)",
    )

    parser.add_argument(
        "--warm-up-ops",
        dest="warm_up_operations",
        type=int,
        default=0,
        help=(
            "Warm-up count per trial before measurement "
            "(communication: operations, hierarchy: episodes; default: 0)"
        ),
    )

    # Benchmark type selection arguments
    parser.add_argument(
        "--hierarchy-only",
        action="store_true",
        help="Run only hierarchy strategy benchmarks (skip\
            communication benchmarks)",
    )

    parser.add_argument(
        "--communication-only",
        action="store_true",
        help="Run only communication protocol benchmarks (skip \
            hierarchy benchmarks)",
    )

    parser.add_argument(
        "--hierarchy-types",
        nargs="+",
        choices=["tree", "peer_to_peer", "hybrid"],
        default=["tree", "peer_to_peer", "hybrid"],
        help="Hierarchy strategies to benchmark (default: all)",
    )

    parser.add_argument(
        "--hierarchy-environments",
        nargs="+",
        choices=[
            "task_distribution",
            "resource_allocation",
            "collaborative",
            "fault_recovery",
            "scalability",
        ],
        default=["task_distribution", "resource_allocation", "collaborative"],
        help=(
            "Hierarchy environments to test "
            "(default: task_distribution, resource_allocation, collaborative)"
        ),
    )

    parser.add_argument(
        "--hierarchy-episodes",
        type=int,
        default=10,
        help="Number of episodes per hierarchy benchmark (default: 10)",
    )

    parser.add_argument(
        "--hierarchy-agent-counts",
        nargs="+",
        type=int,
        help=(
            "Override hierarchy benchmark agent counts "
            "(defaults: simple=[5,8], extensive=[3,5,8,12])"
        ),
    )

    parser.add_argument(
        "--hierarchy-ablation",
        action="store_true",
        help="Run hierarchy ablation study",
    )

    default_protocols = parser.get_default("protocols")
    default_scenarios = parser.get_default("scenarios")
    default_latency_mode = parser.get_default("latency_mode")
    default_output_dir = parser.get_default("output_dir")
    default_num_trials = parser.get_default("num_trials")
    default_warm_up_operations = parser.get_default("warm_up_operations")
    default_hierarchy_only = parser.get_default("hierarchy_only")
    default_communication_only = parser.get_default("communication_only")
    default_hierarchy_types = parser.get_default("hierarchy_types")
    default_hierarchy_environments = parser.get_default(
        "hierarchy_environments"
    )
    default_hierarchy_episodes = parser.get_default("hierarchy_episodes")
    default_hierarchy_agent_counts = parser.get_default(
        "hierarchy_agent_counts"
    )
    default_hierarchy_ablation = parser.get_default("hierarchy_ablation")

    args = parser.parse_args()

    yaml_config: Dict[str, Any] = {}
    if args.config_file:
        yaml_config = load_config_from_yaml(args.config_file)

    def _coerce_bool(value: Any, field_name: str) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "y", "on"}:
                return True
            if lowered in {"0", "false", "no", "n", "off"}:
                return False
        raise ValueError(f"{field_name} must be a boolean")

    # YAML-driven configuration (defaults)
    protocols = yaml_config.get("protocols", args.protocols)
    if args.protocols != default_protocols:
        protocols = args.protocols
    elif isinstance(protocols, str):
        protocols = [protocols]
    protocols = [str(p).lower() for p in protocols]

    scenarios = yaml_config.get("scenarios", args.scenarios)
    if args.scenarios != default_scenarios:
        scenarios = args.scenarios
    elif isinstance(scenarios, str):
        scenarios = [scenarios]
    scenarios = [str(s) for s in scenarios]

    latency_mode = yaml_config.get("latency_mode", args.latency_mode)
    if args.latency_mode != default_latency_mode:
        latency_mode = args.latency_mode

    output_dir = yaml_config.get("output_dir", args.output_dir)
    if args.output_dir != default_output_dir:
        output_dir = args.output_dir

    num_trials = yaml_config.get("num_trials", args.num_trials)
    if args.num_trials != default_num_trials:
        num_trials = args.num_trials
    try:
        num_trials = int(num_trials)
    except (TypeError, ValueError) as exc:
        raise ValueError("num_trials must be an integer") from exc
    if num_trials < 1:
        raise ValueError("num_trials must be >= 1")

    warm_up_operations = yaml_config.get(
        "warm_up_operations", args.warm_up_operations
    )
    if args.warm_up_operations != default_warm_up_operations:
        warm_up_operations = args.warm_up_operations
    try:
        warm_up_operations = int(warm_up_operations)
    except (TypeError, ValueError) as exc:
        raise ValueError("warm_up_operations must be an integer") from exc
    if warm_up_operations < 0:
        raise ValueError("warm_up_operations must be >= 0")

    comm_agent_counts = yaml_config.get("agent_counts")
    if comm_agent_counts is not None and not isinstance(
        comm_agent_counts, list
    ):
        comm_agent_counts = [comm_agent_counts]
    if comm_agent_counts is not None:
        try:
            comm_agent_counts = [int(value) for value in comm_agent_counts]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "All agent_counts entries must be integers"
            ) from exc

    yaml_protocol_variants = yaml_config.get("protocol_variants", {})
    yaml_variant_settings = yaml_config.get("variant_settings", {})

    hierarchy_only = args.hierarchy_only
    if (
        args.hierarchy_only == default_hierarchy_only
        and "hierarchy_only" in yaml_config
    ):
        hierarchy_only = _coerce_bool(
            yaml_config["hierarchy_only"], "hierarchy_only"
        )

    communication_only = args.communication_only
    if (
        args.communication_only == default_communication_only
        and "communication_only" in yaml_config
    ):
        communication_only = _coerce_bool(
            yaml_config["communication_only"], "communication_only"
        )

    hierarchy_types_raw = yaml_config.get(
        "hierarchy_types", args.hierarchy_types
    )
    if args.hierarchy_types != default_hierarchy_types:
        hierarchy_types_raw = args.hierarchy_types
    elif isinstance(hierarchy_types_raw, str):
        hierarchy_types_raw = [hierarchy_types_raw]
    hierarchy_types_raw = [str(ht) for ht in hierarchy_types_raw]

    hierarchy_environments = yaml_config.get(
        "hierarchy_environments", args.hierarchy_environments
    )
    if args.hierarchy_environments != default_hierarchy_environments:
        hierarchy_environments = args.hierarchy_environments
    elif isinstance(hierarchy_environments, str):
        hierarchy_environments = [hierarchy_environments]
    hierarchy_environments = [str(env) for env in hierarchy_environments]

    hierarchy_episodes = yaml_config.get(
        "hierarchy_episodes", args.hierarchy_episodes
    )
    if args.hierarchy_episodes != default_hierarchy_episodes:
        hierarchy_episodes = args.hierarchy_episodes
    try:
        hierarchy_episodes = int(hierarchy_episodes)
    except (TypeError, ValueError) as exc:
        raise ValueError("hierarchy_episodes must be an integer") from exc
    if hierarchy_episodes < 1:
        raise ValueError("hierarchy_episodes must be >= 1")

    hierarchy_agent_counts = yaml_config.get("hierarchy_agent_counts")
    if (
        args.hierarchy_agent_counts != default_hierarchy_agent_counts
        and args.hierarchy_agent_counts is not None
    ):
        hierarchy_agent_counts = args.hierarchy_agent_counts
    if hierarchy_agent_counts is not None and not isinstance(
        hierarchy_agent_counts, list
    ):
        hierarchy_agent_counts = [hierarchy_agent_counts]
    if hierarchy_agent_counts is not None:
        try:
            hierarchy_agent_counts = [
                int(value) for value in hierarchy_agent_counts
            ]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "All hierarchy_agent_counts entries must be integers"
            ) from exc
        if any(count < 1 for count in hierarchy_agent_counts):
            raise ValueError("All hierarchy_agent_counts entries must be >= 1")

    hierarchy_ablation = args.hierarchy_ablation
    if (
        args.hierarchy_ablation == default_hierarchy_ablation
        and "hierarchy_ablation" in yaml_config
    ):
        hierarchy_ablation = _coerce_bool(
            yaml_config["hierarchy_ablation"], "hierarchy_ablation"
        )

    hierarchy_ablation_params = copy.deepcopy(
        yaml_config.get("hierarchy_ablation_params", {})
    )
    hierarchy_ablation_base_configs = copy.deepcopy(
        yaml_config.get("hierarchy_ablation_base_configs", [])
    )

    # Validate conflicting flags
    if hierarchy_only and communication_only:
        parser.error(
            "Cannot specify both --hierarchy-only and --communication-only"
        )

    # Set default mode if none specified
    simple_mode = yaml_config.get("simple_mode")
    extensive_mode = yaml_config.get("extensive_mode")

    if args.simple:
        simple_mode = True
        extensive_mode = False
    elif args.extensive:
        extensive_mode = True
        simple_mode = False

    if simple_mode is None and extensive_mode is None:
        simple_mode = True
        extensive_mode = False
    elif simple_mode is None:
        simple_mode = not extensive_mode
    elif extensive_mode is None:
        extensive_mode = not simple_mode

    # Parse protocol variant overrides
    protocol_variants = copy.deepcopy(yaml_protocol_variants)
    variant_settings = copy.deepcopy(yaml_variant_settings)

    variant_overrides: Dict[str, List[str]] = {}
    if args.protocol_variants:
        for entry in args.protocol_variants:
            if "=" not in entry:
                print(
                    f"Warning: ignoring invalid protocol-variant entry '{entry}'"
                )
                continue
            proto_key, variants_str = entry.split("=", 1)
            proto_key = proto_key.strip().lower()
            if not variants_str:
                continue
            variants = [
                v.strip() for v in variants_str.split(",") if v.strip()
            ]
            if variants:
                variant_overrides[proto_key] = variants

    if variant_overrides:
        protocol_variants = protocol_variants or {}
        for proto_key, variants in variant_overrides.items():
            protocol_variants[proto_key] = variants

    if protocol_variants:
        protocol_variants = {
            str(proto).lower(): variants
            for proto, variants in protocol_variants.items()
        }
    if variant_settings:
        variant_settings = {
            str(proto).lower(): settings
            for proto, settings in variant_settings.items()
        }

    export_csv = not args.no_csv
    if not args.no_csv and "export_csv" in yaml_config:
        export_csv = bool(yaml_config["export_csv"])

    export_json = not args.no_json
    if not args.no_json and "export_json" in yaml_config:
        export_json = bool(yaml_config["export_json"])

    # Run hierarchy benchmarks unless --communication-only is specified
    if not communication_only:
        hierarchy_type_map = {
            "tree": HierarchyType.TREE,
            "peer_to_peer": HierarchyType.PEER_TO_PEER,
            "hybrid": HierarchyType.HYBRID,
        }
        hierarchy_types = [
            hierarchy_type_map[ht] for ht in hierarchy_types_raw
        ]

        hierarchy_runner = HierarchyComparisonBenchmark(output_dir=output_dir)

        if hierarchy_agent_counts is None:
            if extensive_mode:
                hierarchy_agent_counts = [3, 5, 8, 12]
            else:
                hierarchy_agent_counts = [5, 8]

        hierarchy_runner.run_comparison(
            hierarchy_types=hierarchy_types,
            environment_types=hierarchy_environments,
            agent_counts=hierarchy_agent_counts,
            num_episodes=hierarchy_episodes,
            num_trials=num_trials,
            warm_up_operations=warm_up_operations,
        )

        # Run ablation study if requested
        if hierarchy_ablation:
            from benchmarks.local.hierarchy_benchmarks.hierarchy_benchmark_scenarios import (
                BenchmarkConfiguration as HierarchyConfig,
            )

            default_ablation_params: Dict[str, List[Any]] = {
                "hierarchy_depth": [1, 2, 3],
                "planning_frequency": [1, 5, 10],
                "communication_limit": [None, 50, 100],
            }
            raw_ablation_params = (
                hierarchy_ablation_params
                if hierarchy_ablation_params
                else default_ablation_params
            )

            ablation_params: Dict[str, List[Any]] = {}
            for key, values in raw_ablation_params.items():
                if isinstance(values, list):
                    ablation_params[str(key)] = values
                else:
                    ablation_params[str(key)] = [values]

            base_cfg_map: Dict[str, Dict[str, Any]] = {}
            for entry in hierarchy_ablation_base_configs:
                if not isinstance(entry, dict):
                    continue
                entry_type = str(entry.get("hierarchy_type", "")).strip()
                if entry_type:
                    base_cfg_map[entry_type] = entry

            default_ablation_agents = (
                hierarchy_agent_counts[-1]
                if hierarchy_agent_counts
                else (8 if extensive_mode else 5)
            )
            default_ablation_episodes = max(5, min(hierarchy_episodes, 10))

            for hierarchy_type in hierarchy_types:
                strategy_key = hierarchy_type.value
                strategy_base = base_cfg_map.get(strategy_key, {})

                num_agents = int(
                    strategy_base.get("num_agents", default_ablation_agents)
                )
                environment_type = str(
                    strategy_base.get("environment_type", "task_distribution")
                )
                num_episodes_for_ablation = int(
                    strategy_base.get(
                        "num_episodes", default_ablation_episodes
                    )
                )
                num_tasks_per_episode = int(
                    strategy_base.get("num_tasks_per_episode", 5)
                )
                max_steps = int(strategy_base.get("max_steps", 100))
                hierarchy_depth = int(strategy_base.get("hierarchy_depth", 2))
                planning_frequency = int(
                    strategy_base.get("planning_frequency", 1)
                )
                communication_limit = strategy_base.get("communication_limit")
                if communication_limit is not None:
                    communication_limit = int(communication_limit)
                static_roles = bool(strategy_base.get("static_roles", True))
                env_params = strategy_base.get("env_params", {}) or {}
                if not isinstance(env_params, dict):
                    raise ValueError(
                        "env_params in hierarchy_ablation_base_configs "
                        f"for '{strategy_key}' must be a mapping"
                    )

                base_config = HierarchyConfig(
                    hierarchy_type=hierarchy_type,
                    num_agents=num_agents,
                    environment_type=environment_type,
                    num_episodes=num_episodes_for_ablation,
                    num_tasks_per_episode=num_tasks_per_episode,
                    max_steps=max_steps,
                    hierarchy_depth=hierarchy_depth,
                    static_roles=static_roles,
                    communication_limit=communication_limit,
                    planning_frequency=planning_frequency,
                    env_params=env_params,
                )

                hierarchy_runner.run_ablation_study(
                    base_config=base_config,
                    ablation_params=ablation_params,
                    num_trials=num_trials,
                    warm_up_operations=warm_up_operations,
                    output_suffix=strategy_key,
                )

        print("\nHierarchy benchmarks completed!")

    # Run communication benchmarks unless --hierarchy-only is specified
    if not hierarchy_only:
        # Create configuration for protocol benchmarks
        config = BenchmarkConfig(
            protocols=protocols,
            scenarios=scenarios,
            simple_mode=simple_mode,
            extensive_mode=extensive_mode,
            output_dir=output_dir,
            export_csv=export_csv,
            export_json=export_json,
            latency_mode=latency_mode,
            num_trials=num_trials,
            warm_up_operations=warm_up_operations,
            agent_counts=(
                comm_agent_counts
                if comm_agent_counts is not None
                else [5, 10, 15, 20]
            ),
            protocol_variants=protocol_variants if protocol_variants else None,
            variant_settings=variant_settings if variant_settings else None,
        )

        # Run protocol benchmarks
        runner = ProtocolBenchmarkRunner(config)
        results = runner.run_all_benchmarks()

        print("\nCommunication benchmarks completed!")
        return results

    return None


if __name__ == "__main__":
    main()
