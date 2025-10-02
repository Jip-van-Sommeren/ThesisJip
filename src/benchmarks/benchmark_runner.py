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
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from enum import Enum
from communication.communication_config import TopologyPattern


# Import protocol-specific benchmark scenarios
from benchmarks.rest_benchmark_scenarios import create_rest_benchmark_scenarios
from benchmarks.grpc_benchmark_scenarios import create_grpc_benchmark_scenarios
from benchmarks.mqtt_benchmark_scenarios import (
    create_mqtt_benchmark_scenarios,
)
from benchmarks.kafka_benchmark_scenarios import (
    create_kafka_benchmark_scenarios,
)
from benchmarks.hierarchy_benchmark_scenarios import (
    HierarchyComparisonBenchmark,
)
from benchmarks.hierarchy_strategies import HierarchyType


class EnumEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles Enum types."""

    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


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

        # Protocol factory mapping
        self.protocol_factories = {
            "rest": create_rest_benchmark_scenarios,
            "grpc": create_grpc_benchmark_scenarios,
            "mqtt": create_mqtt_benchmark_scenarios,
            "kafka": create_kafka_benchmark_scenarios,
        }

        # Ensure output directory exists
        os.makedirs(self.config.output_dir, exist_ok=True)

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

        # Check and start required brokers
        # self._ensure_brokers_running()

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

        # Generate comparison data
        self._generate_comparison_data()

        # Export results
        if self.config.export_json:
            self._export_json_results(total_time)
        if self.config.export_csv:
            self._export_csv_results()

        # Print summary
        self._print_final_summary(total_time)

        return self.results

    def _run_protocol_benchmarks(self, protocol: str) -> Dict[str, Any]:
        """Run benchmarks for a specific protocol."""
        benchmark = self.protocol_factories[protocol]()
        protocol_results = {
            "scenarios": {},
            "topology_comparison": {},
            "scalability_analysis": {},
            "metadata": {
                "protocol": protocol,
                "timestamp": time.time(),
                "config": asdict(self.config),
            },
        }

        # Run individual scenarios
        for scenario in self.config.scenarios:
            print(f"Running {scenario} scenario...")

            if self.config.simple_mode:
                # Simple mode: basic parameters
                params = self._get_simple_params(scenario)
                result = benchmark.run_scenario(scenario, **params)
                protocol_results["scenarios"][scenario] = (
                    self._extract_metrics(result)
                )

            elif self.config.extensive_mode:
                # Extensive mode: multiple configurations
                scenario_results = {}

                # Test different agent counts
                for agent_count in self.config.agent_counts:
                    print(f"Testing with {agent_count} agents...")
                    params = self._get_extensive_params(
                        scenario, agent_count=agent_count
                    )
                    result = benchmark.run_scenario(scenario, **params)
                    scenario_results[f"{agent_count}_agents"] = (
                        self._extract_metrics(result)
                    )

                protocol_results["scenarios"][scenario] = scenario_results

        # Run topology comparison (extensive mode only)
        if self.config.extensive_mode:
            print("Running topology comparison...")
            topology_results = {}
            for topology in self.config.topologies:
                print(f"Testing {topology.value} topology...")
                result = benchmark.run_scenario(
                    "concurrent_messaging",
                    agent_count=20,
                    topology_pattern=topology,
                    messages_per_agent=50,
                )
                topology_results[topology.value] = self._extract_metrics(
                    result
                )
            protocol_results["topology_comparison"] = topology_results

        return protocol_results

    def _get_simple_params(self, scenario: str) -> Dict[str, Any]:
        """Get simple benchmark parameters."""
        base_params = {
            "agent_count": 5,
            "topology_pattern": TopologyPattern.FULLY_CONNECTED,
        }

        scenario_params = {
            "point_to_point_latency": {"agent_count": 2, "message_count": 50},
            "broadcast_throughput": {"agent_count": 5, "message_count": 30},
            "concurrent_messaging": {
                "agent_count": 4,
                "messages_per_agent": 10,
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
        }

        scenario_params = {
            "point_to_point_latency": {"message_count": 100},
            "broadcast_throughput": {"message_count": 50},
            "concurrent_messaging": {"messages_per_agent": 20},
            "scalability_stress": {"stress_duration": 5.0},
        }

        base_params.update(scenario_params.get(scenario, {}))
        base_params.update(overrides)
        return base_params

    def _extract_metrics(self, result) -> Dict[str, float]:
        """Extract key metrics from benchmark result."""
        if result is None:
            return {}

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
        }

    def _generate_comparison_data(self):
        """Generate cross-protocol comparison data."""
        for scenario in self.config.scenarios:
            scenario_comparison = {}

            for protocol, protocol_results in self.results.items():
                if "error" in protocol_results:
                    continue

                if self.config.simple_mode:
                    metrics = protocol_results.get("scenarios", {}).get(
                        scenario, {}
                    )
                    scenario_comparison[protocol] = metrics
                else:
                    # For extensive mode, use average across agent counts
                    scenario_data = protocol_results.get("scenarios", {}).get(
                        scenario, {}
                    )
                    if scenario_data:
                        avg_metrics = self._average_metrics(
                            scenario_data.values()
                        )
                        scenario_comparison[protocol] = avg_metrics

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
            values = [m.get(key, 0) for m in metrics_list if key in m]
            avg_metrics[key] = sum(values) / len(values) if values else 0

        return avg_metrics

    def _export_json_results(self, total_time: float):
        """Export results to JSON format."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        export_data = {
            "benchmark_metadata": {
                "timestamp": timestamp,
                "total_duration_sec": total_time,
                "mode": (
                    "extensive" if self.config.extensive_mode else "simple"
                ),
                "protocols_tested": self.config.protocols,
                "scenarios_tested": self.config.scenarios,
            },
            "protocol_results": self.results,
            "cross_protocol_comparison": self.comparison_data,
            "configuration": asdict(self.config),
        }

        # Main results file
        main_file = os.path.join(
            self.config.output_dir, f"benchmark_results_{timestamp}.json"
        )
        with open(main_file, "w") as f:
            json.dump(export_data, f, indent=2, cls=EnumEncoder)

        # Summary comparison file
        summary_file = os.path.join(
            self.config.output_dir, f"comparison_summary_{timestamp}.json"
        )
        with open(summary_file, "w") as f:
            json.dump(
                {
                    "timestamp": timestamp,
                    "comparison_data": self.comparison_data,
                    "protocol_ranking": self._rank_protocols(),
                },
                f,
                cls=EnumEncoder,
                indent=2,
            )

        print("\n Results exported to:")
        print(f"{main_file}")
        print(f"{summary_file}")

    def _export_csv_results(self):
        """Export results to CSV format for plotting tools."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Latency comparison CSV
        latency_file = os.path.join(
            self.config.output_dir, f"latency_comparison_{timestamp}.csv"
        )
        with open(latency_file, "w") as f:
            f.write(
                "Protocol,Scenario,AvgLatency_ms,P95Latency_ms,P99Latency_ms\n"
            )
            for scenario, protocol_data in self.comparison_data.items():
                for protocol, metrics in protocol_data.items():
                    f.write(
                        f"{protocol},{scenario},\
                            {metrics.get('avg_latency_ms', 0):.2f},"
                        f"{metrics.get('p95_latency_ms', 0):.2f},\
                            {metrics.get('p99_latency_ms', 0):.2f}\n"
                    )

        # Throughput comparison CSV
        throughput_file = os.path.join(
            self.config.output_dir, f"throughput_comparison_{timestamp}.csv"
        )
        with open(throughput_file, "w") as f:
            f.write(
                "Protocol,Scenario,AvgThroughput_msg_per_sec,\
                    PeakThroughput_msg_per_sec,SuccessRate_percent\n"
            )
            for scenario, protocol_data in self.comparison_data.items():
                for protocol, metrics in protocol_data.items():
                    f.write(
                        f"{protocol},{scenario},\
                            {metrics.get('throughput_msg_per_sec', 0):.1f},"
                        f"{metrics.get('peak_throughput_msg_per_sec', 0):.1f}\
                            ,{metrics.get('success_rate_percent', 0):.1f}\n"
                    )

        # Resource usage CSV
        resource_file = os.path.join(
            self.config.output_dir, f"resource_usage_{timestamp}.csv"
        )
        with open(resource_file, "w") as f:
            f.write("Protocol,Scenario,CPU_percent,Memory_mb\n")
            for scenario, protocol_data in self.comparison_data.items():
                for protocol, metrics in protocol_data.items():
                    f.write(
                        f"{protocol},{scenario},\
                            {metrics.get('cpu_usage_percent', 0):.1f},"
                        f"{metrics.get('memory_usage_mb', 0):.1f}\n"
                    )

        print(f"{latency_file}")
        print(f"{throughput_file}")
        print(f"{resource_file}")

    def _rank_protocols(self) -> Dict[str, Dict[str, int]]:
        """Rank protocols by performance metrics."""
        rankings = {}

        for scenario, protocol_data in self.comparison_data.items():
            if not protocol_data:
                continue

            scenario_rankings = {}

            # Rank by latency (lower is better)
            latency_ranking = sorted(
                protocol_data.items(),
                key=lambda x: x[1].get("avg_latency_ms", float("inf")),
            )
            for i, (protocol, _) in enumerate(latency_ranking):
                scenario_rankings[f"{protocol}_latency_rank"] = i + 1

            # Rank by throughput (higher is better)
            throughput_ranking = sorted(
                protocol_data.items(),
                key=lambda x: x[1].get("throughput_msg_per_sec", 0),
                reverse=True,
            )
            for i, (protocol, _) in enumerate(throughput_ranking):
                scenario_rankings[f"{protocol}_throughput_rank"] = i + 1

            # Rank by success rate (higher is better)
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

                # Sort by average latency for display
                sorted_protocols = sorted(
                    protocol_data.items(),
                    key=lambda x: x[1].get("avg_latency_ms", float("inf")),
                )

                for protocol, metrics in sorted_protocols:
                    print(
                        f"   {protocol.upper():>8}: "
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
  # Simple protocol benchmarks
  python benchmark_runner.py --simple

  # Extensive protocol benchmarks
  python benchmark_runner.py --extensive --protocols rest grpc

  # Hierarchy strategy benchmarks (quick test)
  python benchmark_runner.py --hierarchy

  # Hierarchy benchmarks with specific strategies
  python benchmark_runner.py --hierarchy --hierarchy-types tree hybrid

  # Full hierarchy comparison with ablation study
  python benchmark_runner.py --hierarchy --extensive --hierarchy-ablation

  # Hierarchy benchmarks for specific environments
  python benchmark_runner.py --hierarchy \\
      --hierarchy-environments task_distribution fault_recovery
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

    # Hierarchy benchmark arguments
    parser.add_argument(
        "--hierarchy",
        action="store_true",
        help="Run hierarchy strategy benchmarks",
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
        "--hierarchy-ablation",
        action="store_true",
        help="Run hierarchy ablation study",
    )

    args = parser.parse_args()

    # Default to simple if neither mode specified
    if not args.simple and not args.extensive and not args.hierarchy:
        args.simple = True

    # Run hierarchy benchmarks if requested
    if args.hierarchy:
        # Map string hierarchy types to enum
        hierarchy_type_map = {
            "tree": HierarchyType.TREE,
            "peer_to_peer": HierarchyType.PEER_TO_PEER,
            "hybrid": HierarchyType.HYBRID,
        }
        hierarchy_types = [
            hierarchy_type_map[ht] for ht in args.hierarchy_types
        ]

        # Create hierarchy benchmark runner
        hierarchy_runner = HierarchyComparisonBenchmark(
            output_dir=os.path.join(args.output_dir, "hierarchy")
        )

        # Determine agent counts
        if args.extensive:
            agent_counts = [3, 5, 8, 12]
        else:
            agent_counts = [5, 8]

        # Run hierarchy comparison
        hierarchy_runner.run_comparison(
            hierarchy_types=hierarchy_types,
            environment_types=args.hierarchy_environments,
            agent_counts=agent_counts,
            num_episodes=args.hierarchy_episodes,
        )

        # Run ablation study if requested
        if args.hierarchy_ablation:
            from benchmarks.hierarchy_benchmark_scenarios import (
                BenchmarkConfiguration as HierarchyConfig,
            )

            tree_config = HierarchyConfig(
                hierarchy_type=HierarchyType.TREE,
                num_agents=8,
                environment_type="task_distribution",
                num_episodes=5,
            )

            hierarchy_runner.run_ablation_study(
                base_config=tree_config,
                ablation_params={
                    "hierarchy_depth": [1, 2, 3],
                    "planning_frequency": [1, 5, 10],
                    "communication_limit": [None, 50, 100],
                },
            )

        print("\nHierarchy benchmarks completed!")
        return None

    # Create configuration for protocol benchmarks
    config = BenchmarkConfig(
        protocols=args.protocols,
        scenarios=args.scenarios,
        simple_mode=args.simple,
        extensive_mode=args.extensive,
        output_dir=args.output_dir,
        export_csv=not args.no_csv,
        export_json=not args.no_json,
        agent_counts=[5, 10, 15, 20],
    )

    # Run protocol benchmarks
    runner = ProtocolBenchmarkRunner(config)
    results = runner.run_all_benchmarks()
    return results


if __name__ == "__main__":
    main()
