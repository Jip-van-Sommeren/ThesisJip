#!/usr/bin/env python3
"""
Test Script for Benchmark Consistency
Verifies that all communication protocols have consistent benchmark
implementations
and can be executed without errors.
"""

import sys
import os
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import benchmark scenarios
try:
    from benchmarks.communication_benchmarks.rest_benchmark_scenarios import (
        create_rest_benchmark_scenarios as create_rest_scenarios,
    )
    from benchmarks.communication_benchmarks.grpc_benchmark_scenarios import (
        create_grpc_benchmark_scenarios,
    )
    from benchmarks.communication_benchmarks.mqtt_benchmark_scenarios import (
        create_mqtt_benchmark_scenarios,
    )
    from benchmarks.communication_benchmarks.kafka_benchmark_scenarios import (
        create_kafka_benchmark_scenarios,
    )
    from benchmarks.communication.communication_config import TopologyPattern
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running from the correct directory")
    sys.exit(1)


class BenchmarkConsistencyTester:
    """Tests consistency across all communication protocol benchmarks."""

    def __init__(self):
        self.protocols = {
            "REST": create_rest_scenarios,
            "gRPC": create_grpc_benchmark_scenarios,
            "MQTT": create_mqtt_benchmark_scenarios,
            "Kafka": create_kafka_benchmark_scenarios,
        }

        self.expected_scenarios = [
            "point_to_point_latency",
            "broadcast_throughput",
            "concurrent_messaging",
            "scalability_stress",
        ]

        self.test_results = {}

    def test_scenario_consistency(self) -> Dict[str, List[str]]:
        """Test that all protocols implement the same scenarios."""
        print("🔍 Testing scenario consistency across protocols...")

        results = {}

        for protocol_name, factory_func in self.protocols.items():
            try:
                benchmark = factory_func()
                available_scenarios = list(benchmark.scenarios.keys())
                results[protocol_name] = available_scenarios

                missing_scenarios = set(self.expected_scenarios) - set(
                    available_scenarios
                )
                extra_scenarios = set(available_scenarios) - set(
                    self.expected_scenarios
                )

                print(
                    f"  {protocol_name:>6}: {len(available_scenarios)}\
                        scenarios"
                )
                if missing_scenarios:
                    print(f"    Missing: {list(missing_scenarios)}")
                if extra_scenarios:
                    print(f"    Extra: {list(extra_scenarios)}")
                if not missing_scenarios and not extra_scenarios:
                    print("    All expected scenarios present")

            except Exception as e:
                print(f"  {protocol_name:>6}: Error creating benchmark: {e}")
                results[protocol_name] = []

        return results

    def test_scenario_parameters(self) -> Dict[str, Dict[str, bool]]:
        """Test that scenario setup functions accept standard parameters."""
        print("\n🔧 Testing scenario parameter consistency...")

        results = {}
        test_params = {
            "agent_count": 3,
            "topology_pattern": TopologyPattern.FULLY_CONNECTED,
            "message_count": 10,
            "messages_per_agent": 5,
            "stress_duration": 1.0,
        }

        for protocol_name, factory_func in self.protocols.items():
            protocol_results = {}
            print(f"  Testing {protocol_name} scenarios:")

            try:
                benchmark = factory_func()

                for scenario_name in self.expected_scenarios:
                    try:
                        if scenario_name in benchmark.scenarios:
                            scenario = benchmark.scenarios[scenario_name]

                            # Test if setup function can handle parameters
                            if scenario.setup_func:
                                # This should not raise an exception with
                                # standard params
                                setup_result = scenario.setup_func(
                                    test_params.copy()
                                )
                                protocol_results[scenario_name] = True
                                print(f"    {scenario_name}")

                                # Clean up if teardown function exists
                                if scenario.teardown_func and setup_result:
                                    try:
                                        scenario.teardown_func(setup_result)
                                    except Exception:
                                        pass
                            else:
                                protocol_results[scenario_name] = False
                                print(
                                    f"    {scenario_name}: No setup function"
                                )
                        else:
                            protocol_results[scenario_name] = False
                            print(f"    {scenario_name}: Not implemented")

                    except Exception as e:
                        protocol_results[scenario_name] = False
                        print(f"    {scenario_name}: Error - {str(e)[:50]}...")

            except Exception as e:
                print(f"    Failed to create benchmark: {e}")
                for scenario_name in self.expected_scenarios:
                    protocol_results[scenario_name] = False

            results[protocol_name] = protocol_results

        return results

    def test_benchmark_framework_integration(self) -> Dict[str, bool]:
        """Test that all protocols integrate properly with the benchmark
        framework."""
        print("\nTesting benchmark framework integration...")

        results = {}

        for protocol_name, factory_func in self.protocols.items():
            try:
                benchmark = factory_func()

                # Check if benchmark has required attributes/methods
                required_attributes = [
                    "scenarios",
                    "run_scenario",
                    "latency_tracker",
                    "throughput_tracker",
                ]
                has_all_attributes = all(
                    hasattr(benchmark, attr) for attr in required_attributes
                )

                if has_all_attributes:
                    print(f"  {protocol_name}: Framework integration OK")
                    results[protocol_name] = True
                else:
                    missing_attrs = [
                        attr
                        for attr in required_attributes
                        if not hasattr(benchmark, attr)
                    ]
                    print(
                        f"  {protocol_name}: Missing attributes:\
                            {missing_attrs}"
                    )
                    results[protocol_name] = False

            except Exception as e:
                print(f"  {protocol_name}: Integration error - {e}")
                results[protocol_name] = False

        return results

    def test_metric_consistency(self) -> Dict[str, Dict[str, bool]]:
        """Test that all protocols return consistent metrics structure."""
        print("\nTesting metric structure consistency...")

        results = {}
        _ = [
            "message_latency_avg",
            "message_latency_p95",
            "message_latency_p99",
            "throughput_avg",
            "success_rate",
            "cpu_usage_avg",
            "memory_usage_avg",
            "total_messages",
            "test_duration",
        ]

        for protocol_name, factory_func in self.protocols.items():
            protocol_results = {}
            print(f"  Testing {protocol_name} metrics:")

            try:
                benchmark = factory_func()

                # Try to get a dummy result structure
                # We can't run full tests here, so we check if the trackers
                # exist
                if hasattr(benchmark, "latency_tracker") and hasattr(
                    benchmark, "throughput_tracker"
                ):
                    # Check if trackers have expected methods
                    latency_methods = [
                        "start_message_timing",
                        "end_message_timing",
                        "get_latency_stats",
                    ]
                    throughput_methods = [
                        "record_message",
                        "get_current_throughput",
                    ]

                    has_latency_methods = all(
                        hasattr(benchmark.latency_tracker, method)
                        for method in latency_methods
                    )
                    has_throughput_methods = all(
                        hasattr(benchmark.throughput_tracker, method)
                        for method in throughput_methods
                    )

                    if has_latency_methods and has_throughput_methods:
                        print("    Metric tracking structure OK")
                        protocol_results["metric_structure"] = True
                    else:
                        print("    Missing metric tracking methods")
                        protocol_results["metric_structure"] = False
                else:
                    print("    Missing metric trackers")
                    protocol_results["metric_structure"] = False

            except Exception as e:
                print(f"    Error testing metrics: {e}")
                protocol_results["metric_structure"] = False

            results[protocol_name] = protocol_results

        return results

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all consistency tests."""
        print("Starting benchmark consistency tests...")
        print("=" * 60)

        all_results = {
            "scenario_consistency": self.test_scenario_consistency(),
            "parameter_consistency": self.test_scenario_parameters(),
            "framework_integration": self.test_benchmark_framework_integration(),
            "metric_consistency": self.test_metric_consistency(),
        }

        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        # Calculate overall success rates
        total_protocols = len(self.protocols)

        # Scenario consistency
        scenario_success = sum(
            1
            for p, scenarios in all_results["scenario_consistency"].items()
            if set(scenarios) == set(self.expected_scenarios)
        )
        print(
            f"Scenario Consistency:    {scenario_success}/{total_protocols}\
                protocols"
        )

        # Parameter consistency
        param_success = sum(
            1
            for p, scenarios in all_results["parameter_consistency"].items()
            if all(scenarios.values())
        )
        print(
            f"Parameter Consistency:   {param_success}/{total_protocols}\
                protocols"
        )

        # Framework integration
        framework_success = sum(all_results["framework_integration"].values())
        print(
            f"Framework Integration:   {framework_success}/{total_protocols}\
                protocols"
        )

        # Metric consistency
        metric_success = sum(
            1
            for p, metrics in all_results["metric_consistency"].items()
            if all(metrics.values())
        )
        print(
            f"Metric Consistency:      {metric_success}/{total_protocols}\
                protocols"
        )

        overall_success_rate = (
            (
                scenario_success
                + param_success
                + framework_success
                + metric_success
            )
            / (4 * total_protocols)
            * 100
        )

        print(f"\nOverall Success Rate:    {overall_success_rate:.1f}%")

        if overall_success_rate >= 90:
            print("Benchmarks are highly consistent across protocols.")
        elif overall_success_rate >= 75:
            print("Minor inconsistencies that should be addressed.")
        elif overall_success_rate >= 50:
            print("Warning! Significant inconsistencies detected.")
        else:
            print("Error! Major inconsistencies need immediate attention.")

        return all_results

    def generate_test_report(
        self,
        results: Dict[str, Any],
        output_file: str = "benchmark_consistency_report.txt",
    ):
        """Generate a detailed test report."""
        with open(output_file, "w") as f:
            f.write("BENCHMARK CONSISTENCY TEST REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write("PROTOCOLS TESTED:\n")
            for protocol in self.protocols.keys():
                f.write(f"  • {protocol}\n")
            f.write("\n")

            f.write("EXPECTED SCENARIOS:\n")
            for scenario in self.expected_scenarios:
                f.write(f"  • {scenario}\n")
            f.write("\n")

            # Detailed results for each test
            for test_name, test_results in results.items():
                f.write(f"{test_name.upper().replace('_', ' ')}:\n")
                f.write("-" * 30 + "\n")

                if test_name == "scenario_consistency":
                    for protocol, scenarios in test_results.items():
                        missing = set(self.expected_scenarios) - set(scenarios)
                        extra = set(scenarios) - set(self.expected_scenarios)
                        f.write(f"  {protocol}:\n")
                        f.write(f"    Scenarios: {len(scenarios)}\n")
                        if missing:
                            f.write(f"    Missing: {list(missing)}\n")
                        if extra:
                            f.write(f"    Extra: {list(extra)}\n")
                        if not missing and not extra:
                            f.write("    Status: ✓ All scenarios present\n")
                        f.write("\n")

                elif test_name == "parameter_consistency":
                    for protocol, scenarios in test_results.items():
                        f.write(f"  {protocol}:\n")
                        for scenario, success in scenarios.items():
                            status = "✓" if success else "✗"
                            f.write(f"    {scenario}: {status}\n")
                        f.write("\n")

                else:
                    for protocol, result in test_results.items():
                        status = "✓" if result else "✗"
                        f.write(f"  {protocol}: {status}\n")
                    f.write("\n")

        print(f"\nDetailed test report saved to: {output_file}")


def main():
    """Main entry point for consistency testing."""
    tester = BenchmarkConsistencyTester()
    results = tester.run_all_tests()
    tester.generate_test_report(results)


if __name__ == "__main__":
    main()
