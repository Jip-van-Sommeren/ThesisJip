#!/usr/bin/env python3
"""
Quick test script to verify hierarchy benchmark metrics are working correctly.
"""

from src.benchmarks.hierarchy_benchmark_scenarios import (
    HierarchyBenchmarkScenario,
    BenchmarkConfiguration,
)
from src.benchmarks.hierarchy_strategies import HierarchyType


def test_metrics():
    """Test that metrics are being tracked properly."""

    print("Testing hierarchy benchmark metrics...\n")

    # Create a simple test config
    config = BenchmarkConfiguration(
        hierarchy_type=HierarchyType.TREE,
        num_agents=5,
        environment_type="task_distribution",
        num_episodes=3,
        num_tasks_per_episode=3,
        max_steps=50,
    )

    # Run benchmark
    scenario = HierarchyBenchmarkScenario(config)
    result = scenario.run_benchmark()

    # Check metrics
    metrics = result.metrics

    print("\n" + "=" * 60)
    print("METRICS CHECK")
    print("=" * 60)

    print("\nHierarchy Overhead Metrics:")
    print(f"  Manager Utilization: {metrics.manager_utilization:.2f}")
    print(f"  Delegation Success Rate: {metrics.delegation_success_rate:.2%}")
    print(f"  Preemption Rate: {metrics.preemption_rate:.3f}")

    print("\nCommunication Metrics:")
    print(f"  Messages/Episode: {metrics.messages_per_episode:.1f}")
    print(
        f"  Coordination Latency:\
        {metrics.coordination_latency_mean*1000:.2f}ms"
    )

    # Validation
    issues = []

    if metrics.manager_utilization == 0:
        issues.append("Manager utilization is 0")

    if metrics.delegation_success_rate == 0:
        issues.append("Delegation success rate is 0")

    if metrics.coordination_latency_mean == 0:
        issues.append("Coordination latency is 0")

    if metrics.messages_per_episode == 0:
        issues.append("Messages per episode is 0")

    if issues:
        print("\n Issues found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n All metrics are being tracked correctly!")

    print("\n" + "=" * 60)

    return len(issues) == 0


if __name__ == "__main__":
    success = test_metrics()
    exit(0 if success else 1)
