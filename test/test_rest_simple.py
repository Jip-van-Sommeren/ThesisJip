#!/usr/bin/env python3
"""
Simple benchmark test to verify benchmarks work without port conflicts.
"""

from benchmarks.communication_benchmarks.rest_benchmark_scenarios import create_rest_benchmark_scenarios


def test_simple_benchmark():
    """Test a simple benchmark scenario."""
    print("Testing Simple Benchmark Scenario")
    print("=" * 40)

    try:
        # Create benchmark suite
        benchmark = create_rest_benchmark_scenarios()

        # Run a quick latency test
        print("Running point-to-point latency test...")
        result = benchmark.run_scenario(
            "point_to_point_latency",
            agent_count=2,
            message_count=10,  # Small number for quick test
        )

        # Print results
        benchmark.print_summary(result)

        print("Benchmark test completed")
        return True

    except Exception as e:
        print(f"Benchmark test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_simple_benchmark()
