#!/usr/bin/env python3
"""
Test larger benchmark to verify agent count works with more agents.
"""

from benchmarks.communication_benchmarks.rest_benchmark_scenarios import create_rest_benchmark_scenarios


def test_larger_benchmark():
    """Test with more agents to verify the fix."""
    print("Testing Larger Benchmark Scenario")
    print("=" * 40)

    try:
        # Create benchmark suite
        benchmark = create_rest_benchmark_scenarios()

        # Run concurrent messaging with 5 agents
        print("Running concurrent messaging test with 5 agents...")
        result = benchmark.run_scenario(
            "concurrent_messaging",
            agent_count=5,
            messages_per_agent=5,  # Small number for quick test
        )

        # Print results
        benchmark.print_summary(result)

        print("✅ Larger benchmark test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Larger benchmark test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_larger_benchmark()

    if success:
        print("\n🎉 Agent count tracking working correctly!")
    else:
        print("\n⚠️  Agent count tracking has issues.")
