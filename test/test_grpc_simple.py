#!/usr/bin/env python3
"""
Simple gRPC test to verify the implementation works correctly.
"""

from benchmarks.communication_benchmarks.grpc_benchmark_scenarios import create_grpc_benchmark_scenarios


def test_grpc_simple_benchmark():
    """Test a simple gRPC benchmark scenario."""
    print("Testing Simple gRPC Benchmark Scenario")
    print("=" * 40)

    try:
        # Create gRPC benchmark suite
        benchmark = create_grpc_benchmark_scenarios()

        # Run a quick latency test
        print("Running gRPC point-to-point latency test...")
        result = benchmark.run_scenario(
            "grpc_point_to_point_latency",
            agent_count=2,
            message_count=10,  # Small number for quick test
        )

        # Print results
        benchmark.print_summary(result)

        print("✅ gRPC benchmark test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ gRPC benchmark test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_grpc_simple_benchmark()

    if success:
        print("\n🎉 gRPC system working correctly!")
    else:
        print("\n⚠️  gRPC system has issues.")
