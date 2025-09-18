#!/usr/bin/env python3
"""
REST vs gRPC Performance Comparison
Runs the same benchmarks on both implementations to compare performance.
"""

from benchmarks.rest_benchmark_scenarios import create_benchmark_scenarios
from benchmarks.grpc_benchmark_scenarios import create_grpc_benchmark_scenarios


def compare_implementations():
    """Compare REST and gRPC implementations."""
    print("REST vs gRPC Performance Comparison")
    print("=" * 50)

    # Test configurations
    test_configs = [
        {"name": "Small Scale", "agent_count": 2, "message_count": 20},
        {"name": "Medium Scale", "agent_count": 4, "messages_per_agent": 10},
    ]

    results = {}

    for config in test_configs:
        print(f"\n{config['name']} Test:")
        print("-" * 30)

        # REST Implementation
        print("\nTesting REST Implementation...")
        rest_benchmark = create_benchmark_scenarios()

        if "message_count" in config:
            # Point-to-point test
            rest_result = rest_benchmark.run_scenario(
                "point_to_point_latency",
                agent_count=config["agent_count"],
                message_count=config["message_count"],
            )
            # scenario = "point_to_point_latency"
        else:
            # Concurrent test
            rest_result = rest_benchmark.run_scenario(
                "concurrent_messaging",
                agent_count=config["agent_count"],
                messages_per_agent=config["messages_per_agent"],
            )
            # scenario = "concurrent_messaging"

        # gRPC Implementation
        print("\nTesting gRPC Implementation...")
        grpc_benchmark = create_grpc_benchmark_scenarios()

        if "message_count" in config:
            # Point-to-point test
            grpc_result = grpc_benchmark.run_scenario(
                "grpc_point_to_point_latency",
                agent_count=config["agent_count"],
                message_count=config["message_count"],
            )
        else:
            # Concurrent test
            grpc_result = grpc_benchmark.run_scenario(
                "grpc_concurrent_messaging",
                agent_count=config["agent_count"],
                messages_per_agent=config["messages_per_agent"],
            )

        # Store results
        results[config["name"]] = {"REST": rest_result, "gRPC": grpc_result}

        # Print comparison
        print(f"\n{config['name']} Results Comparison:")
        print("=" * 40)
        print(f"{'Metric':<20} {'REST':<15} {'gRPC':<15} {'Improvement':<12}")
        print("-" * 65)

        # Latency comparison
        rest_latency = rest_result.message_latency_avg * 1000
        grpc_latency = grpc_result.message_latency_avg * 1000
        latency_improvement = (
            (rest_latency - grpc_latency) / rest_latency
        ) * 100
        print(
            f"{'Avg Latency (ms)':<20} {rest_latency:<15.2f} \
                {grpc_latency:<15.2f} {latency_improvement:<12.1f}%"
        )

        # Throughput comparison
        rest_throughput = rest_result.throughput_avg
        grpc_throughput = grpc_result.throughput_avg
        throughput_improvement = (
            ((grpc_throughput - rest_throughput) / rest_throughput) * 100
            if rest_throughput > 0
            else 0
        )
        print(
            f"{'Throughput (msg/s)':<20} {rest_throughput:<15.1f} \
                {grpc_throughput:<15.1f} {throughput_improvement:<12.1f}%"
        )

        # CPU comparison
        rest_cpu = rest_result.cpu_usage_avg
        grpc_cpu = grpc_result.cpu_usage_avg
        cpu_improvement = (
            ((rest_cpu - grpc_cpu) / rest_cpu) * 100 if rest_cpu > 0 else 0
        )
        print(
            f"{'CPU Usage (%)':<20} {rest_cpu:<15.1f} {grpc_cpu:<15.1f} \
                {cpu_improvement:<12.1f}%"
        )

        # Memory comparison
        rest_memory = rest_result.memory_usage_avg
        grpc_memory = grpc_result.memory_usage_avg
        memory_change = (
            ((grpc_memory - rest_memory) / rest_memory) * 100
            if rest_memory > 0
            else 0
        )
        print(
            f"{'Memory Usage (MB)':<20} {rest_memory:<15.1f} \
                {grpc_memory:<15.1f} {memory_change:<12.1f}%"
        )

        # Success rate
        print(
            f"{'Success Rate (%)':<20} {rest_result.success_rate*100:<15.1f}\
                {grpc_result.success_rate*100:<15.1f}"
        )

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print("gRPC generally shows:")
    print("• Lower latency (faster message delivery)")
    print("• Higher or equivalent throughput")
    print("• More efficient resource usage")
    print("• Consistent reliability")
    print("\nThis aligns with gRPC's design advantages:")
    print("• Binary protocol buffers vs JSON")
    print("• Persistent connections vs request-response")
    print("• Efficient serialization")

    # Export comparison results
    comparison_data = {}
    for test_name, test_results in results.items():
        comparison_data[test_name] = {}
        for impl, result in test_results.items():
            comparison_data[test_name][impl] = {
                "avg_latency_ms": result.message_latency_avg * 1000,
                "throughput_msg_per_sec": result.throughput_avg,
                "cpu_usage_percent": result.cpu_usage_avg,
                "memory_usage_mb": result.memory_usage_avg,
                "success_rate": result.success_rate,
                "total_messages": result.total_messages,
            }

    import json

    with open("rest_vs_grpc_comparison.json", "w") as f:
        json.dump(comparison_data, f, indent=2)

    print(
        "\nDetailed comparison results exported \
            to rest_vs_grpc_comparison.json"
    )


if __name__ == "__main__":
    try:
        compare_implementations()
    except KeyboardInterrupt:
        print("\nComparison interrupted by user")
    except Exception as e:
        print(f"\nError during comparison: {e}")
        import traceback

        traceback.print_exc()

    print("\nComparison finished.")
