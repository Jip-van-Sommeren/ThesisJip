#!/usr/bin/env python3
"""
Debug script to see what's happening with the agent count parameter.
"""

from benchmarks.rest_benchmark_scenarios import create_benchmark_scenarios


def debug_benchmark():
    """Debug the benchmark parameter passing."""
    print("Debugging Benchmark Parameters")
    print("=" * 40)

    # Create benchmark suite
    benchmark = create_benchmark_scenarios()

    # Get the scenario
    scenario = benchmark.scenarios["point_to_point_latency"]

    print(f"Initial scenario parameters: {scenario.parameters}")

    # Set parameters like the benchmark would
    scenario.parameters.update({"agent_count": 2, "message_count": 10})
    print(f"After update: {scenario.parameters}")

    # Run setup function
    if scenario.setup_func:
        setup_result = scenario.setup_func(scenario.parameters)
        print(f"Setup result: {setup_result}")

        if setup_result:
            scenario.parameters.update(setup_result)
            print(f"After setup update: {scenario.parameters}")

    # Check final agent count
    agent_count = scenario.parameters.get("agent_count", 0)
    print(f"Final agent count: {agent_count}")

    # Check if agents were actually created
    agents = scenario.parameters.get("agents", [])
    print(f"Number of agents created: {len(agents)}")

    # Cleanup
    if "environment" in scenario.parameters:
        scenario.parameters["environment"].stop_service()


if __name__ == "__main__":
    debug_benchmark()
