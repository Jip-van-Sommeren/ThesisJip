"""
Hierarchy Benchmark Scenarios
Complete benchmark scenarios for testing hierarchy strategies.

Implements comprehensive benchmarks comparing:
- Tree hierarchy
- Peer-to-peer
- Hybrid hierarchy

Across multiple metrics and ablation configurations.
"""

import time
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

from benchmarks.hierarchy_benchmarks.hierarchy_metrics import (
    HierarchyBenchmarkTracker,
    AggregatedMetrics,
)
from benchmarks.hierarchy_benchmarks.hierarchy_strategies import (
    HierarchyType,
    HierarchyStrategy,
    create_hierarchy_strategy,
)
from benchmarks.hierarchy_benchmarks.hierarchy_environments import (
    HierarchyEnvironment,
    create_environment,
)


@dataclass
class BenchmarkConfiguration:
    """Configuration for a hierarchy benchmark."""

    hierarchy_type: HierarchyType
    num_agents: int
    environment_type: str
    num_episodes: int = 10
    num_tasks_per_episode: int = 5
    max_steps: int = 100

    # Ablation parameters
    hierarchy_depth: int = 2
    static_roles: bool = True
    communication_limit: Optional[int] = None  # None = unlimited
    planning_frequency: int = 1  # Manager plans every k steps

    # Environment-specific params
    env_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.env_params is None:
            self.env_params = {}


@dataclass
class BenchmarkResult:
    """Results from a hierarchy benchmark."""

    configuration: BenchmarkConfiguration
    metrics: AggregatedMetrics
    episode_data: List[Dict[str, Any]]
    execution_time: float
    timestamp: float


class HierarchyBenchmarkScenario:
    """
    Complete benchmark scenario for a hierarchy strategy.
    Runs episodes and collects comprehensive metrics.
    """

    def __init__(self, config: BenchmarkConfiguration):
        self.config = config
        self.tracker = HierarchyBenchmarkTracker(r_min=0.0, r_max=100.0)
        self.hierarchy: Optional[HierarchyStrategy] = None
        self.environment: Optional[HierarchyEnvironment] = None

    def setup(self):
        """Setup hierarchy and environment."""
        # Create hierarchy strategy
        self.hierarchy = create_hierarchy_strategy(
            self.config.hierarchy_type,
            self.config.num_agents,
            hierarchy_depth=self.config.hierarchy_depth,
        )

        # Initialize hierarchy
        self.hierarchy.initialize_hierarchy()

        # Apply communication limits if specified
        if self.config.communication_limit is not None:
            self.hierarchy.message_limit = self.config.communication_limit

        # Set planning frequency for tree/hybrid
        if hasattr(self.hierarchy, "manager_planning_frequency"):
            self.hierarchy.manager_planning_frequency = (
                self.config.planning_frequency
            )

        # Create environment
        self.environment = create_environment(
            self.config.environment_type,
            self.config.num_agents,
            max_steps=self.config.max_steps,
            **self.config.env_params,
        )

    def run_episode(self) -> Dict[str, Any]:
        """Run a single episode."""
        state = self.environment.reset()

        tasks = self.environment.generate_tasks(
            self.config.num_tasks_per_episode
        )
        tasks_per_episode = len(tasks)

        for task in tasks:
            self.hierarchy.tasks[task.task_id] = task

        self.tracker.start_episode()

        task_delegation_step = {}
        for task in tasks:
            self.tracker.record_delegation(issued=True)

            success = self.hierarchy.allocate_task(task)

            if success:
                task_delegation_step[task.task_id] = 0
                self.tracker.start_coordination(task.task_id)

        # Count initial delegation coordination records before stepping.
        initial_messages = len(self.hierarchy.message_queue)
        for _ in range(initial_messages):
            self.tracker.record_message(message_bytes=100)
        if initial_messages:
            self.hierarchy.message_queue.clear()

        # Run episode
        done = False
        total_steps = 0
        primitive_actions = 0
        previous_task_status = {task.task_id: task.status for task in tasks}

        while not done and total_steps < self.config.max_steps:
            state, _, done = self.environment.step(self.hierarchy)

            self.tracker.record_step()

            for task in tasks:
                prev_status = previous_task_status.get(task.task_id)
                curr_status = self.hierarchy.tasks[task.task_id].status
                task_id = task.task_id

                if (
                    task_id in task_delegation_step
                    and task_delegation_step[task_id] is not None
                ):
                    worker = self.hierarchy.tasks[task_id].assigned_to
                    if worker and worker in self.hierarchy.agents:
                        agent = self.hierarchy.agents[worker]
                        # Check if worker has started (has progress > 0)
                        if (
                            hasattr(agent, "task_progress")
                            and agent.task_progress > 0
                        ):
                            # First action taken - end coordination
                            self.tracker.end_coordination(task_id)
                            task_delegation_step[task_id] = None

                # Task completed successfully
                if prev_status == "in_progress" and curr_status == "completed":
                    self.tracker.record_delegation(completed=True)
                    # Clear coordination if still tracking
                    if task_id in task_delegation_step:
                        task_delegation_step.pop(task_id, None)

                # Task preempted (needs reassignment)
                elif prev_status == "in_progress" and curr_status == "pending":
                    self.tracker.record_delegation(preempted=True)
                    # New delegation issued
                    self.tracker.record_delegation(issued=True)
                    # Restart coordination tracking
                    task_delegation_step[task_id] = total_steps
                    self.tracker.start_coordination(task_id)

                # Task failed
                elif prev_status == "in_progress" and curr_status == "failed":
                    self.tracker.record_delegation(failed=True)
                    if task_id in task_delegation_step:
                        task_delegation_step.pop(task_id, None)

                previous_task_status[task.task_id] = curr_status

            # Track actions
            if hasattr(self.hierarchy, "managers"):
                # Tree/Hybrid: track manager vs worker actions
                active_managers = [
                    manager_id
                    for manager_id in self.hierarchy.managers
                    if self.hierarchy.agents[manager_id].is_active
                ]
                manager_should_act = True
                if hasattr(self.hierarchy, "manager_planning_frequency"):
                    # Only count manager actions on planning steps.
                    manager_should_act = (
                        self.hierarchy.step_count
                        % self.hierarchy.manager_planning_frequency
                        == 0
                    )
                if manager_should_act:
                    for _ in active_managers:
                        self.tracker.record_action(
                            is_manager=True, execution_time=0.001
                        )

                for worker_id in self.hierarchy.workers:
                    if self.hierarchy.agents[worker_id].is_active:
                        self.tracker.record_action(
                            is_manager=False, execution_time=0.002
                        )
                        primitive_actions += 1
            else:
                # Peer-to-peer: all actions are peer actions
                for agent_id in self.hierarchy.agents:
                    if self.hierarchy.agents[agent_id].is_active:
                        self.tracker.record_action(
                            is_manager=False, execution_time=0.002
                        )
                        primitive_actions += 1

            # Track messages (coordination records) generated this step
            num_messages = len(self.hierarchy.message_queue)
            for _ in range(num_messages):
                self.tracker.record_message(message_bytes=100)
            if num_messages:
                self.hierarchy.message_queue.clear()

            total_steps += 1

        # End episode
        success = self.environment.check_success()
        total_return = state.total_return

        self.tracker.end_episode(
            success=success,
            total_return=total_return,
            steps=total_steps,
            primitive_actions=primitive_actions,
            tasks_per_episode=tasks_per_episode,
        )

        return {
            "success": success,
            "total_return": total_return,
            "steps": total_steps,
            "primitive_actions": primitive_actions,
        }

    def run_benchmark(self) -> BenchmarkResult:
        """Run full benchmark with multiple episodes."""
        print(f"\n{'='*60}")
        print("Running Hierarchy Benchmark")
        print(f"Strategy: {self.config.hierarchy_type.value}")
        print(f"Environment: {self.config.environment_type}")
        print(f"Agents: {self.config.num_agents}")
        print(f"Episodes: {self.config.num_episodes}")
        print(f"{'='*60}\n")

        start_time = time.perf_counter()

        # Setup
        self.setup()

        # Run episodes
        episode_results = []
        for episode_num in range(self.config.num_episodes):
            print(
                f"Episode {episode_num + 1}/{self.config.num_episodes}...",
                end=" ",
            )

            episode_result = self.run_episode()
            episode_results.append(episode_result)

            print(
                f"✓ Success: {episode_result['success']}, "
                f"Return: {episode_result['total_return']:.1f}, "
                f"Steps: {episode_result['steps']}"
            )

        execution_time = time.perf_counter() - start_time

        # Get aggregated metrics
        metrics = self.tracker.get_aggregated_metrics()

        # Create result
        result = BenchmarkResult(
            configuration=self.config,
            metrics=metrics,
            episode_data=self.tracker.export_results()["episodes"],
            execution_time=execution_time,
            timestamp=time.time(),
        )

        self._print_summary(result)

        return result

    def _print_summary(self, result: BenchmarkResult):
        """Print benchmark summary."""
        print(f"\n{'='*60}")
        print(
            f"BENCHMARK RESULTS - {self.config.hierarchy_type.value.upper()}"
        )
        print(f"{'='*60}")

        metrics = result.metrics

        print("\n Task Effectiveness:")
        print(f"  Success Rate: {metrics.success_rate*100:.1f}%")
        print(
            f"  Normalized Return: {metrics.normalized_return_mean:.3f} ±\
                {metrics.normalized_return_std:.3f}"
        )

        print("\n Time & Resource Efficiency:")
        print(f"  Makespan (avg): {metrics.makespan_mean:.1f} steps")
        print(
            f"  Action Efficiency: {metrics.action_efficiency:.1f}\
                actions/task"
        )
        print(
            f"  Wall Clock: {metrics.wall_clock_per_episode:.2f}s per episode"
        )

        print("\n Hierarchy Overhead:")
        print(
            f"  Manager Utilization: {metrics.manager_utilization:.1f} \
                actions/100 steps"
        )
        print(
            f"  Delegation Success: {metrics.delegation_success_rate*100:.1f}%"
        )
        print(f"  Preemption Rate: {metrics.preemption_rate:.2f}")
        print(f"  Manager Time: {metrics.manager_time_percent:.1f}%")
        print(f"  Worker Time: {metrics.worker_time_percent:.1f}%")

        print("\n Communication Cost:")
        print(f"  Messages/Episode: {metrics.messages_per_episode:.1f}")
        print(f"  Bytes/Step: {metrics.bytes_per_step:.1f}")
        print(
            f"  Coordination Latency:\
                {metrics.coordination_latency_mean*1000:.2f}ms"
        )

        print(f"\n Total Execution Time: {result.execution_time:.2f}s")
        print(f"{'='*60}\n")


class HierarchyComparisonBenchmark:
    """
    Compares multiple hierarchy strategies across different scenarios.
    """

    def __init__(self, output_dir: str = "results/hierarchy"):
        self.output_dir = output_dir
        self.results: List[BenchmarkResult] = []
        os.makedirs(self.output_dir, exist_ok=True)

    def run_comparison(
        self,
        hierarchy_types: List[HierarchyType],
        environment_types: List[str],
        agent_counts: List[int],
        num_episodes: int = 10,
    ):
        """Run comprehensive comparison across strategies and environments."""

        print("\n" + "=" * 70)
        print("HIERARCHY STRATEGY COMPARISON BENCHMARK")
        print("=" * 70)
        print(f"Strategies: {[h.value for h in hierarchy_types]}")
        print(f"Environments: {environment_types}")
        print(f"Agent counts: {agent_counts}")
        print(f"Episodes per config: {num_episodes}")
        print("=" * 70 + "\n")

        # Run benchmarks for each combination
        for hierarchy_type in hierarchy_types:
            for env_type in environment_types:
                for num_agents in agent_counts:
                    config = BenchmarkConfiguration(
                        hierarchy_type=hierarchy_type,
                        num_agents=num_agents,
                        environment_type=env_type,
                        num_episodes=num_episodes,
                    )

                    scenario = HierarchyBenchmarkScenario(config)
                    result = scenario.run_benchmark()
                    self.results.append(result)

        # Export results
        self.export_results()
        self.generate_comparison_report()

    def run_ablation_study(
        self,
        base_config: BenchmarkConfiguration,
        ablation_params: Dict[str, List[Any]],
    ):
        """Run ablation study varying specific parameters."""

        print("\n" + "=" * 70)
        print("HIERARCHY ABLATION STUDY")
        print("=" * 70)
        print(f"Base strategy: {base_config.hierarchy_type.value}")
        print(f"Ablation parameters: {list(ablation_params.keys())}")
        print("=" * 70 + "\n")

        start_idx = len(self.results)

        # Run baseline
        baseline_scenario = HierarchyBenchmarkScenario(base_config)
        baseline_result = baseline_scenario.run_benchmark()
        self.results.append(baseline_result)

        # Run ablations
        for param_name, param_values in ablation_params.items():
            print(f"\nAblating {param_name}:")
            for value in param_values:
                # Create modified config
                config = BenchmarkConfiguration(
                    hierarchy_type=base_config.hierarchy_type,
                    num_agents=base_config.num_agents,
                    environment_type=base_config.environment_type,
                    num_episodes=base_config.num_episodes,
                    num_tasks_per_episode=base_config.num_tasks_per_episode,
                    max_steps=base_config.max_steps,
                )

                # Set ablation parameter
                setattr(config, param_name, value)

                # Run benchmark
                scenario = HierarchyBenchmarkScenario(config)
                result = scenario.run_benchmark()
                self.results.append(result)

        # Export ablation results only (exclude prior comparison runs)
        ablation_results = self.results[start_idx:]
        self.export_ablation_results(
            ablation_params,
            ablation_results,
            base_config=base_config,
        )

    def export_results(self):
        """Export all results to JSON and CSV."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Export full results to JSON
        json_file = os.path.join(
            self.output_dir, f"hierarchy_results_{timestamp}.json"
        )

        export_data = {
            "timestamp": timestamp,
            "benchmarks": [
                {
                    "configuration": {
                        "hierarchy_type": r.configuration.hierarchy_type.value,
                        "num_agents": r.configuration.num_agents,
                        "environment_type": r.configuration.environment_type,
                        "num_episodes": r.configuration.num_episodes,
                    },
                    "metrics": asdict(r.metrics),
                    "execution_time": r.execution_time,
                }
                for r in self.results
            ],
        }

        with open(json_file, "w") as f:
            json.dump(export_data, f, indent=2)

        print(f"\n Results exported to: {json_file}")

        # Export CSV for easy plotting
        self._export_csv(timestamp)

    def _export_csv(self, timestamp: str):
        """Export results to CSV format."""
        csv_file = os.path.join(
            self.output_dir, f"hierarchy_metrics_{timestamp}.csv"
        )

        with open(csv_file, "w") as f:
            # Header
            f.write(
                "Strategy,Environment,Agents,SuccessRate,NormReturn,Makespan,"
            )
            f.write(
                "ActionEfficiency,ManagerUtil,DelegationSuccess,MessagesPerEp,"
            )
            f.write("BytesPerStep,CoordLatency\n")

            # Data rows
            for result in self.results:
                m = result.metrics
                c = result.configuration
                f.write(
                    f"{c.hierarchy_type.value},{c.environment_type},\
                        {c.num_agents},"
                )
                f.write(
                    f"{m.success_rate},{m.normalized_return_mean},\
                        {m.makespan_mean},"
                )
                f.write(
                    f"{m.action_efficiency},{m.manager_utilization},\
                        {m.delegation_success_rate},"
                )
                f.write(
                    f"{m.messages_per_episode},{m.bytes_per_step},\
                        {m.coordination_latency_mean}\n"
                )

        print(f" CSV exported to: {csv_file}")

    def export_ablation_results(
        self,
        ablation_params: Dict[str, List[Any]],
        results: Optional[List[BenchmarkResult]] = None,
        base_config: Optional[BenchmarkConfiguration] = None,
    ):
        """Export ablation study results."""
        if results is None:
            results = self.results

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ablation_file = os.path.join(
            self.output_dir, f"ablation_study_{timestamp}.json"
        )
        results = [
            {
                "config": {
                    "hierarchy_depth": r.configuration.hierarchy_depth,
                    "planning_frequency": r.configuration.planning_frequency,
                    "communication_limit": r.configuration.communication_limit,
                    "static_roles": r.configuration.static_roles,
                },
                "metrics": asdict(r.metrics),
            }
            for r in results
        ]
        export_data = {
            "timestamp": timestamp,
            "ablation_parameters": {
                k: [str(v) for v in vals]
                for k, vals in ablation_params.items()
            },
            "base_config": (
                {
                    "hierarchy_depth": base_config.hierarchy_depth,
                    "planning_frequency": base_config.planning_frequency,
                    "communication_limit": base_config.communication_limit,
                    "static_roles": base_config.static_roles,
                }
                if base_config
                else None
            ),
            "results": results,
        }

        with open(ablation_file, "w") as f:
            json.dump(export_data, f, indent=2)

        print(f"\n Ablation results exported to: {ablation_file}")

    def generate_comparison_report(self):
        """Generate comparison report across strategies."""
        print("\n" + "=" * 70)
        print("HIERARCHY STRATEGY COMPARISON REPORT")
        print("=" * 70)

        # Group results by environment and agent count
        grouped: Dict[Tuple[str, int], List[BenchmarkResult]] = {}

        for result in self.results:
            key = (
                result.configuration.environment_type,
                result.configuration.num_agents,
            )
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(result)

        # Print comparison for each group
        for (env_type, num_agents), results in grouped.items():
            print(f"\n{env_type.upper()} - {num_agents} agents:")
            print("-" * 70)

            # Sort by success rate
            results_sorted = sorted(
                results, key=lambda r: r.metrics.success_rate, reverse=True
            )

            for result in results_sorted:
                strategy = result.configuration.hierarchy_type.value
                m = result.metrics
                print(
                    f"  {strategy:15s} | \
                        Success: {m.success_rate*100:5.1f}% | "
                    f"Return: {m.normalized_return_mean:5.3f} | "
                    f"Makespan: {m.makespan_mean:6.1f} | "
                    f"Messages: {m.messages_per_episode:6.1f}"
                )

        print("=" * 70 + "\n")


def run_quick_test():
    """Run a quick test of all three strategies."""
    print("\nRunning quick hierarchy benchmark test...\n")

    comparison = HierarchyComparisonBenchmark(output_dir="results/hierarchy")

    comparison.run_comparison(
        hierarchy_types=[
            HierarchyType.TREE,
            HierarchyType.PEER_TO_PEER,
            HierarchyType.HYBRID,
        ],
        environment_types=["task_distribution"],
        agent_counts=[5],
        num_episodes=3,
    )


def run_full_benchmark():
    """Run comprehensive hierarchy benchmark."""
    print("\nRunning full hierarchy benchmark suite...\n")

    comparison = HierarchyComparisonBenchmark(output_dir="results/hierarchy")

    # Main comparison
    comparison.run_comparison(
        hierarchy_types=[
            HierarchyType.TREE,
            HierarchyType.PEER_TO_PEER,
            HierarchyType.HYBRID,
        ],
        environment_types=[
            "task_distribution",
            "resource_allocation",
            "collaborative",
            "fault_recovery",
        ],
        agent_counts=[3, 5, 8, 12],
        num_episodes=10,
    )

    # Ablation study for tree hierarchy
    tree_config = BenchmarkConfiguration(
        hierarchy_type=HierarchyType.TREE,
        num_agents=8,
        environment_type="task_distribution",
        num_episodes=5,
    )

    comparison.run_ablation_study(
        base_config=tree_config,
        ablation_params={
            "hierarchy_depth": [1, 2, 3],
            "planning_frequency": [1, 5, 10],
            "communication_limit": [None, 50, 100],
        },
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        run_full_benchmark()
    else:
        run_quick_test()
