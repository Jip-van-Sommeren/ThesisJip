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
import csv
import math
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field

from benchmarks.local.hierarchy_benchmarks.hierarchy_metrics import (
    HierarchyBenchmarkTracker,
    AggregatedMetrics,
)
from benchmarks.local.hierarchy_benchmarks.hierarchy_strategies import (
    HierarchyType,
    HierarchyStrategy,
    create_hierarchy_strategy,
)
from benchmarks.local.hierarchy_benchmarks.hierarchy_environments import (
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
    trial_count: int = 1
    warm_up_operations: int = 0
    trial_metric_summary: Dict[str, Dict[str, float]] = field(
        default_factory=dict
    )
    trials: List[Dict[str, Any]] = field(default_factory=list)


METRIC_FIELDS: Tuple[str, ...] = tuple(AggregatedMetrics.__dataclass_fields__.keys())


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

    def _reset_trial_state(self):
        """Reset mutable state before a new trial."""
        self.tracker = HierarchyBenchmarkTracker(r_min=0.0, r_max=100.0)
        self.hierarchy = None
        self.environment = None

    def _compute_trial_metric_summary(
        self, trial_metrics: List[Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """Compute mean/std/95% CI across trials for each metric field."""
        summary: Dict[str, Dict[str, float]] = {}
        for field_name in METRIC_FIELDS:
            values = []
            for metric in trial_metrics:
                value = metric.get(field_name)
                if value is None:
                    continue
                values.append(float(value))

            if not values:
                continue

            mean_value = statistics.mean(values)
            std_value = statistics.stdev(values) if len(values) > 1 else 0.0
            margin = (
                1.96 * std_value / math.sqrt(len(values))
                if len(values) > 1
                else 0.0
            )
            summary[field_name] = {
                "mean": mean_value,
                "std": std_value,
                "ci95_lower": mean_value - margin,
                "ci95_upper": mean_value + margin,
            }

        return summary

    def _metrics_from_summary(
        self, summary: Dict[str, Dict[str, float]]
    ) -> AggregatedMetrics:
        """Materialize AggregatedMetrics from trial summary means."""
        kwargs: Dict[str, Any] = {}
        for field_name in METRIC_FIELDS:
            mean_value = summary.get(field_name, {}).get("mean", 0.0)
            if field_name in {"total_episodes", "successful_episodes"}:
                kwargs[field_name] = int(round(mean_value))
            else:
                kwargs[field_name] = float(mean_value)
        return AggregatedMetrics(**kwargs)

    def _run_single_trial(
        self, trial_index: int, total_trials: int
    ) -> BenchmarkResult:
        """Run one measured trial."""
        self._reset_trial_state()
        self.setup()

        start_time = time.perf_counter()

        for episode_num in range(self.config.num_episodes):
            episode_label = (
                f"Trial {trial_index}/{total_trials} - "
                if total_trials > 1
                else ""
            )
            print(
                f"{episode_label}Episode {episode_num + 1}/{self.config.num_episodes}...",
                end=" ",
            )

            episode_result = self.run_episode()

            print(
                f"✓ Success: {episode_result['success']}, "
                f"Return: {episode_result['total_return']:.1f}, "
                f"Steps: {episode_result['steps']}"
            )

        execution_time = time.perf_counter() - start_time
        metrics = self.tracker.get_aggregated_metrics()
        trial_summary = self._compute_trial_metric_summary([asdict(metrics)])

        return BenchmarkResult(
            configuration=self.config,
            metrics=metrics,
            episode_data=self.tracker.export_results()["episodes"],
            execution_time=execution_time,
            timestamp=time.time(),
            trial_count=1,
            warm_up_operations=0,
            trial_metric_summary=trial_summary,
            trials=[],
        )

    def run_benchmark(
        self, num_trials: int = 1, warm_up_operations: int = 0
    ) -> BenchmarkResult:
        """Run full benchmark with optional multi-trial aggregation."""
        if num_trials < 1:
            raise ValueError("num_trials must be >= 1")
        if warm_up_operations < 0:
            raise ValueError("warm_up_operations must be >= 0")

        print(f"\n{'='*60}")
        print("Running Hierarchy Benchmark")
        print(f"Strategy: {self.config.hierarchy_type.value}")
        print(f"Environment: {self.config.environment_type}")
        print(f"Agents: {self.config.num_agents}")
        print(f"Episodes: {self.config.num_episodes}")
        print(f"Trials: {num_trials}")
        print(f"Warm-up Episodes/Trial: {warm_up_operations}")
        print(f"{'='*60}\n")

        total_start = time.perf_counter()
        trial_results: List[BenchmarkResult] = []

        for trial_idx in range(1, num_trials + 1):
            if num_trials > 1:
                print(f"\n--- Trial {trial_idx}/{num_trials} ---")

            if warm_up_operations > 0:
                print(
                    f"Running {warm_up_operations} warm-up episode(s) "
                    f"for trial {trial_idx}..."
                )
                self._reset_trial_state()
                self.setup()
                for _ in range(warm_up_operations):
                    self.run_episode()

            trial_result = self._run_single_trial(trial_idx, num_trials)
            trial_results.append(trial_result)

        trial_entries = [
            {
                "trial_id": idx + 1,
                "metrics": asdict(trial.metrics),
                "execution_time": trial.execution_time,
                "episode_data": trial.episode_data,
            }
            for idx, trial in enumerate(trial_results)
        ]

        trial_metric_summary = self._compute_trial_metric_summary(
            [asdict(trial.metrics) for trial in trial_results]
        )
        metrics = self._metrics_from_summary(trial_metric_summary)

        result = BenchmarkResult(
            configuration=self.config,
            metrics=metrics,
            episode_data=trial_results[0].episode_data if trial_results else [],
            execution_time=time.perf_counter() - total_start,
            timestamp=time.time(),
            trial_count=num_trials,
            warm_up_operations=warm_up_operations,
            trial_metric_summary=trial_metric_summary,
            trials=trial_entries,
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
        print(f" Trials: {result.trial_count}")
        print(f" Warm-up Episodes/Trial: {result.warm_up_operations}")

        metrics = result.metrics
        success_summary = result.trial_metric_summary.get("success_rate", {})
        success_ci_lower = success_summary.get("ci95_lower")
        success_ci_upper = success_summary.get("ci95_upper")
        makespan_summary = result.trial_metric_summary.get("makespan_mean", {})
        makespan_ci_lower = makespan_summary.get("ci95_lower")
        makespan_ci_upper = makespan_summary.get("ci95_upper")

        print("\n Task Effectiveness:")
        if (
            result.trial_count > 1
            and success_ci_lower is not None
            and success_ci_upper is not None
        ):
            print(
                "  Success Rate: "
                f"{metrics.success_rate*100:.1f}% "
                f"(95% CI: {success_ci_lower*100:.1f}% - {success_ci_upper*100:.1f}%)"
            )
        else:
            print(f"  Success Rate: {metrics.success_rate*100:.1f}%")
        print(
            f"  Normalized Return: {metrics.normalized_return_mean:.3f} ±\
                {metrics.normalized_return_std:.3f}"
        )

        print("\n Time & Resource Efficiency:")
        if (
            result.trial_count > 1
            and makespan_ci_lower is not None
            and makespan_ci_upper is not None
        ):
            print(
                "  Makespan (avg): "
                f"{metrics.makespan_mean:.1f} steps "
                f"(95% CI: {makespan_ci_lower:.1f} - {makespan_ci_upper:.1f})"
            )
        else:
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
        num_trials: int = 1,
        warm_up_operations: int = 0,
    ):
        """Run comprehensive comparison across strategies and environments."""

        print("\n" + "=" * 70)
        print("HIERARCHY STRATEGY COMPARISON BENCHMARK")
        print("=" * 70)
        print(f"Strategies: {[h.value for h in hierarchy_types]}")
        print(f"Environments: {environment_types}")
        print(f"Agent counts: {agent_counts}")
        print(f"Episodes per config: {num_episodes}")
        print(f"Trials per config: {num_trials}")
        print(f"Warm-up episodes per trial: {warm_up_operations}")
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
                    result = scenario.run_benchmark(
                        num_trials=num_trials,
                        warm_up_operations=warm_up_operations,
                    )
                    self.results.append(result)

        # Export results
        self.export_results()
        self.generate_comparison_report()

    def run_ablation_study(
        self,
        base_config: BenchmarkConfiguration,
        ablation_params: Dict[str, List[Any]],
        num_trials: int = 1,
        warm_up_operations: int = 0,
        output_suffix: Optional[str] = None,
    ):
        """Run ablation study varying specific parameters."""

        print("\n" + "=" * 70)
        print("HIERARCHY ABLATION STUDY")
        print("=" * 70)
        print(f"Base strategy: {base_config.hierarchy_type.value}")
        print(f"Ablation parameters: {list(ablation_params.keys())}")
        print(f"Trials per config: {num_trials}")
        print(f"Warm-up episodes per trial: {warm_up_operations}")
        print("=" * 70 + "\n")

        start_idx = len(self.results)

        # Run baseline
        baseline_scenario = HierarchyBenchmarkScenario(base_config)
        baseline_result = baseline_scenario.run_benchmark(
            num_trials=num_trials,
            warm_up_operations=warm_up_operations,
        )
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
                    hierarchy_depth=base_config.hierarchy_depth,
                    static_roles=base_config.static_roles,
                    communication_limit=base_config.communication_limit,
                    planning_frequency=base_config.planning_frequency,
                    env_params=dict(base_config.env_params),
                )

                # Set ablation parameter
                setattr(config, param_name, value)

                # Run benchmark
                scenario = HierarchyBenchmarkScenario(config)
                result = scenario.run_benchmark(
                    num_trials=num_trials,
                    warm_up_operations=warm_up_operations,
                )
                self.results.append(result)

        # Export ablation results only (exclude prior comparison runs)
        ablation_results = self.results[start_idx:]
        self.export_ablation_results(
            ablation_params,
            ablation_results,
            base_config=base_config,
            output_suffix=output_suffix,
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
                        "num_tasks_per_episode": r.configuration.num_tasks_per_episode,
                        "max_steps": r.configuration.max_steps,
                        "hierarchy_depth": r.configuration.hierarchy_depth,
                        "static_roles": r.configuration.static_roles,
                        "communication_limit": r.configuration.communication_limit,
                        "planning_frequency": r.configuration.planning_frequency,
                        "env_params": r.configuration.env_params,
                    },
                    "metrics": asdict(r.metrics),
                    "execution_time": r.execution_time,
                    "trial_count": r.trial_count,
                    "warm_up_operations": r.warm_up_operations,
                    "trial_metric_summary": r.trial_metric_summary,
                    "trials": r.trials,
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

        fieldnames = [
            "Strategy",
            "Environment",
            "Agents",
            "Trials",
            "WarmUpEpisodesPerTrial",
            "SuccessRate",
            "SuccessRateStd",
            "SuccessRateCI95Lower",
            "SuccessRateCI95Upper",
            "NormReturn",
            "NormReturnStd",
            "NormReturnCI95Lower",
            "NormReturnCI95Upper",
            "Makespan",
            "MakespanStd",
            "MakespanCI95Lower",
            "MakespanCI95Upper",
            "ActionEfficiency",
            "ActionEfficiencyStd",
            "ManagerUtil",
            "ManagerUtilStd",
            "DelegationSuccess",
            "DelegationSuccessStd",
            "MessagesPerEp",
            "MessagesPerEpStd",
            "BytesPerStep",
            "BytesPerStepStd",
            "CoordLatency",
            "CoordLatencyStd",
        ]

        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in self.results:
                metrics = result.metrics
                config = result.configuration
                summary = result.trial_metric_summary

                def _summary(metric_name: str, field_name: str) -> float:
                    return float(
                        summary.get(metric_name, {}).get(field_name, 0.0)
                    )

                writer.writerow(
                    {
                        "Strategy": config.hierarchy_type.value,
                        "Environment": config.environment_type,
                        "Agents": config.num_agents,
                        "Trials": result.trial_count,
                        "WarmUpEpisodesPerTrial": result.warm_up_operations,
                        "SuccessRate": metrics.success_rate,
                        "SuccessRateStd": _summary("success_rate", "std"),
                        "SuccessRateCI95Lower": _summary(
                            "success_rate", "ci95_lower"
                        ),
                        "SuccessRateCI95Upper": _summary(
                            "success_rate", "ci95_upper"
                        ),
                        "NormReturn": metrics.normalized_return_mean,
                        "NormReturnStd": _summary(
                            "normalized_return_mean", "std"
                        ),
                        "NormReturnCI95Lower": _summary(
                            "normalized_return_mean", "ci95_lower"
                        ),
                        "NormReturnCI95Upper": _summary(
                            "normalized_return_mean", "ci95_upper"
                        ),
                        "Makespan": metrics.makespan_mean,
                        "MakespanStd": _summary("makespan_mean", "std"),
                        "MakespanCI95Lower": _summary(
                            "makespan_mean", "ci95_lower"
                        ),
                        "MakespanCI95Upper": _summary(
                            "makespan_mean", "ci95_upper"
                        ),
                        "ActionEfficiency": metrics.action_efficiency,
                        "ActionEfficiencyStd": _summary(
                            "action_efficiency", "std"
                        ),
                        "ManagerUtil": metrics.manager_utilization,
                        "ManagerUtilStd": _summary(
                            "manager_utilization", "std"
                        ),
                        "DelegationSuccess": metrics.delegation_success_rate,
                        "DelegationSuccessStd": _summary(
                            "delegation_success_rate", "std"
                        ),
                        "MessagesPerEp": metrics.messages_per_episode,
                        "MessagesPerEpStd": _summary(
                            "messages_per_episode", "std"
                        ),
                        "BytesPerStep": metrics.bytes_per_step,
                        "BytesPerStepStd": _summary("bytes_per_step", "std"),
                        "CoordLatency": metrics.coordination_latency_mean,
                        "CoordLatencyStd": _summary(
                            "coordination_latency_mean", "std"
                        ),
                    }
                )

        print(f" CSV exported to: {csv_file}")

    def export_ablation_results(
        self,
        ablation_params: Dict[str, List[Any]],
        results: Optional[List[BenchmarkResult]] = None,
        base_config: Optional[BenchmarkConfiguration] = None,
        output_suffix: Optional[str] = None,
    ):
        """Export ablation study results."""
        if results is None:
            results = self.results

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        suffix = f"_{output_suffix}" if output_suffix else ""
        ablation_file = os.path.join(
            self.output_dir, f"ablation_study{suffix}_{timestamp}.json"
        )
        serialized_results = []
        for result in results:
            serialized_results.append(
                {
                    "configuration": {
                        "hierarchy_type": result.configuration.hierarchy_type.value,
                        "num_agents": result.configuration.num_agents,
                        "environment_type": result.configuration.environment_type,
                        "num_episodes": result.configuration.num_episodes,
                        "num_tasks_per_episode": result.configuration.num_tasks_per_episode,
                        "max_steps": result.configuration.max_steps,
                        "hierarchy_depth": result.configuration.hierarchy_depth,
                        "planning_frequency": result.configuration.planning_frequency,
                        "communication_limit": result.configuration.communication_limit,
                        "static_roles": result.configuration.static_roles,
                    },
                    # Backward-compatible field used by existing analysis scripts.
                    "config": {
                        "hierarchy_depth": result.configuration.hierarchy_depth,
                        "planning_frequency": result.configuration.planning_frequency,
                        "communication_limit": result.configuration.communication_limit,
                        "static_roles": result.configuration.static_roles,
                        "hierarchy_type": result.configuration.hierarchy_type.value,
                    },
                    "metrics": asdict(result.metrics),
                    "trial_count": result.trial_count,
                    "warm_up_operations": result.warm_up_operations,
                    "trial_metric_summary": result.trial_metric_summary,
                    "trials": result.trials,
                }
            )
        export_data = {
            "timestamp": timestamp,
            "ablation_parameters": {
                k: [str(v) for v in vals]
                for k, vals in ablation_params.items()
            },
            "base_config": (
                {
                    "hierarchy_type": base_config.hierarchy_type.value,
                    "num_agents": base_config.num_agents,
                    "environment_type": base_config.environment_type,
                    "num_episodes": base_config.num_episodes,
                    "num_tasks_per_episode": base_config.num_tasks_per_episode,
                    "max_steps": base_config.max_steps,
                    "hierarchy_depth": base_config.hierarchy_depth,
                    "planning_frequency": base_config.planning_frequency,
                    "communication_limit": base_config.communication_limit,
                    "static_roles": base_config.static_roles,
                }
                if base_config
                else None
            ),
            "results": serialized_results,
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
