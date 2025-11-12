"""
Hierarchy Strategy Metrics Module
Comprehensive metric tracking for comparing hierarchy strategies (tree, peer-to-peer, hybrid).

Implements metrics for:
- Task effectiveness (success rate, normalized return)
- Time and resource efficiency (makespan, action efficiency, compute overhead)
- Hierarchy overhead (manager utilization, delegation success, preemption count)
- Communication cost (messages per episode, bits per step, coordination latency)
- Generalization and scalability (team-size scaling, domain shift)
- Robustness (fault injection, non-stationarity)
"""

import time
import statistics
import threading
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque, defaultdict


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""

    episode_id: int
    success: bool
    total_return: float
    steps_to_success: Optional[int] = None
    total_steps: int = 0
    primitive_actions: int = 0
    manager_actions: int = 0
    worker_actions: int = 0
    total_messages: int = 0
    total_bytes: int = 0
    delegations_issued: int = 0
    delegations_completed: int = 0
    delegations_preempted: int = 0
    coordination_latencies: List[float] = field(default_factory=list)
    wall_clock_time: float = 0.0
    manager_time: float = 0.0
    worker_time: float = 0.0


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple episodes."""

    # Task effectiveness
    success_rate: float = 0.0
    normalized_return_mean: float = 0.0
    normalized_return_std: float = 0.0

    # Time and resource efficiency
    makespan_mean: float = 0.0  # steps-to-success
    makespan_std: float = 0.0
    action_efficiency: float = 0.0  # primitive actions per task
    wall_clock_per_episode: float = 0.0
    manager_time_percent: float = 0.0
    worker_time_percent: float = 0.0

    # Hierarchy overhead
    manager_utilization: float = 0.0  # manager actions per 100 env steps
    delegation_success_rate: float = 0.0
    preemption_rate: float = 0.0  # preemptions per delegation

    # Communication cost
    messages_per_episode: float = 0.0
    bytes_per_step: float = 0.0
    coordination_latency_mean: float = 0.0
    coordination_latency_std: float = 0.0

    # Additional stats
    total_episodes: int = 0
    successful_episodes: int = 0


class TaskEffectivenessTracker:
    """Tracks task effectiveness metrics: success rate, normalized return."""

    def __init__(self, r_min: float = 0.0, r_max: float = 100.0):
        self.r_min = r_min
        self.r_max = r_max
        self.episodes: List[EpisodeMetrics] = []
        self.current_episode: Optional[EpisodeMetrics] = None
        self.episode_counter = 0

    def start_episode(self):
        """Start tracking a new episode."""
        self.episode_counter += 1
        self.current_episode = EpisodeMetrics(
            episode_id=self.episode_counter, success=False, total_return=0.0
        )

    def end_episode(self, success: bool, total_return: float, steps: int):
        """End current episode and record results."""
        if self.current_episode:
            self.current_episode.success = success
            self.current_episode.total_return = total_return
            self.current_episode.total_steps = steps
            if success:
                self.current_episode.steps_to_success = steps
            self.episodes.append(self.current_episode)
            self.current_episode = None

    def get_success_rate(self) -> float:
        """Calculate success rate across all episodes."""
        if not self.episodes:
            return 0.0
        successful = sum(1 for ep in self.episodes if ep.success)
        return successful / len(self.episodes)

    def get_normalized_return(self) -> Tuple[float, float]:
        """Calculate mean and std of normalized return."""
        if not self.episodes:
            return 0.0, 0.0

        normalized_returns = [
            (
                (ep.total_return - self.r_min) / (self.r_max - self.r_min)
                if (self.r_max - self.r_min) > 0
                else 0.0
            )
            for ep in self.episodes
        ]

        mean_return = statistics.mean(normalized_returns)
        std_return = (
            statistics.stdev(normalized_returns)
            if len(normalized_returns) > 1
            else 0.0
        )

        return mean_return, std_return

    def get_makespan_stats(self) -> Tuple[float, float]:
        """Get mean and std of steps-to-success for successful episodes."""
        successful_episodes = [ep for ep in self.episodes if ep.success]
        if not successful_episodes:
            return 0.0, 0.0

        makespans = [
            ep.steps_to_success
            for ep in successful_episodes
            if ep.steps_to_success
        ]
        if not makespans:
            return 0.0, 0.0

        mean_makespan = statistics.mean(makespans)
        std_makespan = (
            statistics.stdev(makespans) if len(makespans) > 1 else 0.0
        )

        return mean_makespan, std_makespan


class HierarchyOverheadTracker:
    """Tracks hierarchy-specific overhead metrics."""

    def __init__(self):
        self.manager_actions_count = 0
        self.total_env_steps = 0
        self.delegations_issued = 0
        self.delegations_completed = 0
        self.delegations_failed = 0
        self.preemptions = 0
        self.lock = threading.Lock()

    def record_manager_action(self):
        """Record a manager action."""
        with self.lock:
            self.manager_actions_count += 1

    def record_env_step(self):
        """Record an environment step."""
        with self.lock:
            self.total_env_steps += 1

    def record_delegation(
        self,
        issued: bool = True,
        completed: bool = False,
        failed: bool = False,
        preempted: bool = False,
    ):
        """Record delegation event."""
        with self.lock:
            if issued:
                self.delegations_issued += 1
            if completed:
                self.delegations_completed += 1
            if failed:
                self.delegations_failed += 1
            if preempted:
                self.preemptions += 1

    def get_manager_utilization(self) -> float:
        """Calculate manager actions per 100 environment steps."""
        with self.lock:
            if self.total_env_steps == 0:
                return 0.0
            return (self.manager_actions_count / self.total_env_steps) * 100

    def get_delegation_success_rate(self) -> float:
        """Calculate fraction of delegations completed successfully."""
        with self.lock:
            if self.delegations_issued == 0:
                return 0.0
            return self.delegations_completed / self.delegations_issued

    def get_preemption_rate(self) -> float:
        """Calculate preemptions per delegation."""
        with self.lock:
            if self.delegations_issued == 0:
                return 0.0
            return self.preemptions / self.delegations_issued

    def get_stats(self) -> Dict[str, float]:
        """Get all hierarchy overhead statistics."""
        return {
            "manager_utilization": self.get_manager_utilization(),
            "delegation_success_rate": self.get_delegation_success_rate(),
            "preemption_rate": self.get_preemption_rate(),
            "total_delegations": self.delegations_issued,
            "completed_delegations": self.delegations_completed,
            "preemptions": self.preemptions,
        }


class CommunicationCostTracker:
    """Tracks communication costs: messages, bytes, latency."""

    def __init__(self):
        self.message_count = 0
        self.total_bytes = 0
        self.total_steps = 0
        self.coordination_events: Dict[str, float] = (
            {}
        )  # task_id -> start_time
        self.coordination_latencies: List[float] = []
        self.lock = threading.Lock()

    def record_message(self, message_bytes: int = 100):
        """Record a message sent."""
        with self.lock:
            self.message_count += 1
            self.total_bytes += message_bytes

    def record_step(self):
        """Record an environment step."""
        with self.lock:
            self.total_steps += 1

    def start_coordination(self, task_id: str):
        """Start tracking coordination for a task."""
        with self.lock:
            self.coordination_events[task_id] = time.perf_counter()

    def end_coordination(self, task_id: str) -> Optional[float]:
        """End coordination tracking and record latency."""
        with self.lock:
            if task_id in self.coordination_events:
                start_time = self.coordination_events.pop(task_id)
                latency = time.perf_counter() - start_time
                self.coordination_latencies.append(latency)
                return latency
        return None

    def get_messages_per_episode(self, num_episodes: int) -> float:
        """Calculate messages per episode."""
        with self.lock:
            if num_episodes == 0:
                return 0.0
            return self.message_count / num_episodes

    def get_bytes_per_step(self) -> float:
        """Calculate average bytes per step."""
        with self.lock:
            if self.total_steps == 0:
                return 0.0
            return self.total_bytes / self.total_steps

    def get_coordination_latency_stats(self) -> Tuple[float, float]:
        """Get mean and std of coordination latencies."""
        with self.lock:
            if not self.coordination_latencies:
                return 0.0, 0.0

            mean_latency = statistics.mean(self.coordination_latencies)
            std_latency = (
                statistics.stdev(self.coordination_latencies)
                if len(self.coordination_latencies) > 1
                else 0.0
            )

            return mean_latency, std_latency

    def get_stats(self, num_episodes: int = 1) -> Dict[str, float]:
        """Get all communication cost statistics."""
        mean_latency, std_latency = self.get_coordination_latency_stats()
        return {
            "messages_per_episode": self.get_messages_per_episode(
                num_episodes
            ),
            "bytes_per_step": self.get_bytes_per_step(),
            "coordination_latency_mean": mean_latency,
            "coordination_latency_std": std_latency,
            "total_messages": self.message_count,
            "total_bytes": self.total_bytes,
        }


class ResourceEfficiencyTracker:
    """Tracks resource usage: compute time, action counts."""

    def __init__(self):
        self.wall_clock_times: List[float] = []
        self.manager_times: List[float] = []
        self.worker_times: List[float] = []
        self.primitive_actions: List[int] = []
        self.current_episode_start: Optional[float] = None
        self.lock = threading.Lock()

    def start_episode(self):
        """Start tracking episode timing."""
        with self.lock:
            self.current_episode_start = time.perf_counter()

    def end_episode(
        self, manager_time: float, worker_time: float, primitive_actions: int
    ):
        """End episode and record resource usage."""
        with self.lock:
            if self.current_episode_start:
                wall_clock = time.perf_counter() - self.current_episode_start
                self.wall_clock_times.append(wall_clock)
                self.manager_times.append(manager_time)
                self.worker_times.append(worker_time)
                self.primitive_actions.append(primitive_actions)
                self.current_episode_start = None

    def get_stats(self) -> Dict[str, float]:
        """Get resource efficiency statistics."""
        with self.lock:
            if not self.wall_clock_times:
                return {
                    "wall_clock_per_episode": 0.0,
                    "manager_time_percent": 0.0,
                    "worker_time_percent": 0.0,
                    "action_efficiency": 0.0,
                }

            avg_wall_clock = statistics.mean(self.wall_clock_times)
            avg_manager_time = statistics.mean(self.manager_times)
            avg_worker_time = statistics.mean(self.worker_times)
            total_compute = avg_manager_time + avg_worker_time

            manager_percent = (
                (avg_manager_time / total_compute * 100)
                if total_compute > 0
                else 0.0
            )
            worker_percent = (
                (avg_worker_time / total_compute * 100)
                if total_compute > 0
                else 0.0
            )

            avg_actions = (
                statistics.mean(self.primitive_actions)
                if self.primitive_actions
                else 0.0
            )

            return {
                "wall_clock_per_episode": avg_wall_clock,
                "manager_time_percent": manager_percent,
                "worker_time_percent": worker_percent,
                "action_efficiency": avg_actions,
            }


class RobustnessTracker:
    """Tracks robustness metrics: fault tolerance, recovery time."""

    def __init__(self):
        self.fault_events: List[Dict[str, Any]] = []
        self.recovery_times: List[float] = []
        self.performance_degradation: List[float] = []
        self.active_faults: Dict[str, float] = {}  # fault_id -> start_time
        self.lock = threading.Lock()

    def inject_fault(self, fault_id: str, fault_type: str, severity: float):
        """Record a fault injection event."""
        with self.lock:
            self.active_faults[fault_id] = time.perf_counter()
            self.fault_events.append(
                {
                    "fault_id": fault_id,
                    "fault_type": fault_type,
                    "severity": severity,
                    "timestamp": time.time(),
                }
            )

    def record_recovery(self, fault_id: str, performance_drop: float):
        """Record recovery from a fault."""
        with self.lock:
            if fault_id in self.active_faults:
                start_time = self.active_faults.pop(fault_id)
                recovery_time = time.perf_counter() - start_time
                self.recovery_times.append(recovery_time)
                self.performance_degradation.append(performance_drop)

    def get_stats(self) -> Dict[str, float]:
        """Get robustness statistics."""
        with self.lock:
            avg_recovery = (
                statistics.mean(self.recovery_times)
                if self.recovery_times
                else 0.0
            )
            avg_degradation = (
                statistics.mean(self.performance_degradation)
                if self.performance_degradation
                else 0.0
            )

            return {
                "fault_count": len(self.fault_events),
                "avg_recovery_time": avg_recovery,
                "avg_performance_degradation": avg_degradation,
                "unrecovered_faults": len(self.active_faults),
            }


class HierarchyBenchmarkTracker:
    """
    Comprehensive tracker for hierarchy strategy benchmarks.
    Aggregates all metric trackers and provides unified interface.
    """

    def __init__(self, r_min: float = 0.0, r_max: float = 100.0):
        self.task_tracker = TaskEffectivenessTracker(r_min, r_max)
        self.hierarchy_tracker = HierarchyOverheadTracker()
        self.communication_tracker = CommunicationCostTracker()
        self.resource_tracker = ResourceEfficiencyTracker()
        self.robustness_tracker = RobustnessTracker()

        # Episode tracking
        self.current_episode_metrics: Optional[EpisodeMetrics] = None
        self.episode_start_time: Optional[float] = None
        self.manager_action_time: float = 0.0
        self.worker_action_time: float = 0.0

    def start_episode(self):
        """Start tracking a new episode."""
        self.task_tracker.start_episode()
        self.resource_tracker.start_episode()
        self.episode_start_time = time.perf_counter()
        self.manager_action_time = 0.0
        self.worker_action_time = 0.0
        self.current_episode_metrics = self.task_tracker.current_episode

    def end_episode(
        self,
        success: bool,
        total_return: float,
        steps: int,
        primitive_actions: int,
    ):
        """End current episode and finalize metrics."""
        self.task_tracker.end_episode(success, total_return, steps)
        self.resource_tracker.end_episode(
            self.manager_action_time,
            self.worker_action_time,
            primitive_actions,
        )

        if self.current_episode_metrics and self.episode_start_time:
            self.current_episode_metrics.wall_clock_time = (
                time.perf_counter() - self.episode_start_time
            )
            self.current_episode_metrics.manager_time = (
                self.manager_action_time
            )
            self.current_episode_metrics.worker_time = self.worker_action_time
            self.current_episode_metrics.primitive_actions = primitive_actions

    def record_action(self, is_manager: bool, execution_time: float):
        """Record an action execution."""
        if is_manager:
            self.hierarchy_tracker.record_manager_action()
            self.manager_action_time += execution_time
        else:
            self.worker_action_time += execution_time

    def record_step(self):
        """Record an environment step."""
        self.hierarchy_tracker.record_env_step()
        self.communication_tracker.record_step()

    def record_message(self, message_bytes: int = 100):
        """Record a message sent."""
        self.communication_tracker.record_message(message_bytes)
        if self.current_episode_metrics:
            self.current_episode_metrics.total_messages += 1
            self.current_episode_metrics.total_bytes += message_bytes

    def record_delegation(
        self,
        issued: bool = False,
        completed: bool = False,
        failed: bool = False,
        preempted: bool = False,
    ):
        """Record a delegation event."""
        self.hierarchy_tracker.record_delegation(
            issued, completed, failed, preempted
        )
        if self.current_episode_metrics:
            if issued:
                self.current_episode_metrics.delegations_issued += 1
            if completed:
                self.current_episode_metrics.delegations_completed += 1
            if preempted:
                self.current_episode_metrics.delegations_preempted += 1

    def start_coordination(self, task_id: str):
        """Start tracking coordination latency."""
        self.communication_tracker.start_coordination(task_id)

    def end_coordination(self, task_id: str):
        """End coordination tracking."""
        latency = self.communication_tracker.end_coordination(task_id)
        if latency and self.current_episode_metrics:
            self.current_episode_metrics.coordination_latencies.append(latency)

    def inject_fault(self, fault_id: str, fault_type: str, severity: float):
        """Inject a fault for robustness testing."""
        self.robustness_tracker.inject_fault(fault_id, fault_type, severity)

    def record_recovery(self, fault_id: str, performance_drop: float):
        """Record recovery from fault."""
        self.robustness_tracker.record_recovery(fault_id, performance_drop)

    def get_aggregated_metrics(self) -> AggregatedMetrics:
        """Get aggregated metrics across all episodes."""
        num_episodes = len(self.task_tracker.episodes)

        # Task effectiveness
        success_rate = self.task_tracker.get_success_rate()
        norm_return_mean, norm_return_std = (
            self.task_tracker.get_normalized_return()
        )
        makespan_mean, makespan_std = self.task_tracker.get_makespan_stats()

        # Hierarchy overhead
        hierarchy_stats = self.hierarchy_tracker.get_stats()

        # Communication cost
        comm_stats = self.communication_tracker.get_stats(num_episodes)

        # Resource efficiency
        resource_stats = self.resource_tracker.get_stats()

        return AggregatedMetrics(
            success_rate=success_rate,
            normalized_return_mean=norm_return_mean,
            normalized_return_std=norm_return_std,
            makespan_mean=makespan_mean,
            makespan_std=makespan_std,
            action_efficiency=resource_stats["action_efficiency"],
            wall_clock_per_episode=resource_stats["wall_clock_per_episode"],
            manager_time_percent=resource_stats["manager_time_percent"],
            worker_time_percent=resource_stats["worker_time_percent"],
            manager_utilization=hierarchy_stats["manager_utilization"],
            delegation_success_rate=hierarchy_stats["delegation_success_rate"],
            preemption_rate=hierarchy_stats["preemption_rate"],
            messages_per_episode=comm_stats["messages_per_episode"],
            bytes_per_step=comm_stats["bytes_per_step"],
            coordination_latency_mean=comm_stats["coordination_latency_mean"],
            coordination_latency_std=comm_stats["coordination_latency_std"],
            total_episodes=num_episodes,
            successful_episodes=(
                int(success_rate * num_episodes) if num_episodes > 0 else 0
            ),
        )

    def export_results(self) -> Dict[str, Any]:
        """Export all metrics and episode data."""
        aggregated = self.get_aggregated_metrics()

        return {
            "aggregated_metrics": {
                "task_effectiveness": {
                    "success_rate": aggregated.success_rate,
                    "normalized_return_mean": aggregated.normalized_return_mean,
                    "normalized_return_std": aggregated.normalized_return_std,
                },
                "time_efficiency": {
                    "makespan_mean": aggregated.makespan_mean,
                    "makespan_std": aggregated.makespan_std,
                    "action_efficiency": aggregated.action_efficiency,
                    "wall_clock_per_episode": aggregated.wall_clock_per_episode,
                },
                "hierarchy_overhead": {
                    "manager_utilization": aggregated.manager_utilization,
                    "delegation_success_rate": aggregated.delegation_success_rate,
                    "preemption_rate": aggregated.preemption_rate,
                    "manager_time_percent": aggregated.manager_time_percent,
                    "worker_time_percent": aggregated.worker_time_percent,
                },
                "communication_cost": {
                    "messages_per_episode": aggregated.messages_per_episode,
                    "bytes_per_step": aggregated.bytes_per_step,
                    "coordination_latency_mean": aggregated.coordination_latency_mean,
                    "coordination_latency_std": aggregated.coordination_latency_std,
                },
                "robustness": self.robustness_tracker.get_stats(),
            },
            "episode_count": aggregated.total_episodes,
            "successful_episodes": aggregated.successful_episodes,
            "episodes": [
                {
                    "episode_id": ep.episode_id,
                    "success": ep.success,
                    "total_return": ep.total_return,
                    "steps": ep.total_steps,
                    "steps_to_success": ep.steps_to_success,
                    "primitive_actions": ep.primitive_actions,
                    "messages": ep.total_messages,
                    "delegations": ep.delegations_issued,
                    "delegations_completed": ep.delegations_completed,
                }
                for ep in self.task_tracker.episodes
            ],
        }
