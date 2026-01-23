"""
Hierarchy Task Environments
Goal-based task environments for testing hierarchy strategies.

Provides measurable multi-agent coordination tasks:
- Resource allocation
- Task distribution
- Collaborative problem solving
- Fault recovery scenarios
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Set
from dataclasses import dataclass, field
import random
import time
from benchmarks.hierarchy_benchmarks.hierarchy_strategies import Task, HierarchyStrategy


@dataclass
class EnvironmentState:
    """State of the task environment."""

    step_count: int = 0
    total_return: float = 0.0
    episode_complete: bool = False
    success: bool = False
    resources: Dict[str, int] = field(default_factory=dict)
    shared_state: Dict[str, Any] = field(default_factory=dict)


class HierarchyEnvironment(ABC):
    """Abstract base class for hierarchy task environments."""

    def __init__(self, num_agents: int, max_steps: int = 100):
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.state = EnvironmentState()
        self.tasks: List[Task] = []
        self.r_min = 0.0
        self.r_max = 100.0

    @abstractmethod
    def reset(self) -> EnvironmentState:
        """Reset environment to initial state."""
        pass

    @abstractmethod
    def generate_tasks(self, num_tasks: int) -> List[Task]:
        """Generate tasks for the episode."""
        pass

    @abstractmethod
    def step(
        self, hierarchy: HierarchyStrategy
    ) -> Tuple[EnvironmentState, float, bool]:
        """
        Execute one environment step.
        Returns: (new_state, reward, done)
        """
        pass

    @abstractmethod
    def check_success(self) -> bool:
        """Check if episode goal is achieved."""
        pass

    def get_normalized_return(self) -> float:
        """Get normalized return for current episode."""
        if self.r_max == self.r_min:
            return 0.0
        return (self.state.total_return - self.r_min) / (
            self.r_max - self.r_min
        )


class ResourceAllocationEnvironment(HierarchyEnvironment):
    """
    Resource Allocation Environment
    Agents must coordinate to efficiently allocate limited resources to tasks.

    Goal: Maximize resource utilization while completing all tasks.
    Success: All tasks completed within resource constraints.
    """

    def __init__(
        self,
        num_agents: int,
        num_resources: int = 3,
        max_steps: int = 100,
        *,
        completion_threshold: float = 0.9,
        over_allocation_tolerance: float = 0.05,
    ):
        super().__init__(num_agents, max_steps)
        self.num_resources = num_resources
        self.resource_capacity: Dict[str, int] = {}
        self.resource_usage: Dict[str, int] = {}
        self.task_requirements: Dict[str, Dict[str, int]] = {}
        self.active_allocations: Set[str] = set()
        self.completed_task_ids: Set[str] = set()
        self.over_allocation_violation = False
        self.completion_threshold = completion_threshold
        self.over_allocation_tolerance = over_allocation_tolerance
        self.r_max = 100.0  # Perfect allocation

    def reset(self) -> EnvironmentState:
        """Reset environment."""
        self.state = EnvironmentState()
        self.tasks = []
        self.task_requirements = {}
        self.active_allocations = set()
        self.completed_task_ids = set()
        self.over_allocation_violation = False
        # Initialize resources
        self.resource_capacity = {
            f"resource_{i}": random.randint(10, 20)
            for i in range(self.num_resources)
        }
        self.resource_usage = {k: 0 for k in self.resource_capacity}
        self.state.resources = self.resource_capacity.copy()

        return self.state

    def generate_tasks(self, num_tasks: int) -> List[Task]:
        """Generate resource allocation tasks."""
        self.tasks = []
        self.task_requirements = {}
        for i in range(num_tasks):
            required_resources = {}
            for resource_name in random.sample(
                list(self.resource_capacity.keys()),
                k=random.randint(1, self.num_resources),
            ):
                required_resources[resource_name] = random.randint(1, 5)

            task = Task(
                task_id=f"resource_task_{i}",
                task_type="resource_allocation",
                complexity=sum(required_resources.values()),
                status="pending",
            )
            task.subtasks = [f"{k}:{v}" for k, v in required_resources.items()]
            self.tasks.append(task)
            self.task_requirements[task.task_id] = required_resources

        return self.tasks

    def step(
        self, hierarchy: HierarchyStrategy
    ) -> Tuple[EnvironmentState, float, bool]:
        """Execute one step."""
        self.state.step_count += 1
        reward = 0.0

        # Process hierarchy step
        hierarchy.process_step()

        # Track resource allocations based on task state
        for task in self.tasks:
            requirements = self.task_requirements.get(task.task_id, {})

            if (
                task.status == "in_progress"
                and task.task_id not in self.active_allocations
            ):
                if self._can_allocate(requirements):
                    self._allocate_resources(task.task_id, requirements)
                else:
                    # Penalize attempting to exceed capacity
                    reward -= 2.0
                    self.over_allocation_violation = True

            if (
                task.status not in {"in_progress"}
                and task.task_id in self.active_allocations
            ):
                self._release_resources(task.task_id)

            if (
                task.status == "completed"
                and task.task_id not in self.completed_task_ids
            ):
                reward += 10.0
                self.completed_task_ids.add(task.task_id)
                if task.task_id in self.active_allocations:
                    self._release_resources(task.task_id)

        # Penalty for resource over-allocation
        for resource_name, usage in self.resource_usage.items():
            capacity = self.resource_capacity[resource_name]
            if usage > capacity * (1 + self.over_allocation_tolerance):
                reward -= 5.0
                self.over_allocation_violation = True

        # Bonus for efficient utilization
        total_capacity = sum(self.resource_capacity.values())
        total_usage = sum(self.resource_usage.values())
        if total_capacity > 0:
            utilization = total_usage / total_capacity
            if 0.7 <= utilization <= 0.9:  # Optimal range
                reward += 5.0

        self.state.total_return += reward

        # Check if done
        done = self.state.step_count >= self.max_steps or self.check_success()
        self.state.episode_complete = done
        self.state.success = self.check_success()

        return self.state, reward, done

    def check_success(self) -> bool:
        """Success if all tasks completed without over-allocation."""
        total_tasks = len(self.tasks)
        if total_tasks == 0:
            return True

        completion_rate = len(self.completed_task_ids) / total_tasks
        return (
            completion_rate >= self.completion_threshold
            and not self.over_allocation_violation
        )

    def _can_allocate(self, requirements: Dict[str, int]) -> bool:
        """Check if resources can be allocated within tolerance."""
        for resource, amount in requirements.items():
            capacity = self.resource_capacity.get(resource, 0)
            current = self.resource_usage.get(resource, 0)
            if current + amount > capacity * (1 + self.over_allocation_tolerance):
                return False
        return True

    def _allocate_resources(self, task_id: str, requirements: Dict[str, int]):
        """Allocate resources for a task."""
        for resource, amount in requirements.items():
            self.resource_usage[resource] = self.resource_usage.get(resource, 0) + amount
        self.active_allocations.add(task_id)

    def _release_resources(self, task_id: str):
        """Release resources when a task stops using them."""
        requirements = self.task_requirements.get(task_id, {})
        for resource, amount in requirements.items():
            self.resource_usage[resource] = max(
                0, self.resource_usage.get(resource, 0) - amount
            )
        self.active_allocations.discard(task_id)


class TaskDistributionEnvironment(HierarchyEnvironment):
    """
    Task Distribution Environment
    Agents must distribute and complete tasks efficiently.

    Goal: Complete maximum tasks with minimum makespan.
    Success: All tasks completed efficiently.
    """

    def __init__(self, num_agents: int, max_steps: int = 100):
        super().__init__(num_agents, max_steps)
        self.task_completion_times: List[float] = []
        self.r_max = 100.0

    def reset(self) -> EnvironmentState:
        """Reset environment."""
        self.state = EnvironmentState()
        self.tasks = []
        self.task_completion_times = []
        return self.state

    def generate_tasks(self, num_tasks: int) -> List[Task]:
        """Generate distribution tasks."""
        self.tasks = []

        for i in range(num_tasks):
            complexity = random.uniform(0.5, 2.0)
            task = Task(
                task_id=f"dist_task_{i}",
                task_type="distribution",
                complexity=complexity,
                status="pending",
            )
            self.tasks.append(task)

        return self.tasks

    def step(
        self, hierarchy: HierarchyStrategy
    ) -> Tuple[EnvironmentState, float, bool]:
        """Execute one step."""
        self.state.step_count += 1
        reward = 0.0

        # Process hierarchy step
        actions = hierarchy.process_step()

        # Reward for task completion
        for task in self.tasks:
            if task.status == "completed" and task.end_time:
                if task.task_id not in [
                    f"dist_task_{i}"
                    for i in range(len(self.task_completion_times))
                ]:
                    completion_time = self.state.step_count
                    self.task_completion_times.append(completion_time)

                    # Reward inversely proportional to completion time
                    time_bonus = max(0, 10.0 - (completion_time / 10.0))
                    reward += 10.0 + time_bonus

        # Penalty for inefficiency (too many manager actions)
        if actions.get("manager", 0) > self.num_agents:
            reward -= 1.0

        self.state.total_return += reward

        # Check if done
        done = self.state.step_count >= self.max_steps or self.check_success()
        self.state.episode_complete = done
        self.state.success = self.check_success()

        return self.state, reward, done

    def check_success(self) -> bool:
        """Success if all tasks completed."""
        return all(task.status == "completed" for task in self.tasks)


class CollaborativeProblemSolving(HierarchyEnvironment):
    """
    Collaborative Problem Solving Environment
    Multi-step tasks requiring coordination between agents.

    Goal: Solve complex problems through agent collaboration.
    Success: All sub-problems solved in correct order.
    """

    def __init__(
        self,
        num_agents: int,
        problem_complexity: int = 3,
        max_steps: int = 150,
        *,
        success_threshold: float = 0.9,
        dependency_penalty: float = 5.0,
        max_dependency_violation_rate: float = 0.4,
    ):
        super().__init__(num_agents, max_steps)
        self.problem_complexity = problem_complexity
        self.problem_graph: Dict[str, List[str]] = (
            {}
        )  # task_id -> dependencies
        self.success_threshold = success_threshold
        self.dependency_penalty = dependency_penalty
        self.out_of_order_requeues = 0
        self.max_dependency_violation_rate = max_dependency_violation_rate
        self.r_max = 100.0

    def reset(self) -> EnvironmentState:
        """Reset environment."""
        self.state = EnvironmentState()
        self.tasks = []
        self.problem_graph = {}
        self.out_of_order_requeues = 0
        return self.state

    def generate_tasks(self, num_tasks: int) -> List[Task]:
        """Generate collaborative tasks with dependencies."""
        self.tasks = []
        self.problem_graph = {}

        # Create task dependency graph
        for i in range(num_tasks):
            task = Task(
                task_id=f"collab_task_{i}",
                task_type="collaborative",
                complexity=random.uniform(0.5, 1.5),
                status="pending",
            )

            # Add dependencies (tasks depend on previous tasks)
            dependencies = []
            if i > 0:
                # Depend on 1-2 previous tasks
                num_deps = min(i, random.randint(1, 2))
                dependencies = random.sample(
                    [f"collab_task_{j}" for j in range(i)], k=num_deps
                )

            self.problem_graph[task.task_id] = dependencies
            task.subtasks = dependencies
            self.tasks.append(task)

        return self.tasks

    def step(
        self, hierarchy: HierarchyStrategy
    ) -> Tuple[EnvironmentState, float, bool]:
        """Execute one step."""
        self.state.step_count += 1
        reward = 0.0

        # Process hierarchy step
        hierarchy.process_step()

        # Check task completion with dependency validation
        for task in self.tasks:
            deps_satisfied = self._dependencies_satisfied(task.task_id, hierarchy)

            if task.status == "in_progress" and not deps_satisfied:
                # Pause progress until dependencies are satisfied
                self._reset_task_assignment(hierarchy, task, requeue=True)
                reward -= self.dependency_penalty / 2
                continue

            if task.status == "completed":
                if deps_satisfied:
                    reward += 15.0
                else:
                    reward -= self.dependency_penalty
                    self._reset_task_assignment(hierarchy, task, requeue=True)
                    self.out_of_order_requeues += 1

        self.state.total_return += reward

        # Check if done
        done = self.state.step_count >= self.max_steps or self.check_success()
        self.state.episode_complete = done
        self.state.success = self.check_success()

        return self.state, reward, done

    def check_success(self) -> bool:
        """Success if sufficient tasks complete respecting dependencies."""
        total = len(self.tasks)
        if total == 0:
            return True

        completed = sum(1 for task in self.tasks if task.status == "completed")
        completion_rate = completed / total

        if completion_rate < self.success_threshold:
            return False

        allowed_requeues = max(
            1, int(self.max_dependency_violation_rate * total)
        )
        return self.out_of_order_requeues <= allowed_requeues

    def _dependencies_satisfied(
        self, task_id: str, hierarchy: HierarchyStrategy
    ) -> bool:
        """Check whether dependencies are complete."""
        dependencies = self.problem_graph.get(task_id, [])
        if not dependencies:
            return True
        return all(
            hierarchy.tasks.get(dep_id, Task("", "")).status == "completed"
            for dep_id in dependencies
        )

    def _reset_task_assignment(
        self,
        hierarchy: HierarchyStrategy,
        task: Task,
        *,
        requeue: bool = False,
    ):
        """Reset task state and free worker when dependencies are not met."""
        assigned_agent = task.assigned_to
        if assigned_agent and assigned_agent in hierarchy.agents:
            hierarchy.agents[assigned_agent].current_task = None
        if requeue:
            task.status = "pending"
            task.assigned_to = None
            task.start_time = None
            task.end_time = None


class FaultRecoveryEnvironment(HierarchyEnvironment):
    """
    Fault Recovery Environment
    Tests robustness by injecting agent failures during execution.

    Goal: Complete tasks despite agent failures.
    Success: Recovery from faults with minimal performance degradation.
    """

    def __init__(
        self,
        num_agents: int,
        failure_probability: float = 0.1,
        max_steps: int = 100,
    ):
        super().__init__(num_agents, max_steps)
        self.failure_probability = failure_probability
        self.failed_agents: Set[str] = set()
        self.recovery_count = 0
        self.r_max = 100.0

    def reset(self) -> EnvironmentState:
        """Reset environment."""
        self.state = EnvironmentState()
        self.tasks = []
        self.failed_agents = set()
        self.recovery_count = 0
        return self.state

    def generate_tasks(self, num_tasks: int) -> List[Task]:
        """Generate fault recovery tasks."""
        self.tasks = []

        for i in range(num_tasks):
            task = Task(
                task_id=f"fault_task_{i}",
                task_type="fault_recovery",
                complexity=random.uniform(0.5, 1.5),
                status="pending",
            )
            self.tasks.append(task)

        return self.tasks

    def step(
        self, hierarchy: HierarchyStrategy
    ) -> Tuple[EnvironmentState, float, bool]:
        """Execute one step with potential failures."""
        self.state.step_count += 1
        reward = 0.0

        # Random agent failures
        if random.random() < self.failure_probability:
            active_agents = hierarchy.get_active_agents()
            if active_agents:
                failed_agent = random.choice(active_agents)
                if hierarchy.handle_agent_failure(failed_agent):
                    self.failed_agents.add(failed_agent)
                    reward -= 5.0  # Penalty for failure

        # Process hierarchy step
        hierarchy.process_step()

        # Reward for task completion
        for task in self.tasks:
            if task.status == "completed":
                reward += 10.0

                # Bonus if completed despite failures
                if len(self.failed_agents) > 0:
                    reward += 5.0
                    self.recovery_count += 1

        self.state.total_return += reward
        self.state.shared_state["failed_agents"] = len(self.failed_agents)
        self.state.shared_state["recovery_count"] = self.recovery_count

        # Check if done
        done = self.state.step_count >= self.max_steps or self.check_success()
        self.state.episode_complete = done
        self.state.success = self.check_success()

        return self.state, reward, done

    def check_success(self) -> bool:
        """Success if majority of tasks completed despite failures."""
        completed_count = sum(
            1 for task in self.tasks if task.status == "completed"
        )
        completion_rate = (
            completed_count / len(self.tasks) if self.tasks else 0.0
        )

        # Success if >80% tasks completed
        return completion_rate >= 0.8


class ScalabilityTestEnvironment(HierarchyEnvironment):
    """
    Scalability Test Environment
    Tests performance scaling with increasing team sizes.

    Goal: Maintain efficiency as team size increases.
    Success: Performance scales sub-linearly with team size.
    """

    def __init__(
        self, num_agents: int, tasks_per_agent: int = 2, max_steps: int = 100
    ):
        super().__init__(num_agents, max_steps)
        self.tasks_per_agent = tasks_per_agent
        self.coordination_overhead: List[int] = []
        self.r_max = 100.0

    def reset(self) -> EnvironmentState:
        """Reset environment."""
        self.state = EnvironmentState()
        self.tasks = []
        self.coordination_overhead = []
        return self.state

    def generate_tasks(self, num_tasks: int = None) -> List[Task]:
        """Generate scalability tasks."""
        if num_tasks is None:
            num_tasks = self.num_agents * self.tasks_per_agent

        self.tasks = []

        for i in range(num_tasks):
            task = Task(
                task_id=f"scale_task_{i}",
                task_type="scalability",
                complexity=1.0,
                status="pending",
            )
            self.tasks.append(task)

        return self.tasks

    def step(
        self, hierarchy: HierarchyStrategy
    ) -> Tuple[EnvironmentState, float, bool]:
        """Execute one step."""
        self.state.step_count += 1
        reward = 0.0

        # Process hierarchy step
        actions = hierarchy.process_step()

        # Track coordination overhead
        total_actions = sum(actions.values())
        self.coordination_overhead.append(total_actions)

        # Reward for task completion
        completed_this_step = sum(
            1
            for task in self.tasks
            if task.status == "completed" and task.end_time == time.time()
        )
        reward += completed_this_step * 5.0

        # Penalty for excessive coordination overhead
        avg_overhead = sum(self.coordination_overhead) / len(
            self.coordination_overhead
        )
        if avg_overhead > self.num_agents * 2:  # More than 2x agent count
            reward -= 2.0

        self.state.total_return += reward
        self.state.shared_state["coordination_overhead"] = avg_overhead

        # Check if done
        done = self.state.step_count >= self.max_steps or self.check_success()
        self.state.episode_complete = done
        self.state.success = self.check_success()

        return self.state, reward, done

    def check_success(self) -> bool:
        """Success if all tasks completed efficiently."""
        all_completed = all(task.status == "completed" for task in self.tasks)

        # Check coordination efficiency
        if self.coordination_overhead:
            avg_overhead = sum(self.coordination_overhead) / len(
                self.coordination_overhead
            )
            efficient = avg_overhead <= self.num_agents * 3
        else:
            efficient = False

        return all_completed and efficient


def create_environment(
    env_type: str, num_agents: int, **kwargs
) -> HierarchyEnvironment:
    """Factory function to create hierarchy environments."""

    if env_type == "resource_allocation":
        num_resources = kwargs.get("num_resources", 3)
        max_steps = kwargs.get("max_steps", 100)
        completion_threshold = kwargs.get("completion_threshold", 0.9)
        over_allocation_tolerance = kwargs.get(
            "over_allocation_tolerance", 0.05
        )
        return ResourceAllocationEnvironment(
            num_agents,
            num_resources,
            max_steps,
            completion_threshold=completion_threshold,
            over_allocation_tolerance=over_allocation_tolerance,
        )

    elif env_type == "task_distribution":
        max_steps = kwargs.get("max_steps", 100)
        return TaskDistributionEnvironment(num_agents, max_steps)

    elif env_type == "collaborative":
        complexity = kwargs.get("problem_complexity", 3)
        max_steps = kwargs.get("max_steps", 150)
        success_threshold = kwargs.get("success_threshold", 0.9)
        dependency_penalty = kwargs.get("dependency_penalty", 5.0)
        violation_rate = kwargs.get("dependency_violation_rate", 0.4)
        return CollaborativeProblemSolving(
            num_agents,
            complexity,
            max_steps,
            success_threshold=success_threshold,
            dependency_penalty=dependency_penalty,
            max_dependency_violation_rate=violation_rate,
        )

    elif env_type == "fault_recovery":
        failure_prob = kwargs.get("failure_probability", 0.1)
        max_steps = kwargs.get("max_steps", 100)
        return FaultRecoveryEnvironment(num_agents, failure_prob, max_steps)

    elif env_type == "scalability":
        tasks_per_agent = kwargs.get("tasks_per_agent", 2)
        max_steps = kwargs.get("max_steps", 100)
        return ScalabilityTestEnvironment(
            num_agents, tasks_per_agent, max_steps
        )

    else:
        raise ValueError(f"Unknown environment type: {env_type}")
