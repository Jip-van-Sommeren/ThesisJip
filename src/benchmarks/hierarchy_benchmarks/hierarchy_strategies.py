"""
Hierarchy Strategy Implementations
Implements three coordination strategies for multi-agent systems:
1. Tree Hierarchy: Traditional manager-worker with centralized control
2. Peer-to-Peer: Fully distributed with consensus-based coordination
3. Hybrid: Tree structure for task allocation + peer communication for
execution
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import time
import random


class HierarchyType(Enum):
    """Types of hierarchy strategies."""

    TREE = "tree"
    PEER_TO_PEER = "peer_to_peer"
    HYBRID = "hybrid"


@dataclass
class AgentState:
    """State of an agent in the hierarchy."""

    agent_id: str
    role: str  # "manager" or "worker"
    current_task: Optional[str] = None
    task_progress: float = 0.0
    is_active: bool = True
    subordinates: Set[str] = field(default_factory=set)
    peers: Set[str] = field(default_factory=set)
    supervisor: Optional[str] = None
    beliefs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """Represents a task to be completed."""

    task_id: str
    task_type: str
    complexity: float = 1.0
    assigned_to: Optional[str] = None
    status: str = "pending"  # pending, in_progress, completed, failed
    subtasks: List[str] = field(default_factory=list)
    parent_task: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None


@dataclass
class Message:
    """Communication message between agents."""

    sender: str
    receiver: str
    message_type: (
        str  # command, delegation, report, query, response, vote, proposal
    )
    content: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    priority: int = 5


class HierarchyStrategy(ABC):
    """Abstract base class for hierarchy strategies."""

    def __init__(self, num_agents: int, hierarchy_type: HierarchyType):
        self.num_agents = num_agents
        self.hierarchy_type = hierarchy_type
        self.agents: Dict[str, AgentState] = {}
        self.tasks: Dict[str, Task] = {}
        self.message_queue: List[Message] = []
        self.step_count = 0

    @abstractmethod
    def initialize_hierarchy(self) -> Dict[str, AgentState]:
        """Initialize the hierarchy structure."""
        pass

    @abstractmethod
    def allocate_task(self, task: Task) -> bool:
        """Allocate a task to agent(s)."""
        pass

    @abstractmethod
    def process_step(self) -> Dict[str, Any]:
        """Process one step of the hierarchy."""
        pass

    @abstractmethod
    def handle_agent_failure(self, agent_id: str) -> bool:
        """Handle failure of an agent."""
        pass

    def send_message(self, message: Message):
        """Send a message through the hierarchy."""
        self.message_queue.append(message)

    def get_active_agents(self) -> List[str]:
        """Get list of active agents."""
        return [aid for aid, agent in self.agents.items() if agent.is_active]

    def get_stats(self) -> Dict[str, Any]:
        """Get hierarchy statistics."""
        completed_tasks = sum(
            1 for t in self.tasks.values() if t.status == "completed"
        )
        active_tasks = sum(
            1 for t in self.tasks.values() if t.status == "in_progress"
        )

        return {
            "total_agents": self.num_agents,
            "active_agents": len(self.get_active_agents()),
            "total_tasks": len(self.tasks),
            "completed_tasks": completed_tasks,
            "active_tasks": active_tasks,
            "messages_sent": len(self.message_queue),
            "steps": self.step_count,
        }


class TreeHierarchy(HierarchyStrategy):
    """
    Tree Hierarchy Strategy
    Traditional manager-worker hierarchy with centralized control.
    - Single root manager
    - Tree structure with supervisor-subordinate relationships
    - Top-down command flow
    - Bottom-up reporting
    """

    def __init__(self, num_agents: int, hierarchy_depth: int = 2):
        super().__init__(num_agents, HierarchyType.TREE)
        self.hierarchy_depth = hierarchy_depth
        self.managers: Set[str] = set()
        self.workers: Set[str] = set()
        self.delegation_map: Dict[str, str] = {}  # task_id -> agent_id
        self.manager_planning_frequency = (
            1  # Manager acts every step by default
        )

    def initialize_hierarchy(self) -> Dict[str, AgentState]:
        """Initialize tree hierarchy with levels."""
        self.agents.clear()
        self.managers.clear()
        self.workers.clear()

        if self.hierarchy_depth == 1:
            # Flat: 1 manager, rest workers
            num_managers = 1
        elif self.hierarchy_depth == 2:
            # 2 levels: root manager + middle managers
            num_managers = max(1, self.num_agents // 3)
        else:
            # 3+ levels: exponential distribution
            num_managers = max(1, self.num_agents // 2)

        root_id = "agent_0"
        self.agents[root_id] = AgentState(
            agent_id=root_id, role="manager", subordinates=set()
        )
        self.managers.add(root_id)

        agents_created = 1
        if self.hierarchy_depth == 1:
            for i in range(1, self.num_agents):
                worker_id = f"agent_{i}"
                self.agents[worker_id] = AgentState(
                    agent_id=worker_id, role="worker", supervisor=root_id
                )
                self.agents[root_id].subordinates.add(worker_id)
                self.workers.add(worker_id)
                agents_created += 1

        elif self.hierarchy_depth == 2:
            num_middle = min(num_managers - 1, (self.num_agents - 1) // 2)
            for i in range(1, num_middle + 1):
                manager_id = f"agent_{i}"
                self.agents[manager_id] = AgentState(
                    agent_id=manager_id,
                    role="manager",
                    supervisor=root_id,
                    subordinates=set(),
                )
                self.agents[root_id].subordinates.add(manager_id)
                self.managers.add(manager_id)
                agents_created += 1

            manager_list = list(self.managers - {root_id})
            for i in range(agents_created, self.num_agents):
                worker_id = f"agent_{i}"
                # Assign to a middle manager (round-robin)
                supervisor = (
                    manager_list[(i - agents_created) % len(manager_list)]
                    if manager_list
                    else root_id
                )
                self.agents[worker_id] = AgentState(
                    agent_id=worker_id, role="worker", supervisor=supervisor
                )
                self.agents[supervisor].subordinates.add(worker_id)
                self.workers.add(worker_id)

        else:
            # 3+ levels: more complex tree
            # For simplicity, create 3-level tree
            num_level2 = max(2, num_managers // 3)
            level2_managers = []

            for i in range(1, num_level2 + 1):
                manager_id = f"agent_{i}"
                self.agents[manager_id] = AgentState(
                    agent_id=manager_id,
                    role="manager",
                    supervisor=root_id,
                    subordinates=set(),
                )
                self.agents[root_id].subordinates.add(manager_id)
                self.managers.add(manager_id)
                level2_managers.append(manager_id)
                agents_created += 1

            # Create level 3 workers
            for i in range(agents_created, self.num_agents):
                worker_id = f"agent_{i}"
                supervisor = level2_managers[
                    (i - agents_created) % len(level2_managers)
                ]
                self.agents[worker_id] = AgentState(
                    agent_id=worker_id, role="worker", supervisor=supervisor
                )
                self.agents[supervisor].subordinates.add(worker_id)
                self.workers.add(worker_id)

        return self.agents

    def allocate_task(self, task: Task) -> bool:
        """Allocate task through manager delegation."""
        # Root manager makes allocation decision
        root_id = "agent_0"

        # Find available worker (or delegate to sub-manager)
        available_workers = [
            aid
            for aid in self.workers
            if self.agents[aid].is_active
            and self.agents[aid].current_task is None
        ]

        if not available_workers:
            return False

        # Assign to first available worker
        assigned_worker = available_workers[0]
        task.assigned_to = assigned_worker
        task.status = "in_progress"
        task.start_time = time.time()

        self.agents[assigned_worker].current_task = task.task_id
        self.delegation_map[task.task_id] = assigned_worker

        # Send delegation message
        delegation_msg = Message(
            sender=root_id,
            receiver=assigned_worker,
            message_type="delegation",
            content={"task_id": task.task_id, "task_type": task.task_type},
            priority=7,
        )
        self.send_message(delegation_msg)

        return True

    def process_step(self) -> Dict[str, Any]:
        """Process one step: manager plans, workers execute."""
        self.step_count += 1
        actions_taken = {"manager": 0, "worker": 0, "messages": 0}
        message_count_before = len(self.message_queue)

        # Manager planning (every k steps)
        if self.step_count % self.manager_planning_frequency == 0:
            for manager_id in self.managers:
                if self.agents[manager_id].is_active:
                    # Manager checks subordinates and reassigns if needed
                    self._manager_coordinate(manager_id)
                    actions_taken["manager"] += 1

        # Workers execute tasks
        for worker_id in self.workers:
            worker = self.agents[worker_id]
            if worker.is_active and worker.current_task:
                # Simulate task progress
                task = self.tasks.get(worker.current_task)
                if task:
                    # Small chance of task failure (5%)
                    import random

                    if random.random() < 0.05 and worker.task_progress < 0.5:
                        # Task fails early - needs reassignment
                        task.status = "pending"
                        task.assigned_to = None
                        worker.current_task = None
                        worker.task_progress = 0.0
                    else:
                        # Normal progress
                        worker.task_progress += 0.1
                        if worker.task_progress >= 1.0:
                            # Task completed
                            task.status = "completed"
                            task.end_time = time.time()
                            worker.current_task = None
                            worker.task_progress = 0.0

                            # Send completion report
                            report_msg = Message(
                                sender=worker_id,
                                receiver=worker.supervisor,
                                message_type="report",
                                content={
                                    "task_id": task.task_id,
                                    "status": "completed",
                                },
                            )
                            self.send_message(report_msg)

                actions_taken["worker"] += 1

        actions_taken["messages"] = len(self.message_queue) - message_count_before
        return actions_taken

    def _manager_coordinate(self, manager_id: str):
        """Manager coordination logic."""
        manager = self.agents[manager_id]

        # Check for pending tasks that need reassignment
        pending_tasks = [
            task
            for task in self.tasks.values()
            if task.status == "pending" and task.assigned_to is None
        ]

        for task in pending_tasks:
            # Try to reassign
            available_workers = [
                aid
                for aid in self.workers
                if (
                    self.agents[aid].is_active
                    and self.agents[aid].current_task is None
                    and self.agents[aid].supervisor == manager_id
                )
            ]

            if available_workers:
                # Reassign to available worker
                worker_id = available_workers[0]
                task.assigned_to = worker_id
                task.status = "in_progress"
                self.agents[worker_id].current_task = task.task_id

                # Send delegation message
                delegation_msg = Message(
                    sender=manager_id,
                    receiver=worker_id,
                    message_type="delegation",
                    content={
                        "task_id": task.task_id,
                        "task_type": task.task_type,
                        "reassigned": True,
                    },
                    priority=8,
                )
                self.send_message(delegation_msg)

        # Check subordinate status
        for subordinate_id in manager.subordinates:
            subordinate = self.agents[subordinate_id]
            if subordinate.role == "worker" and subordinate.current_task:
                # Query for status update
                query_msg = Message(
                    sender=manager_id,
                    receiver=subordinate_id,
                    message_type="query",
                    content={"query_type": "status"},
                )
                self.send_message(query_msg)

    def handle_agent_failure(self, agent_id: str) -> bool:
        """Handle agent failure by reassigning tasks."""
        if agent_id not in self.agents:
            return False

        agent = self.agents[agent_id]
        agent.is_active = False

        # If worker, reassign task to another worker
        if agent.role == "worker" and agent.current_task:
            task = self.tasks.get(agent.current_task)
            if task:
                task.assigned_to = None
                task.status = "pending"
                agent.current_task = None
                # Manager will reassign in next planning cycle
                return True

        # If manager, promote a subordinate or reassign to supervisor
        if agent.role == "manager" and agent.subordinates:
            supervisor_id = agent.supervisor
            if supervisor_id:
                # Reassign all subordinates to supervisor
                for subordinate_id in agent.subordinates:
                    self.agents[subordinate_id].supervisor = supervisor_id
                    self.agents[supervisor_id].subordinates.add(subordinate_id)

        return True


class PeerToPeerHierarchy(HierarchyStrategy):
    """
    Peer-to-Peer Strategy
    Fully distributed coordination with consensus-based decision making.
    - All agents are peers (no hierarchy)
    - Consensus-based task allocation
    - Voting for decisions
    - Shared state through communication
    """

    def __init__(self, num_agents: int, consensus_threshold: float = 0.5):
        super().__init__(num_agents, HierarchyType.PEER_TO_PEER)
        self.consensus_threshold = consensus_threshold
        self.pending_votes: Dict[str, Dict[str, bool]] = (
            {}
        )  # proposal_id -> {agent_id: vote}
        self.shared_beliefs: Dict[str, Any] = {}

    def initialize_hierarchy(self) -> Dict[str, AgentState]:
        """Initialize peer-to-peer network (fully connected)."""
        self.agents.clear()

        # Create all agents as peers
        all_agent_ids = [f"agent_{i}" for i in range(self.num_agents)]

        for agent_id in all_agent_ids:
            peers = set(all_agent_ids) - {agent_id}
            self.agents[agent_id] = AgentState(
                agent_id=agent_id, role="peer", peers=peers
            )

        return self.agents

    def allocate_task(self, task: Task) -> bool:
        """Allocate task through consensus voting."""
        # Initiate vote for task allocation
        proposal_id = f"task_allocation_{task.task_id}"

        # Find available agents
        available_agents = [
            aid
            for aid in self.agents
            if self.agents[aid].is_active
            and self.agents[aid].current_task is None
        ]

        if not available_agents:
            return False

        # Randomly select candidate
        candidate = random.choice(available_agents)

        # Broadcast proposal
        for agent_id in self.agents:
            if self.agents[agent_id].is_active:
                proposal_msg = Message(
                    sender="system",
                    receiver=agent_id,
                    message_type="proposal",
                    content={
                        "proposal_id": proposal_id,
                        "task_id": task.task_id,
                        "candidate": candidate,
                    },
                )
                self.send_message(proposal_msg)

        # Simulate voting (in real system, agents would vote asynchronously)
        votes = {}
        for agent_id in self.agents:
            if self.agents[agent_id].is_active:
                # Vote based on simple heuristic
                vote = self.agents[candidate].current_task is None
                votes[agent_id] = vote

        self.pending_votes[proposal_id] = votes

        # Check consensus
        if self._check_consensus(proposal_id):
            task.assigned_to = candidate
            task.status = "in_progress"
            task.start_time = time.time()
            self.agents[candidate].current_task = task.task_id
            return True

        return False

    def _check_consensus(self, proposal_id: str) -> bool:
        """Check if consensus is reached for a proposal."""
        if proposal_id not in self.pending_votes:
            return False

        votes = self.pending_votes[proposal_id]
        if not votes:
            return False

        yes_votes = sum(1 for v in votes.values() if v)
        consensus_ratio = yes_votes / len(votes)

        return consensus_ratio >= self.consensus_threshold

    def process_step(self) -> Dict[str, Any]:
        """Process step with peer coordination."""
        self.step_count += 1
        actions_taken = {"peer_actions": 0, "messages": 0}
        message_count_before = len(self.message_queue)

        # All peers execute tasks and coordinate
        for agent_id in self.agents:
            agent = self.agents[agent_id]
            if not agent.is_active:
                continue

            # Execute current task
            if agent.current_task:
                task = self.tasks.get(agent.current_task)
                if task:
                    import random

                    # Small chance of task failure (5%)
                    if random.random() < 0.05 and agent.task_progress < 0.5:
                        # Task fails - needs consensus to reassign
                        task.status = "pending"
                        task.assigned_to = None
                        agent.current_task = None
                        agent.task_progress = 0.0
                    else:
                        agent.task_progress += 0.1
                        if agent.task_progress >= 1.0:
                            task.status = "completed"
                            task.end_time = time.time()
                            agent.current_task = None
                            agent.task_progress = 0.0

                            # Broadcast completion to all peers
                            for peer_id in agent.peers:
                                completion_msg = Message(
                                    sender=agent_id,
                                    receiver=peer_id,
                                    message_type="report",
                                    content={
                                        "task_id": task.task_id,
                                        "status": "completed",
                                    },
                                )
                                self.send_message(completion_msg)

            actions_taken["peer_actions"] += 1

        # Check for pending tasks that need re-allocation
        pending_tasks = [
            task
            for task in self.tasks.values()
            if task.status == "pending" and task.assigned_to is None
        ]

        for task in pending_tasks:
            # Try to reallocate through consensus
            self.allocate_task(task)

        # Periodic state synchronization
        if self.step_count % 5 == 0:
            self._synchronize_state()

        actions_taken["messages"] = len(self.message_queue) - message_count_before
        return actions_taken

    def _synchronize_state(self):
        """Synchronize shared state among peers."""
        # Broadcast state updates to maintain consistency
        for agent_id in self.agents:
            if self.agents[agent_id].is_active:
                state_msg = Message(
                    sender=agent_id,
                    receiver="broadcast",
                    message_type="state_sync",
                    content={"beliefs": self.agents[agent_id].beliefs},
                )
                self.send_message(state_msg)

    def handle_agent_failure(self, agent_id: str) -> bool:
        """Handle peer failure through consensus."""
        if agent_id not in self.agents:
            return False

        agent = self.agents[agent_id]
        agent.is_active = False

        # If peer had a task, initiate re-allocation vote
        if agent.current_task:
            task = self.tasks.get(agent.current_task)
            if task:
                task.assigned_to = None
                task.status = "pending"
                agent.current_task = None
                # Will be reallocated through consensus in next allocation

        # Update peer lists
        for other_id in self.agents:
            if other_id != agent_id:
                self.agents[other_id].peers.discard(agent_id)

        return True


class HybridHierarchy(HierarchyStrategy):
    """
    Hybrid Strategy
    Combines tree hierarchy for task allocation with peer-to-peer for
    execution.
    - Tree structure for strategic planning and task allocation
    - Peer communication among workers for tactical coordination
    - Managers coordinate, workers collaborate
    """

    def __init__(self, num_agents: int, hierarchy_depth: int = 2):
        super().__init__(num_agents, HierarchyType.HYBRID)
        self.hierarchy_depth = hierarchy_depth
        self.managers: Set[str] = set()
        self.workers: Set[str] = set()
        self.worker_peer_network: Dict[str, Set[str]] = (
            {}
        )  # worker_id -> peer_workers

    def initialize_hierarchy(self) -> Dict[str, AgentState]:
        """Initialize hybrid hierarchy."""
        self.agents.clear()
        self.managers.clear()
        self.workers.clear()
        self.worker_peer_network.clear()

        # Create tree structure (similar to TreeHierarchy)
        root_id = "agent_0"

        self.agents[root_id] = AgentState(
            agent_id=root_id, role="manager", subordinates=set()
        )
        self.managers.add(root_id)

        # Create workers
        for i in range(1, self.num_agents):
            worker_id = f"agent_{i}"
            self.agents[worker_id] = AgentState(
                agent_id=worker_id,
                role="worker",
                supervisor=root_id,
                peers=set(),  # Will be populated below
            )
            self.agents[root_id].subordinates.add(worker_id)
            self.workers.add(worker_id)

        # Create peer network among workers (fully connected)
        worker_list = list(self.workers)
        for worker_id in worker_list:
            peers = set(worker_list) - {worker_id}
            self.agents[worker_id].peers = peers
            self.worker_peer_network[worker_id] = peers

        return self.agents

    def allocate_task(self, task: Task) -> bool:
        """Manager allocates task, workers coordinate execution."""
        # Manager makes allocation decision
        root_id = "agent_0"

        available_workers = [
            aid
            for aid in self.workers
            if self.agents[aid].is_active
            and self.agents[aid].current_task is None
        ]

        if not available_workers:
            return False

        # Assign to worker
        assigned_worker = available_workers[0]
        task.assigned_to = assigned_worker
        task.status = "in_progress"
        task.start_time = time.time()

        self.agents[assigned_worker].current_task = task.task_id

        # Manager sends delegation
        delegation_msg = Message(
            sender=root_id,
            receiver=assigned_worker,
            message_type="delegation",
            content={"task_id": task.task_id, "task_type": task.task_type},
            priority=7,
        )
        self.send_message(delegation_msg)

        # Worker broadcasts to peers for collaborative execution
        for peer_id in self.agents[assigned_worker].peers:
            if self.agents[peer_id].is_active:
                collab_msg = Message(
                    sender=assigned_worker,
                    receiver=peer_id,
                    message_type="collaboration_request",
                    content={"task_id": task.task_id, "help_needed": False},
                )
                self.send_message(collab_msg)

        return True

    def process_step(self) -> Dict[str, Any]:
        """Process step with hybrid coordination."""
        self.step_count += 1
        actions_taken = {
            "manager": 0,
            "worker": 0,
            "peer_comm": 0,
            "messages": 0,
        }
        message_count_before = len(self.message_queue)

        # Manager monitors and coordinates
        for manager_id in self.managers:
            if self.agents[manager_id].is_active:
                self._manager_monitor(manager_id)
                actions_taken["manager"] += 1

        # Workers execute and coordinate with peers
        for worker_id in self.workers:
            worker = self.agents[worker_id]
            if not worker.is_active:
                continue

            if worker.current_task:
                task = self.tasks.get(worker.current_task)
                if task:
                    import random

                    # Small chance of task failure (5%)
                    if random.random() < 0.05 and worker.task_progress < 0.5:
                        # Task fails - manager will reassign
                        task.status = "pending"
                        task.assigned_to = None
                        worker.current_task = None
                        worker.task_progress = 0.0
                    else:
                        worker.task_progress += 0.1

                        # Periodic peer coordination
                        if self.step_count % 3 == 0:
                            self._worker_peer_coordinate(worker_id)
                            actions_taken["peer_comm"] += 1

                        if worker.task_progress >= 1.0:
                            task.status = "completed"
                            task.end_time = time.time()
                            worker.current_task = None
                            worker.task_progress = 0.0

                            # Report to manager
                            report_msg = Message(
                                sender=worker_id,
                                receiver=worker.supervisor,
                                message_type="report",
                                content={
                                    "task_id": task.task_id,
                                    "status": "completed",
                                },
                            )
                            self.send_message(report_msg)

                            # Notify peers
                            for peer_id in worker.peers:
                                if self.agents[peer_id].is_active:
                                    peer_msg = Message(
                                        sender=worker_id,
                                        receiver=peer_id,
                                        message_type="peer_update",
                                        content={
                                            "task_id": task.task_id,
                                            "status": "completed",
                                        },
                                    )
                                    self.send_message(peer_msg)

                actions_taken["worker"] += 1

        actions_taken["messages"] = len(self.message_queue) - message_count_before
        return actions_taken

    def _manager_monitor(self, manager_id: str):
        """Manager monitoring and oversight."""
        manager = self.agents[manager_id]

        # Check for pending tasks that need reassignment
        pending_tasks = [
            task
            for task in self.tasks.values()
            if task.status == "pending" and task.assigned_to is None
        ]

        for task in pending_tasks:
            # Try to reassign to available worker
            available_workers = [
                aid
                for aid in self.workers
                if (
                    self.agents[aid].is_active
                    and self.agents[aid].current_task is None
                )
            ]

            if available_workers:
                # Reassign task
                worker_id = available_workers[0]
                task.assigned_to = worker_id
                task.status = "in_progress"
                self.agents[worker_id].current_task = task.task_id

                delegation_msg = Message(
                    sender=manager_id,
                    receiver=worker_id,
                    message_type="delegation",
                    content={
                        "task_id": task.task_id,
                        "task_type": task.task_type,
                        "reassigned": True,
                    },
                    priority=8,
                )
                self.send_message(delegation_msg)

        active_subordinates = [
            sid
            for sid in manager.subordinates
            if self.agents[sid].is_active and self.agents[sid].current_task
        ]

        manager.beliefs["active_workers"] = len(active_subordinates)

    def _worker_peer_coordinate(self, worker_id: str):
        """Worker coordinates with peers."""
        worker = self.agents[worker_id]

        for peer_id in worker.peers:
            if self.agents[peer_id].is_active:
                progress_msg = Message(
                    sender=worker_id,
                    receiver=peer_id,
                    message_type="progress_update",
                    content={
                        "task_id": worker.current_task,
                        "progress": worker.task_progress,
                    },
                )
                self.send_message(progress_msg)

    def handle_agent_failure(self, agent_id: str) -> bool:
        """Handle failure in hybrid system."""
        if agent_id not in self.agents:
            return False

        agent = self.agents[agent_id]
        agent.is_active = False

        if agent.role == "worker":
            if agent.current_task:
                task = self.tasks.get(agent.current_task)
                if task:
                    task.assigned_to = None
                    task.status = "pending"
                    agent.current_task = None

            for peer_id in agent.peers:
                self.agents[peer_id].peers.discard(agent_id)

        elif agent.role == "manager":
            if agent.subordinates:
                root_id = "agent_0"
                if root_id != agent_id and root_id in self.agents:
                    for sub_id in agent.subordinates:
                        self.agents[sub_id].supervisor = root_id
                        self.agents[root_id].subordinates.add(sub_id)

        return True


def create_hierarchy_strategy(
    hierarchy_type: HierarchyType, num_agents: int, **kwargs
) -> HierarchyStrategy:
    """Factory function to create hierarchy strategy."""

    if hierarchy_type == HierarchyType.TREE:
        depth = kwargs.get("hierarchy_depth", 2)
        return TreeHierarchy(num_agents, hierarchy_depth=depth)

    elif hierarchy_type == HierarchyType.PEER_TO_PEER:
        threshold = kwargs.get("consensus_threshold", 0.5)
        return PeerToPeerHierarchy(num_agents, consensus_threshold=threshold)

    elif hierarchy_type == HierarchyType.HYBRID:
        depth = kwargs.get("hierarchy_depth", 2)
        return HybridHierarchy(num_agents, hierarchy_depth=depth)

    else:
        raise ValueError(f"Unknown hierarchy type: {hierarchy_type}")
