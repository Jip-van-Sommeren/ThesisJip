"""
Hierarchy System Implementation

Based on formal definitions: Sup: A → A ∪ {⊥} and Sub: A → 2^A

Implements tree hierarchy structure with organizational positions,
supervisor/subordinate relations, and escalation paths.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Set, Dict, Optional, List, Any

from .role import RoleManager
from .group import GroupManager


class MessageType(Enum):
    """Types of hierarchical messages."""
    COMMAND = "command"
    REPORT = "report"
    ESCALATION = "escalation"
    DELEGATION = "delegation"
    QUERY = "query"
    RESPONSE = "response"


@dataclass
class HierarchyMessage:
    """Message structure for hierarchy communication."""
    message_type: MessageType
    sender: str
    receiver: str
    content: Dict[str, Any]
    priority: float = 1.0
    timestamp: float = field(default_factory=time.time)
    requires_response: bool = False
    escalation_path: List[str] = field(default_factory=list)


@dataclass
class OrganizationalPosition:
    """
    Agent's position in hierarchy: (Roles, Sup, Subs, G)

    Part of agent's internal State component.
    """
    roles: Set[str] = field(default_factory=set)
    supervisor: Optional[str] = None  # Sup(a) → A ∪ {⊥}
    subordinates: Set[str] = field(default_factory=set)  # Sub(a) → 2^A
    groups: Set[str] = field(default_factory=set)
    hierarchy_level: int = 0
    reporting_frequency: float = 300.0  # seconds


class HierarchyManager:
    """
    Manages tree hierarchy structure and organizational relationships.

    Enforces tree invariants and provides hierarchy operations.
    Implements Sup: A → A ∪ {⊥} and Sub: A → 2^A.
    """

    def __init__(
        self,
        role_manager: Optional[RoleManager] = None,
        group_manager: Optional[GroupManager] = None
    ):
        self.role_manager = role_manager
        self.group_manager = group_manager
        self.hierarchy_structure: Dict[str, OrganizationalPosition] = {}
        self.message_queue: List[HierarchyMessage] = []
        self.root_agents: Set[str] = set()

    def add_agent(
        self,
        agent_id: str,
        supervisor: Optional[str] = None,
        roles: Optional[Set[str]] = None,
        groups: Optional[Set[str]] = None,
    ) -> OrganizationalPosition:
        """
        Add agent to hierarchy with optional supervisor.

        Args:
            agent_id: Agent identifier
            supervisor: Supervisor agent ID (None for root)
            roles: Set of role names
            groups: Set of group names

        Returns:
            Created OrganizationalPosition
        """
        # Create organizational position
        position = OrganizationalPosition(
            roles=roles or set(),
            supervisor=supervisor,
            groups=groups or set(),
        )

        # Calculate hierarchy level
        if supervisor:
            supervisor_pos = self.hierarchy_structure.get(supervisor)
            if supervisor_pos:
                position.hierarchy_level = supervisor_pos.hierarchy_level + 1
                # Update supervisor's subordinates
                supervisor_pos.subordinates.add(agent_id)
            else:
                position.hierarchy_level = 1
        else:
            position.hierarchy_level = 0
            self.root_agents.add(agent_id)

        # Store position
        self.hierarchy_structure[agent_id] = position

        # Assign roles
        if roles and self.role_manager:
            for role_name in roles:
                self.role_manager.assign_role(agent_id, role_name)

        # Add to groups
        if groups and self.group_manager:
            for group_name in groups:
                self.group_manager.add_agent_to_group(agent_id, group_name)

        return position

    def remove_agent(self, agent_id: str) -> bool:
        """
        Remove agent from hierarchy.

        Subordinates are reassigned to the removed agent's supervisor.

        Args:
            agent_id: Agent to remove

        Returns:
            True if removed successfully
        """
        if agent_id not in self.hierarchy_structure:
            return False

        position = self.hierarchy_structure[agent_id]

        # Remove from supervisor's subordinates
        if position.supervisor:
            supervisor_pos = self.hierarchy_structure.get(position.supervisor)
            if supervisor_pos:
                supervisor_pos.subordinates.discard(agent_id)

        # Reassign subordinates to supervisor (or make them roots)
        for subordinate_id in position.subordinates:
            if position.supervisor:
                self.change_supervisor(subordinate_id, position.supervisor)
            else:
                self.change_supervisor(subordinate_id, None)

        # Remove from groups
        if self.group_manager:
            for group_name in position.groups:
                self.group_manager.remove_agent_from_group(agent_id, group_name)

        # Clean up
        del self.hierarchy_structure[agent_id]
        self.root_agents.discard(agent_id)

        return True

    def change_supervisor(
        self,
        agent_id: str,
        new_supervisor: Optional[str]
    ) -> bool:
        """
        Change agent's supervisor.

        Args:
            agent_id: Agent whose supervisor to change
            new_supervisor: New supervisor ID (None for root)

        Returns:
            True if changed successfully
        """
        if agent_id not in self.hierarchy_structure:
            return False

        position = self.hierarchy_structure[agent_id]
        old_supervisor = position.supervisor

        # Remove from old supervisor
        if old_supervisor and old_supervisor in self.hierarchy_structure:
            self.hierarchy_structure[old_supervisor].subordinates.discard(agent_id)

        # Add to new supervisor
        if new_supervisor:
            if new_supervisor not in self.hierarchy_structure:
                return False
            self.hierarchy_structure[new_supervisor].subordinates.add(agent_id)
            position.supervisor = new_supervisor

            # Update hierarchy level
            supervisor_level = self.hierarchy_structure[new_supervisor].hierarchy_level
            position.hierarchy_level = supervisor_level + 1
            self.root_agents.discard(agent_id)
        else:
            position.supervisor = None
            position.hierarchy_level = 0
            self.root_agents.add(agent_id)

        return True

    def get_supervisor(self, agent_id: str) -> Optional[str]:
        """
        Get agent's supervisor: Sup(agent_id).

        Args:
            agent_id: Agent identifier

        Returns:
            Supervisor agent ID or None (⊥)
        """
        position = self.hierarchy_structure.get(agent_id)
        return position.supervisor if position else None

    def get_subordinates(self, agent_id: str) -> Set[str]:
        """
        Get agent's subordinates: Sub(agent_id).

        Args:
            agent_id: Agent identifier

        Returns:
            Set of subordinate agent IDs
        """
        position = self.hierarchy_structure.get(agent_id)
        return position.subordinates.copy() if position else set()

    def get_path_to_root(self, agent_id: str) -> List[str]:
        """
        Get path from agent to root of hierarchy.

        Args:
            agent_id: Starting agent

        Returns:
            List of agent IDs from agent to root
        """
        path = []
        current = agent_id

        while current and current in self.hierarchy_structure:
            path.append(current)
            current = self.hierarchy_structure[current].supervisor

            # Prevent infinite loops
            if len(path) > 100:
                break

        return path

    def get_escalation_path(self, agent_id: str, target_level: int) -> List[str]:
        """
        Get escalation path to specific hierarchy level.

        Args:
            agent_id: Starting agent
            target_level: Target hierarchy level

        Returns:
            List of agent IDs to escalate through
        """
        path = self.get_path_to_root(agent_id)

        # Find agents at or above target level
        escalation_path = []
        for agent in path[1:]:  # Skip the agent itself
            agent_level = self.hierarchy_structure[agent].hierarchy_level
            if agent_level <= target_level:
                escalation_path.append(agent)
                break

        return escalation_path

    def find_common_supervisor(
        self,
        agent1_id: str,
        agent2_id: str
    ) -> Optional[str]:
        """
        Find lowest common supervisor of two agents.

        Args:
            agent1_id: First agent
            agent2_id: Second agent

        Returns:
            Common supervisor ID or None
        """
        path1 = set(self.get_path_to_root(agent1_id))
        path2 = self.get_path_to_root(agent2_id)

        for agent in path2:
            if agent in path1:
                return agent

        return None

    def get_subordinates_recursive(self, agent_id: str) -> Set[str]:
        """
        Get all subordinates recursively.

        Args:
            agent_id: Agent identifier

        Returns:
            Set of all subordinate agent IDs (direct and indirect)
        """
        if agent_id not in self.hierarchy_structure:
            return set()

        subordinates = set()
        to_process = [agent_id]

        while to_process:
            current = to_process.pop()
            if current in self.hierarchy_structure:
                direct_subordinates = self.hierarchy_structure[current].subordinates
                subordinates.update(direct_subordinates)
                to_process.extend(direct_subordinates)

        return subordinates

    def get_position(self, agent_id: str) -> Optional[OrganizationalPosition]:
        """Get agent's organizational position."""
        return self.hierarchy_structure.get(agent_id)

    def get_hierarchy_level(self, agent_id: str) -> int:
        """Get agent's level in hierarchy (0 = root)."""
        position = self.hierarchy_structure.get(agent_id)
        return position.hierarchy_level if position else -1

    def send_message(self, message: HierarchyMessage) -> bool:
        """
        Queue a hierarchical message.

        Args:
            message: Message to queue

        Returns:
            True if queued successfully
        """
        if message.receiver not in self.hierarchy_structure:
            return False

        self.message_queue.append(message)
        return True

    def get_pending_messages(self, agent_id: str) -> List[HierarchyMessage]:
        """
        Get pending messages for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            List of pending messages
        """
        messages = [m for m in self.message_queue if m.receiver == agent_id]
        self.message_queue = [m for m in self.message_queue if m.receiver != agent_id]
        return messages

    def get_hierarchy_statistics(self) -> Dict[str, Any]:
        """Get statistics about the hierarchy structure."""
        if not self.hierarchy_structure:
            return {
                "total_agents": 0,
                "total_levels": 0,
                "root_agents": 0,
                "level_distribution": {},
                "pending_messages": 0,
            }

        total_agents = len(self.hierarchy_structure)
        total_levels = (
            max(pos.hierarchy_level for pos in self.hierarchy_structure.values()) + 1
        )

        level_distribution: Dict[int, int] = {}
        for pos in self.hierarchy_structure.values():
            level = pos.hierarchy_level
            level_distribution[level] = level_distribution.get(level, 0) + 1

        return {
            "total_agents": total_agents,
            "total_levels": total_levels,
            "root_agents": len(self.root_agents),
            "level_distribution": level_distribution,
            "pending_messages": len(self.message_queue),
        }

    def list_agents(self) -> List[str]:
        """Get list of all agent IDs in hierarchy."""
        return list(self.hierarchy_structure.keys())


__all__ = [
    "HierarchyManager",
    "OrganizationalPosition",
    "HierarchyMessage",
    "MessageType",
]
