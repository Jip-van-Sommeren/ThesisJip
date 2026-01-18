"""
Group System Implementation

Based on formal definition: G = (Name, Roles_G, Purpose)

A group is a collection of agents/roles that share common objectives.
Groups correspond to sub-teams or subsystems in the modeled organization.
"""

from dataclasses import dataclass, field
from typing import Set, Dict, Optional, List, Any

from .role import RoleManager


@dataclass
class Group:
    """
    Group: G = (Name, Roles_G, Purpose)

    Collection of agents/roles sharing common objective or context.
    Often corresponds to sub-teams or subsystems in organization.
    """
    name: str
    purpose: str
    roles: Set[str] = field(default_factory=set)  # Role names that exist in group
    agents: Set[str] = field(default_factory=set)  # Agent IDs currently in group
    max_size: Optional[int] = None
    properties: Dict[str, Any] = field(default_factory=dict)

    def add_role(self, role_name: str):
        """Add a role type to this group."""
        self.roles.add(role_name)

    def remove_role(self, role_name: str):
        """Remove a role type from this group."""
        self.roles.discard(role_name)

    def add_agent(self, agent_id: str) -> bool:
        """Add an agent to this group."""
        if self.max_size and len(self.agents) >= self.max_size:
            return False
        self.agents.add(agent_id)
        return True

    def remove_agent(self, agent_id: str):
        """Remove an agent from this group."""
        self.agents.discard(agent_id)

    def has_role(self, role_name: str) -> bool:
        """Check if group contains a specific role."""
        return role_name in self.roles

    def has_agent(self, agent_id: str) -> bool:
        """Check if agent is member of this group."""
        return agent_id in self.agents

    def is_full(self) -> bool:
        """Check if group has reached maximum capacity."""
        return self.max_size is not None and len(self.agents) >= self.max_size

    def get_size(self) -> int:
        """Get current number of agents in group."""
        return len(self.agents)

    def __str__(self) -> str:
        return f"Group({self.name}, {len(self.agents)} agents)"

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other) -> bool:
        if isinstance(other, Group):
            return self.name == other.name
        return False


class GroupManager:
    """
    Manages group definitions, membership, and coordination.

    Handles group lifecycle and agent-group relationships.
    """

    def __init__(self, role_manager: Optional[RoleManager] = None):
        self.role_manager = role_manager
        self.group_definitions: Dict[str, Group] = {}
        self.agent_groups: Dict[str, Set[str]] = {}  # agent_id -> group_names

    def register_group(self, group: Group):
        """Register a group definition."""
        self.group_definitions[group.name] = group

    def create_group(
        self,
        name: str,
        purpose: str,
        roles: Optional[Set[str]] = None,
        max_size: Optional[int] = None,
        **properties
    ) -> Group:
        """
        Create and register a new group.

        Args:
            name: Group name
            purpose: Group purpose
            roles: Set of role names
            max_size: Maximum group size
            **properties: Additional properties

        Returns:
            Created group
        """
        group = Group(
            name=name,
            purpose=purpose,
            roles=roles or set(),
            max_size=max_size,
            properties=properties,
        )
        self.register_group(group)
        return group

    def get_group(self, group_name: str) -> Optional[Group]:
        """Get group by name."""
        return self.group_definitions.get(group_name)

    def add_agent_to_group(self, agent_id: str, group_name: str) -> bool:
        """Add an agent to a group."""
        group = self.get_group(group_name)
        if not group:
            return False

        if group.is_full():
            return False

        success = group.add_agent(agent_id)
        if success:
            if agent_id not in self.agent_groups:
                self.agent_groups[agent_id] = set()
            self.agent_groups[agent_id].add(group_name)

        return success

    def remove_agent_from_group(self, agent_id: str, group_name: str) -> bool:
        """Remove an agent from a group."""
        group = self.get_group(group_name)
        if not group or not group.has_agent(agent_id):
            return False

        group.remove_agent(agent_id)

        if agent_id in self.agent_groups:
            self.agent_groups[agent_id].discard(group_name)
            if not self.agent_groups[agent_id]:
                del self.agent_groups[agent_id]

        return True

    def get_agent_groups(self, agent_id: str) -> Set[str]:
        """Get all groups an agent belongs to."""
        return self.agent_groups.get(agent_id, set())

    def get_group_agents(self, group_name: str) -> Set[str]:
        """Get all agents in a group."""
        group = self.get_group(group_name)
        return group.agents.copy() if group else set()

    def validate_agent_for_group(self, agent_id: str, group_name: str) -> bool:
        """Validate if agent can join group based on role compatibility."""
        group = self.get_group(group_name)
        if not group:
            return False

        if not self.role_manager:
            return True  # No role validation if no role manager

        # Check if agent has any role that exists in the group
        agent_roles = self.role_manager.get_agent_roles(agent_id)
        agent_role_names = {role.name for role in agent_roles}

        return bool(agent_role_names.intersection(group.roles))

    def find_groups_needing_role(self, role_name: str) -> List[str]:
        """Find groups that need agents with a specific role."""
        result = []

        for group_name, group in self.group_definitions.items():
            if group.has_role(role_name) and not group.is_full():
                result.append(group_name)

        return result

    def get_group_role_distribution(self, group_name: str) -> Dict[str, int]:
        """Get distribution of roles within a group."""
        group = self.get_group(group_name)
        if not group or not self.role_manager:
            return {}

        role_count: Dict[str, int] = {}
        for agent_id in group.agents:
            agent_roles = self.role_manager.get_agent_roles(agent_id)
            for role in agent_roles:
                role_count[role.name] = role_count.get(role.name, 0) + 1

        return role_count

    def find_compatible_groups(self, agent_id: str) -> List[str]:
        """Find groups where agent could potentially join."""
        compatible_groups = []

        for group_name, group in self.group_definitions.items():
            if not group.is_full() and self.validate_agent_for_group(
                agent_id, group_name
            ):
                compatible_groups.append(group_name)

        return compatible_groups

    def get_group_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics about all groups."""
        stats = {}

        for group_name, group in self.group_definitions.items():
            role_distribution = self.get_group_role_distribution(group_name)

            stats[group_name] = {
                "purpose": group.purpose,
                "current_size": group.get_size(),
                "max_size": group.max_size,
                "utilization": (
                    group.get_size() / group.max_size if group.max_size else None
                ),
                "role_distribution": role_distribution,
                "required_roles": list(group.roles),
                "properties": group.properties,
            }

        return stats

    def list_groups(self) -> List[str]:
        """Get list of all group names."""
        return list(self.group_definitions.keys())


__all__ = ["Group", "GroupManager"]
