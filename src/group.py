"""
Group System Implementation
Based on formal definition: G = (Name, Roles_G, Purpose)

A group is a collection of agents/roles that share common objectives.
Groups correspond to sub-teams or subsystems in the modeled organization.
"""

from dataclasses import dataclass, field
from typing import Set, Dict, Optional, List, Any
from role import RoleManager


@dataclass
class Group:
    """
    Group: G = (Name, Roles_G, Purpose)

    Collection of agents/roles sharing common objective or context.
    Often corresponds to sub-teams or subsystems in organization.
    """

    name: str
    purpose: str
    roles: Set[str] = field(
        default_factory=set
    )  # Role names that exist in group
    agents: Set[str] = field(
        default_factory=set
    )  # Agent IDs currently in group
    max_size: Optional[int] = None
    properties: Dict[str, Any] = field(
        default_factory=dict
    )  # Group-specific properties

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


class GroupTemplate:
    """
    Template for creating groups with common patterns.
    Provides factory methods for standard organizational groups.
    """

    @staticmethod
    def create_production_cell_group() -> Group:
        """Create a production cell group for manufacturing."""
        group = Group(
            name="ProductionCell",
            purpose="Execute manufacturing processes within a cell",
            max_size=10,
        )

        # Typical roles in a production cell
        group.add_role("SensorMonitor")
        group.add_role("Controller")
        group.add_role("Supervisor")

        # Group-specific properties
        group.properties.update(
            {
                "cell_type": "manufacturing",
                "coordination_style": "centralized",
                "response_time_sla": 30.0,
            }
        )

        return group

    @staticmethod
    def create_maintenance_team_group() -> Group:
        """Create a maintenance team group."""
        group = Group(
            name="MaintenanceTeam",
            purpose="Perform predictive and corrective maintenance",
            max_size=5,
        )

        group.add_role("Supervisor")
        group.add_role("Worker")

        group.properties.update(
            {
                "team_type": "maintenance",
                "specialization": "mechanical",
                "availability_schedule": "24/7",
            }
        )

        return group

    @staticmethod
    def create_quality_control_group() -> Group:
        """Create a quality control group."""
        group = Group(
            name="QualityControl",
            purpose="Monitor and ensure product quality",
            max_size=8,
        )

        group.add_role("SensorMonitor")
        group.add_role("Worker")
        group.add_role("Supervisor")

        group.properties.update(
            {
                "inspection_type": "automated",
                "quality_standards": "ISO9001",
                "reporting_frequency": 60.0,
            }
        )

        return group

    @staticmethod
    def create_logistics_group() -> Group:
        """Create a logistics coordination group."""
        group = Group(
            name="Logistics",
            purpose="Coordinate material flow and inventory",
            max_size=6,
        )

        group.add_role("Controller")
        group.add_role("Supervisor")

        group.properties.update(
            {
                "logistics_type": "internal",
                "coordination_scope": "plant_level",
                "optimization_target": "minimize_inventory",
            }
        )

        return group


class GroupManager:
    """
    Manages group definitions, membership, and coordination.
    Handles group lifecycle and agent-group relationships.
    """

    def __init__(self, role_manager: RoleManager):
        self.role_manager = role_manager
        self.group_definitions: Dict[str, Group] = {}
        self.agent_groups: Dict[str, Set[str]] = {}  # agent_id -> group_names

        # Load default group templates
        self._load_default_groups()

    def _load_default_groups(self):
        """Load standard group templates."""
        templates = [
            GroupTemplate.create_production_cell_group(),
            GroupTemplate.create_maintenance_team_group(),
            GroupTemplate.create_quality_control_group(),
            GroupTemplate.create_logistics_group(),
        ]

        for group in templates:
            self.register_group(group)

    def register_group(self, group: Group):
        """Register a group definition."""
        self.group_definitions[group.name] = group

    def create_group_instance(
        self, group_name: str, instance_name: str
    ) -> Optional[Group]:
        """Create a new instance of a group template."""
        template = self.group_definitions.get(group_name)
        if not template:
            return None

        # Create new instance with unique name
        instance = Group(
            name=instance_name,
            purpose=template.purpose,
            roles=template.roles.copy(),
            max_size=template.max_size,
            properties=template.properties.copy(),
        )

        self.group_definitions[instance_name] = instance
        return instance

    def get_group(self, group_name: str) -> Optional[Group]:
        """Get group by name."""
        return self.group_definitions.get(group_name)

    def add_agent_to_group(self, agent_id: str, group_name: str) -> bool:
        """Add an agent to a group."""
        group = self.get_group(group_name)
        if not group:
            return False

        # Check if group is full
        if group.is_full():
            return False

        # Add agent to group
        success = group.add_agent(agent_id)
        if success:
            # Track agent's group membership
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

        # Update agent's group membership
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

        # Check if agent has any role that exists in the group
        agent_roles = self.role_manager.get_agent_roles(agent_id)
        agent_role_names = {role.name for role in agent_roles}

        return bool(agent_role_names.intersection(group.roles))

    def find_groups_needing_role(self, role_name: str) -> List[str]:
        """Find groups that need agents with a specific role."""
        result = []

        for group_name, group in self.group_definitions.items():
            if group.has_role(role_name) and not group.is_full():
                # Check if group already has agents with this role
                agents_with_role = []
                for agent_id in group.agents:
                    agent_roles = self.role_manager.get_agent_roles(agent_id)
                    if any(role.name == role_name for role in agent_roles):
                        agents_with_role.append(agent_id)

                # Group might need more agents with this role
                result.append(group_name)

        return result

    def get_group_role_distribution(self, group_name: str) -> Dict[str, int]:
        """Get distribution of roles within a group."""
        group = self.get_group(group_name)
        if not group:
            return {}

        role_count = {}
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
                    group.get_size() / group.max_size
                    if group.max_size
                    else None
                ),
                "role_distribution": role_distribution,
                "required_roles": list(group.roles),
                "properties": group.properties,
            }

        return stats

    def cleanup_empty_groups(self):
        """Remove groups that have no agents."""
        empty_groups = [
            name
            for name, group in self.group_definitions.items()
            if group.get_size() == 0 and not group.roles
        ]

        for group_name in empty_groups:
            del self.group_definitions[group_name]
