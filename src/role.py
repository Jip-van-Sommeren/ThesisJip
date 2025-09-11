"""
Role System Implementation
Based on formal definition: R = (responsibilities, permissions, requirements,
expectations)

A role defines abstract behavior/function that agents can fulfill.
Roles influence agent goals, actions, and behavior patterns.
"""

from dataclasses import dataclass, field
from typing import Set, Dict, Any, Optional
from enum import Enum


class PermissionType(Enum):
    """Types of permissions a role can have."""

    READ_DATA = "read_data"
    WRITE_DATA = "write_data"
    COMMAND_AGENTS = "command_agents"
    ACCESS_RESOURCES = "access_resources"
    ESCALATE_ISSUES = "escalate_issues"
    DELEGATE_TASKS = "delegate_tasks"
    VIEW_REPORTS = "view_reports"
    MODIFY_PARAMETERS = "modify_parameters"


class ExpectationType(Enum):
    """Types of behavioral expectations for roles."""

    RESPONSE_TIME = "response_time"
    REPORTING_FREQUENCY = "reporting_frequency"
    ESCALATION_THRESHOLD = "escalation_threshold"
    COMPLIANCE_LEVEL = "compliance_level"
    AVAILABILITY = "availability"


@dataclass
class Responsibility:
    """A specific duty or task the role should perform."""

    name: str
    description: str
    priority: float = 1.0
    frequency: Optional[str] = None  # "continuous", "periodic", "on_demand"

    def __str__(self) -> str:
        return f"Responsibility({self.name}, priority={self.priority})"


@dataclass
class Permission:
    """A right or authority the role has."""

    permission_type: PermissionType
    resource: str  # What resource this permission applies to
    scope: str = "local"  # "local", "group", "global"
    constraints: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"Permission({self.permission_type.value}, {self.resource})"


@dataclass
class Requirement:
    """A condition or skill needed to occupy the role."""

    name: str
    requirement_type: str  # "capability", "resource", "knowledge"
    value: Any
    mandatory: bool = True

    def __str__(self) -> str:
        return f"Requirement({self.name}, mandatory={self.mandatory})"


@dataclass
class Expectation:
    """A constraint or norm on the role's behavior."""

    expectation_type: ExpectationType
    value: Any
    enforcement_level: str = "soft"  # "soft", "hard", "critical"

    def __str__(self) -> str:
        return f"Expectation({self.expectation_type.value}={self.value})"


@dataclass
class Role:
    """
    Role: R = (responsibilities, permissions, requirements, expectations)

    Defines abstract behavior/function within organization that agent
    can fulfill.
    Influences agent goals, available actions, and behavioral constraints.
    """

    name: str
    description: str
    responsibilities: Set[Responsibility] = field(default_factory=set)
    permissions: Set[Permission] = field(default_factory=set)
    requirements: Set[Requirement] = field(default_factory=set)
    expectations: Set[Expectation] = field(default_factory=set)

    def add_responsibility(self, responsibility: Responsibility):
        """Add a responsibility to this role."""
        self.responsibilities.add(responsibility)

    def add_permission(self, permission: Permission):
        """Add a permission to this role."""
        self.permissions.add(permission)

    def add_requirement(self, requirement: Requirement):
        """Add a requirement to this role."""
        self.requirements.add(requirement)

    def add_expectation(self, expectation: Expectation):
        """Add a behavioral expectation to this role."""
        self.expectations.add(expectation)

    def has_permission(
        self, permission_type: PermissionType, resource: str = "*"
    ) -> bool:
        """Check if role has specific permission for resource."""
        for perm in self.permissions:
            if perm.permission_type == permission_type and (
                perm.resource == resource
                or perm.resource == "*"
                or resource == "*"
            ):
                return True
        return False

    def get_expectation_value(self, expectation_type: ExpectationType) -> Any:
        """Get the value for a specific expectation type."""
        for exp in self.expectations:
            if exp.expectation_type == expectation_type:
                return exp.value
        return None

    def meets_requirements(self, agent_capabilities: Dict[str, Any]) -> bool:
        """Check if agent capabilities meet role requirements."""
        for req in self.requirements:
            if req.mandatory:
                if req.name not in agent_capabilities:
                    return False
                if agent_capabilities[req.name] != req.value:
                    return False
        return True

    def get_priority_responsibilities(
        self, min_priority: float = 1.0
    ) -> Set[Responsibility]:
        """Get responsibilities above specified priority threshold."""
        return {r for r in self.responsibilities if r.priority >= min_priority}

    def __str__(self) -> str:
        return f"Role({self.name})"

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other) -> bool:
        if isinstance(other, Role):
            return self.name == other.name
        return False


class RoleTemplate:
    """
    Template for creating roles with common patterns.
    Provides factory methods for standard organizational roles.
    """

    @staticmethod
    def create_supervisor_role() -> Role:
        """Create a standard supervisor role template."""
        role = Role(
            name="Supervisor",
            description="Manages and coordinates subordinate agents",
        )

        # Responsibilities
        role.add_responsibility(
            Responsibility(
                "coordinate_subordinates",
                "Coordinate activities of subordinate agents",
                priority=2.0,
                frequency="continuous",
            )
        )
        role.add_responsibility(
            Responsibility(
                "monitor_performance",
                "Monitor performance of supervised agents",
                priority=1.8,
                frequency="continuous",
            )
        )
        role.add_responsibility(
            Responsibility(
                "escalate_issues",
                "Escalate critical issues to higher levels",
                priority=3.0,
                frequency="on_demand",
            )
        )

        # Permissions
        role.add_permission(
            Permission(PermissionType.COMMAND_AGENTS, "subordinates")
        )
        role.add_permission(
            Permission(PermissionType.VIEW_REPORTS, "subordinates")
        )
        role.add_permission(
            Permission(PermissionType.ESCALATE_ISSUES, "hierarchy")
        )
        role.add_permission(
            Permission(PermissionType.DELEGATE_TASKS, "subordinates")
        )

        # Requirements
        role.add_requirement(Requirement("leadership", "capability", True))
        role.add_requirement(
            Requirement("decision_making", "capability", True)
        )

        # Expectations
        role.add_expectation(
            Expectation(ExpectationType.RESPONSE_TIME, 30.0)
        )  # seconds
        role.add_expectation(
            Expectation(ExpectationType.REPORTING_FREQUENCY, 300.0)
        )  # seconds
        role.add_expectation(
            Expectation(ExpectationType.AVAILABILITY, 0.95)
        )  # 95%

        return role

    @staticmethod
    def create_worker_role() -> Role:
        """Create a standard worker role template."""
        role = Role(
            name="Worker",
            description="Executes assigned tasks and reports status",
        )

        # Responsibilities
        role.add_responsibility(
            Responsibility(
                "execute_tasks",
                "Execute assigned tasks efficiently",
                priority=2.0,
                frequency="on_demand",
            )
        )
        role.add_responsibility(
            Responsibility(
                "report_status",
                "Report status to supervisor",
                priority=1.5,
                frequency="periodic",
            )
        )

        # Permissions
        role.add_permission(Permission(PermissionType.READ_DATA, "local"))
        role.add_permission(Permission(PermissionType.WRITE_DATA, "local"))

        # Requirements
        role.add_requirement(Requirement("task_execution", "capability", True))

        # Expectations
        role.add_expectation(Expectation(ExpectationType.RESPONSE_TIME, 60.0))
        role.add_expectation(
            Expectation(ExpectationType.REPORTING_FREQUENCY, 600.0)
        )

        return role

    @staticmethod
    def create_sensor_monitor_role() -> Role:
        """Create a sensor monitoring role for digital twins."""
        role = Role(
            name="SensorMonitor",
            description="Monitor sensor data and detect anomalies",
        )

        # Responsibilities
        role.add_responsibility(
            Responsibility(
                "monitor_sensors",
                "Continuously monitor assigned sensors",
                priority=3.0,
                frequency="continuous",
            )
        )
        role.add_responsibility(
            Responsibility(
                "detect_anomalies",
                "Detect and report sensor anomalies",
                priority=2.5,
                frequency="continuous",
            )
        )

        # Permissions
        role.add_permission(Permission(PermissionType.READ_DATA, "sensors"))
        role.add_permission(
            Permission(PermissionType.ESCALATE_ISSUES, "supervisor")
        )

        # Requirements
        role.add_requirement(Requirement("sensor_access", "resource", True))
        role.add_requirement(
            Requirement("anomaly_detection", "capability", True)
        )

        # Expectations
        role.add_expectation(Expectation(ExpectationType.RESPONSE_TIME, 5.0))
        role.add_expectation(Expectation(ExpectationType.AVAILABILITY, 0.99))

        return role

    @staticmethod
    def create_controller_role() -> Role:
        """Create a controller role for digital twins."""
        role = Role(
            name="Controller",
            description="Control system parameters and actuators",
        )

        # Responsibilities
        role.add_responsibility(
            Responsibility(
                "control_actuators",
                "Control actuators based on system state",
                priority=2.5,
                frequency="on_demand",
            )
        )
        role.add_responsibility(
            Responsibility(
                "maintain_setpoints",
                "Maintain desired system setpoints",
                priority=2.0,
                frequency="continuous",
            )
        )

        # Permissions
        role.add_permission(Permission(PermissionType.READ_DATA, "sensors"))
        role.add_permission(Permission(PermissionType.WRITE_DATA, "actuators"))
        role.add_permission(
            Permission(PermissionType.MODIFY_PARAMETERS, "local")
        )

        # Requirements
        role.add_requirement(Requirement("control_systems", "knowledge", True))
        role.add_requirement(Requirement("actuator_access", "resource", True))

        # Expectations
        role.add_expectation(Expectation(ExpectationType.RESPONSE_TIME, 10.0))
        role.add_expectation(
            Expectation(ExpectationType.COMPLIANCE_LEVEL, 0.98)
        )

        return role


class RoleManager:
    """
    Manages role definitions and assignments within the system.
    Provides role validation and template management.
    """

    def __init__(self):
        self.role_definitions: Dict[str, Role] = {}
        self.agent_roles: Dict[str, Set[Role]] = {}  # agent_id -> roles

        # Load default role templates
        self._load_default_roles()

    def _load_default_roles(self):
        """Load standard role templates."""
        templates = [
            RoleTemplate.create_supervisor_role(),
            RoleTemplate.create_worker_role(),
            RoleTemplate.create_sensor_monitor_role(),
            RoleTemplate.create_controller_role(),
        ]

        for role in templates:
            self.register_role(role)

    def register_role(self, role: Role):
        """Register a role definition."""
        self.role_definitions[role.name] = role

    def get_role(self, role_name: str) -> Optional[Role]:
        """Get role definition by name."""
        return self.role_definitions.get(role_name)

    def assign_role(self, agent_id: str, role_name: str) -> bool:
        """Assign a role to an agent."""
        role = self.get_role(role_name)
        if not role:
            return False

        if agent_id not in self.agent_roles:
            self.agent_roles[agent_id] = set()

        self.agent_roles[agent_id].add(role)
        return True

    def remove_role(self, agent_id: str, role_name: str) -> bool:
        """Remove a role from an agent."""
        if agent_id not in self.agent_roles:
            return False

        role = self.get_role(role_name)
        if role in self.agent_roles[agent_id]:
            self.agent_roles[agent_id].remove(role)
            return True
        return False

    def get_agent_roles(self, agent_id: str) -> Set[Role]:
        """Get all roles assigned to an agent."""
        return self.agent_roles.get(agent_id, set())

    def agent_has_permission(
        self,
        agent_id: str,
        permission_type: PermissionType,
        resource: str = "*",
    ) -> bool:
        """Check if agent has specific permission through any of its roles."""
        roles = self.get_agent_roles(agent_id)
        return any(
            role.has_permission(permission_type, resource) for role in roles
        )

    def get_available_roles(self) -> Set[str]:
        """Get names of all available role definitions."""
        return set(self.role_definitions.keys())

    def validate_role_assignment(
        self, agent_id: str, role_name: str, agent_capabilities: Dict[str, Any]
    ) -> bool:
        """Validate if agent meets requirements for role assignment."""
        role = self.get_role(role_name)
        if not role:
            return False

        return role.meets_requirements(agent_capabilities)
