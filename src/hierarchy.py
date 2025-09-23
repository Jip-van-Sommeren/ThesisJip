"""
Hierarchy System Implementation
Based on formal definitions: Sup: A → A ∪ {⊥} and Sub: A → 2^A

Implements tree hierarchy structure with organizational positions,
supervisor roles, and integration with reactive, BDI, and hybrid agents.
"""

from dataclasses import dataclass, field
from typing import Set, Dict, Optional, List, Any
from enum import Enum
import time

from abstract_agent import AbstractAgent, AgentId, Goal, GoalType, ReactiveRule
from reactive_agent import ReactiveAgent
from bdi_agent import BDIAgent
from hybrid_agent import HybridAgent
from role import RoleManager
from group import GroupManager


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


class OrganizationalMixin:
    """
    Mixin to add organizational capabilities to any agent type.
    Added to ReactiveAgent, BDIAgent, HybridAgent via composition.
    """

    def __init_organizational__(
        self,
        roles: Set[str],
        position: OrganizationalPosition,
        hierarchy_manager: "HierarchyManager",
    ):
        self.organizational_position = position
        self.assigned_roles = roles
        self.hierarchy_manager = hierarchy_manager
        self.message_queue: List[HierarchyMessage] = []
        self.last_report_time = time.time()

        # Apply role-based behavior
        self._apply_roles()
        self._add_organizational_goals()
        self._add_organizational_rules()

    def _apply_roles(self):
        """Apply role responsibilities as goals and permissions as actions."""
        role_manager = self.hierarchy_manager.role_manager

        for role_name in self.assigned_roles:
            role = role_manager.get_role(role_name)
            if role:
                # Add responsibilities as goals
                for responsibility in role.responsibilities:
                    goal = Goal(
                        condition=responsibility.name,
                        goal_type=GoalType.INTRINSIC,
                        priority=responsibility.priority,
                    )
                    self.add_goal(goal)

    def _add_organizational_goals(self):
        """Add hierarchy-specific goals."""
        # If supervisor, add coordination goal
        if self.organizational_position.subordinates:
            coord_goal = Goal(
                condition="coordinate_subordinates",
                goal_type=GoalType.INTRINSIC,
                priority=2.0,
            )
            self.add_goal(coord_goal)

        # Add reporting goal if has supervisor
        if self.organizational_position.supervisor:
            report_goal = Goal(
                condition="report_to_supervisor",
                goal_type=GoalType.INTRINSIC,
                priority=1.5,
            )
            self.add_goal(report_goal)

    def _add_organizational_rules(self):
        """Add hierarchy-specific reactive rules."""
        # Command response rule (highest priority)
        command_rule = ReactiveRule(
            condition=lambda state: self._has_pending_command(state),
            action="execute_command",
            priority=10.0,
        )
        self.add_reactive_rule(command_rule)

        # Reporting deadline rule
        report_rule = ReactiveRule(
            condition=lambda state: self._reporting_overdue(state),
            action="send_status_report",
            priority=8.0,
        )
        self.add_reactive_rule(report_rule)

    def _has_pending_command(self, state_dict: Dict) -> bool:
        """Check if there are pending commands from supervisor."""
        return any(
            msg.message_type == MessageType.COMMAND
            for msg in self.message_queue
        )

    def _reporting_overdue(self, state_dict: Dict) -> bool:
        """Check if status reporting is overdue."""
        if not self.organizational_position.supervisor:
            return False

        time_since_report = time.time() - self.last_report_time
        return (
            time_since_report
            > self.organizational_position.reporting_frequency
        )

    def send_command(
        self, subordinate_id: str, command: Dict[str, Any]
    ) -> bool:
        """Send command to subordinate (requires command permission)."""
        if subordinate_id not in self.organizational_position.subordinates:
            return False

        # Check permission
        role_manager = self.hierarchy_manager.role_manager
        if not role_manager.agent_has_permission(
            str(self.id), role_manager.PermissionType.COMMAND_AGENTS
        ):
            return False

        message = HierarchyMessage(
            message_type=MessageType.COMMAND,
            sender=str(self.id),
            receiver=subordinate_id,
            content=command,
            priority=9.0,
            requires_response=True,
        )

        return self.hierarchy_manager.send_message(message)

    def report_to_supervisor(self, report_content: Dict[str, Any]) -> bool:
        """Send status report to supervisor."""
        if not self.organizational_position.supervisor:
            return False

        message = HierarchyMessage(
            message_type=MessageType.REPORT,
            sender=str(self.id),
            receiver=self.organizational_position.supervisor,
            content=report_content,
            priority=5.0,
        )

        self.last_report_time = time.time()
        return self.hierarchy_manager.send_message(message)

    def escalate_issue(
        self, issue: Dict[str, Any], target_level: Optional[int] = None
    ) -> bool:
        """Escalate issue up the hierarchy."""
        if not self.organizational_position.supervisor:
            return False

        # Determine escalation target
        if target_level is None:
            target_level = self.organizational_position.hierarchy_level + 1

        escalation_path = self.hierarchy_manager.get_escalation_path(
            str(self.id), target_level
        )

        if not escalation_path:
            return False

        message = HierarchyMessage(
            message_type=MessageType.ESCALATION,
            sender=str(self.id),
            receiver=escalation_path[0],
            content=issue,
            priority=8.0,
            escalation_path=escalation_path,
        )

        return self.hierarchy_manager.send_message(message)

    def delegate_goal(self, subordinate_id: str, goal: Goal) -> bool:
        """Delegate goal to subordinate."""
        if subordinate_id not in self.organizational_position.subordinates:
            return False

        delegation_content = {
            "goal_condition": goal.condition,
            "goal_type": goal.goal_type.value,
            "priority": goal.priority,
            "deadline": goal.deadline,
        }

        message = HierarchyMessage(
            message_type=MessageType.DELEGATION,
            sender=str(self.id),
            receiver=subordinate_id,
            content=delegation_content,
            priority=7.0,
            requires_response=True,
        )

        return self.hierarchy_manager.send_message(message)

    def process_message(self, message: HierarchyMessage):
        """Process incoming hierarchical message."""
        if message.message_type == MessageType.COMMAND:
            self._handle_command(message)
        elif message.message_type == MessageType.DELEGATION:
            self._handle_delegation(message)
        elif message.message_type == MessageType.ESCALATION:
            self._handle_escalation(message)
        elif message.message_type == MessageType.QUERY:
            self._handle_query(message)

    def _handle_command(self, message: HierarchyMessage):
        """Handle command from supervisor."""
        command_content = message.content

        # Add command as high priority goal
        command_goal = Goal(
            condition=f"execute_{command_content.get('action', 'command')}",
            goal_type=GoalType.EXTRINSIC,
            priority=9.0,
        )
        self.add_goal(command_goal)

        # Send acknowledgment if required
        if message.requires_response:
            response = HierarchyMessage(
                message_type=MessageType.RESPONSE,
                sender=str(self.id),
                receiver=message.sender,
                content={
                    "status": "acknowledged",
                    "command_id": message.content.get("id"),
                },
            )
            self.hierarchy_manager.send_message(response)

    def _handle_delegation(self, message: HierarchyMessage):
        """Handle goal delegation from supervisor."""
        delegation = message.content

        delegated_goal = Goal(
            condition=delegation["goal_condition"],
            goal_type=GoalType.valueOf(delegation["goal_type"]),
            priority=delegation["priority"],
            deadline=delegation.get("deadline"),
        )

        self.add_goal(delegated_goal)

    def _handle_escalation(self, message: HierarchyMessage):
        """Handle escalated issue."""
        # As supervisor, need to address escalated issue
        escalation_goal = Goal(
            condition=f"resolve_escalation_\
                {message.content.get('type', 'issue')}",
            goal_type=GoalType.EXTRINSIC,
            priority=8.0,
        )
        self.add_goal(escalation_goal)

    def _handle_query(self, message: HierarchyMessage):
        """Handle query from supervisor or subordinate."""
        # Process query and prepare response
        query_response = self._generate_query_response(message.content)

        response = HierarchyMessage(
            message_type=MessageType.RESPONSE,
            sender=str(self.id),
            receiver=message.sender,
            content=query_response,
        )
        self.hierarchy_manager.send_message(response)

    def _generate_query_response(
        self, query: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate response to query."""
        query_type = query.get("type", "status")

        if query_type == "status":
            hierarchy_level = self.organizational_position.hierarchy_level
            return {
                "agent_id": str(self.id),
                "status": "active",
                "current_goals": len(self.goals),
                "hierarchy_level": hierarchy_level,
            }

        return {"error": "unknown_query_type"}


class HierarchicalReactiveAgent(ReactiveAgent):
    """Reactive agent with organizational awareness."""

    def __init__(
        self,
        agent_id: AgentId,
        observable_properties: Set[str],
        roles: Set[str],
        position: OrganizationalPosition,
        hierarchy_manager: "HierarchyManager",
    ):
        super().__init__(agent_id, observable_properties)

        # Add organizational capabilities
        self.org_mixin = OrganizationalMixin()
        self.org_mixin.__init_organizational__(
            roles, position, hierarchy_manager
        )

        # Copy organizational attributes
        self.organizational_position = position
        self.assigned_roles = roles
        self.hierarchy_manager = hierarchy_manager
        self.message_queue = self.org_mixin.message_queue

        # Delegate methods
        self.send_command = self.org_mixin.send_command
        self.report_to_supervisor = self.org_mixin.report_to_supervisor
        self.escalate_issue = self.org_mixin.escalate_issue
        self.delegate_goal = self.org_mixin.delegate_goal
        self.process_message = self.org_mixin.process_message


class HierarchicalBDIAgent(BDIAgent):
    """BDI agent with organizational awareness."""

    def __init__(
        self,
        agent_id: AgentId,
        observable_properties: Set[str],
        roles: Set[str],
        position: OrganizationalPosition,
        hierarchy_manager: "HierarchyManager",
    ):
        super().__init__(agent_id, observable_properties)

        # Add organizational capabilities
        self.org_mixin = OrganizationalMixin()
        self.org_mixin.__init_organizational__(
            roles, position, hierarchy_manager
        )

        # Copy organizational attributes
        self.organizational_position = position
        self.assigned_roles = roles
        self.hierarchy_manager = hierarchy_manager
        self.message_queue = self.org_mixin.message_queue

        # Delegate methods
        self.send_command = self.org_mixin.send_command
        self.report_to_supervisor = self.org_mixin.report_to_supervisor
        self.escalate_issue = self.org_mixin.escalate_issue
        self.delegate_goal = self.org_mixin.delegate_goal
        self.process_message = self.org_mixin.process_message

    def select_intentions(self):
        """Override to consider hierarchy priorities."""
        # Process any pending hierarchy messages first
        for message in self.message_queue[:]:
            if message.priority >= 8.0:  # High priority messages
                self.process_message(message)
                self.message_queue.remove(message)

        # Then do standard intention selection
        super().select_intentions()


class HierarchicalHybridAgent(HybridAgent):
    """Hybrid agent with organizational awareness."""

    def __init__(
        self,
        agent_id: AgentId,
        observable_properties: Set[str],
        roles: Set[str],
        position: OrganizationalPosition,
        hierarchy_manager: "HierarchyManager",
    ):
        super().__init__(agent_id, observable_properties)

        # Add organizational capabilities
        self.org_mixin = OrganizationalMixin()
        self.org_mixin.__init_organizational__(
            roles, position, hierarchy_manager
        )

        # Copy organizational attributes
        self.organizational_position = position
        self.assigned_roles = roles
        self.hierarchy_manager = hierarchy_manager
        self.message_queue = self.org_mixin.message_queue

        # Delegate methods
        self.send_command = self.org_mixin.send_command
        self.report_to_supervisor = self.org_mixin.report_to_supervisor
        self.escalate_issue = self.org_mixin.escalate_issue
        self.delegate_goal = self.org_mixin.delegate_goal
        self.process_message = self.org_mixin.process_message

    def decide_action(self) -> str:
        """Layered decision with hierarchy integration."""
        available_actions = set(self.available_actions.keys())

        # Process high priority messages first
        for message in self.message_queue[:]:
            if message.priority >= 9.0:  # Commands and emergencies
                self.process_message(message)
                self.message_queue.remove(message)

        # Layer 1: Emergency + Supervisor commands (highest priority)
        emergency_rules = [
            r for r in self.decision.reactive_rules if r.priority >= 9.0
        ]

        for rule in sorted(
            emergency_rules, key=lambda r: r.priority, reverse=True
        ):
            state_dict = {
                "internal": {
                    k: v.proposition
                    for k, v in self.state.internal_beliefs.items()
                },
                "external": {
                    k: v.proposition
                    for k, v in self.state.external_beliefs.items()
                },
            }

            if rule.condition(state_dict) and rule.action in available_actions:
                return rule.action

        # Continue with standard hybrid decision making
        return super().decide_action()


class HierarchyManager:
    """
    Manages tree hierarchy structure and organizational relationships.
    Enforces tree invariants and provides hierarchy operations.
    """

    def __init__(self, role_manager: RoleManager, group_manager: GroupManager):
        self.role_manager = role_manager
        self.group_manager = group_manager
        self.agents: Dict[str, AbstractAgent] = {}
        self.hierarchy_structure: Dict[str, OrganizationalPosition] = {}
        self.message_queue: List[HierarchyMessage] = []
        self.root_agents: Set[str] = set()

    def add_agent(
        self,
        agent: AbstractAgent,
        supervisor: Optional[str] = None,
        roles: Optional[Set[str]] = None,
        groups: Optional[Set[str]] = None,
    ):
        """Add agent to hierarchy with optional supervisor."""
        agent_id = str(agent.id)

        # Create organizational position
        position = OrganizationalPosition(
            roles=roles or set(), supervisor=supervisor, groups=groups or set()
        )

        # Calculate hierarchy level
        if supervisor:
            supervisor_level = self.hierarchy_structure.get(
                supervisor, OrganizationalPosition()
            ).hierarchy_level
            position.hierarchy_level = supervisor_level + 1
        else:
            position.hierarchy_level = 0
            self.root_agents.add(agent_id)

        # Update supervisor's subordinates
        if supervisor and supervisor in self.hierarchy_structure:
            self.hierarchy_structure[supervisor].subordinates.add(agent_id)

        # Store agent and position
        self.agents[agent_id] = agent
        self.hierarchy_structure[agent_id] = position

        # Assign roles
        if roles:
            for role_name in roles:
                self.role_manager.assign_role(agent_id, role_name)

        # Add to groups
        if groups:
            for group_name in groups:
                self.group_manager.add_agent_to_group(agent_id, group_name)

    def remove_agent(self, agent_id: str):
        """Remove agent from hierarchy."""
        if agent_id not in self.agents:
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
        for group_name in position.groups:
            self.group_manager.remove_agent_from_group(agent_id, group_name)

        # Clean up
        del self.agents[agent_id]
        del self.hierarchy_structure[agent_id]
        self.root_agents.discard(agent_id)

        return True

    def change_supervisor(self, agent_id: str, new_supervisor: Optional[str]):
        """Change agent's supervisor."""
        if agent_id not in self.hierarchy_structure:
            return False

        position = self.hierarchy_structure[agent_id]
        old_supervisor = position.supervisor

        # Remove from old supervisor
        if old_supervisor and old_supervisor in self.hierarchy_structure:
            self.hierarchy_structure[old_supervisor].subordinates.discard(
                agent_id
            )

        # Add to new supervisor
        if new_supervisor:
            if new_supervisor not in self.hierarchy_structure:
                return False
            self.hierarchy_structure[new_supervisor].subordinates.add(agent_id)
            position.supervisor = new_supervisor

            # Update hierarchy level
            supervisor_level = self.hierarchy_structure[
                new_supervisor
            ].hierarchy_level
            position.hierarchy_level = supervisor_level + 1
            self.root_agents.discard(agent_id)
        else:
            position.supervisor = None
            position.hierarchy_level = 0
            self.root_agents.add(agent_id)

        return True

    def get_path_to_root(self, agent_id: str) -> List[str]:
        """Get path from agent to root of hierarchy."""
        path = []
        current = agent_id

        while current and current in self.hierarchy_structure:
            path.append(current)
            current = self.hierarchy_structure[current].supervisor

            # Prevent infinite loops
            if len(path) > 100:
                break

        return path

    def get_escalation_path(
        self, agent_id: str, target_level: int
    ) -> List[str]:
        """Get escalation path to specific hierarchy level."""
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
        self, agent1_id: str, agent2_id: str
    ) -> Optional[str]:
        """Find lowest common supervisor of two agents."""
        path1 = set(self.get_path_to_root(agent1_id))
        path2 = self.get_path_to_root(agent2_id)

        for agent in path2:
            if agent in path1:
                return agent

        return None

    def get_subordinates_recursive(self, agent_id: str) -> Set[str]:
        """Get all subordinates recursively."""
        if agent_id not in self.hierarchy_structure:
            return set()

        subordinates = set()
        to_process = [agent_id]

        while to_process:
            current = to_process.pop()
            if current in self.hierarchy_structure:
                direct_subordinates = self.hierarchy_structure[
                    current
                ].subordinates
                subordinates.update(direct_subordinates)
                to_process.extend(direct_subordinates)

        return subordinates

    def send_message(self, message: HierarchyMessage) -> bool:
        """Send message through hierarchy."""
        receiver_agent = self.agents.get(message.receiver)
        if not receiver_agent:
            return False

        # Add to receiver's message queue
        if hasattr(receiver_agent, "message_queue"):
            receiver_agent.message_queue.append(message)

        return True

    def get_hierarchy_statistics(self) -> Dict[str, Any]:
        """Get statistics about the hierarchy structure."""
        total_agents = len(self.agents)
        total_levels = (
            max(
                [
                    pos.hierarchy_level
                    for pos in self.hierarchy_structure.values()
                ]
            )
            + 1
        )

        level_distribution = {}
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


class OrganizationalAgentFactory:
    """Factory for creating hierarchical agents with proper setup."""

    def __init__(self, hierarchy_manager: HierarchyManager):
        self.hierarchy_manager = hierarchy_manager

    def create_agent(
        self,
        agent_type: str,
        agent_id: AgentId,
        observable_properties: Set[str],
        roles: Set[str],
        supervisor: Optional[str] = None,
        groups: Optional[Set[str]] = None,
    ) -> AbstractAgent:
        """Create hierarchical agent of specified type."""

        # Create organizational position
        position = OrganizationalPosition(
            roles=roles, supervisor=supervisor, groups=groups or set()
        )

        # Create agent based on type
        if agent_type == "reactive":
            agent = HierarchicalReactiveAgent(
                agent_id,
                observable_properties,
                roles,
                position,
                self.hierarchy_manager,
            )
        elif agent_type == "bdi":
            agent = HierarchicalBDIAgent(
                agent_id,
                observable_properties,
                roles,
                position,
                self.hierarchy_manager,
            )
        elif agent_type == "hybrid":
            agent = HierarchicalHybridAgent(
                agent_id,
                observable_properties,
                roles,
                position,
                self.hierarchy_manager,
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        # Register in hierarchy
        self.hierarchy_manager.add_agent(agent, supervisor, roles, groups)

        return agent
