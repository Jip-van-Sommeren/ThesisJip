"""
BDI Agent Implementation
Based on the formal definition: A_BDI = ⟨ID, State_beliefs, Goals_desires,
Decision_BDI, Perception, Action⟩

BDI agents have:
- Rich state (beliefs about environment and digital twin)
- Explicit goals (desires from which intentions are selected)
- Deliberative decision function with plan generation
- Suited for reasoning, coordination, optimization, scheduling
"""

from abstract_agent import AbstractAgent, AgentId, Goal
from typing import Dict, Set, List, Optional
from dataclasses import dataclass
from enum import Enum


class IntentionStatus(Enum):
    """Status of an intention in the BDI agent."""

    SELECTED = "selected"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Intention:
    """
    An intention represents a committed goal with associated plan.
    In BDI terms, intentions are desires the agent commits to pursue.
    """

    goal: Goal
    plan: List[str]  # Sequence of action IDs
    current_step: int = 0
    status: IntentionStatus = IntentionStatus.SELECTED

    def get_next_action(self) -> Optional[str]:
        """Get the next action in the plan."""
        if self.current_step < len(self.plan):
            return self.plan[self.current_step]
        return None

    def advance_plan(self):
        """Move to the next step in the plan."""
        if self.current_step < len(self.plan):
            self.current_step += 1

        if self.current_step >= len(self.plan):
            self.status = IntentionStatus.COMPLETED


class BDIAgent(AbstractAgent):
    """
    BDI Agent: A_BDI = ⟨ID, State_beliefs, Goals_desires, Decision_BDI,
    Perception, Action⟩

    Characteristics:
    - State represents beliefs (model of environment and digital twin)
    - Goals correspond to desires (from which intentions are selected)
    - Decision_BDI uses deliberative reasoning with plan generation
    - Maintains intentions (committed subset of desires)
    - Suited for reasoning, coordination, optimization, predictive maintenance
    """

    def __init__(self, agent_id: AgentId, observable_properties: Set[str]):
        super().__init__(agent_id, observable_properties)

        # BDI-specific components
        self.intentions: List[Intention] = []  # Current committed intentions
        self.plan_library: Dict[str, List[str]] = {}  # Pre-defined plans
        self.max_intentions: int = 3  # Limit concurrent intentions

        # Initialize BDI agent
        self.initialize_agent()

    def initialize_agent(self):
        """
        Initialize BDI agent with basic plan library.
        Concrete implementations should extend this.
        """
        # Basic plan library - can be extended by subclasses
        self.plan_library = {
            "default_plan": ["noop"],
            "exploration_plan": ["observe", "analyze", "report"],
            "optimization_plan": ["assess", "optimize", "validate"],
        }

    def add_plan(self, plan_name: str, actions: List[str]):
        """Add a plan to the agent's plan library."""
        self.plan_library[plan_name] = actions

    def select_intentions(self):
        """
        Intention selection: choose desires to commit to as intentions.
        This implements the desire-to-intention filtering in BDI.
        """
        # Remove completed or failed intentions
        self.intentions = [
            i
            for i in self.intentions
            if i.status
            not in [IntentionStatus.COMPLETED, IntentionStatus.FAILED]
        ]

        # If we have room for more intentions, select from active goals
        if len(self.intentions) < self.max_intentions:
            # Sort goals by priority
            active_goals = [g for g in self.goals if g.active]
            active_goals.sort(key=lambda g: g.priority, reverse=True)

            # Select goals that aren't already intentions
            existing_goal_conditions = {
                i.goal.condition for i in self.intentions
            }

            for goal in active_goals:
                if len(self.intentions) >= self.max_intentions:
                    break

                if goal.condition not in existing_goal_conditions:
                    # Generate or select a plan for this goal
                    plan = self.generate_plan(goal)
                    if plan:
                        intention = Intention(
                            goal=goal,
                            plan=plan,
                            status=IntentionStatus.SELECTED,
                        )
                        self.intentions.append(intention)

    def generate_plan(self, goal: Goal) -> List[str]:
        """
        Plan generation: create a sequence of actions to achieve a goal.
        Simple implementation - can be extended with more sophisticated
        planning.
        """
        # Simple plan selection based on goal type
        if goal.goal_type.value in self.plan_library:
            return self.plan_library[goal.goal_type.value].copy()

        # Check if goal condition suggests a specific plan
        for plan_name, plan_actions in self.plan_library.items():
            if plan_name in goal.condition.lower():
                return plan_actions.copy()

        # Default plan
        return self.plan_library.get("default_plan", ["noop"])

    def decide_action(self) -> str:
        """
        BDI decision function: deliberative reasoning with intention execution.
        No reactive rules - pure deliberation based on beliefs and intentions.
        """
        # First, perform intention selection (desire filtering)
        self.select_intentions()

        # If no intentions, fall back to basic deliberation
        if not self.intentions:
            available_actions = set(self.available_actions.keys())
            return self.decision.deliberate(
                self.state, self.goals, available_actions
            )

        # Execute current intentions
        for intention in self.intentions:
            if intention.status == IntentionStatus.SELECTED:
                intention.status = IntentionStatus.ACTIVE

            if intention.status == IntentionStatus.ACTIVE:
                next_action = intention.get_next_action()
                if next_action and next_action in self.available_actions:
                    return next_action

        # Default action if no intention can proceed
        return "noop" if "noop" in self.available_actions else ""

    def execute_action(self, action_id: str, environment_state: Dict) -> Dict:
        """
        Execute action and update intentions based on results.
        """
        new_environment = super().execute_action(action_id, environment_state)

        # Update intentions based on action execution
        for intention in self.intentions:
            if (
                intention.status == IntentionStatus.ACTIVE
                and intention.get_next_action() == action_id
            ):

                intention.advance_plan()

                # Check if goal is achieved
                if self.goal_achieved(intention.goal, new_environment):
                    intention.status = IntentionStatus.COMPLETED

        return new_environment

    def goal_achieved(self, goal: Goal, environment_state: Dict) -> bool:
        """
        Check if a goal has been achieved based on current state.
        Simple implementation - can be extended with more sophisticated
        reasoning.
        """
        # Check if goal condition is satisfied by current beliefs
        return self.state.satisfies(goal.condition)

    def perceive(self, environment_state: Dict):
        """
        BDI perception: maintain rich belief state about environment.
        Updates beliefs and may trigger intention revision.
        """
        super().perceive(environment_state)

        # Update belief history for richer reasoning
        timestamp = len(self.state.history)
        belief_snapshot = {
            "internal": dict(self.state.internal_beliefs),
            "external": dict(self.state.external_beliefs),
        }
        self.state.history.append((timestamp, belief_snapshot))

        # Limit history to prevent unbounded growth
        if len(self.state.history) > 100:
            self.state.history.pop(0)

    def revise_intentions(self):
        """
        Intention revision: modify intentions based on changed beliefs.
        This is called when significant belief changes occur.
        """
        for intention in self.intentions:
            # Check if intention is still feasible given current beliefs
            if not self.intention_feasible(intention):
                intention.status = IntentionStatus.SUSPENDED
            elif intention.status == IntentionStatus.SUSPENDED:
                # Reactivate if now feasible
                intention.status = IntentionStatus.ACTIVE

    def intention_feasible(self, intention: Intention) -> bool:
        """
        Check if an intention is still feasible given current beliefs.
        Simple implementation - can be extended.
        """
        # Basic feasibility: goal is still active and not achieved
        return intention.goal.active and not self.goal_achieved(
            intention.goal, {}
        )

    def get_bdi_status(self) -> Dict:
        """Get current BDI-specific status information."""
        return {
            "agent_type": "bdi",
            "beliefs_count": {
                "internal": len(self.state.internal_beliefs),
                "external": len(self.state.external_beliefs),
                "history_length": len(self.state.history),
            },
            "desires_count": len([g for g in self.goals if g.active]),
            "intentions_count": len(self.intentions),
            "intentions_status": {
                status.value: len(
                    [i for i in self.intentions if i.status == status]
                )
                for status in IntentionStatus
            },
            "plan_library_size": len(self.plan_library),
        }

    def get_agent_type(self) -> str:
        """Return agent type identifier."""
        return "bdi"

    def __str__(self) -> str:
        return f"BDIAgent({self.id})"
