"""
Hybrid Agent Implementation

Based on the formal definition:
A_hybrid = ⟨ID, State_reactive+deliberative, Goals, Decision_layered, Perception, Action⟩

Hybrid agents combine:
- Reactive layer for immediate stimulus-response
- Deliberative layer for reasoning and planning
- Layered decision structure with reactive precedence
- Both short-term perceptual data and structured internal models

This is the default agent type for digital twins.
"""

from enum import Enum
from typing import Dict, Set, List, Tuple

from .agent import (
    Agent,
    AgentId,
    Goal,
    Action,
    ActionType,
    ReactiveRule,
)
from .bdi_agent import Intention, IntentionStatus


class LayerPriority(Enum):
    """Priority levels for different decision layers."""
    EMERGENCY = 10.0    # Highest priority reactive rules
    REACTIVE = 5.0      # Standard reactive rules
    DELIBERATIVE = 1.0  # Deliberative reasoning
    DEFAULT = 0.1       # Default/fallback actions


class HybridAgent(Agent):
    """
    Hybrid Agent: A_hybrid = ⟨ID, State_reactive+deliberative, Goals,
    Decision_layered, Perception, Action⟩

    Characteristics:
    - State maintains both reactive perceptions and deliberative beliefs
    - Goals guide deliberative layer behavior
    - Decision_layered: reactive layer handles immediate stimuli,
      deliberative layer plans
    - Combines immediacy of reactive behavior with flexibility of
      deliberative reasoning
    - Reactive rules take precedence over deliberative actions
    """

    def __init__(self, agent_id: AgentId, observable_properties: Set[str]):
        super().__init__(agent_id, observable_properties)

        # Hybrid-specific components
        self.intentions: List[Intention] = []
        self.plan_library: Dict[str, List[str]] = {}
        self.max_intentions: int = 2  # Fewer than pure BDI for efficiency

        # Layer configuration
        self.reactive_layer_enabled: bool = True
        self.deliberative_layer_enabled: bool = True
        self.emergency_threshold: float = 8.0

        # Performance tracking
        self.layer_stats = {
            "reactive_actions": 0,
            "deliberative_actions": 0,
            "emergency_actions": 0,
        }

        self.initialize_agent()

    def initialize_agent(self):
        """Initialize hybrid agent with both reactive rules and deliberative capabilities."""
        # Default plan library for deliberative layer
        self.plan_library = {
            "default_plan": ["noop"],
            "optimization_plan": ["analyze", "optimize", "validate"],
            "maintenance_plan": ["diagnose", "repair", "test"],
        }

        # Default emergency reactive rule
        emergency_rule = ReactiveRule(
            condition=lambda state: self._emergency_condition(state),
            action="emergency_stop",
            priority=LayerPriority.EMERGENCY.value,
        )
        self.add_reactive_rule(emergency_rule)

        # Add default emergency action
        self.add_action(
            Action(
                action_id="emergency_stop",
                action_type=ActionType.TRANSIENT,
                preconditions=lambda env: True,
                effects=lambda env: {**env, "emergency_stop": True},
            )
        )

        # Add default noop action
        self.add_action(
            Action(
                action_id="noop",
                action_type=ActionType.TRANSIENT,
                preconditions=lambda env: True,
                effects=lambda env: env,
            )
        )

    def _emergency_condition(self, state_dict: Dict) -> bool:
        """
        Default emergency condition - can be overridden by subclasses.

        Checks for critical alerts in external beliefs.
        """
        for belief in state_dict.get("external", {}).values():
            if isinstance(belief, str):
                if "critical" in belief.lower() or "emergency" in belief.lower():
                    return True
        return False

    def add_plan(self, plan_name: str, actions: List[str]):
        """Add a plan to the deliberative layer's plan library."""
        self.plan_library[plan_name] = actions

    def enable_layer(self, reactive: bool = True, deliberative: bool = True):
        """Enable or disable specific layers."""
        self.reactive_layer_enabled = reactive
        self.deliberative_layer_enabled = deliberative

    def decide_action(self) -> str:
        """
        Hybrid decision function: layered decision structure.

        1. Check emergency reactive rules (highest priority)
        2. Check standard reactive rules
        3. Use deliberative reasoning if no reactive rules fire
        """
        available_actions = set(self.available_actions.keys())

        # Layer 1: Emergency reactive rules (highest priority)
        if self.reactive_layer_enabled:
            emergency_rules = [
                r for r in self.decision.reactive_rules
                if r.priority >= self.emergency_threshold
            ]

            for rule in sorted(emergency_rules, key=lambda r: r.priority, reverse=True):
                state_dict = self.state.to_dict()

                if rule.condition(state_dict) and rule.action in available_actions:
                    self.layer_stats["emergency_actions"] += 1
                    return rule.action

        # Layer 2: Standard reactive rules
        if self.reactive_layer_enabled:
            standard_rules = [
                r for r in self.decision.reactive_rules
                if r.priority < self.emergency_threshold
            ]

            for rule in sorted(standard_rules, key=lambda r: r.priority, reverse=True):
                state_dict = self.state.to_dict()

                if rule.condition(state_dict) and rule.action in available_actions:
                    self.layer_stats["reactive_actions"] += 1
                    return rule.action

        # Layer 3: Deliberative reasoning
        if self.deliberative_layer_enabled:
            deliberative_action = self._deliberative_decision()
            if deliberative_action and deliberative_action in available_actions:
                self.layer_stats["deliberative_actions"] += 1
                return deliberative_action

        # Default fallback
        return "noop" if "noop" in available_actions else ""

    def _deliberative_decision(self) -> str:
        """
        Deliberative decision making similar to BDI agents.

        Manages intentions and executes plans.
        """
        # Select intentions from active goals
        self._select_intentions()

        # Execute current intentions
        for intention in self.intentions:
            if intention.status == IntentionStatus.SELECTED:
                intention.status = IntentionStatus.ACTIVE

            if intention.status == IntentionStatus.ACTIVE:
                next_action = intention.get_next_action()
                if next_action and next_action in self.available_actions:
                    return next_action

        # If no intentions, use basic goal-driven selection
        if self.goals:
            return self.decision.deliberate(
                self.state, self.goals, set(self.available_actions.keys())
            )

        return "noop"

    def _select_intentions(self):
        """Select intentions from active goals (simplified BDI-style)."""
        # Clean up completed intentions
        self.intentions = [
            i for i in self.intentions
            if i.status not in [IntentionStatus.COMPLETED, IntentionStatus.FAILED]
        ]

        # Add new intentions if we have capacity
        if len(self.intentions) < self.max_intentions:
            active_goals = [g for g in self.goals if g.active]
            active_goals.sort(key=lambda g: g.priority, reverse=True)

            existing_conditions = {i.goal.condition for i in self.intentions}

            for goal in active_goals:
                if len(self.intentions) >= self.max_intentions:
                    break

                if goal.condition not in existing_conditions:
                    plan = self._generate_plan(goal)
                    if plan:
                        intention = Intention(goal=goal, plan=plan)
                        self.intentions.append(intention)

    def _generate_plan(self, goal: Goal) -> List[str]:
        """Generate a plan for a goal (simplified planning)."""
        for plan_name, actions in self.plan_library.items():
            if plan_name in goal.condition.lower():
                return actions.copy()

        return self.plan_library.get("default_plan", ["noop"])

    def execute_action(self, action_id: str, environment_state: Dict) -> Dict:
        """Execute action and update both reactive and deliberative components."""
        new_environment = super().execute_action(action_id, environment_state)

        # Update intentions (deliberative layer)
        for intention in self.intentions:
            if (
                intention.status == IntentionStatus.ACTIVE
                and intention.get_next_action() == action_id
            ):
                intention.advance_plan()

                # Simple goal achievement check
                if self.state.satisfies(intention.goal.condition):
                    intention.status = IntentionStatus.COMPLETED

        return new_environment

    def perceive(self, environment_state: Dict):
        """Hybrid perception: maintain both reactive perceptions and deliberative beliefs."""
        super().perceive(environment_state)

        timestamp = len(self.state.history)
        belief_snapshot = {
            "internal": dict(self.state.internal_beliefs),
            "external": dict(self.state.external_beliefs),
        }
        self.state.history.append((timestamp, belief_snapshot))

        if len(self.state.history) > 50:  # Less than BDI agent's 100
            self.state.history.pop(0)

    def step(self, environment_state: Dict) -> Tuple[str, Dict]:
        """Hybrid agent step: optimized for both reactive speed and deliberative reasoning."""
        # Perception phase
        self.perceive(environment_state)

        # Layered decision phase
        chosen_action = self.decide_action()

        # Action execution phase
        new_environment = self.execute_action(chosen_action, environment_state)

        return chosen_action, new_environment

    def get_hybrid_status(self) -> Dict:
        """Get hybrid agent-specific status information."""
        return {
            "agent_type": "hybrid",
            "layers_enabled": {
                "reactive": self.reactive_layer_enabled,
                "deliberative": self.deliberative_layer_enabled,
            },
            "layer_statistics": self.layer_stats.copy(),
            "reactive_rules": len(self.decision.reactive_rules),
            "emergency_rules": len([
                r for r in self.decision.reactive_rules
                if r.priority >= self.emergency_threshold
            ]),
            "intentions": len(self.intentions),
            "active_goals": len([g for g in self.goals if g.active]),
            "plan_library_size": len(self.plan_library),
            "belief_history_length": len(self.state.history),
        }

    def reset_layer_stats(self):
        """Reset performance statistics."""
        self.layer_stats = {
            "reactive_actions": 0,
            "deliberative_actions": 0,
            "emergency_actions": 0,
        }

    def get_agent_type(self) -> str:
        """Return agent type identifier."""
        return "hybrid"

    def __str__(self) -> str:
        return f"HybridAgent({self.id})"


__all__ = ["HybridAgent", "LayerPriority"]
