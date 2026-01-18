"""
Reactive Agent Implementation

Based on the formal definition:
A_reactive = ⟨ID, State, ∅, Decision_reactive, Perception, Action⟩

Reactive agents:
- Have minimal state (essentially latest perceptions)
- No explicit goals (empty set)
- Fixed mapping from state/percepts to action (stimulus-response)
- Direct coupling enables fast responses

Use for: High-speed safeguards, anomaly triggers, fail-safe loops.
"""

from typing import Dict, Set, Tuple

from .agent import (
    Agent,
    AgentId,
    Action,
    ActionType,
    ReactiveRule,
)


class ReactiveAgent(Agent):
    """
    Reactive Agent: A_reactive = ⟨ID, State, ∅, Decision_reactive, Perception, Action⟩

    Characteristics:
    - State: Minimal, essentially the latest perceptions
    - Goals: None (empty set) - behavior is purely rule-driven
    - Decision: Fixed mapping from state/percepts to action (stimulus-response)
    - Perception/Action: Direct coupling enables fast responses

    Suitable for time-critical control loops and immediate response scenarios.
    """

    def __init__(self, agent_id: AgentId, observable_properties: Set[str]):
        super().__init__(agent_id, observable_properties)
        self.initialize_agent()

    def initialize_agent(self):
        """
        Initialize reactive agent with default actions.

        Subclasses should override to add specific reactive rules.
        """
        # Add default noop action
        self.add_action(
            Action(
                action_id="noop",
                action_type=ActionType.TRANSIENT,
                preconditions=lambda env: True,
                effects=lambda env: env,
            )
        )

    def decide_action(self) -> str:
        """
        Reactive decision: pure stimulus-response.

        Only checks reactive rules, no deliberation.
        Returns first matching rule's action or 'noop'.
        """
        available_actions = set(self.available_actions.keys())

        # Check reactive rules in priority order
        reactive_action = self.decision.check_reactive_rules(self.state)
        if reactive_action and reactive_action in available_actions:
            return reactive_action

        # Default fallback
        return "noop" if "noop" in available_actions else ""

    def step(self, environment_state: Dict) -> Tuple[str, Dict]:
        """
        Reactive agent step: minimal perceive -> react cycle.

        Optimized for fast response times.
        """
        # Perception phase
        self.perceive(environment_state)

        # Reactive decision phase (no deliberation)
        chosen_action = self.decide_action()

        # Action execution phase
        new_environment = self.execute_action(chosen_action, environment_state)

        return chosen_action, new_environment

    def add_stimulus_response(
        self,
        condition: callable,
        action_id: str,
        priority: float = 1.0
    ):
        """
        Convenience method to add a stimulus-response rule.

        Args:
            condition: Function(state_dict) -> bool
            action_id: Action to perform when condition is true
            priority: Rule priority (higher = evaluated first)
        """
        rule = ReactiveRule(
            condition=condition,
            action=action_id,
            priority=priority,
        )
        self.add_reactive_rule(rule)

    def get_agent_type(self) -> str:
        """Return agent type identifier."""
        return "reactive"

    def __str__(self) -> str:
        return f"ReactiveAgent({self.id})"


__all__ = ["ReactiveAgent"]
