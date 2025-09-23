"""
Reactive Agent Implementation
Based on the formal definition: A_reactive = ⟨ID, State, ∅, Decision_reactive,
Perception, Action⟩

Reactive agents have:
- Minimal state (most recent perceptions only)
- No explicit goals (Goals = ∅)
- Direct perception-to-action mapping
- Fast response for immediate stimuli
"""

from abstract_agent import AbstractAgent, AgentId
from typing import Dict, Set


class ReactiveAgent(AbstractAgent):
    """
    Reactive Agent: A_reactive = ⟨ID, State, ∅, Decision_reactive, Perception,
    Action⟩

    Characteristics:
    - Goals = ∅ (empty set - no explicit goals)
    - State is minimal (only recent perceptions)
    - Decision function uses only reactive rules (no deliberation)
    - Direct stimulus-response behavior
    - Suited for high-speed responses, anomaly detection, fail-safe mechanisms
    """

    def __init__(self, agent_id: AgentId, observable_properties: Set[str]):
        super().__init__(agent_id, observable_properties)

        # Reactive agents have no goals by definition
        self.goals = set()  # Goals = ∅

        # Initialize reactive-specific components
        self.initialize_agent()

    def initialize_agent(self):
        """
        Initialize reactive agent with only reactive rules and actions.
        No goals or deliberative reasoning.
        """
        # Default reactive agent has minimal setup
        # Concrete implementations should override this method
        pass

    def add_goal(self, goal):
        """
        Reactive agents cannot have goals by definition.
        This method is disabled for reactive agents.
        """
        raise NotImplementedError(
            "Reactive agents cannot have goals (Goals = ∅). "
            "Use BDI or Hybrid agents for goal-driven behavior."
        )

    def decide_action(self) -> str:
        """
        Reactive decision function: only uses reactive rules.
        No deliberative reasoning since Goals = ∅.
        """
        # Only check reactive rules - no deliberation
        available_action_ids = set(self.available_actions.keys())
        reactive_action = self.decision.check_reactive_rules(self.state)

        if reactive_action and reactive_action in available_action_ids:
            return reactive_action

        # Default action if no rules fire
        return "noop" if "noop" in available_action_ids else ""

    def perceive(self, environment_state: Dict):
        """
        Minimal perception: only store most recent observations.
        No historical state maintenance.
        """
        # Clear previous external beliefs to keep state minimal
        self.state.external_beliefs.clear()

        # Update with current observations only
        observations = self.perception.observe(environment_state)
        self.state = self.perception.update_state(self.state, observations)

    def step(self, environment_state: Dict) -> tuple[str, Dict]:
        """
        Reactive agent step: direct perception-to-action mapping.
        Optimized for fast response.
        """
        # Minimal perception (current observations only)
        self.perceive(environment_state)

        # Reactive decision (rules only, no deliberation)
        chosen_action = self.decide_action()

        # Execute action
        new_environment = self.execute_action(chosen_action, environment_state)

        return chosen_action, new_environment

    def get_agent_type(self) -> str:
        """Return agent type identifier."""
        return "reactive"

    def __str__(self) -> str:
        return f"ReactiveAgent({self.id})"
