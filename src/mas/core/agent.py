"""
Core Agent Abstractions

Based on the formal definition: A = (Id, State, Goal, Perception, Action, Decision)

This module provides the foundational components for building agents:
- AgentId: Hierarchical unique identifier
- State: Internal/external beliefs
- Goal: Agent objectives
- Belief: Probabilistic knowledge representation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Set, Tuple, Optional, List, Any
import uuid


class GoalType(Enum):
    """Types of goals an agent can have."""

    INTRINSIC = "intrinsic"  # Built-in from role/design
    EXTRINSIC = "extrinsic"  # Assigned by scenario/application
    WORKFLOW = "workflow"  # Execution-oriented
    PERFORMANCE = "performance"  # Optimization objectives


class ActionType(Enum):
    """Types of actions an agent can perform."""

    TRANSIENT = "transient"  # One-shot execution
    PERSISTENT = "persistent"  # Long-running
    PERIODIC = "periodic"  # Repeating at intervals


@dataclass
class AgentId:
    """
    Agent Identity: Unique identifier with hierarchical structure.

    id_A = (app, type, instance) for global uniqueness and context.

    Examples:
        AgentId("battery_twin", "telemetry_ingestor", "001")
        AgentId("factory", "controller", "zone_a")
    """

    app: str  # Application/domain
    type: str  # Agent type within application
    instance: str  # Unique instance identifier

    def __post_init__(self):
        if not self.instance:
            self.instance = str(uuid.uuid4())[:8]

    def __str__(self) -> str:
        return f"{self.app}.{self.type}.{self.instance}"

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other) -> bool:
        if isinstance(other, AgentId):
            return str(self) == str(other)
        return False


@dataclass
class Belief:
    """
    A belief with proposition and confidence level.

    Belief_A(p, c) means agent A believes proposition p with confidence c.
    Confidence c ∈ [0, 1] where 1 = certain, 0 = no confidence.
    """

    proposition: str
    confidence: float = 1.0
    timestamp: float = 0.0

    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0 and 1")


@dataclass(frozen=True)
class Goal:
    """
    Represents an agent goal with type and priority.

    Goals represent desired world states the agent aims to achieve.
    """

    condition: str
    goal_type: GoalType
    priority: float = 1.0
    deadline: Optional[float] = None
    active: bool = True


class State:
    """
    Internal State: State_A = ⟨B^int_A, B^ext_A⟩

    Captures both internal beliefs (about self) and external beliefs
    (about environment and other agents).
    """

    def __init__(self):
        self.internal_beliefs: Dict[str, Belief] = {}
        self.external_beliefs: Dict[str, Belief] = {}
        self.history: List[Tuple[float, Dict]] = []

    def add_internal_belief(self, key: str, belief: Belief):
        """Add internal belief about agent's own condition."""
        self.internal_beliefs[key] = belief

    def add_external_belief(self, key: str, belief: Belief):
        """Add external belief about environment/other agents."""
        self.external_beliefs[key] = belief

    def get_belief(self, key: str) -> Optional[Belief]:
        """Get belief by key, checking both internal and external."""
        return self.internal_beliefs.get(key) or self.external_beliefs.get(key)

    def update_belief(
        self,
        key: str,
        proposition: str,
        confidence: float = 1.0,
        is_internal: bool = False,
    ):
        """Update or create a belief."""
        belief = Belief(proposition, confidence)
        if is_internal:
            self.internal_beliefs[key] = belief
        else:
            self.external_beliefs[key] = belief

    def satisfies(self, condition: str) -> bool:
        """Check if current state satisfies a condition."""
        for belief_dict in [self.internal_beliefs, self.external_beliefs]:
            for belief in belief_dict.values():
                if condition in belief.proposition and belief.confidence > 0.5:
                    return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary representation."""
        return {
            "internal": {
                k: v.proposition for k, v in self.internal_beliefs.items()
            },
            "external": {
                k: v.proposition for k, v in self.external_beliefs.items()
            },
        }


@dataclass
class ReactiveRule:
    """
    Condition-action rule: (φ → a)

    If condition φ holds then perform action a.
    Higher priority rules are evaluated first.
    """

    condition: callable  # Function: Dict -> bool
    action: str
    priority: float = 1.0


class Perception:
    """
    Perception component: Ω_A: E → O_A

    Maps the global environment state to an agent's observable percepts.
    Includes belief update function: upd: State × O → State
    """

    def __init__(self, observable_properties: Set[str]):
        self.observable_properties = observable_properties

    def observe(self, environment_state: Dict) -> Dict:
        """
        Observation function Ω_A(e): filters environment to agent's perspective.

        Returns observable key-value pairs for this agent.
        """
        observations = {}
        for prop in self.observable_properties:
            if prop in environment_state:
                observations[prop] = environment_state[prop]
        return observations

    def update_state(self, state: State, observations: Dict) -> State:
        """
        Belief update function: upd: State × O → State

        Updates agent's external beliefs based on new observations.
        """
        for key, value in observations.items():
            state.update_belief(
                key, str(value), confidence=1.0, is_internal=False
            )
        return state


class Action:
    """
    Action component: Defines agent's interaction capabilities.

    Act_A ⊆ A - the collection of actions agent A can perform.
    Actions transform environment state e ∈ E to e'.
    """

    def __init__(
        self,
        action_id: str,
        action_type: ActionType,
        preconditions: Optional[callable] = None,
        effects: Optional[callable] = None,
    ):
        self.action_id = action_id
        self.action_type = action_type
        self.preconditions = preconditions or (lambda _: True)
        self.effects = effects or (lambda x: x)

    def can_execute(self, environment_state: Dict) -> bool:
        """Check if action preconditions are met."""
        return self.preconditions(environment_state)

    def execute(self, environment_state: Dict) -> Dict:
        """Execute action and return new environment state."""
        if self.can_execute(environment_state):
            return self.effects(environment_state)
        return environment_state


class Decision:
    """
    Decision component: δ_A implements sense-think-act cycle.

    Combines reactive rules R_A and deliberative reasoning f_BDI.
    """

    def __init__(self):
        self.reactive_rules: List[ReactiveRule] = []

    def add_reactive_rule(self, rule: ReactiveRule):
        """Add a reactive rule to the rule set R_A."""
        self.reactive_rules.append(rule)

    def check_reactive_rules(self, state: State) -> Optional[str]:
        """
        Check if reactive rule fires: (φ → a) ∈ R_A such that State_A ⊨ φ

        Returns action if rule fires, None otherwise.
        """
        sorted_rules = sorted(
            self.reactive_rules, key=lambda r: r.priority, reverse=True
        )

        state_dict = state.to_dict()

        for rule in sorted_rules:
            if rule.condition(state_dict):
                return rule.action

        return None

    def deliberate(
        self, state: State, goals: Set[Goal], available_actions: Set[str]
    ) -> str:
        """
        f_BDI: Deliberative reasoning function.

        Uses beliefs, goals, and available actions to choose an action.
        """
        if not goals or not available_actions:
            return "noop"

        active_goals = [g for g in goals if g.active]
        if not active_goals:
            return "noop"

        active_goals.sort(key=lambda g: g.priority, reverse=True)

        # Simple heuristic: choose first available action
        return next(iter(available_actions), "noop")

    def decide(
        self, state: State, goals: Set[Goal], available_actions: Set[str]
    ) -> str:
        """
        Main decision function δ_A(State_A).

        Returns action choice based on reactive/deliberative reasoning.
        """
        reactive_action = self.check_reactive_rules(state)
        if reactive_action and reactive_action in available_actions:
            return reactive_action

        return self.deliberate(state, goals, available_actions)


class Agent(ABC):
    """
    Abstract Agent: A = (Id, State, Goal, Perception, Action, Decision)

    Base class for all agent types (Reactive, BDI, Hybrid).

    Components:
    - Id: Unique hierarchical identifier
    - State: Internal belief state (internal + external beliefs)
    - Goal: Set of objectives to achieve
    - Perception: Observation and belief update capabilities
    - Action: Available actions and execution
    - Decision: Reactive rules + deliberative reasoning
    """

    def __init__(self, agent_id: AgentId, observable_properties: Set[str]):
        self.id = agent_id
        self.state = State()
        self.goals: Set[Goal] = set()
        self.perception = Perception(observable_properties)
        self.available_actions: Dict[str, Action] = {}
        self.decision = Decision()

        # Initialize basic beliefs
        self.state.add_internal_belief(
            "agent_id", Belief(f"agent_id={self.id}", 1.0)
        )
        self.state.add_internal_belief("status", Belief("active", 1.0))

    def add_goal(self, goal: Goal):
        """Add a goal to the agent's goal set."""
        self.goals.add(goal)

    def remove_goal(self, goal: Goal):
        """Remove a goal from the agent's goal set."""
        self.goals.discard(goal)

    def add_action(self, action: Action):
        """Add an action to the agent's capability set Act_A."""
        self.available_actions[action.action_id] = action

    def add_reactive_rule(self, rule: ReactiveRule):
        """Add a reactive rule to the decision component."""
        self.decision.add_reactive_rule(rule)

    def perceive(self, environment_state: Dict):
        """
        Perception phase: observe environment and update beliefs.

        Implements Ω_A(e) followed by upd(State_A, o).
        """
        observations = self.perception.observe(environment_state)
        self.state = self.perception.update_state(self.state, observations)

    def decide_action(self) -> str:
        """
        Decision phase: choose action using δ_A(State_A).

        Uses reactive rules or deliberative reasoning.
        """
        available_action_ids = set(self.available_actions.keys())
        return self.decision.decide(
            self.state, self.goals, available_action_ids
        )

    def execute_action(self, action_id: str, environment_state: Dict) -> Dict:
        """
        Action phase: execute chosen action if available and applicable.

        Returns new environment state.
        """
        if action_id in self.available_actions:
            action = self.available_actions[action_id]
            return action.execute(environment_state)
        return environment_state

    def step(self, environment_state: Dict) -> Tuple[str, Dict]:
        """
        Complete agent step: perceive -> decide -> act.

        Returns (chosen_action, new_environment_state).
        """
        self.perceive(environment_state)
        chosen_action = self.decide_action()
        new_environment = self.execute_action(chosen_action, environment_state)
        return chosen_action, new_environment

    @abstractmethod
    def initialize_agent(self):
        """Initialize agent-specific goals, actions, and rules."""
        pass

    def get_agent_id(self) -> AgentId:
        """Return agent's unique identifier."""
        return self.id

    def __str__(self) -> str:
        return f"Agent({self.id})"

    def __repr__(self) -> str:
        return self.__str__()


__all__ = [
    "AgentId",
    "Agent",
    "State",
    "Belief",
    "Goal",
    "GoalType",
    "Action",
    "ActionType",
    "Perception",
    "Decision",
    "ReactiveRule",
]
