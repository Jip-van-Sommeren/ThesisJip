"""
MAS Core Layer

Core agent abstractions based on the formal definition:
A = (Id, State, Goal, Perception, Action, Decision)

Provides three agent types:
- ReactiveAgent: Stimulus-response for fast control
- BDIAgent: Beliefs-Desires-Intentions for planning
- HybridAgent: Layered reactive + deliberative (default for digital twins)
"""

from .agent import (
    Agent,
    AgentId,
    State,
    Belief,
    Goal,
    GoalType,
    Action,
    ActionType,
    Perception,
    Decision,
    ReactiveRule,
)
from .reactive_agent import ReactiveAgent
from .bdi_agent import BDIAgent, Intention, IntentionStatus
from .hybrid_agent import HybridAgent, LayerPriority

__all__ = [
    # Base agent components
    "Agent",
    "AgentId",
    "State",
    "Belief",
    "Goal",
    "GoalType",
    "Action",
    "ActionType",
    "Perception",
    "Decision",
    "ReactiveRule",
    # Agent types
    "ReactiveAgent",
    "BDIAgent",
    "HybridAgent",
    # BDI components
    "Intention",
    "IntentionStatus",
    # Hybrid components
    "LayerPriority",
]
