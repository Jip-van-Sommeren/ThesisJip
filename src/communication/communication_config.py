"""
Communication Configuration Module
Provides utilities for configuring message space and communication topology
according to the formal definitions from the thesis.

Implements:
- Message Space (M) configuration and validation
- Communication Topology (Comm) patterns
- Topology generators for different network structures
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import networkx as nx
from .rest.rest_communication import MessageType, CommunicationTopology


class TopologyPattern(Enum):
    """Predefined communication topology patterns."""

    FULLY_CONNECTED = "fully_connected"
    STAR = "star"
    RING = "ring"
    CHAIN = "chain"
    HIERARCHICAL = "hierarchical"
    SMALL_WORLD = "small_world"
    SCALE_FREE = "scale_free"
    CUSTOM = "custom"


@dataclass
class MessageTemplate:
    """
    Template for messages in the message space M.
    Defines structure and validation for message content.
    """

    message_type: MessageType
    required_fields: Set[str]
    optional_fields: Set[str] = None
    field_validators: Dict[str, callable] = None

    def __post_init__(self):
        if self.optional_fields is None:
            self.optional_fields = set()
        if self.field_validators is None:
            self.field_validators = {}

    def validate_message_content(
        self, content: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Validate message content against template."""
        # Check required fields
        missing_fields = self.required_fields - set(content.keys())
        if missing_fields:
            return False, f"Missing required fields: {missing_fields}"

        # Check field validators
        for field, validator in self.field_validators.items():
            if field in content:
                try:
                    if not validator(content[field]):
                        return False, f"Validation failed for field '{field}'"
                except Exception as e:
                    return (
                        False,
                        f"Validator error for field '{field}': {str(e)}",
                    )

        return True, "Valid"


class MessageSpace:
    """
    Implementation of Message Space (M).
    Manages all possible messages and their templates.
    """

    def __init__(self):
        self.message_templates: Dict[MessageType, MessageTemplate] = {}
        self._setup_default_templates()

    def _setup_default_templates(self):
        """Setup default message templates for common message types."""

        # INFORM messages
        self.add_template(
            MessageTemplate(
                message_type=MessageType.INFORM,
                required_fields={"information"},
                optional_fields={"context", "priority"},
                field_validators={
                    "priority": lambda x: isinstance(x, (int, float))
                    and 0 <= x <= 1
                },
            )
        )

        # REQUEST messages
        self.add_template(
            MessageTemplate(
                message_type=MessageType.REQUEST,
                required_fields={"request_type", "parameters"},
                optional_fields={"timeout", "callback"},
                field_validators={
                    "timeout": lambda x: isinstance(x, (int, float)) and x > 0
                },
            )
        )

        # REPLY messages
        self.add_template(
            MessageTemplate(
                message_type=MessageType.REPLY,
                required_fields={"response"},
                optional_fields={"status", "error"},
                field_validators={
                    "status": lambda x: x in ["success", "failure", "partial"]
                },
            )
        )

        # BROADCAST messages
        self.add_template(
            MessageTemplate(
                message_type=MessageType.BROADCAST,
                required_fields={"announcement"},
                optional_fields={"scope", "urgency"},
            )
        )

        # ERROR messages
        self.add_template(
            MessageTemplate(
                message_type=MessageType.ERROR,
                required_fields={"error_code", "description"},
                optional_fields={"details", "suggestions"},
            )
        )

    def add_template(self, template: MessageTemplate):
        """Add a message template to the message space."""
        self.message_templates[template.message_type] = template

    def validate_message(
        self, message_type: MessageType, content: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Validate a message against its template."""
        if message_type not in self.message_templates:
            return False, f"Unknown message type: {message_type}"

        template = self.message_templates[message_type]
        return template.validate_message_content(content)

    def get_template(
        self, message_type: MessageType
    ) -> Optional[MessageTemplate]:
        """Get message template for a given type."""
        return self.message_templates.get(message_type)

    def get_all_message_types(self) -> Set[MessageType]:
        """Get all supported message types."""
        return set(self.message_templates.keys())


class TopologyBuilder:
    """
    Builder for creating communication topologies (Comm ⊆ A × A).
    Provides methods to generate different network patterns.
    """

    @staticmethod
    def create_fully_connected(agent_ids: List[str]) -> CommunicationTopology:
        """Create fully connected topology: Comm = A × A (minus self-loops)."""
        topology = CommunicationTopology()
        topology.create_fully_connected(agent_ids)
        return topology

    @staticmethod
    def create_star(
        agent_ids: List[str], center_agent: str = None
    ) -> CommunicationTopology:
        """Create star topology with one central agent."""
        if not agent_ids:
            return CommunicationTopology()

        topology = CommunicationTopology()

        if center_agent is None:
            center_agent = agent_ids[0]
        elif center_agent not in agent_ids:
            raise ValueError(f"Center agent {center_agent} not in agent list")

        # Center can communicate with all others, others can communicate
        # with center
        for agent_id in agent_ids:
            if agent_id != center_agent:
                topology.add_link(center_agent, agent_id)
                topology.add_link(agent_id, center_agent)

        return topology

    @staticmethod
    def create_ring(
        agent_ids: List[str], bidirectional: bool = True
    ) -> CommunicationTopology:
        """Create ring topology."""
        if len(agent_ids) < 2:
            return CommunicationTopology()

        topology = CommunicationTopology()

        for i in range(len(agent_ids)):
            next_i = (i + 1) % len(agent_ids)
            topology.add_link(agent_ids[i], agent_ids[next_i])

            if bidirectional:
                topology.add_link(agent_ids[next_i], agent_ids[i])

        return topology

    @staticmethod
    def create_chain(
        agent_ids: List[str], bidirectional: bool = True
    ) -> CommunicationTopology:
        """Create chain topology."""
        if len(agent_ids) < 2:
            return CommunicationTopology()

        topology = CommunicationTopology()

        for i in range(len(agent_ids) - 1):
            topology.add_link(agent_ids[i], agent_ids[i + 1])

            if bidirectional:
                topology.add_link(agent_ids[i + 1], agent_ids[i])

        return topology

    @staticmethod
    def create_hierarchical(
        agent_ids: List[str], levels: int = 3
    ) -> CommunicationTopology:
        """Create hierarchical topology with multiple levels."""
        if not agent_ids or levels < 1:
            return CommunicationTopology()

        topology = CommunicationTopology()
        agents_per_level = max(1, len(agent_ids) // levels)

        for level in range(levels - 1):
            start_idx = level * agents_per_level
            end_idx = min((level + 1) * agents_per_level, len(agent_ids))

            # Agents in current level communicate with agents in next level
            for i in range(start_idx, end_idx):
                next_level_start = end_idx
                next_level_end = min(
                    next_level_start + agents_per_level, len(agent_ids)
                )

                for j in range(next_level_start, next_level_end):
                    if j < len(agent_ids):
                        topology.add_link(agent_ids[i], agent_ids[j])
                        topology.add_link(agent_ids[j], agent_ids[i])

        return topology

    @staticmethod
    def create_small_world(
        agent_ids: List[str], k: int = 4, p: float = 0.3
    ) -> CommunicationTopology:
        """Create small-world topology using Watts-Strogatz model."""
        if len(agent_ids) < k + 1:
            return TopologyBuilder.create_fully_connected(agent_ids)

        # Use NetworkX to generate small-world graph
        G = nx.watts_strogatz_graph(len(agent_ids), k, p)

        topology = CommunicationTopology()
        for edge in G.edges():
            i, j = edge
            topology.add_link(agent_ids[i], agent_ids[j])
            topology.add_link(agent_ids[j], agent_ids[i])  # Bidirectional

        return topology

    @staticmethod
    def create_scale_free(
        agent_ids: List[str], m: int = 2
    ) -> CommunicationTopology:
        """Create scale-free topology using Barabási-Albert model."""
        if len(agent_ids) < m + 1:
            return TopologyBuilder.create_fully_connected(agent_ids)

        # Use NetworkX to generate scale-free graph
        G = nx.barabasi_albert_graph(len(agent_ids), m)

        topology = CommunicationTopology()
        for edge in G.edges():
            i, j = edge
            topology.add_link(agent_ids[i], agent_ids[j])
            topology.add_link(agent_ids[j], agent_ids[i])  # Bidirectional

        return topology


class CommunicationConfiguration:
    """
    Complete communication configuration combining message space and topology.
    Provides high-level interface for setting up communication systems.
    """

    def __init__(self):
        self.message_space = MessageSpace()
        self.topology: Optional[CommunicationTopology] = None
        self.agent_ids: List[str] = []

    def set_agents(self, agent_ids: List[str]):
        """Set the list of agents in the system."""
        self.agent_ids = agent_ids.copy()

    def set_topology(
        self, pattern: TopologyPattern, **kwargs
    ) -> CommunicationTopology:
        """Set communication topology using predefined pattern."""
        if not self.agent_ids:
            raise ValueError("Agent IDs must be set before topology")

        if pattern == TopologyPattern.FULLY_CONNECTED:
            self.topology = TopologyBuilder.create_fully_connected(
                self.agent_ids
            )
        elif pattern == TopologyPattern.STAR:
            center = kwargs.get("center_agent")
            self.topology = TopologyBuilder.create_star(self.agent_ids, center)
        elif pattern == TopologyPattern.RING:
            bidirectional = kwargs.get("bidirectional", True)
            self.topology = TopologyBuilder.create_ring(
                self.agent_ids, bidirectional
            )
        elif pattern == TopologyPattern.CHAIN:
            bidirectional = kwargs.get("bidirectional", True)
            self.topology = TopologyBuilder.create_chain(
                self.agent_ids, bidirectional
            )
        elif pattern == TopologyPattern.HIERARCHICAL:
            levels = kwargs.get("levels", 3)
            self.topology = TopologyBuilder.create_hierarchical(
                self.agent_ids, levels
            )
        elif pattern == TopologyPattern.SMALL_WORLD:
            k = kwargs.get("k", 4)
            p = kwargs.get("p", 0.3)
            self.topology = TopologyBuilder.create_small_world(
                self.agent_ids, k, p
            )
        elif pattern == TopologyPattern.SCALE_FREE:
            m = kwargs.get("m", 2)
            self.topology = TopologyBuilder.create_scale_free(
                self.agent_ids, m
            )
        else:
            raise ValueError(f"Unsupported topology pattern: {pattern}")

        return self.topology

    def add_custom_message_template(self, template: MessageTemplate):
        """Add custom message template to message space."""
        self.message_space.add_template(template)

    def get_topology_stats(self) -> Dict[str, Any]:
        """Get statistics about the current topology."""
        if not self.topology:
            return {"error": "No topology set"}

        total_links = len(self.topology.links)
        max_possible_links = len(self.agent_ids) * (len(self.agent_ids) - 1)
        density = (
            total_links / max_possible_links if max_possible_links > 0 else 0
        )

        # Calculate agent connectivity
        connectivity = {}
        for agent_id in self.agent_ids:
            outgoing = len(self.topology.get_reachable_agents(agent_id))
            incoming = len(
                [link for link in self.topology.links if link[1] == agent_id]
            )
            connectivity[agent_id] = {
                "outgoing": outgoing,
                "incoming": incoming,
            }

        return {
            "total_agents": len(self.agent_ids),
            "total_links": total_links,
            "max_possible_links": max_possible_links,
            "density": density,
            "agent_connectivity": connectivity,
        }

    def export_config(self) -> Dict[str, Any]:
        """Export configuration to dictionary."""
        return {
            "agent_ids": self.agent_ids,
            "topology_links": (
                list(self.topology.links) if self.topology else []
            ),
            "message_types": [
                mt.value for mt in self.message_space.get_all_message_types()
            ],
            "stats": self.get_topology_stats(),
        }

    def save_config(self, filename: str):
        """Save configuration to JSON file."""
        config = self.export_config()
        with open(filename, "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load_config(cls, filename: str) -> "CommunicationConfiguration":
        """Load configuration from JSON file."""
        with open(filename, "r") as f:
            config = json.load(f)

        comm_config = cls()
        comm_config.set_agents(config["agent_ids"])

        # Recreate topology from links
        if config["topology_links"]:
            comm_config.topology = CommunicationTopology()
            for sender, receiver in config["topology_links"]:
                comm_config.topology.add_link(sender, receiver)

        return comm_config
