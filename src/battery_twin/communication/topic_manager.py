"""
MQTT Topic Manager

Manages MQTT topic templates, formatting, and routing for battery twin.
Loads topic definitions from mqtt_topics.yaml and provides helper methods
for topic formatting and parsing.
"""

import re
import yaml
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class TopicManager:
    """
    Manages MQTT topic templates and routing.

    Loads topic definitions from mqtt_topics.yaml and provides methods for:
    - Formatting topics with variables (e.g., {battery_id}, {agent_id})
    - Parsing topics to extract variables
    - Generating subscription wildcards
    - Validating topic patterns
    """

    def __init__(self, config_path: str = "src/battery_twin/config/mqtt_topics.yaml"):
        """
        Initialize topic manager.

        Args:
            config_path: Path to mqtt_topics.yaml configuration file
        """
        self.config_path = config_path
        self.topics: Dict[str, str] = {}
        self.topic_patterns: Dict[str, re.Pattern] = {}
        self._load_topics()

    def _load_topics(self):
        """Load topic definitions from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            if 'topics' not in config:
                raise ValueError(f"No 'topics' key found in {self.config_path}")

            self.topics = config['topics']

            # Compile regex patterns for each topic template
            for topic_name, topic_template in self.topics.items():
                pattern = self._template_to_regex(topic_template)
                self.topic_patterns[topic_name] = re.compile(pattern)

        except FileNotFoundError:
            raise FileNotFoundError(f"MQTT topics config not found: {self.config_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load MQTT topics: {e}")

    def _template_to_regex(self, template: str) -> str:
        """
        Convert topic template to regex pattern.

        Args:
            template: Topic template (e.g., "battery/{battery_id}/telemetry")

        Returns:
            Regex pattern string
        """
        # Escape special regex characters except {}
        pattern = re.escape(template)

        # Replace escaped placeholders with named groups
        pattern = pattern.replace(r'\{battery_id\}', r'(?P<battery_id>[^/]+)')
        pattern = pattern.replace(r'\{agent_id\}', r'(?P<agent_id>[^/]+)')

        return f"^{pattern}$"

    def get_topic(self, topic_name: str, **kwargs) -> str:
        """
        Get formatted topic by name with variable substitution.

        Args:
            topic_name: Name of the topic (e.g., "raw_telemetry")
            **kwargs: Variables to substitute (e.g., battery_id="B0005")

        Returns:
            Formatted topic string

        Raises:
            KeyError: If topic_name doesn't exist
            ValueError: If required variables are missing

        Example:
            >>> tm = TopicManager()
            >>> tm.get_topic("raw_telemetry", battery_id="B0005")
            'battery/B0005/raw'
        """
        if topic_name not in self.topics:
            raise KeyError(f"Topic '{topic_name}' not found in configuration")

        template = self.topics[topic_name]

        # Find all placeholders in template
        placeholders = re.findall(r'\{(\w+)\}', template)

        # Check if all required variables are provided
        missing = [p for p in placeholders if p not in kwargs]
        if missing:
            raise ValueError(f"Missing required variables for topic '{topic_name}': {missing}")

        # Format the topic
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Invalid variable in topic template: {e}")

    def parse_topic(self, topic: str) -> Optional[Tuple[str, Dict[str, str]]]:
        """
        Parse a topic string to extract variables.

        Args:
            topic: MQTT topic string (e.g., "battery/B0005/telemetry/clean")

        Returns:
            Tuple of (topic_name, variables_dict) if match found, None otherwise

        Example:
            >>> tm = TopicManager()
            >>> tm.parse_topic("battery/B0005/telemetry/clean")
            ('clean_telemetry', {'battery_id': 'B0005'})
        """
        for topic_name, pattern in self.topic_patterns.items():
            match = pattern.match(topic)
            if match:
                variables = match.groupdict()
                return topic_name, variables

        return None

    def get_subscription_pattern(self, topic_name: str, **kwargs) -> str:
        """
        Get MQTT subscription pattern with wildcards.

        Args:
            topic_name: Name of the topic
            **kwargs: Variables to substitute. Use None to create wildcard.

        Returns:
            MQTT subscription pattern with wildcards

        Example:
            >>> tm = TopicManager()
            >>> tm.get_subscription_pattern("raw_telemetry", battery_id=None)
            'battery/+/raw'
            >>> tm.get_subscription_pattern("raw_telemetry", battery_id="B0005")
            'battery/B0005/raw'
        """
        if topic_name not in self.topics:
            raise KeyError(f"Topic '{topic_name}' not found in configuration")

        template = self.topics[topic_name]

        # Find all placeholders
        placeholders = re.findall(r'\{(\w+)\}', template)

        # Build substitution dict with wildcards for None values
        substitutions = {}
        for placeholder in placeholders:
            if placeholder in kwargs:
                value = kwargs[placeholder]
                substitutions[placeholder] = '+' if value is None else value
            else:
                # Default to wildcard if not provided
                substitutions[placeholder] = '+'

        return template.format(**substitutions)

    def get_all_topics(self) -> Dict[str, str]:
        """
        Get all topic templates.

        Returns:
            Dictionary of topic_name -> topic_template
        """
        return self.topics.copy()

    def list_topics(self) -> List[str]:
        """
        Get list of all topic names.

        Returns:
            List of topic names
        """
        return list(self.topics.keys())

    def get_topics_by_prefix(self, prefix: str) -> Dict[str, str]:
        """
        Get all topics that start with a given prefix.

        Args:
            prefix: Topic prefix (e.g., "battery/")

        Returns:
            Dictionary of matching topic_name -> topic_template
        """
        return {
            name: template
            for name, template in self.topics.items()
            if template.startswith(prefix)
        }

    def get_battery_topics(self) -> Dict[str, str]:
        """Get all battery-related topics."""
        return self.get_topics_by_prefix("battery/")

    def get_agent_topics(self) -> Dict[str, str]:
        """Get all agent-related topics."""
        return self.get_topics_by_prefix("agent/")

    def validate_topic(self, topic: str) -> bool:
        """
        Validate if a topic matches any known pattern.

        Args:
            topic: MQTT topic string

        Returns:
            True if topic is valid, False otherwise
        """
        return self.parse_topic(topic) is not None

    def get_topic_variables(self, topic_name: str) -> List[str]:
        """
        Get list of variables required by a topic.

        Args:
            topic_name: Name of the topic

        Returns:
            List of variable names (e.g., ['battery_id'])

        Example:
            >>> tm = TopicManager()
            >>> tm.get_topic_variables("raw_telemetry")
            ['battery_id']
        """
        if topic_name not in self.topics:
            raise KeyError(f"Topic '{topic_name}' not found in configuration")

        template = self.topics[topic_name]
        return re.findall(r'\{(\w+)\}', template)


# Convenience functions for common operations

def format_battery_topic(topic_name: str, battery_id: str,
                         config_path: str = "src/battery_twin/config/mqtt_topics.yaml") -> str:
    """
    Quick helper to format a battery topic.

    Args:
        topic_name: Topic name (e.g., "raw_telemetry")
        battery_id: Battery ID (e.g., "B0005")
        config_path: Path to MQTT topics config

    Returns:
        Formatted topic string
    """
    tm = TopicManager(config_path)
    return tm.get_topic(topic_name, battery_id=battery_id)


def format_agent_topic(topic_name: str, agent_id: str,
                      config_path: str = "src/battery_twin/config/mqtt_topics.yaml") -> str:
    """
    Quick helper to format an agent topic.

    Args:
        topic_name: Topic name (e.g., "agent_register")
        agent_id: Agent ID (e.g., "telemetry.ingestor.1")
        config_path: Path to MQTT topics config

    Returns:
        Formatted topic string
    """
    tm = TopicManager(config_path)
    return tm.get_topic(topic_name, agent_id=agent_id)


def subscribe_to_all_batteries(topic_name: str,
                               config_path: str = "src/battery_twin/config/mqtt_topics.yaml") -> str:
    """
    Get subscription pattern for all batteries.

    Args:
        topic_name: Topic name (e.g., "raw_telemetry")
        config_path: Path to MQTT topics config

    Returns:
        MQTT subscription pattern with wildcard (e.g., "battery/+/raw")
    """
    tm = TopicManager(config_path)
    return tm.get_subscription_pattern(topic_name, battery_id=None)


def subscribe_to_all_agents(topic_name: str,
                            config_path: str = "src/battery_twin/config/mqtt_topics.yaml") -> str:
    """
    Get subscription pattern for all agents.

    Args:
        topic_name: Topic name (e.g., "agent_heartbeat")
        config_path: Path to MQTT topics config

    Returns:
        MQTT subscription pattern with wildcard (e.g., "agent/+/heartbeat")
    """
    tm = TopicManager(config_path)
    return tm.get_subscription_pattern(topic_name, agent_id=None)


__all__ = [
    'TopicManager',
    'format_battery_topic',
    'format_agent_topic',
    'subscribe_to_all_batteries',
    'subscribe_to_all_agents',
]
