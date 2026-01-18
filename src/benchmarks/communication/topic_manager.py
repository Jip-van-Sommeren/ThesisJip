"""
Generic Topic Manager for Digital Twins

Manages MQTT topic templates, formatting, and routing for any digital twin.
Loads topic definitions from YAML and provides helper methods for topic
formatting, parsing, and subscription pattern generation.

This is a twin-agnostic base that can be used across different digital twin
implementations.
"""

import re
import yaml
from typing import Dict, List, Optional, Tuple


class TopicManager:
    """
    Generic topic manager for digital twins.

    Loads topic definitions from YAML configuration and provides methods for:
    - Formatting topics with variables (auto-detects placeholders like {entity_id})
    - Parsing topics to extract variables
    - Generating subscription wildcards
    - Validating topic patterns

    This base class is twin-agnostic and can be extended for specific twin types.
    """

    def __init__(self, config_path: str):
        """
        Initialize topic manager.

        Args:
            config_path: Path to YAML configuration file containing topic templates
        """
        self.config_path = config_path
        self.topics: Dict[str, str] = {}
        self.topic_patterns: Dict[str, re.Pattern] = {}
        self._load_topics()

    def _load_topics(self):
        """Load topic definitions from YAML file."""
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)

            if "topics" not in config:
                raise ValueError(
                    f"No 'topics' key found in {self.config_path}"
                )

            self.topics = config["topics"]

            # Compile regex patterns for each topic template
            for topic_name, topic_template in self.topics.items():
                pattern = self._template_to_regex(topic_template)
                self.topic_patterns[topic_name] = re.compile(pattern)

        except FileNotFoundError:
            raise FileNotFoundError(
                f"MQTT topics config not found: {self.config_path}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load MQTT topics: {e}")

    def _template_to_regex(self, template: str) -> str:
        """
        Convert topic template to regex pattern with auto-detected placeholders.

        Automatically detects any {placeholder} in the template and converts
        it to a named capture group.

        Args:
            template: Topic template (e.g., "battery/{battery_id}/telemetry")

        Returns:
            Regex pattern string with named groups

        Example:
            "battery/{battery_id}/data" -> "^battery/(?P<battery_id>[^/]+)/data$"
            "system/{entity_type}/{entity_id}" ->
                "^system/(?P<entity_type>[^/]+)/(?P<entity_id>[^/]+)$"
        """
        # Escape special regex characters except {}
        pattern = re.escape(template)

        # Find all escaped placeholders like \{placeholder_name\}
        # and replace them with named regex groups
        def replace_placeholder(match):
            placeholder_name = match.group(1)
            return f"(?P<{placeholder_name}>[^/]+)"

        # Replace \{word\} with (?P<word>[^/]+)
        pattern = re.sub(r"\\{(\w+)\\}", replace_placeholder, pattern)

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
            >>> tm = TopicManager("config/topics.yaml")
            >>> tm.get_topic("raw_telemetry", battery_id="B0005")
            'battery/B0005/raw'
        """
        if topic_name not in self.topics:
            raise KeyError(f"Topic '{topic_name}' not found in configuration")

        template = self.topics[topic_name]

        # Find all placeholders in template
        placeholders = re.findall(r"\{(\w+)\}", template)

        # Check if all required variables are provided
        missing = [p for p in placeholders if p not in kwargs]
        if missing:
            raise ValueError(
                f"Missing required variables for topic '{topic_name}': {missing}"
            )

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
            >>> tm = TopicManager("config/topics.yaml")
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
            >>> tm = TopicManager("config/topics.yaml")
            >>> tm.get_subscription_pattern("raw_telemetry", battery_id=None)
            'battery/+/raw'
            >>> tm.get_subscription_pattern("raw_telemetry", battery_id="B0005")
            'battery/B0005/raw'
        """
        if topic_name not in self.topics:
            raise KeyError(f"Topic '{topic_name}' not found in configuration")

        template = self.topics[topic_name]

        # Find all placeholders
        placeholders = re.findall(r"\{(\w+)\}", template)

        # Build substitution dict with wildcards for None values
        substitutions = {}
        for placeholder in placeholders:
            if placeholder in kwargs:
                value = kwargs[placeholder]
                substitutions[placeholder] = "+" if value is None else value
            else:
                # Default to wildcard if not provided
                substitutions[placeholder] = "+"

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
            prefix: Topic prefix (e.g., "battery/", "system/")

        Returns:
            Dictionary of matching topic_name -> topic_template
        """
        return {
            name: template
            for name, template in self.topics.items()
            if template.startswith(prefix)
        }

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
            >>> tm = TopicManager("config/topics.yaml")
            >>> tm.get_topic_variables("raw_telemetry")
            ['battery_id']
        """
        if topic_name not in self.topics:
            raise KeyError(f"Topic '{topic_name}' not found in configuration")

        template = self.topics[topic_name]
        return re.findall(r"\{(\w+)\}", template)


# Generalized convenience functions


def format_entity_topic(
    topic_name: str,
    entity_type: str,
    entity_id: str,
    config_path: str,
) -> str:
    """
    Quick helper to format an entity topic.

    This is a generalized version that works for any entity type
    (battery, agent, device, etc.).

    Args:
        topic_name: Topic name (e.g., "raw_telemetry")
        entity_type: Type of entity (e.g., "battery_id", "agent_id", "device_id")
        entity_id: Entity identifier (e.g., "B0005", "agent.1")
        config_path: Path to MQTT topics config

    Returns:
        Formatted topic string

    Example:
        >>> format_entity_topic("raw_telemetry", "battery_id", "B0005", "config.yaml")
        'battery/B0005/raw'
    """
    tm = TopicManager(config_path)
    return tm.get_topic(topic_name, **{entity_type: entity_id})


def subscribe_to_entity_pattern(
    topic_name: str,
    entity_type: str,
    config_path: str,
) -> str:
    """
    Get subscription pattern for all entities of a given type.

    This is a generalized version that works for any entity type.

    Args:
        topic_name: Topic name (e.g., "raw_telemetry")
        entity_type: Type of entity variable (e.g., "battery_id", "agent_id")
        config_path: Path to MQTT topics config

    Returns:
        MQTT subscription pattern with wildcard (e.g., "battery/+/raw")

    Example:
        >>> subscribe_to_entity_pattern("raw_telemetry", "battery_id", "config.yaml")
        'battery/+/raw'
    """
    tm = TopicManager(config_path)
    return tm.get_subscription_pattern(topic_name, **{entity_type: None})


__all__ = [
    "TopicManager",
    "format_entity_topic",
    "subscribe_to_entity_pattern",
]
