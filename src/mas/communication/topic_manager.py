"""
Topic Manager for Multi-Agent Systems

Manages MQTT topic templates, formatting, and routing for any digital twin.
Loads topic definitions from YAML and provides helper methods for topic
formatting, parsing, and subscription pattern generation.

This is a twin-agnostic component that can be used across different
digital twin implementations.
"""

import re
import yaml
from typing import Dict, List, Optional, Tuple


class TopicManager:
    """
    Topic manager for MQTT-based agent communication.

    Loads topic definitions from YAML configuration and provides methods for:
    - Formatting topics with variables (auto-detects placeholders like {entity_id})
    - Parsing topics to extract variables
    - Generating subscription wildcards
    - Validating topic patterns

    Example YAML configuration:
        topics:
          raw_telemetry: "battery/{battery_id}/raw"
          clean_telemetry: "battery/{battery_id}/telemetry/clean"
          agent_status: "agent/{agent_id}/status"
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
                raise ValueError(f"No 'topics' key found in {self.config_path}")

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

        Args:
            template: Topic template (e.g., "battery/{battery_id}/telemetry")

        Returns:
            Regex pattern string with named groups

        Example:
            "battery/{battery_id}/data" -> "^battery/(?P<battery_id>[^/]+)/data$"
        """
        # Escape special regex characters except {}
        pattern = re.escape(template)

        # Replace escaped placeholders with named regex groups
        def replace_placeholder(match):
            placeholder_name = match.group(1)
            return f"(?P<{placeholder_name}>[^/]+)"

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
        placeholders = re.findall(r"\{(\w+)\}", template)

        missing = [p for p in placeholders if p not in kwargs]
        if missing:
            raise ValueError(
                f"Missing required variables for topic '{topic_name}': {missing}"
            )

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
            >>> tm.parse_topic("battery/B0005/telemetry/clean")
            ('clean_telemetry', {'battery_id': 'B0005'})
        """
        for topic_name, pattern in self.topic_patterns.items():
            match = pattern.match(topic)
            if match:
                return topic_name, match.groupdict()
        return None

    def get_subscription_pattern(self, topic_name: str, **kwargs) -> str:
        """
        Get MQTT subscription pattern with wildcards.

        Args:
            topic_name: Name of the topic
            **kwargs: Variables to substitute. Use None for wildcard (+).

        Returns:
            MQTT subscription pattern with wildcards

        Example:
            >>> tm.get_subscription_pattern("raw_telemetry", battery_id=None)
            'battery/+/raw'
            >>> tm.get_subscription_pattern("raw_telemetry", battery_id="B0005")
            'battery/B0005/raw'
        """
        if topic_name not in self.topics:
            raise KeyError(f"Topic '{topic_name}' not found in configuration")

        template = self.topics[topic_name]
        placeholders = re.findall(r"\{(\w+)\}", template)

        substitutions = {}
        for placeholder in placeholders:
            if placeholder in kwargs:
                value = kwargs[placeholder]
                substitutions[placeholder] = "+" if value is None else value
            else:
                substitutions[placeholder] = "+"

        return template.format(**substitutions)

    def get_all_topics(self) -> Dict[str, str]:
        """Get all topic templates."""
        return self.topics.copy()

    def list_topics(self) -> List[str]:
        """Get list of all topic names."""
        return list(self.topics.keys())

    def get_topics_by_prefix(self, prefix: str) -> Dict[str, str]:
        """Get all topics that start with a given prefix."""
        return {
            name: template
            for name, template in self.topics.items()
            if template.startswith(prefix)
        }

    def validate_topic(self, topic: str) -> bool:
        """Validate if a topic matches any known pattern."""
        return self.parse_topic(topic) is not None

    def get_topic_variables(self, topic_name: str) -> List[str]:
        """
        Get list of variables required by a topic.

        Args:
            topic_name: Name of the topic

        Returns:
            List of variable names (e.g., ['battery_id'])
        """
        if topic_name not in self.topics:
            raise KeyError(f"Topic '{topic_name}' not found in configuration")

        template = self.topics[topic_name]
        return re.findall(r"\{(\w+)\}", template)


__all__ = ["TopicManager"]
