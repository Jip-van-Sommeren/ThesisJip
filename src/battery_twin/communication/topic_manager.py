"""
Battery Twin Topic Manager

Battery-specific extension of the generic TopicManager that provides
convenience methods and defaults for battery digital twin applications.
"""

from typing import Dict
from communication.topic_manager import (
    TopicManager,
    format_entity_topic as _format_entity_topic,
    subscribe_to_entity_pattern as _subscribe_to_entity_pattern,
)


class BatteryTopicManager(TopicManager):
    """
    Battery-specific topic manager.

    Extends the generic TopicManager with battery twin defaults and
    convenience methods for battery and agent topics.
    """

    def __init__(
        self, config_path: str = "src/battery_twin/config/mqtt_topics.yaml"
    ):
        """
        Initialize battery topic manager.

        Args:
            config_path: Path to mqtt_topics.yaml configuration file.
                        Defaults to battery twin config location.
        """
        super().__init__(config_path)

    def get_battery_topics(self) -> Dict[str, str]:
        """
        Get all battery-related topics.

        Returns:
            Dictionary of battery topic_name -> topic_template
        """
        return self.get_topics_by_prefix("battery/")

    def get_agent_topics(self) -> Dict[str, str]:
        """
        Get all agent-related topics.

        Returns:
            Dictionary of agent topic_name -> topic_template
        """
        return self.get_topics_by_prefix("system/agents/")


# Battery-specific convenience functions for backward compatibility


def format_battery_topic(
    topic_name: str,
    battery_id: str,
    config_path: str = "src/battery_twin/config/mqtt_topics.yaml",
) -> str:
    """
    Quick helper to format a battery topic.

    Args:
        topic_name: Topic name (e.g., "raw_telemetry")
        battery_id: Battery ID (e.g., "B0005")
        config_path: Path to MQTT topics config

    Returns:
        Formatted topic string

    Example:
        >>> format_battery_topic("raw_telemetry", "B0005")
        'battery/B0005/raw'
    """
    return _format_entity_topic(topic_name, "battery_id", battery_id, config_path)


def format_agent_topic(
    topic_name: str,
    agent_id: str,
    config_path: str = "src/battery_twin/config/mqtt_topics.yaml",
) -> str:
    """
    Quick helper to format an agent topic.

    Args:
        topic_name: Topic name (e.g., "agent_heartbeat")
        agent_id: Agent ID (e.g., "telemetry.ingestor.1")
        config_path: Path to MQTT topics config

    Returns:
        Formatted topic string

    Example:
        >>> format_agent_topic("agent_heartbeat", "telemetry.ingestor.1")
        'system/agents/telemetry.ingestor.1/heartbeat'
    """
    return _format_entity_topic(topic_name, "agent_id", agent_id, config_path)


def subscribe_to_all_batteries(
    topic_name: str,
    config_path: str = "src/battery_twin/config/mqtt_topics.yaml",
) -> str:
    """
    Get subscription pattern for all batteries.

    Args:
        topic_name: Topic name (e.g., "raw_telemetry")
        config_path: Path to MQTT topics config

    Returns:
        MQTT subscription pattern with wildcard (e.g., "battery/+/raw")

    Example:
        >>> subscribe_to_all_batteries("raw_telemetry")
        'battery/+/raw'
    """
    return _subscribe_to_entity_pattern(topic_name, "battery_id", config_path)


def subscribe_to_all_agents(
    topic_name: str,
    config_path: str = "src/battery_twin/config/mqtt_topics.yaml",
) -> str:
    """
    Get subscription pattern for all agents.

    Args:
        topic_name: Topic name (e.g., "agent_heartbeat")
        config_path: Path to MQTT topics config

    Returns:
        MQTT subscription pattern with wildcard (e.g., "system/agents/+/heartbeat")

    Example:
        >>> subscribe_to_all_agents("agent_heartbeat")
        'system/agents/+/heartbeat'
    """
    return _subscribe_to_entity_pattern(topic_name, "agent_id", config_path)


__all__ = [
    "BatteryTopicManager",
    "format_battery_topic",
    "format_agent_topic",
    "subscribe_to_all_batteries",
    "subscribe_to_all_agents",
]
