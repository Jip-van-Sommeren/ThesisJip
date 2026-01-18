"""
Base Message Schema

Foundation message types for agent communication.
Domain-specific messages should extend these base types.
"""

import time
import uuid
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class AgentMessage(BaseModel):
    """
    Base message type for all agent communication.

    Provides common fields for message identification and tracking.
    Domain-specific messages should extend this class.

    Example:
        class TelemetryMessage(AgentMessage):
            battery_id: str
            voltage: float
            current: float
    """

    message_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Unique message identifier",
    )
    sender_id: Optional[str] = Field(
        default=None,
        description="ID of the sending agent",
    )
    timestamp: float = Field(
        default_factory=time.time,
        description="Unix timestamp when message was created",
    )

    model_config = {
        "extra": "allow",  # Allow domain-specific fields
    }


class AgentStatusMessage(AgentMessage):
    """
    Agent status update message.

    Published periodically by agents to report their operational status.
    """

    agent_id: str = Field(..., description="Agent identifier")
    status: str = Field(
        ...,
        description="Agent status: active, idle, busy, degraded, failed",
    )
    current_task: Optional[str] = Field(
        default=None,
        description="Current task being executed",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if status is failed",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional status metadata",
    )


class AgentHeartbeatMessage(AgentMessage):
    """
    Agent heartbeat for liveness monitoring.

    Published periodically by agents to indicate they are alive.
    """

    agent_id: str = Field(..., description="Agent identifier")
    uptime: float = Field(
        ...,
        ge=0,
        description="Agent uptime in seconds",
    )
    status: str = Field(
        default="active",
        description="Current status",
    )


class CommandMessage(AgentMessage):
    """
    Command message for agent control.

    Used to send commands from orchestrators/supervisors to agents.
    """

    command_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Unique command identifier",
    )
    target_agent: str = Field(
        ...,
        description="Target agent ID or 'all' for broadcast",
    )
    command_type: str = Field(
        ...,
        description="Command type: start, stop, configure, etc.",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Command parameters",
    )
    priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Command priority (1=low, 10=high)",
    )
    requires_ack: bool = Field(
        default=True,
        description="Whether acknowledgment is required",
    )


class ResponseMessage(AgentMessage):
    """
    Response to a command message.

    Sent by agents in response to CommandMessage.
    """

    command_id: str = Field(
        ...,
        description="ID of the command being responded to",
    )
    success: bool = Field(
        ...,
        description="Whether command executed successfully",
    )
    result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Command result data",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if success is False",
    )


__all__ = [
    "AgentMessage",
    "AgentStatusMessage",
    "AgentHeartbeatMessage",
    "CommandMessage",
    "ResponseMessage",
]
