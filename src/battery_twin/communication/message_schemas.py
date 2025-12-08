"""
Battery Twin Message Schemas

Pydantic models for all MQTT message types used in the battery digital twin.
These schemas ensure type safety and validation for inter-agent communication.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


# ============================================================================
# Battery Data Messages
# ============================================================================


class TelemetryMessage(BaseModel):
    """
    Raw or cleaned telemetry message from battery sensors.

    Published to:
    - battery/{battery_id}/raw (from replay engine)
    - battery/{battery_id}/telemetry/clean (from TelemetryIngestorAgent)
    """

    battery_id: str = Field(
        ..., description="Battery identifier (e.g., B0005)"
    )
    timestamp: float = Field(..., description="Unix timestamp in seconds")
    cycle: int = Field(..., ge=0, description="Cycle number")
    voltage: float = Field(..., ge=0, description="Battery voltage in V")
    current: float = Field(
        ..., description="Battery current in A (negative for discharge)"
    )
    temperature: float = Field(..., description="Cell temperature in °C")
    ambient_temperature: Optional[float] = Field(
        None, description="Ambient temperature in °C"
    )

    @field_validator("voltage")
    @classmethod
    def validate_voltage(cls, v: float) -> float:
        if not (2.0 <= v <= 5.0):
            raise ValueError(f"Voltage {v}V outside typical range [2.0, 5.0]V")
        return v

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not (-20 <= v <= 80):
            raise ValueError(
                f"Temperature {v}°C outside valid range [-20, 80]°C"
            )
        return v


class CapacityMessage(BaseModel):
    """
    Measured battery capacity (typically at end of discharge cycle).

    Published to: battery/{battery_id}/capacity
    """

    battery_id: str
    timestamp: float
    cycle: int = Field(..., ge=0)
    capacity: float = Field(..., ge=0, description="Measured capacity in Ah")
    measurement_type: str = Field(
        default="measured", description="measured, estimated, or predicted"
    )


class PredictionMessage(BaseModel):
    """
    Capacity prediction from physics, ML, or hybrid model.

    Published to:
    - battery/{battery_id}/prediction/physics
    - battery/{battery_id}/prediction/ml
    - battery/{battery_id}/prediction/hybrid
    """

    battery_id: str
    timestamp: float
    cycle: int = Field(..., ge=0)
    prediction_type: str = Field(..., description="physics, ml, or hybrid")
    predicted_capacity: float = Field(
        ..., ge=0, description="Predicted capacity in Ah"
    )
    uncertainty: Optional[float] = Field(
        None, ge=0, description="Prediction uncertainty"
    )
    horizon: int = Field(
        default=0, ge=0, description="Prediction horizon in seconds"
    )
    agent_id: str = Field(..., description="Agent that made the prediction")

    @field_validator("prediction_type")
    @classmethod
    def validate_prediction_type(cls, v: str) -> str:
        if v not in ["physics", "ml", "hybrid"]:
            raise ValueError(
                f"Invalid prediction_type: {v}. Must be physics, ml, or hybrid"
            )
        return v


class StateEstimateMessage(BaseModel):
    """
    Battery state estimate (SoC, SoH, internal resistance).

    Published to: battery/{battery_id}/state/estimate
    """

    battery_id: str
    timestamp: float
    soc: float = Field(..., ge=0, le=1, description="State of Charge (0-1)")
    soh: float = Field(..., ge=0, le=1, description="State of Health (0-1)")
    internal_resistance: Dict[str, float] = Field(
        ..., description="Internal resistance components (R0, R1, C1)"
    )
    uncertainty: Optional[Dict[str, float]] = Field(
        None, description="Uncertainty estimates for soc, soh"
    )
    agent_id: str = Field(..., description="State estimator agent ID")

    @field_validator("internal_resistance")
    @classmethod
    def validate_resistance(cls, v: Dict[str, float]) -> Dict[str, float]:
        required_keys = ["R0", "R1", "C1"]
        if not all(key in v for key in required_keys):
            raise ValueError(
                f"internal_resistance must contain keys: {required_keys}"
            )
        return v


class ParameterMessage(BaseModel):
    """
    Updated battery model parameters.

    Published to: battery/{battery_id}/parameters/update
    """

    battery_id: str
    timestamp: float
    cycle: int = Field(..., ge=0)
    parameters: Dict[str, float] = Field(
        ..., description="Model parameters (k, C0, R0, R1, C1, etc.)"
    )
    confidence: float = Field(
        ..., ge=0, le=1, description="Confidence in parameters"
    )
    fit_quality: Optional[Dict[str, float]] = Field(
        None, description="Fit quality metrics (RMSE, R2, etc.)"
    )
    agent_id: str = Field(..., description="Parameter identification agent ID")


class FaultMessage(BaseModel):
    """
    Battery fault detection event.

    Published to: battery/{battery_id}/fault/detected
    """

    battery_id: str
    timestamp: float
    severity: str = Field(
        ..., description="Severity level: warning, fault, critical"
    )
    fault_type: str = Field(..., description="Type of fault detected")
    cause: str = Field(..., description="Human-readable cause description")
    residual_magnitude: float = Field(
        ..., description="Magnitude of residual or anomaly"
    )
    agent_id: str = Field(..., description="Fault detection agent ID")

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        if v not in ["warning", "fault", "critical"]:
            raise ValueError(
                f"Invalid severity: {v}. Must be warning, fault, or critical"
            )
        return v


class ControlMessage(BaseModel):
    """
    Battery control command (charging/discharging setpoint).

    Published to: battery/{battery_id}/control/command
    """

    battery_id: str
    timestamp: float
    control_mode: str = Field(
        ..., description="Control mode: CC, CV, CCCV, idle"
    )
    current_setpoint: Optional[float] = Field(
        None, description="Current setpoint in A"
    )
    voltage_limit: Optional[float] = Field(
        None, description="Voltage limit in V"
    )
    power_limit: Optional[float] = Field(None, description="Power limit in W")
    agent_id: str = Field(..., description="Control agent ID")

    @field_validator("control_mode")
    @classmethod
    def validate_control_mode(cls, v: str) -> str:
        valid_modes = ["CC", "CV", "CCCV", "idle", "discharge", "charge"]
        if v not in valid_modes:
            raise ValueError(
                f"Invalid control_mode: {v}. Must be one of {valid_modes}"
            )
        return v


# ============================================================================
# Agent Coordination Messages
# ============================================================================


class AgentRegistrationMessage(BaseModel):
    """
    Agent registration message for RegistryAgent.

    Published to: agent/{agent_id}/register
    """

    agent_id: str
    agent_type: str = Field(
        ..., description="Agent type: BDI, Reactive, Hybrid"
    )
    capabilities: List[str] = Field(
        default_factory=list, description="Agent capabilities"
    )
    supervisor: Optional[str] = Field(None, description="Supervisor agent ID")
    roles: List[str] = Field(default_factory=list, description="Agent roles")
    groups: List[str] = Field(default_factory=list, description="Agent groups")
    timestamp: float


class AgentHeartbeatMessage(BaseModel):
    """
    Agent heartbeat for health monitoring.

    Published to: agent/{agent_id}/heartbeat
    """

    agent_id: str
    timestamp: float
    status: str = Field(
        ..., description="Status: active, idle, busy, degraded"
    )
    uptime: float = Field(..., ge=0, description="Agent uptime in seconds")

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        valid_statuses = ["active", "idle", "busy", "degraded", "failed"]
        if v not in valid_statuses:
            raise ValueError(
                f"Invalid status: {v}. Must be one of {valid_statuses}"
            )
        return v


class AgentStatusMessage(BaseModel):
    """
    Detailed agent status update.

    Published to: agent/{agent_id}/status
    """

    agent_id: str
    timestamp: float
    status: str
    current_task: Optional[str] = Field(
        None, description="Current task description"
    )
    last_action: Optional[str] = Field(
        None, description="Last action performed"
    )
    error_message: Optional[str] = Field(
        None, description="Error message if failed"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class AgentMetricsMessage(BaseModel):
    """
    Agent performance metrics.

    Published to: agent/{agent_id}/metrics
    """

    agent_id: str
    timestamp: float
    messages_processed: int = Field(default=0, ge=0)
    actions_executed: int = Field(default=0, ge=0)
    avg_processing_time: float = Field(
        default=0.0, ge=0, description="Average time in ms"
    )
    error_count: int = Field(default=0, ge=0)
    custom_metrics: Dict[str, float] = Field(
        default_factory=dict, description="Agent-specific metrics"
    )


class CoordinationCommandMessage(BaseModel):
    """
    Coordination command from OrchestratorAgent.

    Published to: agent/orchestrator/command
    """

    command_id: str
    timestamp: float
    target_agent: str = Field(..., description="Target agent ID or 'all'")
    command_type: str = Field(
        ..., description="Command type: start, stop, restart, configure"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Command parameters"
    )
    priority: int = Field(default=5, ge=1, le=10, description="Priority 1-10")

    @field_validator("command_type")
    @classmethod
    def validate_command_type(cls, v: str) -> str:
        valid_commands = [
            "start",
            "stop",
            "restart",
            "configure",
            "optimize",
            "reset",
        ]
        if v not in valid_commands:
            raise ValueError(
                f"Invalid command_type: {v}. Must be one of {valid_commands}"
            )
        return v


class AgentDirectoryMessage(BaseModel):
    """
    Agent directory update from RegistryAgent.

    Published to: agent/registry/directory
    """

    timestamp: float
    agents: List[Dict[str, Any]] = Field(
        ..., description="List of registered agents with their metadata"
    )
    total_agents: int = Field(..., ge=0)


# ============================================================================
# Message Factory and Utilities
# ============================================================================


class MessageFactory:
    """Factory for creating and parsing messages."""

    # Map message types to Pydantic models
    MESSAGE_TYPES = {
        "telemetry": TelemetryMessage,
        "capacity": CapacityMessage,
        "prediction": PredictionMessage,
        "state_estimate": StateEstimateMessage,
        "parameters": ParameterMessage,
        "fault": FaultMessage,
        "control": ControlMessage,
        "agent_registration": AgentRegistrationMessage,
        "agent_heartbeat": AgentHeartbeatMessage,
        "agent_status": AgentStatusMessage,
        "agent_metrics": AgentMetricsMessage,
        "coordination_command": CoordinationCommandMessage,
        "agent_directory": AgentDirectoryMessage,
    }

    @classmethod
    def create_message(
        cls, message_type: str, data: Dict[str, Any]
    ) -> BaseModel:
        """
        Create a message from a dictionary.

        Args:
            message_type: Type of message to create
            data: Message data as dictionary

        Returns:
            Validated Pydantic model instance

        Raises:
            ValueError: If message_type is unknown
            ValidationError: If data doesn't match schema
        """
        if message_type not in cls.MESSAGE_TYPES:
            raise ValueError(f"Unknown message type: {message_type}")

        model_class = cls.MESSAGE_TYPES[message_type]
        return model_class(**data)

    @classmethod
    def parse_message(cls, message_type: str, json_str: str) -> BaseModel:
        """
        Parse a JSON string into a message.

        Args:
            message_type: Type of message
            json_str: JSON string

        Returns:
            Validated Pydantic model instance
        """
        if message_type not in cls.MESSAGE_TYPES:
            raise ValueError(f"Unknown message type: {message_type}")

        model_class = cls.MESSAGE_TYPES[message_type]
        return model_class.model_validate_json(json_str)

    @classmethod
    def to_json(cls, message: BaseModel) -> str:
        """
        Convert a message to JSON string.

        Args:
            message: Pydantic message instance

        Returns:
            JSON string
        """
        return message.model_dump_json()

    @classmethod
    def to_dict(cls, message: BaseModel) -> Dict[str, Any]:
        """
        Convert a message to dictionary.

        Args:
            message: Pydantic message instance

        Returns:
            Dictionary representation
        """
        return message.model_dump()


# Export all message types
__all__ = [
    "TelemetryMessage",
    "CapacityMessage",
    "PredictionMessage",
    "StateEstimateMessage",
    "ParameterMessage",
    "FaultMessage",
    "ControlMessage",
    "AgentRegistrationMessage",
    "AgentHeartbeatMessage",
    "AgentStatusMessage",
    "AgentMetricsMessage",
    "CoordinationCommandMessage",
    "AgentDirectoryMessage",
    "MessageFactory",
]
