"""
Unified configuration schema for the Battery Digital Twin.

This module defines a structured configuration model that mirrors the nested
YAML documents under ``src/battery_twin/config/``.  It combines system,
messaging, storage, data, and agent settings into a single Pydantic model and
provides helper utilities to load the schema from disk and adapt it to runtime
objects (e.g., ``BatteryTwinConfig`` for the orchestrator).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator

DEFAULT_BATTERY_ID = "B0005"


# ---------------------------------------------------------------------------
# Core configuration sections
# ---------------------------------------------------------------------------


class SystemConfig(BaseModel):
    name: str = "battery_digital_twin"
    mode: str = "batch_replay"
    log_level: str = "INFO"


class MQTTAuthConfig(BaseModel):
    username: Optional[str] = None
    password: Optional[str] = None


class MQTTConfig(BaseModel):
    broker: str = "localhost"
    port: int = 1883
    qos: int = 1
    keepalive: int = 60
    client_id_prefix: str = "battery_agent_"
    auth: MQTTAuthConfig = Field(default_factory=MQTTAuthConfig)


class InfluxDBConfig(BaseModel):
    host: str = "localhost"
    port: int = 8086
    database: str = "battery_metrics"
    measurement_prefix: str = "battery_"
    retention_policy: str = "30d"


class MongoDBCollections(BaseModel):
    models: str = "trained_models"
    predictions: str = "predictions"
    configs: str = "agent_configs"
    parameters: str = "parameter_history"
    faults: str = "fault_events"


class MongoDBConfig(BaseModel):
    host: str = "localhost"
    port: int = 27017
    database: str = "battery_twin"
    collections: MongoDBCollections = Field(default_factory=MongoDBCollections)


class Neo4jConfig(BaseModel):
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password123"
    database: str = "battery_hierarchy"


class RedisConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379
    db: int = 1
    state_ttl: int = 3600


class StorageConfig(BaseModel):
    enable_time_series: bool = True
    enable_document_store: bool = True
    enable_graph_store: bool = True
    enable_cache: bool = True

    influxdb: InfluxDBConfig = Field(default_factory=InfluxDBConfig)
    mongodb: MongoDBConfig = Field(default_factory=MongoDBConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)

    def any_persistence_enabled(self) -> bool:
        return (
            self.enable_time_series
            or self.enable_document_store
            or self.enable_graph_store
            or self.enable_cache
        )


class DataConfig(BaseModel):
    source: str = "Digital-Twin-in-python/data/raw/discharge.csv"
    replay_speed: float = 1.0
    batch_size: int = 100
    batteries: List[str] = Field(default_factory=lambda: [DEFAULT_BATTERY_ID])

    @field_validator("batteries", mode="before")
    @classmethod
    def _ensure_list(cls, value: Union[str, List[str]]) -> List[str]:
        if value is None:
            return [DEFAULT_BATTERY_ID]
        if isinstance(value, list):
            return value or [DEFAULT_BATTERY_ID]
        return [str(value)]

    def primary_battery_id(self) -> str:
        return self.batteries[0] if self.batteries else DEFAULT_BATTERY_ID


# ---------------------------------------------------------------------------
# Agent configuration sections
# ---------------------------------------------------------------------------


class AgentToggle(BaseModel):
    enabled: bool = True


class ValidationRulesConfig(BaseModel):
    voltage_range: List[float] = Field(default_factory=lambda: [2.5, 4.5])
    current_range: List[float] = Field(default_factory=lambda: [-5.0, 5.0])
    temperature_range: List[float] = Field(default_factory=lambda: [-10.0, 60.0])

    @field_validator("voltage_range", "current_range", "temperature_range")
    @classmethod
    def _validate_range(cls, value: List[float]) -> List[float]:
        if len(value) != 2:
            raise ValueError("Range fields must contain exactly 2 elements")
        return value


class TelemetryIngestorConfig(AgentToggle):
    validation_rules: ValidationRulesConfig = Field(
        default_factory=ValidationRulesConfig
    )
    resampling_rate: float = 1.0
    outlier_threshold: float = 3.0


class PhysicsModelConfig(AgentToggle):
    model_type: str = "exponential_degradation"
    degradation_coefficient: float = 0.13
    prediction_horizon: int = 60
    temperature_ref: float = 25.0


class MLResidualConfig(AgentToggle):
    architecture: List[int] = Field(default_factory=lambda: [64, 64])
    dropout_rate: float = 0.1
    learning_rate: float = 0.001
    uncertainty_samples: int = 100
    retrain_threshold: float = 0.1
    feature_set: List[str] = Field(
        default_factory=lambda: ["physics_pred", "temperature", "cycle", "time"]
    )


class StateEstimatorConfig(AgentToggle):
    filter_type: str = "kalman"
    state_variables: List[str] = Field(
        default_factory=lambda: ["SoC", "SoH", "R0", "R1", "C1"]
    )
    process_noise: float = 0.01
    measurement_noise: float = 0.05
    initial_soc: float = 0.8
    initial_soh: float = 1.0
    capacity_nominal: float = 2.0


class HealthMonitorConfig(AgentToggle):
    initial_soh: float = 1.0
    initial_r0: float = 0.01
    eol_threshold: float = 0.8


class ParameterIdentificationConfig(AgentToggle):
    window_size: int = 50
    update_frequency: int = 10
    min_confidence: float = 0.8


class FaultDetectionConfig(AgentToggle):
    residual_threshold: float = 3.0
    temperature_alarm: float = 50.0
    voltage_alarm_low: float = 2.5
    voltage_alarm_high: float = 4.3
    drift_window: int = 100


class ControlAgentConfig(AgentToggle):
    mode: str = "rule_based"
    target_soc: float = 0.8
    constraints: Dict[str, float] = Field(
        default_factory=lambda: {
            "max_current": 3.0,
            "max_voltage": 4.2,
            "min_voltage": 2.5,
            "max_temperature": 45.0,
        }
    )
    control_frequency: float = 1.0


class OrchestratorAgentConfig(AgentToggle):
    heartbeat_timeout: int = 30
    health_check_interval: int = 5
    restart_attempts: int = 3


class RegistryAgentConfig(AgentToggle):
    schema_validation: bool = True
    announce_interval: int = 60


class AgentsConfig(BaseModel):
    telemetry_ingestor: TelemetryIngestorConfig = Field(
        default_factory=TelemetryIngestorConfig
    )
    state_estimator: StateEstimatorConfig = Field(
        default_factory=StateEstimatorConfig
    )
    health_monitor: HealthMonitorConfig = Field(
        default_factory=HealthMonitorConfig
    )
    physics_model: PhysicsModelConfig = Field(default_factory=PhysicsModelConfig)
    ml_residual: MLResidualConfig = Field(default_factory=MLResidualConfig)
    parameter_id: ParameterIdentificationConfig = Field(
        default_factory=ParameterIdentificationConfig
    )
    fault_detection: FaultDetectionConfig = Field(
        default_factory=FaultDetectionConfig
    )
    control: ControlAgentConfig = Field(default_factory=ControlAgentConfig)
    orchestrator: OrchestratorAgentConfig = Field(
        default_factory=OrchestratorAgentConfig
    )
    registry: RegistryAgentConfig = Field(default_factory=RegistryAgentConfig)


# ---------------------------------------------------------------------------
# Root configuration model
# ---------------------------------------------------------------------------


class TwinConfig(BaseModel):
    system: SystemConfig = Field(default_factory=SystemConfig)
    mqtt: MQTTConfig = Field(default_factory=MQTTConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)

    @classmethod
    def from_dict(cls, data: Dict) -> "TwinConfig":
        return cls(**data)

    source_path: Optional[str] = Field(default=None, exclude=True)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TwinConfig":
        resolved = Path(path).expanduser()
        with resolved.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        config = cls.from_dict(payload)
        object.__setattr__(config, "source_path", str(resolved))
        return config

    def to_battery_twin_config(self):
        """
        Build a ``BatteryTwinConfig`` dataclass suitable for the orchestrator.

        The import is performed lazily to avoid circular dependencies for
        callers that only need the configuration schema.
        """

        from src.battery_twin.orchestrator import BatteryTwinConfig

        storage_config_path = self.source_path if self.source_path else None

        return BatteryTwinConfig(
            battery_id=self.data.primary_battery_id(),
            mqtt_broker=self.mqtt.broker,
            mqtt_port=self.mqtt.port,
            mqtt_keepalive=self.mqtt.keepalive,
            enable_storage=self.storage.any_persistence_enabled(),
            enable_telemetry_ingestor=self.agents.telemetry_ingestor.enabled,
            enable_state_estimator=self.agents.state_estimator.enabled,
            enable_health_monitor=self.agents.health_monitor.enabled,
            enable_physics_model=self.agents.physics_model.enabled,
            enable_ml_residual=self.agents.ml_residual.enabled,
            ekf_initial_soc=self.agents.state_estimator.initial_soc,
            ekf_initial_soh=self.agents.state_estimator.initial_soh,
            ekf_capacity_nominal=self.agents.state_estimator.capacity_nominal,
            health_initial_soh=self.agents.health_monitor.initial_soh,
            health_initial_r0=self.agents.health_monitor.initial_r0,
            health_eol_threshold=self.agents.health_monitor.eol_threshold,
            log_level=self.system.log_level,
            log_file=None,
            enable_metrics=self.agents.orchestrator.enabled,
            metrics_interval=float(
                max(self.agents.orchestrator.health_check_interval, 1)
            ),
            storage_config_path=storage_config_path,
        )

    def to_storage_config(self) -> "BatteryStorageConfig":
        from src.battery_twin.storage.battery_storage_config import (
            BatteryStorageConfig,
        )

        if self.source_path:
            return BatteryStorageConfig.from_yaml(self.source_path)
        return BatteryStorageConfig()


# ---------------------------------------------------------------------------
# Loader helper
# ---------------------------------------------------------------------------


def load_unified_config(path: Union[str, Path]) -> TwinConfig:
    """
    Load a unified configuration document and return the parsed ``TwinConfig``.
    """

    return TwinConfig.load(path)


__all__ = [
    "TwinConfig",
    "load_unified_config",
    "SystemConfig",
    "MQTTConfig",
    "StorageConfig",
    "DataConfig",
    "AgentsConfig",
]
