"""
Battery-Specific Storage Configuration

Extends the base storage configuration with battery twin specific
measurements, schemas, and retention policies.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from config.storage_config import StorageConfig, InfluxConfig, MongoConfig, Neo4jConfig, RedisConfig


@dataclass
class BatteryInfluxConfig(InfluxConfig):
    """InfluxDB configuration for battery metrics."""

    # Battery-specific measurements
    measurements: Dict[str, Dict] = field(default_factory=lambda: {
        "battery_voltage": {
            "tags": ["battery_id", "measurement_type"],
            "fields": ["voltage", "timestamp"],
            "retention": "30d"
        },
        "battery_current": {
            "tags": ["battery_id", "measurement_type"],
            "fields": ["current", "timestamp"],
            "retention": "30d"
        },
        "battery_temperature": {
            "tags": ["battery_id", "measurement_type"],
            "fields": ["temperature", "ambient_temperature", "timestamp"],
            "retention": "30d"
        },
        "battery_capacity": {
            "tags": ["battery_id", "cycle"],
            "fields": ["capacity", "timestamp"],
            "retention": "90d"
        },
        "battery_predictions": {
            "tags": ["battery_id", "prediction_type", "agent_id"],
            "fields": ["predicted_capacity", "uncertainty", "horizon", "timestamp"],
            "retention": "60d"
        },
        "battery_state_estimates": {
            "tags": ["battery_id", "agent_id"],
            "fields": ["soc", "soh", "r0", "r1", "c1", "timestamp"],
            "retention": "60d"
        },
        "battery_faults": {
            "tags": ["battery_id", "fault_type", "severity"],
            "fields": ["residual_magnitude", "description", "timestamp"],
            "retention": "90d"
        },
        "battery_control": {
            "tags": ["battery_id", "control_mode"],
            "fields": ["current_setpoint", "voltage_limit", "timestamp"],
            "retention": "30d"
        },
        "battery_parameters": {
            "tags": ["battery_id", "agent_id"],
            "fields": ["k", "c0", "r0", "r1", "c1", "confidence", "timestamp"],
            "retention": "90d"
        },
    })


@dataclass
class BatteryMongoConfig(MongoConfig):
    """MongoDB configuration for battery twin."""

    # Battery-specific collections
    collections: Dict[str, Dict] = field(default_factory=lambda: {
        "trained_models": {
            "indexes": [
                {"keys": [("agent_id", 1), ("version", -1)], "unique": True},
                {"keys": [("battery_id", 1), ("timestamp", -1)]},
                {"keys": [("model_type", 1)]}
            ],
            "schema": {
                "agent_id": str,
                "battery_id": str,
                "model_type": str,  # "physics", "ml", "kalman"
                "version": str,
                "model_data": bytes,  # Serialized model
                "metadata": dict,
                "timestamp": float,
                "performance_metrics": dict
            }
        },
        "predictions": {
            "indexes": [
                {"keys": [("battery_id", 1), ("timestamp", -1)]},
                {"keys": [("prediction_type", 1)]}
            ],
            "schema": {
                "battery_id": str,
                "timestamp": float,
                "cycle": int,
                "physics_prediction": float,
                "ml_correction": float,
                "hybrid_prediction": float,
                "uncertainty": float,
                "actual_capacity": Optional[float]
            }
        },
        "hybrid_training_samples": {
            "indexes": [
                {"keys": [("battery_id", 1), ("cycle", 1)]},
                {"keys": [("timestamp", -1)]}
            ],
            "schema": {
                "battery_id": str,
                "cycle": int,
                "temperature": float,
                "duration": float,
                "capacity": float,
                "source": str,
                "timestamp": float
            }
        },
        "parameter_history": {
            "indexes": [
                {"keys": [("battery_id", 1), ("timestamp", -1)]},
                {"keys": [("agent_id", 1)]}
            ],
            "schema": {
                "battery_id": str,
                "agent_id": str,
                "timestamp": float,
                "cycle": int,
                "parameters": dict,  # k, C0, R0, R1, C1, etc.
                "confidence": float,
                "fit_quality": dict
            }
        },
        "fault_events": {
            "indexes": [
                {"keys": [("battery_id", 1), ("timestamp", -1)]},
                {"keys": [("severity", 1), ("timestamp", -1)]},
                {"keys": [("fault_type", 1)]}
            ],
            "schema": {
                "battery_id": str,
                "timestamp": float,
                "severity": str,  # "warning", "fault", "critical"
                "fault_type": str,
                "cause": str,
                "residual_magnitude": float,
                "agent_id": str
            }
        },
        "agent_configs": {
            "indexes": [
                {"keys": [("agent_id", 1)], "unique": True},
                {"keys": [("battery_id", 1)]}
            ],
            "schema": {
                "agent_id": str,
                "battery_id": str,
                "agent_type": str,
                "config": dict,
                "created_at": float,
                "updated_at": float
            }
        },
        "battery_metadata": {
            "indexes": [
                {"keys": [("battery_id", 1)], "unique": True}
            ],
            "schema": {
                "battery_id": str,
                "battery_type": str,
                "nominal_capacity": float,
                "nominal_voltage": float,
                "chemistry": str,
                "manufacturing_date": str,
                "first_cycle_date": float,
                "metadata": dict
            }
        }
    })


@dataclass
class BatteryStorageConfig(StorageConfig):
    """
    Extended storage configuration for battery digital twin.
    Includes all base storage configs plus battery-specific settings.
    """

    # Override with battery-specific configs
    influx_config: BatteryInfluxConfig = field(default_factory=BatteryInfluxConfig)
    mongo_config: BatteryMongoConfig = field(default_factory=BatteryMongoConfig)

    # Battery-specific settings
    battery_ids: List[str] = field(default_factory=lambda: ["B0005"])
    enable_real_time_caching: bool = True
    cache_ttl_seconds: int = 300  # 5 minutes

    # Batch processing for battery data
    batch_size: int = 100
    flush_interval: float = 1.0  # seconds

    @classmethod
    def from_yaml(cls, config_path: str) -> "BatteryStorageConfig":
        """Load battery storage config from YAML file."""
        import yaml

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        storage_config = config.get('storage', {})

        # Create InfluxDB config
        influx_cfg = storage_config.get('influxdb', {})
        influx_config = BatteryInfluxConfig(
            host=influx_cfg.get('host', 'localhost'),
            port=influx_cfg.get('port', 8086),
            database=influx_cfg.get('database', 'battery_metrics'),
            username=influx_cfg.get('username', 'admin'),
            password=influx_cfg.get('password', 'password123')
        )

        # Create MongoDB config
        mongo_cfg = storage_config.get('mongodb', {})
        mongo_config = BatteryMongoConfig(
            host=mongo_cfg.get('host', 'localhost'),
            port=mongo_cfg.get('port', 27017),
            database=mongo_cfg.get('database', 'battery_twin'),
            username=mongo_cfg.get('username', 'admin'),
            password=mongo_cfg.get('password', 'password123')
        )

        # Create Neo4j config
        neo4j_cfg = storage_config.get('neo4j', {})
        neo4j_config = Neo4jConfig(
            uri=neo4j_cfg.get('uri', 'bolt://localhost:7687'),
            username=neo4j_cfg.get('username', neo4j_cfg.get('user', 'neo4j')),
            password=neo4j_cfg.get('password', 'password123'),
            database=neo4j_cfg.get('database', 'battery_hierarchy')
        )

        # Create Redis config
        redis_cfg = storage_config.get('redis', {})
        redis_config = RedisConfig(
            host=redis_cfg.get('host', 'localhost'),
            port=redis_cfg.get('port', 6379),
            db=redis_cfg.get('db', 1),
            password=redis_cfg.get('password')
        )

        # Get battery IDs from data config
        battery_ids = config.get('data', {}).get('batteries', ['B0005'])

        return cls(
            enable_time_series=storage_config.get('enable_time_series', True),
            enable_document_store=storage_config.get('enable_document_store', True),
            enable_graph_store=storage_config.get('enable_graph_store', True),
            enable_cache=storage_config.get('enable_cache', True),
            influx_config=influx_config,
            mongo_config=mongo_config,
            neo4j_config=neo4j_config,
            redis_config=redis_config,
            battery_ids=battery_ids
        )
