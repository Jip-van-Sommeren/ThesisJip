"""
Storage Configuration System
Manages configuration for multi-backend storage architecture.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import json
import yaml


@dataclass
class InfluxConfig:
    """InfluxDB configuration."""

    host: str = "localhost"
    port: int = 8086
    database: str = "agent_metrics"
    username: str = "admin"
    password: str = "password"
    ssl: bool = False
    verify_ssl: bool = False
    timeout: int = 30
    retention_policy: str = "autogen"

    def get_url(self) -> str:
        protocol = "https" if self.ssl else "http"
        return f"{protocol}://{self.host}:{self.port}"


@dataclass
class MongoConfig:
    """MongoDB configuration."""

    host: str = "localhost"
    port: int = 27017
    database: str = "agent_system"
    username: Optional[str] = None
    password: Optional[str] = None
    auth_source: str = "admin"
    replica_set: Optional[str] = None
    ssl: bool = False
    connection_timeout: int = 30000
    server_selection_timeout: int = 30000

    def get_connection_string(self) -> str:
        if self.username and self.password:
            auth = f"{self.username}:{self.password}@"
        else:
            auth = ""

        options = []
        if self.auth_source:
            options.append(f"authSource={self.auth_source}")
        if self.replica_set:
            options.append(f"replicaSet={self.replica_set}")
        if self.ssl:
            options.append("ssl=true")

        option_string = "&".join(options)
        if option_string:
            option_string = "?" + option_string

        return f"mongodb://{auth}{self.host}:{self.port}/{self.database}{option_string}"


@dataclass
class Neo4jConfig:
    """Neo4j configuration."""

    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password"
    database: str = "agent_hierarchy"
    encrypted: bool = False
    trust: str = "TRUST_SYSTEM_CA_SIGNED_CERTIFICATES"
    max_connection_lifetime: int = 3600
    max_connection_pool_size: int = 50
    connection_acquisition_timeout: int = 60

    def get_auth(self) -> tuple:
        return (self.username, self.password)


@dataclass
class RedisConfig:
    """Redis configuration."""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ssl: bool = False
    ssl_cert_reqs: str = "required"
    connection_pool_max_connections: int = 50
    socket_timeout: int = 30
    socket_connect_timeout: int = 30
    decode_responses: bool = True

    def get_connection_kwargs(self) -> Dict[str, Any]:
        kwargs = {
            "host": self.host,
            "port": self.port,
            "db": self.db,
            "decode_responses": self.decode_responses,
            "socket_timeout": self.socket_timeout,
            "socket_connect_timeout": self.socket_connect_timeout,
        }

        if self.password:
            kwargs["password"] = self.password
        if self.ssl:
            kwargs["ssl"] = True
            kwargs["ssl_cert_reqs"] = self.ssl_cert_reqs

        return kwargs


@dataclass
class RetentionPolicy:
    """Data retention policies."""

    belief_retention_days: int = 30
    goal_retention_days: int = 90
    message_retention_days: int = 90
    hierarchy_retention_days: int = 365
    performance_retention_days: int = 7
    agent_profile_retention_days: int = 1095  # 3 years


@dataclass
class StorageConfig:
    """Main storage configuration."""

    time_series_backend: str = "influxdb"  # influxdb, timescaledb, prometheus
    document_backend: str = "mongodb"  # mongodb, couchdb
    graph_backend: str = "neo4j"  # neo4j, arangodb
    cache_backend: str = "redis"  # redis, memcached

    # Backend configurations
    influx_config: InfluxConfig = field(default_factory=InfluxConfig)
    mongo_config: MongoConfig = field(default_factory=MongoConfig)
    neo4j_config: Neo4jConfig = field(default_factory=Neo4jConfig)
    redis_config: RedisConfig = field(default_factory=RedisConfig)

    # Data retention policies
    retention: RetentionPolicy = field(default_factory=RetentionPolicy)

    # Performance settings
    batch_size: int = 1000
    flush_interval: int = 5  # seconds
    max_retries: int = 3
    retry_delay: int = 1  # seconds

    # Feature flags
    enable_time_series: bool = True
    enable_document_store: bool = True
    enable_graph_store: bool = True
    enable_cache: bool = True
    auto_persist_beliefs: bool = True
    auto_persist_goals: bool = True
    auto_persist_messages: bool = True

    @classmethod
    def from_file(cls, config_path: str) -> "StorageConfig":
        """Load configuration from file (JSON or YAML)."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            if config_path.endswith(".yaml") or config_path.endswith(".yml"):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StorageConfig":
        """Create configuration from dictionary."""
        config = cls()

        # Update basic settings
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Handle nested configurations
        if "influx_config" in data:
            config.influx_config = InfluxConfig(**data["influx_config"])

        if "mongo_config" in data:
            config.mongo_config = MongoConfig(**data["mongo_config"])

        if "neo4j_config" in data:
            config.neo4j_config = Neo4jConfig(**data["neo4j_config"])

        if "redis_config" in data:
            config.redis_config = RedisConfig(**data["redis_config"])

        if "retention" in data:
            config.retention = RetentionPolicy(**data["retention"])

        return config

    @classmethod
    def from_env(cls) -> "StorageConfig":
        """Create configuration from environment variables."""
        config = cls()

        # InfluxDB from environment
        config.influx_config.host = os.getenv(
            "INFLUX_HOST", config.influx_config.host
        )
        config.influx_config.port = int(
            os.getenv("INFLUX_PORT", str(config.influx_config.port))
        )
        config.influx_config.database = os.getenv(
            "INFLUX_DATABASE", config.influx_config.database
        )
        config.influx_config.username = os.getenv(
            "INFLUX_USERNAME", config.influx_config.username
        )
        config.influx_config.password = os.getenv(
            "INFLUX_PASSWORD", config.influx_config.password
        )

        # MongoDB from environment
        config.mongo_config.host = os.getenv(
            "MONGO_HOST", config.mongo_config.host
        )
        config.mongo_config.port = int(
            os.getenv("MONGO_PORT", str(config.mongo_config.port))
        )
        config.mongo_config.database = os.getenv(
            "MONGO_DATABASE", config.mongo_config.database
        )
        config.mongo_config.username = os.getenv(
            "MONGO_USERNAME", config.mongo_config.username
        )
        config.mongo_config.password = os.getenv(
            "MONGO_PASSWORD", config.mongo_config.password
        )

        # Neo4j from environment
        config.neo4j_config.uri = os.getenv(
            "NEO4J_URI", config.neo4j_config.uri
        )
        config.neo4j_config.username = os.getenv(
            "NEO4J_USERNAME", config.neo4j_config.username
        )
        config.neo4j_config.password = os.getenv(
            "NEO4J_PASSWORD", config.neo4j_config.password
        )
        config.neo4j_config.database = os.getenv(
            "NEO4J_DATABASE", config.neo4j_config.database
        )

        # Redis from environment
        config.redis_config.host = os.getenv(
            "REDIS_HOST", config.redis_config.host
        )
        config.redis_config.port = int(
            os.getenv("REDIS_PORT", str(config.redis_config.port))
        )
        config.redis_config.db = int(
            os.getenv("REDIS_DB", str(config.redis_config.db))
        )
        config.redis_config.password = os.getenv(
            "REDIS_PASSWORD", config.redis_config.password
        )

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "time_series_backend": self.time_series_backend,
            "document_backend": self.document_backend,
            "graph_backend": self.graph_backend,
            "cache_backend": self.cache_backend,
            "influx_config": {
                "host": self.influx_config.host,
                "port": self.influx_config.port,
                "database": self.influx_config.database,
                "username": self.influx_config.username,
                "password": self.influx_config.password,
                "ssl": self.influx_config.ssl,
                "timeout": self.influx_config.timeout,
            },
            "mongo_config": {
                "host": self.mongo_config.host,
                "port": self.mongo_config.port,
                "database": self.mongo_config.database,
                "username": self.mongo_config.username,
                "password": self.mongo_config.password,
                "auth_source": self.mongo_config.auth_source,
            },
            "neo4j_config": {
                "uri": self.neo4j_config.uri,
                "username": self.neo4j_config.username,
                "password": self.neo4j_config.password,
                "database": self.neo4j_config.database,
            },
            "redis_config": {
                "host": self.redis_config.host,
                "port": self.redis_config.port,
                "db": self.redis_config.db,
                "password": self.redis_config.password,
            },
            "retention": {
                "belief_retention_days": self.retention.belief_retention_days,
                "goal_retention_days": self.retention.goal_retention_days,
                "message_retention_days": self.retention.message_retention_days,
                "hierarchy_retention_days": self.retention.hierarchy_retention_days,
            },
            "batch_size": self.batch_size,
            "flush_interval": self.flush_interval,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "enable_time_series": self.enable_time_series,
            "enable_document_store": self.enable_document_store,
            "enable_graph_store": self.enable_graph_store,
            "enable_cache": self.enable_cache,
        }

    def save_to_file(self, config_path: str):
        """Save configuration to file."""
        data = self.to_dict()

        with open(config_path, "w") as f:
            if config_path.endswith(".yaml") or config_path.endswith(".yml"):
                yaml.dump(data, f, default_flow_style=False, indent=2)
            else:
                json.dump(data, f, indent=2)

    def validate(self) -> bool:
        """Validate configuration."""
        errors = []

        # Check backend selections
        valid_ts_backends = ["influxdb", "timescaledb", "prometheus"]
        if self.time_series_backend not in valid_ts_backends:
            errors.append(
                f"Invalid time series backend: {self.time_series_backend}"
            )

        valid_doc_backends = ["mongodb", "couchdb"]
        if self.document_backend not in valid_doc_backends:
            errors.append(f"Invalid document backend: {self.document_backend}")

        valid_graph_backends = ["neo4j", "arangodb"]
        if self.graph_backend not in valid_graph_backends:
            errors.append(f"Invalid graph backend: {self.graph_backend}")

        valid_cache_backends = ["redis", "memcached"]
        if self.cache_backend not in valid_cache_backends:
            errors.append(f"Invalid cache backend: {self.cache_backend}")

        # Check required fields
        if not self.influx_config.database:
            errors.append("InfluxDB database name is required")

        if not self.mongo_config.database:
            errors.append("MongoDB database name is required")

        if not self.neo4j_config.database:
            errors.append("Neo4j database name is required")

        if errors:
            raise ValueError(
                f"Configuration validation errors: {', '.join(errors)}"
            )

        return True
