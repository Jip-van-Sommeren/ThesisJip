"""
Distributed benchmark configuration loader.

Parses YAML configuration for distributed benchmark runs across
multiple hosts (e.g., AWS EC2 instances). Separate from the local
benchmark config_loader to avoid modifying existing code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class HostConfig:
    """Configuration for a single host in the distributed benchmark."""

    name: str
    ip: str
    role: str  # "broker", "agent", or "orchestrator"
    instance_type: str = "t3.medium"

    def __post_init__(self):
        valid_roles = {"broker", "agent", "orchestrator"}
        if self.role not in valid_roles:
            raise ValueError(
                f"Invalid host role '{self.role}'. Must be one of: {valid_roles}"
            )


@dataclass
class TimeSyncConfig:
    """Clock synchronization configuration."""

    enabled: bool = True
    max_offset_ms: float = 1.0
    ntp_source: str = "169.254.169.123"  # Amazon Time Sync Service


@dataclass
class DistributedConfig:
    """Full distributed benchmark configuration."""

    enabled: bool = True
    ssh_user: str = "ubuntu"
    ssh_key: str = "~/.ssh/benchmark-key.pem"
    code_path: str = "/home/ubuntu/thesis"
    hosts: Dict[str, HostConfig] = field(default_factory=dict)
    agent_placement: str = "round_robin"  # or "all_on_separate"
    time_sync: TimeSyncConfig = field(default_factory=TimeSyncConfig)

    # Benchmark parameters (forwarded to workers)
    protocols: List[str] = field(default_factory=lambda: ["rest"])
    protocol_variants: Dict[str, List[str]] = field(default_factory=dict)
    variant_settings: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    scenarios: List[str] = field(
        default_factory=lambda: ["point_to_point_latency"]
    )
    agent_counts: List[int] = field(default_factory=lambda: [2])
    num_trials: int = 3
    latency_mode: str = "app_ack"
    output_dir: str = "results/distributed_benchmarks"

    # Derived helpers
    @property
    def broker_host(self) -> Optional[HostConfig]:
        for h in self.hosts.values():
            if h.role == "broker":
                return h
        return None

    @property
    def agent_hosts(self) -> List[HostConfig]:
        return [h for h in self.hosts.values() if h.role == "agent"]

    @property
    def all_remote_hosts(self) -> List[HostConfig]:
        return [h for h in self.hosts.values() if h.role != "orchestrator"]

    def get_broker_address(self, port: int = 9092) -> str:
        broker = self.broker_host
        if broker is None:
            raise ValueError("No broker host configured")
        return f"{broker.ip}:{port}"

    def validate(self):
        """Validate the distributed configuration."""
        if not self.hosts:
            raise ValueError("No hosts configured")

        broker_count = sum(1 for h in self.hosts.values() if h.role == "broker")
        agent_count = sum(1 for h in self.hosts.values() if h.role == "agent")

        if broker_count > 1:
            raise ValueError("At most one broker host is supported")
        if agent_count < 1:
            raise ValueError("At least one agent host is required")

        # Broker is required for MQTT/Kafka
        broker_protocols = {"mqtt", "kafka"}
        if broker_count == 0 and broker_protocols & set(self.protocols):
            raise ValueError(
                f"A broker host is required for protocols: "
                f"{broker_protocols & set(self.protocols)}"
            )

        ssh_key_path = os.path.expanduser(self.ssh_key)
        if not os.path.exists(ssh_key_path):
            raise FileNotFoundError(f"SSH key not found: {ssh_key_path}")

        valid_placements = {"round_robin", "all_on_separate"}
        if self.agent_placement not in valid_placements:
            raise ValueError(
                f"Invalid agent_placement '{self.agent_placement}'. "
                f"Must be one of: {valid_placements}"
            )


def load_distributed_config(path: str) -> DistributedConfig:
    """Load distributed benchmark configuration from a YAML file."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required. Install with: pip install pyyaml"
        ) from exc

    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    dist = data.get("distributed", {})
    if not dist:
        raise ValueError("No 'distributed' section found in config")

    # Parse hosts
    hosts = {}
    for name, host_data in dist.get("hosts", {}).items():
        hosts[name] = HostConfig(
            name=name,
            ip=host_data["ip"],
            role=host_data.get("role", "agent"),
            instance_type=host_data.get("instance_type", "t3.medium"),
        )

    # Parse time sync
    ts_data = dist.get("time_sync", {})
    time_sync = TimeSyncConfig(
        enabled=ts_data.get("enabled", True),
        max_offset_ms=float(ts_data.get("max_offset_ms", 1.0)),
        ntp_source=ts_data.get("ntp_source", "169.254.169.123"),
    )

    # Parse protocols
    protocols = []
    protocol_variants: Dict[str, List[str]] = {}
    variant_settings: Dict[str, Dict[str, Any]] = {}
    protocols_section = data.get("protocols", {})
    if isinstance(protocols_section, dict):
        protocols = list(protocols_section.keys())
        for proto, proto_cfg in protocols_section.items():
            if proto_cfg and isinstance(proto_cfg, dict):
                variants = proto_cfg.get("variants", {})
                if isinstance(variants, dict):
                    protocol_variants[proto] = list(variants.keys())
                    for vname, vcfg in variants.items():
                        variant_settings.setdefault(proto, {})[vname] = vcfg or {}
                elif isinstance(variants, list):
                    protocol_variants[proto] = [str(v) for v in variants]
    elif isinstance(protocols_section, list):
        protocols = protocols_section

    config = DistributedConfig(
        enabled=dist.get("enabled", True),
        ssh_user=dist.get("ssh_user", "ubuntu"),
        ssh_key=dist.get("ssh_key", "~/.ssh/benchmark-key.pem"),
        code_path=dist.get("code_path", "/home/ubuntu/thesis"),
        hosts=hosts,
        agent_placement=dist.get("agent_placement", "round_robin"),
        time_sync=time_sync,
        protocols=protocols,
        protocol_variants=protocol_variants,
        variant_settings=variant_settings,
        scenarios=data.get("scenarios", ["point_to_point_latency"]),
        agent_counts=data.get("agent_counts", [2]),
        num_trials=int(data.get("num_trials", 3)),
        latency_mode=data.get("latency_mode", "app_ack"),
        output_dir=data.get("output_dir", "results/distributed_benchmarks"),
    )

    return config
