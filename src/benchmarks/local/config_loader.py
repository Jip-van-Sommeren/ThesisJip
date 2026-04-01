"""
Benchmark YAML configuration loader.

Allows defining benchmark matrices (protocol variants, concurrency levels,
scenario overrides, etc.) via declarative YAML files.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _coerce_concurrency(values: Any) -> List[int]:
    levels = []
    for item in _as_list(values):
        try:
            levels.append(int(item))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Concurrency level '{item}' must be an integer"
            ) from exc
    return levels


def _coerce_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError(f"{field_name} must be a boolean")


def load_config_from_yaml(path: str) -> Dict[str, Any]:
    """Load benchmark configuration from a YAML file."""
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PyYAML is required to load benchmark configuration files. "
            "Install it with `pip install pyyaml`."
        ) from exc

    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    config: Dict[str, Any] = {}

    # Trials / warm-up (optional)
    if "num_trials" in data:
        try:
            config["num_trials"] = int(data["num_trials"])
        except (TypeError, ValueError) as exc:
            raise ValueError("num_trials must be an integer") from exc
        if config["num_trials"] < 1:
            raise ValueError("num_trials must be >= 1")

    if "warm_up_operations" in data:
        try:
            config["warm_up_operations"] = int(data["warm_up_operations"])
        except (TypeError, ValueError) as exc:
            raise ValueError("warm_up_operations must be an integer") from exc
        if config["warm_up_operations"] < 0:
            raise ValueError("warm_up_operations must be >= 0")

    # Benchmark mode (simple/extensive)
    mode = data.get("mode")
    simple_mode = True
    extensive_mode = False
    if isinstance(mode, str):
        if mode.lower() == "extensive":
            simple_mode = False
            extensive_mode = True
    elif isinstance(mode, dict):
        simple_mode = mode.get("simple", simple_mode)
        extensive_mode = mode.get("extensive", extensive_mode)

    config["simple_mode"] = simple_mode
    config["extensive_mode"] = extensive_mode

    # Latency mode
    if "latency_mode" in data:
        config["latency_mode"] = data["latency_mode"]

    # Scenarios
    scenarios = data.get("scenarios")
    if scenarios:
        config["scenarios"] = _as_list(scenarios)

    # Agent counts (optional)
    agent_counts = data.get("agent_counts")
    if agent_counts:
        config["agent_counts"] = _as_list(agent_counts)

    # Output directory override
    if "output_dir" in data:
        config["output_dir"] = data["output_dir"]

    # Hierarchy benchmark controls
    for key in ("hierarchy_only", "communication_only", "hierarchy_ablation"):
        if key in data:
            config[key] = _coerce_bool(data[key], key)

    if "hierarchy_types" in data:
        hierarchy_types = [str(v) for v in _as_list(data["hierarchy_types"])]
        allowed = {"tree", "peer_to_peer", "hybrid"}
        invalid = [value for value in hierarchy_types if value not in allowed]
        if invalid:
            raise ValueError(
                f"Unsupported hierarchy_types: {invalid}. Allowed: {sorted(allowed)}"
            )
        config["hierarchy_types"] = hierarchy_types

    if "hierarchy_environments" in data:
        hierarchy_envs = [
            str(v) for v in _as_list(data["hierarchy_environments"])
        ]
        allowed_envs = {
            "task_distribution",
            "resource_allocation",
            "collaborative",
            "fault_recovery",
            "scalability",
        }
        invalid = [value for value in hierarchy_envs if value not in allowed_envs]
        if invalid:
            raise ValueError(
                "Unsupported hierarchy_environments: "
                f"{invalid}. Allowed: {sorted(allowed_envs)}"
            )
        config["hierarchy_environments"] = hierarchy_envs

    if "hierarchy_episodes" in data:
        try:
            config["hierarchy_episodes"] = int(data["hierarchy_episodes"])
        except (TypeError, ValueError) as exc:
            raise ValueError("hierarchy_episodes must be an integer") from exc
        if config["hierarchy_episodes"] < 1:
            raise ValueError("hierarchy_episodes must be >= 1")

    if "hierarchy_agent_counts" in data:
        counts = _as_list(data["hierarchy_agent_counts"])
        try:
            parsed_counts = [int(count) for count in counts]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "hierarchy_agent_counts must contain integers"
            ) from exc
        if any(count < 1 for count in parsed_counts):
            raise ValueError("hierarchy_agent_counts entries must be >= 1")
        config["hierarchy_agent_counts"] = parsed_counts

    if "hierarchy_ablation_params" in data:
        ablation_params = data["hierarchy_ablation_params"] or {}
        if not isinstance(ablation_params, dict):
            raise ValueError("hierarchy_ablation_params must be a mapping")
        normalized_params: Dict[str, List[Any]] = {}
        for key, values in ablation_params.items():
            normalized_params[str(key)] = _as_list(values)
        config["hierarchy_ablation_params"] = normalized_params

    if "hierarchy_ablation_base_configs" in data:
        base_configs = data["hierarchy_ablation_base_configs"] or []
        if not isinstance(base_configs, list):
            raise ValueError(
                "hierarchy_ablation_base_configs must be a list"
            )
        for idx, base_cfg in enumerate(base_configs):
            if not isinstance(base_cfg, dict):
                raise ValueError(
                    "hierarchy_ablation_base_configs entries must be mappings "
                    f"(invalid entry at index {idx})"
                )
        config["hierarchy_ablation_base_configs"] = base_configs

    # Protocols and variant matrices
    protocols_section = data.get("protocols", {})

    if isinstance(protocols_section, dict):
        protocols = list(protocols_section.keys())
        variant_map: Dict[str, List[str]] = {}
        variant_settings: Dict[str, Dict[str, Any]] = {}

        for protocol, proto_cfg in protocols_section.items():
            if proto_cfg is None:
                proto_cfg = {}
            if not isinstance(proto_cfg, dict):
                raise ValueError(
                    f"Protocol '{protocol}' configuration must be a mapping."
                )

            proto_variants, proto_settings = _parse_protocol_variants(
                proto_cfg
            )
            if proto_variants:
                variant_map[protocol] = proto_variants
            if proto_settings:
                variant_settings[protocol] = proto_settings

        config["protocols"] = protocols
        if variant_map:
            config["protocol_variants"] = variant_map
        if variant_settings:
            config["variant_settings"] = variant_settings

    else:
        config["protocols"] = _as_list(protocols_section)
    return config


def _parse_protocol_variants(
    proto_cfg: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """Parse per-protocol variant information from YAML."""
    variants_section = proto_cfg.get("variants")

    proto_defaults = {
        "parameters": proto_cfg.get("parameters", {}) or {},
        "concurrency_levels": proto_cfg.get("concurrency_levels"),
        "scenarios": proto_cfg.get("scenarios", {}) or {},
    }

    variant_names: List[str] = []
    variant_settings: Dict[str, Dict[str, Any]] = {}

    if variants_section is None:
        default_entry = _build_variant_entry("__default__", {}, proto_defaults)
        if default_entry:
            variant_settings["__default__"] = default_entry
        return variant_names, variant_settings

    if isinstance(variants_section, list):
        for variant in variants_section:
            name = str(variant)
            variant_names.append(name)
            variant_settings[name] = _build_variant_entry(
                name, {}, proto_defaults
            )
        return variant_names, variant_settings

    if isinstance(variants_section, dict):
        for name, cfg in variants_section.items():
            variant_names.append(str(name))
            cfg_dict = cfg or {}
            if not isinstance(cfg_dict, dict):
                raise ValueError(
                    f"Variant '{name}' configuration must be a mapping."
                )
            variant_settings[str(name)] = _build_variant_entry(
                name, cfg_dict, proto_defaults
            )
        return variant_names, variant_settings

    raise ValueError("Protocol 'variants' must be a mapping or a list.")


def _build_variant_entry(
    name: str, cfg: Dict[str, Any], defaults: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge variant-specific configuration with protocol defaults."""
    entry: Dict[str, Any] = {}

    parameters = {}
    parameters.update(defaults.get("parameters", {}))
    parameters.update(cfg.get("parameters", {}) or {})
    if parameters:
        entry["parameters"] = parameters

    concurrency_levels = cfg.get("concurrency_levels")
    if concurrency_levels is None:
        concurrency_levels = defaults.get("concurrency_levels")
    if concurrency_levels:
        entry["concurrency_levels"] = _coerce_concurrency(concurrency_levels)

    scenarios = {}
    scenarios.update(defaults.get("scenarios", {}) or {})
    scenarios_cfg = cfg.get("scenarios", {}) or {}
    for scenario_name, scenario_params in scenarios_cfg.items():
        existing = scenarios.get(scenario_name, {})
        merged = {}
        merged.update(existing)
        if scenario_params:
            if not isinstance(scenario_params, dict):
                raise ValueError(
                    f"Scenario override for '{scenario_name}' "
                    f"in variant '{name}' must be a mapping."
                )
            merged.update(scenario_params)
        scenarios[scenario_name] = merged
    if scenarios:
        entry["scenarios"] = scenarios

    return entry


# if __name__ == "__main__":
#     import sys
#     import pprint

#     if len(sys.argv) != 2:
#         print("Usage: python config_loader.py <config.yaml>")
#         sys.exit(1)

#     config_path = sys.argv[1]
#     config = load_config_from_yaml(config_path)
#     pprint.pprint(config)
