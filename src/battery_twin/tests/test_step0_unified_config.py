import pytest
import yaml

from src.battery_twin.config import load_unified_config


def test_unified_config_loader_parses_nested_yaml():
    config = load_unified_config("src/battery_twin/config/battery_twin_config.yaml")

    assert config.system.name == "battery_digital_twin"
    assert config.mqtt.broker == "localhost"
    assert config.data.batteries[0] == "B0005"
    assert config.agents.telemetry_ingestor.enabled is True

    runtime_cfg = config.to_battery_twin_config()

    assert runtime_cfg.battery_id == "B0005"
    assert runtime_cfg.mqtt_port == 1883
    assert runtime_cfg.enable_telemetry_ingestor is True
    assert runtime_cfg.storage_config_path.endswith(
        "src/battery_twin/config/battery_twin_config.yaml"
    )


@pytest.mark.parametrize(
    "input_value, expected",
    [
        ("B1234", ["B1234"]),
        (["B1", "B2"], ["B1", "B2"]),
        (None, ["B0005"]),
    ],
)
def test_unified_config_batteries_coerce_to_list(tmp_path, input_value, expected):
    sample = {
        "system": {"name": "test", "mode": "demo", "log_level": "DEBUG"},
        "mqtt": {"broker": "localhost"},
        "storage": {},
        "data": {"batteries": input_value},
        "agents": {},
    }

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(sample, sort_keys=False), encoding="utf-8")

    config = load_unified_config(path)
    assert config.data.batteries == expected
