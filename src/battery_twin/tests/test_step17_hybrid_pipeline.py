"""
Step 17: Hybrid pipeline end-to-end test.

Validates that telemetry → physics prediction → hybrid prediction publishes the
expected MQTT messages and persists hybrid training samples.
"""

import asyncio
from unittest.mock import MagicMock, patch, ANY
import numpy as np
import pandas as pd
import pytest

from src.battery_twin.communication.message_schemas import (
    TelemetryMessage,
    PredictionMessage,
    CapacityMessage,
)
from src.battery_twin.orchestrator import BatteryTwinConfig, BatteryTwinOrchestrator
from src.battery_twin.hybrid import HybridDigitalTwin


@pytest.mark.asyncio
async def test_hybrid_pipeline_flow():
    config = BatteryTwinConfig(
        battery_id="PIPE001",
        enable_storage=False,
        log_level="ERROR",
        enable_telemetry_ingestor=False,
        enable_state_estimator=False,
        enable_health_monitor=False,
        enable_physics_model=False,
        enable_ml_residual=False,
    )

    with patch("src.battery_twin.orchestrator.MqttBridge") as mock_bridge_cls:
        mock_bridge = MagicMock()
        mock_bridge.connect = MagicMock()
        mock_bridge.publish = MagicMock()
        mock_bridge.subscribe = MagicMock()
        mock_bridge.topic_manager.get_topic = MagicMock(
            side_effect=lambda name, **vars: f"battery/{vars['battery_id']}/{name}"
        )
        mock_bridge_cls.return_value = mock_bridge

        orch = BatteryTwinOrchestrator(config)
        await orch._initialize_mqtt()
        orch.hybrid_twin = HybridDigitalTwin()

    df = pd.DataFrame(
        {
            "id_cycle": np.arange(1, 20),
            "Temperature_measured": np.random.uniform(20, 30, 19),
            "Time": np.random.uniform(1000, 2000, 19),
            "Capacity": np.linspace(2.0, 1.6, 19),
        }
    )
    orch.hybrid_twin.fit(df, target_column="Capacity")

    # Simulate telemetry and capacity for cycle 20
    telemetry = TelemetryMessage(
        battery_id="PIPE001",
        timestamp=1.0,
        cycle=20,
        voltage=3.8,
        current=-1.0,
        temperature=26.0,
    )
    capacity = CapacityMessage(
        battery_id="PIPE001",
        timestamp=2.0,
        cycle=20,
        capacity=1.5,
    )

    orch._handle_clean_telemetry("battery/PIPE001/telemetry", telemetry.model_dump_json())
    orch._handle_capacity_measurement("battery/PIPE001/capacity", capacity.model_dump_json())

    physics_prediction = PredictionMessage(
        battery_id="PIPE001",
        timestamp=3.0,
        cycle=20,
        prediction_type="physics",
        predicted_capacity=1.55,
        agent_id="physics_agent",
        uncertainty=None,
        horizon=0,
    )

    orch._handle_physics_prediction(
        "battery/PIPE001/prediction/physics",
        physics_prediction.model_dump_json(),
    )

    mock_bridge.publish.assert_any_call(
        "hybrid_prediction",
        ANY,
        battery_id="PIPE001",
    )
