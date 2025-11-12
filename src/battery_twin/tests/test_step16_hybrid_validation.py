"""
Step 16: Hybrid prediction parity validation.

Compares orchestrator-managed hybrid predictions against direct HybridDigitalTwin
baseline predictions to guard against regression.
"""

import asyncio
import numpy as np
import pandas as pd
import pytest

from src.battery_twin.orchestrator import BatteryTwinConfig, BatteryTwinOrchestrator
from src.battery_twin.hybrid import HybridDigitalTwin
from src.battery_twin.communication.message_schemas import PredictionMessage


async def _setup_orchestrator():
    config = BatteryTwinConfig(
        enable_telemetry_ingestor=False,
        enable_state_estimator=False,
        enable_health_monitor=False,
        enable_physics_model=False,
        enable_ml_residual=False,
        log_level="ERROR",
    )
    orch = BatteryTwinOrchestrator(config)
    await orch._initialize_mqtt()
    orch.hybrid_twin = HybridDigitalTwin()
    return orch


@pytest.mark.asyncio
async def test_hybrid_prediction_matches_baseline(monkeypatch):
    orch = await _setup_orchestrator()
    baseline = HybridDigitalTwin()

    # Synthetic training data
    df = pd.DataFrame(
        {
            "id_cycle": np.arange(1, 51),
            "Temperature_measured": np.random.uniform(20, 30, 50),
            "Time": np.random.uniform(1000, 2000, 50),
            "Capacity": np.linspace(2.0, 1.5, 50),
        }
    )

    orch.hybrid_twin.fit(df, target_column="Capacity")
    baseline.fit(df, target_column="Capacity")

    # Prepare a physics prediction message
    message = PredictionMessage(
        battery_id="TEST",
        timestamp=0.0,
        cycle=60,
        prediction_type="physics",
        predicted_capacity=1.4,
        uncertainty=None,
        horizon=0,
        agent_id="physics_agent",
    )

    # Capture published hybrid prediction
    published = {}

    def fake_publish(topic_name, msg, **kwargs):
        published["message"] = msg
        return True

    orch.mqtt_bridge.publish = fake_publish  # type: ignore
    orch._handle_physics_prediction(
        "battery/TEST/prediction/physics",
        message.model_dump_json(),
    )

    assert "message" in published

    # Baseline prediction for comparison
    feature = pd.DataFrame(
        {
            "id_cycle": [message.cycle],
            "Temperature_measured": [25.0],
            "Time": [1.0],
            "Capacity": [message.predicted_capacity],
        }
    )
    baseline_result = baseline.predict(
        feature, return_uncertainty=True, return_components=True
    )

    baseline_value = max(0.0, float(baseline_result.hybrid_prediction[0]))
    orch_value = published["message"].predicted_capacity

    assert orch_value >= 0
    assert pytest.approx(orch_value, rel=0.1) == baseline_value
