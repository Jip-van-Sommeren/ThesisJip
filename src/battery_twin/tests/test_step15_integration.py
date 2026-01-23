"""
Integration tests for Battery Digital Twin System (Step 15)

Tests cover:
- System orchestrator functionality
- Multi-agent coordination
- End-to-end data flow
- MQTT communication
- Graceful shutdown
- Configuration management

Author: Battery Twin Development Team
Date: 2025-03-01
"""

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, AsyncMock

import pytest
import yaml

from mas.core import AgentId
from src.battery_twin.orchestrator import (
    BatteryTwinConfig,
    BatteryTwinOrchestrator,
    SystemState,
    AgentStatus,
    AgentInfo
)
from src.battery_twin.communication.message_schemas import (
    TelemetryMessage,
    StateEstimateMessage,
    CapacityMessage,
    MessageFactory
)
from src.battery_twin.agents.health_monitor_agent import HealthReportMessage


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_config():
    """Create test configuration."""
    return BatteryTwinConfig(
        battery_id="TEST001",
        mqtt_broker="localhost",
        mqtt_port=1883,
        enable_storage=False,
        enable_telemetry_ingestor=True,
        enable_state_estimator=True,
        enable_health_monitor=True,
        enable_physics_model=False,
        enable_ml_residual=False,
        log_level="DEBUG",
        enable_metrics=False  # Disable for faster tests
    )


@pytest.fixture
def mock_mqtt_bridge():
    """Create mock MQTT bridge."""
    mqtt = AsyncMock()
    mqtt.connect = AsyncMock()
    mqtt.disconnect = AsyncMock()
    mqtt.publish = AsyncMock()
    mqtt.subscribe = Mock()
    mqtt.topic_manager = Mock()
    mqtt.topic_manager.get_topic = Mock(
        side_effect=lambda name, **vars: f"mock/{name}/{vars.get('battery_id', '')}"
    )
    return mqtt


@pytest.fixture
async def orchestrator(test_config):
    """Create orchestrator for testing."""
    orch = BatteryTwinOrchestrator(test_config)
    yield orch
    # Cleanup
    if orch.state == SystemState.RUNNING:
        await orch.shutdown()


# ============================================================================
# Test 1-5: Configuration Management
# ============================================================================

def test_01_config_creation():
    """Test configuration creation with defaults."""
    config = BatteryTwinConfig()
    assert config.battery_id == "B0005"
    assert config.mqtt_broker == "localhost"
    assert config.mqtt_port == 1883
    assert config.enable_telemetry_ingestor is True


def test_02_config_from_dict():
    """Test configuration from dictionary."""
    config = BatteryTwinConfig(
        battery_id="TEST123",
        mqtt_broker="test.mqtt.com",
        log_level="WARNING"
    )
    assert config.battery_id == "TEST123"
    assert config.mqtt_broker == "test.mqtt.com"
    assert config.log_level == "WARNING"


def test_03_config_from_yaml(tmp_path):
    """Test configuration loading from YAML file."""
    config_file = tmp_path / "test_config.yaml"
    config_dict = {
        'battery_id': 'YAML_TEST',
        'mqtt_broker': 'yaml.broker.com',
        'mqtt_port': 8883,
        'log_level': 'ERROR'
    }

    with open(config_file, 'w') as f:
        yaml.dump(config_dict, f)

    config = BatteryTwinConfig.from_yaml(str(config_file))
    assert config.battery_id == 'YAML_TEST'
    assert config.mqtt_broker == 'yaml.broker.com'
    assert config.mqtt_port == 8883
    assert config.log_level == 'ERROR'


def test_04_config_agent_flags():
    """Test agent enable/disable flags."""
    # All enabled
    config = BatteryTwinConfig(
        enable_telemetry_ingestor=True,
        enable_state_estimator=True,
        enable_health_monitor=True,
        enable_physics_model=True,
        enable_ml_residual=True
    )
    assert config.enable_telemetry_ingestor is True
    assert config.enable_physics_model is True

    # Minimal
    config = BatteryTwinConfig(
        enable_telemetry_ingestor=True,
        enable_state_estimator=False,
        enable_health_monitor=False
    )
    assert config.enable_state_estimator is False
    assert config.enable_health_monitor is False


def test_05_config_ekf_parameters():
    """Test EKF configuration parameters."""
    config = BatteryTwinConfig(
        ekf_initial_soc=0.7,
        ekf_initial_soh=0.95,
        ekf_capacity_nominal=2.5
    )
    assert config.ekf_initial_soc == 0.7
    assert config.ekf_initial_soh == 0.95
    assert config.ekf_capacity_nominal == 2.5


# ============================================================================
# Test 6-10: Orchestrator Initialization
# ============================================================================

@pytest.mark.asyncio
async def test_06_orchestrator_creation(test_config):
    """Test orchestrator creation."""
    orch = BatteryTwinOrchestrator(test_config)
    assert orch.config == test_config
    assert orch.battery_id == "TEST001"
    assert orch.state == SystemState.INITIALIZING
    assert len(orch.agents) == 0


@pytest.mark.asyncio
async def test_07_mqtt_initialization(test_config, mock_mqtt_bridge):
    """Test MQTT infrastructure initialization."""
    with patch('src.battery_twin.orchestrator.MqttBridge', return_value=mock_mqtt_bridge):
        orch = BatteryTwinOrchestrator(test_config)
        await orch._initialize_mqtt()

        assert orch.mqtt_bridge is not None
        assert orch.mqtt_config is not None
        mock_mqtt_bridge.connect.assert_called_once()


@pytest.mark.asyncio
async def test_08_agent_creation(test_config, mock_mqtt_bridge):
    """Test agent creation during initialization."""
    with patch('src.battery_twin.orchestrator.MqttBridge', return_value=mock_mqtt_bridge):
        orch = BatteryTwinOrchestrator(test_config)
        await orch.initialize()

        # Should create 3 agents (telemetry, state, health)
        assert len(orch.agents) == 3
        assert "telemetry_ingestor" in orch.agents
        assert "state_estimator" in orch.agents
        assert "health_monitor" in orch.agents


@pytest.mark.asyncio
async def test_09_agent_creation_selective(mock_mqtt_bridge):
    """Test selective agent creation based on config."""
    config = BatteryTwinConfig(
        battery_id="SEL_TEST",
        enable_telemetry_ingestor=True,
        enable_state_estimator=True,
        enable_health_monitor=False,  # Disabled
        enable_storage=False
    )

    with patch('src.battery_twin.orchestrator.MqttBridge', return_value=mock_mqtt_bridge):
        orch = BatteryTwinOrchestrator(config)
        await orch.initialize()

        # Should only create 2 agents
        assert len(orch.agents) == 2
        assert "health_monitor" not in orch.agents


@pytest.mark.asyncio
async def test_10_initialization_state_transitions(test_config, mock_mqtt_bridge):
    """Test state transitions during initialization."""
    with patch('src.battery_twin.orchestrator.MqttBridge', return_value=mock_mqtt_bridge):
        orch = BatteryTwinOrchestrator(test_config)

        # Initial state
        assert orch.state == SystemState.INITIALIZING

        # After initialization
        await orch.initialize()
        assert orch.state == SystemState.INITIALIZING  # Still initializing until start()


# ============================================================================
# Test 11-15: Agent Lifecycle Management
# ============================================================================

@pytest.mark.asyncio
async def test_11_agent_start(test_config, mock_mqtt_bridge):
    """Test starting all agents."""
    with patch('src.battery_twin.orchestrator.MqttBridge', return_value=mock_mqtt_bridge):
        orch = BatteryTwinOrchestrator(test_config)
        await orch.initialize()
        await orch.start()

        # Check system state
        assert orch.state == SystemState.RUNNING
        assert orch.start_time is not None

        # Check all agents running
        for name, agent_info in orch.agents.items():
            assert agent_info.status == AgentStatus.RUNNING
            assert agent_info.start_time is not None


@pytest.mark.asyncio
async def test_12_agent_info_tracking(test_config, mock_mqtt_bridge):
    """Test agent information tracking."""
    with patch('src.battery_twin.orchestrator.MqttBridge', return_value=mock_mqtt_bridge):
        orch = BatteryTwinOrchestrator(test_config)
        await orch.initialize()
        await orch.start()

        # Check agent info structure
        telemetry_info = orch.agents["telemetry_ingestor"]
        assert telemetry_info.agent_type == "TelemetryIngestorAgent"
        assert telemetry_info.instance is not None
        assert telemetry_info.status == AgentStatus.RUNNING


@pytest.mark.asyncio
async def test_13_graceful_shutdown(test_config, mock_mqtt_bridge):
    """Test graceful system shutdown."""
    with patch('src.battery_twin.orchestrator.MqttBridge', return_value=mock_mqtt_bridge):
        orch = BatteryTwinOrchestrator(test_config)
        await orch.initialize()
        await orch.start()

        # Shutdown
        await orch.shutdown()

        # Check state
        assert orch.state == SystemState.STOPPED

        # Check agents stopped
        for agent_info in orch.agents.values():
            assert agent_info.status == AgentStatus.STOPPED

        # Check MQTT disconnected
        mock_mqtt_bridge.disconnect.assert_called_once()


@pytest.mark.asyncio
async def test_14_shutdown_idempotent(test_config, mock_mqtt_bridge):
    """Test that shutdown can be called multiple times safely."""
    with patch('src.battery_twin.orchestrator.MqttBridge', return_value=mock_mqtt_bridge):
        orch = BatteryTwinOrchestrator(test_config)
        await orch.initialize()
        await orch.start()

        # Shutdown twice
        await orch.shutdown()
        await orch.shutdown()  # Should not raise

        assert orch.state == SystemState.STOPPED


@pytest.mark.asyncio
async def test_15_run_with_duration(test_config, mock_mqtt_bridge):
    """Test running system for specific duration."""
    with patch('src.battery_twin.orchestrator.MqttBridge', return_value=mock_mqtt_bridge):
        orch = BatteryTwinOrchestrator(test_config)
        await orch.initialize()
        await orch.start()

        # Run for short duration
        start = time.time()
        await orch.run(duration=0.5)  # 0.5 seconds
        elapsed = time.time() - start

        # Should have run for approximately 0.5 seconds
        assert 0.4 < elapsed < 0.7
        assert orch.state == SystemState.STOPPED


# ============================================================================
# Test 16-20: Status and Monitoring
# ============================================================================

@pytest.mark.asyncio
async def test_16_get_status(test_config, mock_mqtt_bridge):
    """Test getting system status."""
    with patch('src.battery_twin.orchestrator.MqttBridge', return_value=mock_mqtt_bridge):
        orch = BatteryTwinOrchestrator(test_config)
        await orch.initialize()
        await orch.start()

        status = orch.get_status()

        assert status['system_state'] == SystemState.RUNNING.value
        assert status['battery_id'] == 'TEST001'
        assert status['start_time'] is not None
        assert 'agents' in status
        assert len(status['agents']) == 3


@pytest.mark.asyncio
async def test_17_status_includes_agents(test_config, mock_mqtt_bridge):
    """Test status includes all agent information."""
    with patch('src.battery_twin.orchestrator.MqttBridge', return_value=mock_mqtt_bridge):
        orch = BatteryTwinOrchestrator(test_config)
        await orch.initialize()
        await orch.start()

        status = orch.get_status()

        # Check telemetry agent status
        telemetry_status = status['agents']['telemetry_ingestor']
        assert telemetry_status['type'] == 'TelemetryIngestorAgent'
        assert telemetry_status['status'] == AgentStatus.RUNNING.value
        assert telemetry_status['start_time'] is not None


@pytest.mark.asyncio
async def test_18_status_uptime(test_config, mock_mqtt_bridge):
    """Test status reports system uptime."""
    with patch('src.battery_twin.orchestrator.MqttBridge', return_value=mock_mqtt_bridge):
        orch = BatteryTwinOrchestrator(test_config)
        await orch.initialize()
        await orch.start()

        await asyncio.sleep(0.1)  # Wait a bit

        status = orch.get_status()
        assert status['uptime_seconds'] > 0.0


@pytest.mark.asyncio
async def test_19_status_before_start(test_config, mock_mqtt_bridge):
    """Test status before system is started."""
    with patch('src.battery_twin.orchestrator.MqttBridge', return_value=mock_mqtt_bridge):
        orch = BatteryTwinOrchestrator(test_config)
        await orch.initialize()

        status = orch.get_status()

        assert status['system_state'] == SystemState.INITIALIZING.value
        assert status['start_time'] is None
        assert status['uptime_seconds'] == 0.0


@pytest.mark.asyncio
async def test_20_status_after_shutdown(test_config, mock_mqtt_bridge):
    """Test status after shutdown."""
    with patch('src.battery_twin.orchestrator.MqttBridge', return_value=mock_mqtt_bridge):
        orch = BatteryTwinOrchestrator(test_config)
        await orch.initialize()
        await orch.start()
        await orch.shutdown()

        status = orch.get_status()

        assert status['system_state'] == SystemState.STOPPED.value
        for agent_status in status['agents'].values():
            assert agent_status['status'] == AgentStatus.STOPPED.value


# ============================================================================
# Test 21-25: Error Handling
# ============================================================================

@pytest.mark.asyncio
async def test_21_initialization_error_handling(test_config):
    """Test error handling during initialization."""
    with patch('src.battery_twin.orchestrator.MqttBridge') as mock_bridge:
        mock_bridge.return_value.connect = AsyncMock(side_effect=Exception("Connection failed"))

        orch = BatteryTwinOrchestrator(test_config)

        with pytest.raises(Exception):
            await orch.initialize()

        assert orch.state == SystemState.ERROR


@pytest.mark.asyncio
async def test_22_shutdown_error_handling(test_config, mock_mqtt_bridge):
    """Test error handling during shutdown."""
    with patch('src.battery_twin.orchestrator.MqttBridge', return_value=mock_mqtt_bridge):
        mock_mqtt_bridge.disconnect = AsyncMock(side_effect=Exception("Disconnect failed"))

        orch = BatteryTwinOrchestrator(test_config)
        await orch.initialize()
        await orch.start()

        with pytest.raises(Exception):
            await orch.shutdown()

        assert orch.state == SystemState.ERROR


@pytest.mark.asyncio
async def test_23_missing_mqtt_broker(test_config):
    """Test handling of missing MQTT broker."""
    # Note: This would fail in real scenario, just testing error path
    test_config.mqtt_broker = "nonexistent.broker.com"

    with patch('src.battery_twin.orchestrator.MqttBridge') as mock_bridge:
        mock_bridge.return_value.connect = AsyncMock(side_effect=ConnectionError("Broker not found"))

        orch = BatteryTwinOrchestrator(test_config)

        with pytest.raises(ConnectionError):
            await orch.initialize()


@pytest.mark.asyncio
async def test_24_agent_creation_error(test_config, mock_mqtt_bridge):
    """Test handling of agent creation errors."""
    with patch('src.battery_twin.orchestrator.MqttBridge', return_value=mock_mqtt_bridge):
        with patch('src.battery_twin.orchestrator.StateEstimatorAgent', side_effect=Exception("Agent creation failed")):
            orch = BatteryTwinOrchestrator(test_config)

            with pytest.raises(Exception):
                await orch.initialize()


@pytest.mark.asyncio
async def test_25_partial_initialization(test_config, mock_mqtt_bridge):
    """Test system state after partial initialization failure."""
    with patch('src.battery_twin.orchestrator.MqttBridge', return_value=mock_mqtt_bridge):
        orch = BatteryTwinOrchestrator(test_config)

        # Initialize MQTT only
        await orch._initialize_mqtt()
        assert orch.mqtt_bridge is not None

        # Agents should still be empty
        assert len(orch.agents) == 0


# ============================================================================
# Test 26-30: Integration Scenarios
# ============================================================================

@pytest.mark.asyncio
async def test_26_minimal_system(mock_mqtt_bridge):
    """Test minimal system with only telemetry ingestor."""
    config = BatteryTwinConfig(
        battery_id="MIN_TEST",
        enable_telemetry_ingestor=True,
        enable_state_estimator=False,
        enable_health_monitor=False,
        enable_storage=False
    )

    with patch('src.battery_twin.orchestrator.MqttBridge', return_value=mock_mqtt_bridge):
        orch = BatteryTwinOrchestrator(config)
        await orch.initialize()
        await orch.start()

        assert len(orch.agents) == 1
        assert "telemetry_ingestor" in orch.agents

        await orch.shutdown()


@pytest.mark.asyncio
async def test_27_core_pipeline(mock_mqtt_bridge):
    """Test core monitoring pipeline (telemetry + state + health)."""
    config = BatteryTwinConfig(
        battery_id="CORE_TEST",
        enable_telemetry_ingestor=True,
        enable_state_estimator=True,
        enable_health_monitor=True,
        enable_physics_model=False,
        enable_ml_residual=False,
        enable_storage=False
    )

    with patch('src.battery_twin.orchestrator.MqttBridge', return_value=mock_mqtt_bridge):
        orch = BatteryTwinOrchestrator(config)
        await orch.initialize()
        await orch.start()

        assert len(orch.agents) == 3
        assert "telemetry_ingestor" in orch.agents
        assert "state_estimator" in orch.agents
        assert "health_monitor" in orch.agents

        await orch.shutdown()


@pytest.mark.asyncio
async def test_28_full_system(mock_mqtt_bridge):
    """Test full system with all agents."""
    config = BatteryTwinConfig(
        battery_id="FULL_TEST",
        enable_telemetry_ingestor=True,
        enable_state_estimator=True,
        enable_health_monitor=True,
        enable_physics_model=True,
        enable_ml_residual=True,
        enable_storage=False
    )

    with patch('src.battery_twin.orchestrator.MqttBridge', return_value=mock_mqtt_bridge):
        orch = BatteryTwinOrchestrator(config)
        await orch.initialize()
        await orch.start()

        assert len(orch.agents) == 5
        assert "physics_model" in orch.agents
        assert "ml_residual" in orch.agents

        await orch.shutdown()


@pytest.mark.asyncio
async def test_29_lifecycle_sequence(test_config, mock_mqtt_bridge):
    """Test complete lifecycle sequence."""
    with patch('src.battery_twin.orchestrator.MqttBridge', return_value=mock_mqtt_bridge):
        orch = BatteryTwinOrchestrator(test_config)

        # 1. Initial state
        assert orch.state == SystemState.INITIALIZING

        # 2. Initialize
        await orch.initialize()
        assert orch.mqtt_bridge is not None
        assert len(orch.agents) > 0

        # 3. Start
        await orch.start()
        assert orch.state == SystemState.RUNNING

        # 4. Get status
        status = orch.get_status()
        assert status['system_state'] == SystemState.RUNNING.value

        # 5. Shutdown
        await orch.shutdown()
        assert orch.state == SystemState.STOPPED


@pytest.mark.asyncio
async def test_30_configuration_variants(mock_mqtt_bridge):
    """Test different configuration variants."""
    configs = [
        # Minimal
        BatteryTwinConfig(enable_telemetry_ingestor=True, enable_state_estimator=False, enable_health_monitor=False),
        # Core pipeline
        BatteryTwinConfig(enable_telemetry_ingestor=True, enable_state_estimator=True, enable_health_monitor=True),
        # With physics
        BatteryTwinConfig(enable_telemetry_ingestor=True, enable_state_estimator=True, enable_health_monitor=False, enable_physics_model=True),
    ]

    expected_counts = [1, 3, 3]

    for config, expected in zip(configs, expected_counts):
        config.enable_storage = False

        with patch('src.battery_twin.orchestrator.MqttBridge', return_value=mock_mqtt_bridge):
            orch = BatteryTwinOrchestrator(config)
            await orch.initialize()

            assert len(orch.agents) == expected

            await orch.shutdown()


@pytest.mark.asyncio
async def test_31_hybrid_training_storage(test_config, mock_mqtt_bridge):
    """Verify capacity events trigger storage writes for hybrid samples."""
    with patch('src.battery_twin.orchestrator.MqttBridge', return_value=mock_mqtt_bridge):
        orch = BatteryTwinOrchestrator(test_config)

    orch.storage_manager = MagicMock()
    orch.train_hybrid_twin = MagicMock(return_value=False)
    orch._finalized_cycle_features[5] = {
        "Temperature_measured": 24.5,
        "Time": 900.0,
    }

    message = CapacityMessage(
        battery_id=orch.battery_id,
        timestamp=999.0,
        cycle=5,
        capacity=1.7,
        measurement_type="measured",
    )

    orch._handle_capacity_measurement(
        f"battery/{orch.battery_id}/capacity",
        message.model_dump_json(),
    )

    orch.storage_manager.record_hybrid_training_sample.assert_called_once()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
