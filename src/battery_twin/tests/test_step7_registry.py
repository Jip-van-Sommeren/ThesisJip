"""
Test Step 7: RegistryAgent

Tests for the RegistryAgent implementation:
- Agent registration
- Heartbeat monitoring
- Discovery queries
- Timeout handling
- Directory publishing
- Redis persistence

Run with: pytest src/battery_twin/tests/test_step7_registry.py -v
"""

import pytest
import time
import json
from unittest.mock import Mock, patch, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.abstract_agent import AgentId
from src.battery_twin.agents.registry_agent import (
    RegistryAgent,
    AgentRecord,
    AgentHealth
)
from src.battery_twin.communication.message_schemas import (
    AgentRegistrationMessage,
    AgentHeartbeatMessage,
    AgentDirectoryMessage,
    MessageFactory
)


class TestAgentRecord:
    """Test AgentRecord data class."""

    def test_create_agent_record(self):
        """Test creating an agent record."""
        record = AgentRecord(
            agent_id="test.agent.1",
            agent_type="BDI",
            capabilities=["prediction", "learning"],
            supervisor="orchestrator.1",
            roles=["model"],
            groups=["core"]
        )

        assert record.agent_id == "test.agent.1"
        assert record.agent_type == "BDI"
        assert "prediction" in record.capabilities
        assert record.supervisor == "orchestrator.1"
        assert record.health_status == AgentHealth.UNKNOWN

    def test_agent_record_to_dict(self):
        """Test converting agent record to dictionary."""
        record = AgentRecord(
            agent_id="test.agent.1",
            agent_type="Reactive",
            capabilities=["monitoring"],
            health_status=AgentHealth.ACTIVE
        )

        data = record.to_dict()

        assert data['agent_id'] == "test.agent.1"
        assert data['agent_type'] == "Reactive"
        assert data['health_status'] == "active"
        assert isinstance(data['capabilities'], list)

    def test_agent_record_from_dict(self):
        """Test creating agent record from dictionary."""
        data = {
            'agent_id': 'test.agent.2',
            'agent_type': 'Hybrid',
            'capabilities': ['control', 'optimization'],
            'supervisor': None,
            'roles': ['controller'],
            'groups': [],
            'health_status': 'active',
            'registration_time': time.time(),
            'last_heartbeat_time': time.time(),
            'heartbeat_count': 5,
            'uptime': 120.0,
            'last_status': 'running'
        }

        record = AgentRecord.from_dict(data)

        assert record.agent_id == 'test.agent.2'
        assert record.agent_type == 'Hybrid'
        assert record.health_status == AgentHealth.ACTIVE
        assert record.heartbeat_count == 5


class TestRegistryAgentBasics:
    """Test basic RegistryAgent functionality."""

    def test_create_registry_agent(self):
        """Test creating a RegistryAgent."""
        agent_id = AgentId(app="battery_twin", type="registry", instance="1")

        registry = RegistryAgent(
            agent_id=agent_id,
            heartbeat_timeout=30.0,
            enable_redis_persistence=False  # Disable for testing
        )

        assert registry.id == agent_id  # Use .id instead of .agent_id
        assert registry.heartbeat_timeout == 30.0
        assert len(registry.agent_directory) == 0

    def test_registry_agent_setup_teardown(self):
        """Test registry agent setup and teardown."""
        agent_id = AgentId(app="battery_twin", type="registry", instance="1")

        # Mock MQTT bridge
        mock_mqtt = Mock()
        mock_mqtt.is_connected.return_value = True
        mock_mqtt.connect.return_value = True
        mock_mqtt.subscribe_raw = Mock()
        mock_mqtt.publish = Mock(return_value=True)

        registry = RegistryAgent(
            agent_id=agent_id,
            mqtt_bridge=mock_mqtt,
            heartbeat_timeout=30.0,
            enable_redis_persistence=False
        )

        # Setup
        success = registry.setup()
        assert success
        assert registry.is_initialized
        assert registry.monitor_running
        assert registry.publish_running

        # Teardown
        registry.teardown()
        assert not registry.monitor_running
        assert not registry.publish_running


class TestAgentRegistration:
    """Test agent registration functionality."""

    def test_handle_registration_new_agent(self):
        """Test handling registration of a new agent."""
        agent_id = AgentId(app="battery_twin", type="registry", instance="1")

        # Create registry without MQTT
        registry = RegistryAgent(
            agent_id=agent_id,
            enable_redis_persistence=False
        )

        # Create registration message
        reg_msg = AgentRegistrationMessage(
            agent_id="telemetry.ingestor.1",
            agent_type="Reactive",
            capabilities=["validation", "cleaning"],
            supervisor="orchestrator.1",
            roles=["ingestor"],
            groups=["data_pipeline"],
            timestamp=time.time()
        )

        # Handle registration directly
        payload = MessageFactory.to_json(reg_msg)
        registry._handle_registration("agent/telemetry.ingestor.1/register", payload)

        # Verify agent is registered
        assert registry.get_agent_count() == 1
        record = registry.get_agent("telemetry.ingestor.1")
        assert record is not None
        assert record.agent_type == "Reactive"
        assert "validation" in record.capabilities
        assert record.supervisor == "orchestrator.1"

    def test_handle_registration_update_existing(self):
        """Test updating registration of existing agent."""
        agent_id = AgentId(app="battery_twin", type="registry", instance="1")

        registry = RegistryAgent(
            agent_id=agent_id,
            enable_redis_persistence=False
        )

        # Register agent first time
        reg_msg1 = AgentRegistrationMessage(
            agent_id="test.agent.1",
            agent_type="BDI",
            capabilities=["learning"],
            timestamp=time.time()
        )
        payload1 = MessageFactory.to_json(reg_msg1)
        registry._handle_registration("agent/test.agent.1/register", payload1)

        assert registry.get_agent_count() == 1

        # Update registration
        reg_msg2 = AgentRegistrationMessage(
            agent_id="test.agent.1",
            agent_type="BDI",
            capabilities=["learning", "prediction"],  # Added capability
            supervisor="orchestrator.1",  # Added supervisor
            timestamp=time.time()
        )
        payload2 = MessageFactory.to_json(reg_msg2)
        registry._handle_registration("agent/test.agent.1/register", payload2)

        # Should still be 1 agent, but updated
        assert registry.get_agent_count() == 1
        record = registry.get_agent("test.agent.1")
        assert record is not None
        assert len(record.capabilities) == 2
        assert record.supervisor == "orchestrator.1"


class TestHeartbeatMonitoring:
    """Test heartbeat monitoring functionality."""

    def test_handle_heartbeat(self):
        """Test handling heartbeat messages."""
        agent_id = AgentId(app="battery_twin", type="registry", instance="1")

        registry = RegistryAgent(
            agent_id=agent_id,
            enable_redis_persistence=False
        )

        # Register agent first
        reg_msg = AgentRegistrationMessage(
            agent_id="test.agent.1",
            agent_type="Reactive",
            capabilities=[],
            timestamp=time.time()
        )
        registry._handle_registration("agent/test.agent.1/register", MessageFactory.to_json(reg_msg))

        # Send heartbeat
        hb_msg = AgentHeartbeatMessage(
            agent_id="test.agent.1",
            timestamp=time.time(),
            status="active",
            uptime=60.0
        )
        payload = MessageFactory.to_json(hb_msg)
        registry._handle_heartbeat("agent/test.agent.1/heartbeat", payload)

        # Verify heartbeat recorded
        record = registry.get_agent("test.agent.1")
        assert record is not None
        assert record.heartbeat_count == 1
        assert record.health_status == AgentHealth.ACTIVE
        assert record.uptime == 60.0
        assert record.last_status == "active"

    def test_multiple_heartbeats(self):
        """Test handling multiple heartbeats from same agent."""
        agent_id = AgentId(app="battery_twin", type="registry", instance="1")

        registry = RegistryAgent(
            agent_id=agent_id,
            enable_redis_persistence=False
        )

        # Register agent
        reg_msg = AgentRegistrationMessage(
            agent_id="test.agent.1",
            agent_type="Reactive",
            capabilities=[],
            timestamp=time.time()
        )
        registry._handle_registration("agent/test.agent.1/register", MessageFactory.to_json(reg_msg))

        # Send multiple heartbeats
        for i in range(5):
            hb_msg = AgentHeartbeatMessage(
                agent_id="test.agent.1",
                timestamp=time.time(),
                status="active",
                uptime=60.0 + i * 10
            )
            registry._handle_heartbeat("agent/test.agent.1/heartbeat", MessageFactory.to_json(hb_msg))
            time.sleep(0.1)

        # Verify heartbeats counted
        record = registry.get_agent("test.agent.1")
        assert record.heartbeat_count == 5

    def test_heartbeat_timeout_detection(self):
        """Test detection of heartbeat timeout."""
        agent_id = AgentId(app="battery_twin", type="registry", instance="1")

        # Set very short timeout for testing
        registry = RegistryAgent(
            agent_id=agent_id,
            heartbeat_timeout=1.0,  # 1 second timeout
            enable_redis_persistence=False
        )

        # Register and send initial heartbeat
        reg_msg = AgentRegistrationMessage(
            agent_id="test.agent.1",
            agent_type="Reactive",
            capabilities=[],
            timestamp=time.time()
        )
        registry._handle_registration("agent/test.agent.1/register", MessageFactory.to_json(reg_msg))

        hb_msg = AgentHeartbeatMessage(
            agent_id="test.agent.1",
            timestamp=time.time(),
            status="active",
            uptime=60.0
        )
        registry._handle_heartbeat("agent/test.agent.1/heartbeat", MessageFactory.to_json(hb_msg))

        # Agent should be active
        record = registry.get_agent("test.agent.1")
        assert record.health_status == AgentHealth.ACTIVE

        # Start monitoring
        registry._start_monitoring()

        # Wait for timeout (monitoring loop checks every 5 seconds)
        # We need to wait at least 5 seconds for the monitoring loop to run
        # Plus the 1 second timeout
        time.sleep(6.0)

        # Agent should be marked as failed
        record = registry.get_agent("test.agent.1")
        assert record.health_status == AgentHealth.FAILED

        # Cleanup
        registry._stop_monitoring()

    def test_heartbeat_from_unregistered_agent(self):
        """Test handling heartbeat from unregistered agent."""
        agent_id = AgentId(app="battery_twin", type="registry", instance="1")

        registry = RegistryAgent(
            agent_id=agent_id,
            enable_redis_persistence=False
        )

        # Send heartbeat without registration
        hb_msg = AgentHeartbeatMessage(
            agent_id="unregistered.agent.1",
            timestamp=time.time(),
            status="active",
            uptime=60.0
        )
        registry._handle_heartbeat("agent/unregistered.agent.1/heartbeat", MessageFactory.to_json(hb_msg))

        # Agent should be auto-registered with minimal info
        record = registry.get_agent("unregistered.agent.1")
        assert record is not None
        assert record.agent_type == "unknown"
        assert record.health_status == AgentHealth.ACTIVE


class TestDiscoveryService:
    """Test agent discovery functionality."""

    @pytest.fixture
    def populated_registry(self):
        """Create a registry with multiple registered agents."""
        agent_id = AgentId(app="battery_twin", type="registry", instance="1")
        registry = RegistryAgent(agent_id=agent_id, enable_redis_persistence=False)

        # Register multiple agents
        agents_data = [
            ("telemetry.ingestor.1", "Reactive", ["validation", "cleaning"], "orchestrator.1", ["ingestor"]),
            ("physics.model.1", "Hybrid", ["prediction", "physics"], "orchestrator.1", ["model"]),
            ("ml.residual.1", "BDI", ["prediction", "learning"], "orchestrator.1", ["model"]),
            ("state.estimator.1", "BDI", ["estimation"], "orchestrator.1", ["estimator"]),
            ("fault.detection.1", "Reactive", ["monitoring", "fault_detection"], "orchestrator.1", ["monitor"]),
        ]

        for aid, atype, caps, sup, roles in agents_data:
            reg_msg = AgentRegistrationMessage(
                agent_id=aid,
                agent_type=atype,
                capabilities=caps,
                supervisor=sup,
                roles=roles,
                timestamp=time.time()
            )
            registry._handle_registration(f"agent/{aid}/register", MessageFactory.to_json(reg_msg))

            # Send heartbeat to mark as active
            hb_msg = AgentHeartbeatMessage(
                agent_id=aid,
                timestamp=time.time(),
                status="active",
                uptime=60.0
            )
            registry._handle_heartbeat(f"agent/{aid}/heartbeat", MessageFactory.to_json(hb_msg))

        return registry

    def test_get_all_agents(self, populated_registry):
        """Test getting all registered agents."""
        agents = populated_registry.get_all_agents()
        assert len(agents) == 5

    def test_get_agents_by_type(self, populated_registry):
        """Test querying agents by type."""
        bdi_agents = populated_registry.get_agents_by_type("BDI")
        reactive_agents = populated_registry.get_agents_by_type("Reactive")
        hybrid_agents = populated_registry.get_agents_by_type("Hybrid")

        assert len(bdi_agents) == 2  # ml.residual, state.estimator
        assert len(reactive_agents) == 2  # telemetry.ingestor, fault.detection
        assert len(hybrid_agents) == 1  # physics.model

    def test_get_agents_by_capability(self, populated_registry):
        """Test querying agents by capability."""
        prediction_agents = populated_registry.get_agents_by_capability("prediction")
        monitoring_agents = populated_registry.get_agents_by_capability("monitoring")

        assert len(prediction_agents) == 2  # physics.model, ml.residual
        assert len(monitoring_agents) == 1  # fault.detection

    def test_get_agents_by_role(self, populated_registry):
        """Test querying agents by role."""
        model_agents = populated_registry.get_agents_by_role("model")
        monitor_agents = populated_registry.get_agents_by_role("monitor")

        assert len(model_agents) == 2  # physics.model, ml.residual
        assert len(monitor_agents) == 1  # fault.detection

    def test_get_agents_by_health(self, populated_registry):
        """Test querying agents by health status."""
        active_agents = populated_registry.get_active_agents()
        failed_agents = populated_registry.get_failed_agents()

        assert len(active_agents) == 5  # All agents sent heartbeats
        assert len(failed_agents) == 0

    def test_get_agent_by_id(self, populated_registry):
        """Test getting specific agent by ID."""
        agent = populated_registry.get_agent("physics.model.1")

        assert agent is not None
        assert agent.agent_type == "Hybrid"
        assert "prediction" in agent.capabilities

        # Non-existent agent
        none_agent = populated_registry.get_agent("nonexistent.agent.1")
        assert none_agent is None

    def test_get_statistics(self, populated_registry):
        """Test getting registry statistics."""
        stats = populated_registry.get_statistics()

        assert stats['total_agents'] == 5
        assert stats['active_agents'] == 5
        assert stats['failed_agents'] == 0
        assert stats['agents_by_type']['BDI'] == 2
        assert stats['agents_by_type']['Reactive'] == 2
        assert stats['agents_by_type']['Hybrid'] == 1


class TestDirectoryPublishing:
    """Test directory publishing functionality."""

    def test_publish_directory(self):
        """Test publishing directory to MQTT."""
        agent_id = AgentId(app="battery_twin", type="registry", instance="1")

        # Mock MQTT bridge
        mock_mqtt = Mock()
        mock_mqtt.is_connected.return_value = True
        mock_mqtt.publish = Mock(return_value=True)

        registry = RegistryAgent(
            agent_id=agent_id,
            mqtt_bridge=mock_mqtt,
            enable_redis_persistence=False
        )

        # Register some agents
        for i in range(3):
            reg_msg = AgentRegistrationMessage(
                agent_id=f"test.agent.{i}",
                agent_type="Reactive",
                capabilities=[],
                timestamp=time.time()
            )
            registry._handle_registration(
                f"agent/test.agent.{i}/register",
                MessageFactory.to_json(reg_msg)
            )

        # Publish directory
        registry.publish_directory()

        # Verify MQTT publish was called
        assert mock_mqtt.publish.called

        # Get the call arguments
        call_args = mock_mqtt.publish.call_args
        topic_name = call_args[0][0]
        message = call_args[0][1]

        assert topic_name == "agent_directory"
        assert isinstance(message, AgentDirectoryMessage)
        assert message.total_agents == 3
        assert len(message.agents) == 3


class TestRedisPersistence:
    """Test Redis persistence functionality."""

    def test_save_and_load_directory(self):
        """Test saving and loading directory from Redis."""
        agent_id = AgentId(app="battery_twin", type="registry", instance="1")

        # Mock Redis client
        mock_redis = MagicMock()
        redis_data = {}  # Simulate Redis storage

        def mock_hset(key, mapping):
            redis_data[key] = mapping['data']

        def mock_hget(key, field):
            return redis_data.get(key)

        def mock_keys(pattern):
            return [k for k in redis_data.keys() if k.startswith("registry:agents:")]

        def mock_expire(key, seconds):
            pass

        mock_redis.hset = mock_hset
        mock_redis.hget = mock_hget
        mock_redis.keys = mock_keys
        mock_redis.expire = mock_expire

        # Mock storage manager
        mock_storage = Mock()
        mock_storage.redis_storage = mock_redis

        # Create registry with Redis enabled
        registry = RegistryAgent(
            agent_id=agent_id,
            storage_manager=mock_storage,
            enable_redis_persistence=True
        )

        # Register agents
        for i in range(3):
            reg_msg = AgentRegistrationMessage(
                agent_id=f"test.agent.{i}",
                agent_type="Reactive",
                capabilities=[f"cap_{i}"],
                timestamp=time.time()
            )
            registry._handle_registration(
                f"agent/test.agent.{i}/register",
                MessageFactory.to_json(reg_msg)
            )

        # Save to Redis
        registry._save_directory_to_redis()

        # Verify data was saved
        assert len(redis_data) == 3

        # Create new registry and load
        registry2 = RegistryAgent(
            agent_id=agent_id,
            storage_manager=mock_storage,
            enable_redis_persistence=True
        )

        registry2._load_directory_from_redis()

        # Verify agents were loaded
        assert registry2.get_agent_count() == 3
        for i in range(3):
            agent = registry2.get_agent(f"test.agent.{i}")
            assert agent is not None
            assert f"cap_{i}" in agent.capabilities


class TestIntegration:
    """Integration tests for complete RegistryAgent workflow."""

    def test_complete_workflow(self):
        """Test complete registration, heartbeat, and discovery workflow."""
        agent_id = AgentId(app="battery_twin", type="registry", instance="1")

        # Mock MQTT
        mock_mqtt = Mock()
        mock_mqtt.is_connected.return_value = True
        mock_mqtt.connect.return_value = True
        mock_mqtt.subscribe_raw = Mock()
        mock_mqtt.publish = Mock(return_value=True)

        registry = RegistryAgent(
            agent_id=agent_id,
            mqtt_bridge=mock_mqtt,
            heartbeat_timeout=30.0,
            enable_redis_persistence=False
        )

        # Setup registry
        assert registry.setup()

        try:
            # Step 1: Register agent
            reg_msg = AgentRegistrationMessage(
                agent_id="telemetry.ingestor.1",
                agent_type="Reactive",
                capabilities=["validation", "cleaning"],
                supervisor="orchestrator.1",
                timestamp=time.time()
            )
            registry._handle_registration(
                "agent/telemetry.ingestor.1/register",
                MessageFactory.to_json(reg_msg)
            )

            # Verify registration
            assert registry.get_agent_count() == 1

            # Step 2: Send heartbeat
            hb_msg = AgentHeartbeatMessage(
                agent_id="telemetry.ingestor.1",
                timestamp=time.time(),
                status="active",
                uptime=60.0
            )
            registry._handle_heartbeat(
                "agent/telemetry.ingestor.1/heartbeat",
                MessageFactory.to_json(hb_msg)
            )

            # Verify health status
            agent = registry.get_agent("telemetry.ingestor.1")
            assert agent.health_status == AgentHealth.ACTIVE

            # Step 3: Query by capability
            validators = registry.get_agents_by_capability("validation")
            assert len(validators) == 1
            assert validators[0].agent_id == "telemetry.ingestor.1"

            # Step 4: Get statistics
            stats = registry.get_statistics()
            assert stats['total_agents'] == 1
            assert stats['active_agents'] == 1

            # Step 5: Publish directory
            registry.publish_directory()
            assert mock_mqtt.publish.called

        finally:
            # Cleanup
            registry.teardown()


def test_summary():
    """Print test summary."""
    print("\n" + "="*70)
    print("Step 7: RegistryAgent Implementation - Test Summary")
    print("="*70)
    print("\nTests Cover:")
    print("✓ AgentRecord data structure")
    print("✓ RegistryAgent creation and lifecycle")
    print("✓ Agent registration (new and updates)")
    print("✓ Heartbeat monitoring")
    print("✓ Heartbeat timeout detection")
    print("✓ Discovery queries (by ID, type, capability, role, health)")
    print("✓ Directory publishing")
    print("✓ Redis persistence")
    print("✓ Complete integration workflow")
    print("\nSuccess Criteria:")
    print("✓ Agents can register and be discovered")
    print("✓ Heartbeat monitoring detects failures")
    print("✓ Directory queries work correctly")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Run with pytest
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
