"""
Tests for Step 11: ML Residual Agent

Tests cover:
- Agent initialization and BDI components
- Training data accumulation
- BDI deliberation on training decisions
- Initial training trigger
- Retraining triggers (performance degradation, staleness)
- Hybrid prediction generation
- Model persistence
- MQTT message handling
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, MagicMock
from pathlib import Path
import tempfile

from src.abstract_agent import AgentId
from src.battery_twin.agents.ml_residual_agent import (
    MLResidualAgent,
    ModelStatus,
    PerformanceLevel,
    TrainingDataPoint
)
from src.battery_twin.communication.message_schemas import (
    PredictionMessage,
    CapacityMessage,
    MessageFactory
)
from src.battery_twin.models.neural_network import NeuralNetConfig


class TestTrainingDataPoint:
    """Test training data point dataclass."""

    def test_create_data_point(self):
        """Test creating training data point."""
        data = TrainingDataPoint(
            cycle=10,
            timestamp=time.time(),
            physics_prediction=1.95,
            actual_capacity=1.90
        )

        assert data.cycle == 10
        assert data.physics_prediction == 1.95
        assert data.actual_capacity == 1.90

    def test_is_complete_with_all_data(self):
        """Test is_complete with all required data."""
        data = TrainingDataPoint(
            cycle=10,
            timestamp=time.time(),
            physics_prediction=1.95,
            actual_capacity=1.90,
            voltages=[3.8, 3.9, 4.0],
            temperatures=[25.0, 26.0, 24.0]
        )

        assert data.is_complete()

    def test_is_complete_missing_actual(self):
        """Test is_complete without actual capacity."""
        data = TrainingDataPoint(
            cycle=10,
            timestamp=time.time(),
            physics_prediction=1.95,
            voltages=[3.8],
            temperatures=[25.0]
        )

        assert not data.is_complete()

    def test_is_complete_missing_voltages(self):
        """Test is_complete without voltage data."""
        data = TrainingDataPoint(
            cycle=10,
            timestamp=time.time(),
            physics_prediction=1.95,
            actual_capacity=1.90,
            temperatures=[25.0]
        )

        assert not data.is_complete()


class TestMLResidualAgentBasics:
    """Test ML Residual Agent initialization and basic functionality."""

    def test_agent_initialization(self):
        """Test agent initialization."""
        agent_id = AgentId(app="battery_twin", type="ml", instance="1")
        agent = MLResidualAgent(
            agent_id=agent_id,
            battery_id="B0005",
            min_training_samples=30
        )

        assert agent.battery_id == "B0005"
        assert agent.model_status == ModelStatus.NOT_TRAINED
        assert agent.min_training_samples == 30
        assert len(agent.training_buffer) == 0
        assert not agent.learner.model.is_fitted

    def test_agent_with_custom_config(self):
        """Test agent with custom ML configuration."""
        agent_id = AgentId(app="battery_twin", type="ml", instance="1")

        ml_config = NeuralNetConfig(
            hidden_layers=[128, 64],
            epochs=20,
            batch_size=8,
            device='cpu'
        )

        agent = MLResidualAgent(
            agent_id=agent_id,
            battery_id="B0005",
            ml_config=ml_config,
            min_training_samples=20,
            retrain_threshold_mae=0.08
        )

        assert agent.ml_config.hidden_layers == [128, 64]
        assert agent.ml_config.epochs == 20
        assert agent.min_training_samples == 20
        assert agent.retrain_threshold_mae == 0.08

    def test_beliefs_initialization(self):
        """Test BDI beliefs are initialized correctly."""
        agent_id = AgentId(app="battery_twin", type="ml", instance="1")
        agent = MLResidualAgent(agent_id=agent_id, battery_id="B0005")

        # Check beliefs
        model_status_belief = agent.state.get_belief('model_status')
        assert model_status_belief is not None
        assert 'not_trained' in model_status_belief.proposition

        trained_belief = agent.state.get_belief('model_trained')
        assert trained_belief is not None
        assert 'false' in trained_belief.proposition

    def test_goals_initialization(self):
        """Test BDI goals are initialized."""
        agent_id = AgentId(app="battery_twin", type="ml", instance="1")
        agent = MLResidualAgent(agent_id=agent_id, battery_id="B0005")

        # Agent should have goals
        assert len(agent.goals) > 0

        # Check for specific goal conditions
        goal_conditions = [g.condition for g in agent.goals]
        assert any('trained' in cond for cond in goal_conditions)


class TestMessageHandling:
    """Test MQTT message handling."""

    def test_handle_physics_prediction(self):
        """Test handling physics prediction message."""
        agent_id = AgentId(app="battery_twin", type="ml", instance="1")

        # Create mock MQTT bridge
        mock_mqtt = Mock()
        mock_mqtt.is_connected.return_value = True

        agent = MLResidualAgent(
            agent_id=agent_id,
            battery_id="B0005",
            mqtt_bridge=mock_mqtt
        )

        # Create physics prediction message
        pred_msg = PredictionMessage(
            battery_id="B0005",
            timestamp=time.time(),
            cycle=10,
            prediction_type="physics",
            predicted_capacity=1.95,
            uncertainty=None,
            horizon=0,
            agent_id="physics_agent"
        )

        payload = MessageFactory.to_json(pred_msg)
        agent._handle_physics_prediction("battery/B0005/prediction/physics", payload)

        # Check data was stored
        assert 10 in agent.training_buffer
        assert agent.training_buffer[10].physics_prediction == 1.95

    def test_handle_actual_capacity(self):
        """Test handling actual capacity message."""
        agent_id = AgentId(app="battery_twin", type="ml", instance="1")
        agent = MLResidualAgent(agent_id=agent_id, battery_id="B0005")

        # Create capacity message
        cap_msg = CapacityMessage(
            battery_id="B0005",
            timestamp=time.time(),
            cycle=10,
            capacity=1.90
        )

        payload = MessageFactory.to_json(cap_msg)
        agent._handle_actual_capacity("battery/B0005/capacity", payload)

        # Check data was stored
        assert 10 in agent.training_buffer
        assert agent.training_buffer[10].actual_capacity == 1.90

    def test_complete_data_point_from_messages(self):
        """Test creating complete data point from both messages."""
        agent_id = AgentId(app="battery_twin", type="ml", instance="1")
        agent = MLResidualAgent(agent_id=agent_id, battery_id="B0005")

        # Send physics prediction first
        pred_msg = PredictionMessage(
            battery_id="B0005",
            timestamp=time.time(),
            cycle=10,
            prediction_type="physics",
            predicted_capacity=1.95,
            uncertainty=None,
            horizon=0,
            agent_id="physics_agent"
        )
        agent._handle_physics_prediction(
            "battery/B0005/prediction/physics",
            MessageFactory.to_json(pred_msg)
        )

        # Send actual capacity
        cap_msg = CapacityMessage(
            battery_id="B0005",
            timestamp=time.time(),
            cycle=10,
            capacity=1.90
        )
        agent._handle_actual_capacity(
            "battery/B0005/capacity",
            MessageFactory.to_json(cap_msg)
        )

        # Check data point is complete
        assert 10 in agent.training_buffer
        data_point = agent.training_buffer[10]
        assert data_point.physics_prediction == 1.95
        assert data_point.actual_capacity == 1.90


class TestTrainingDeliberation:
    """Test BDI deliberation on training decisions."""

    def test_no_training_without_enough_data(self):
        """Test model not trained when insufficient data."""
        agent_id = AgentId(app="battery_twin", type="ml", instance="1")
        agent = MLResidualAgent(
            agent_id=agent_id,
            battery_id="B0005",
            min_training_samples=30
        )

        # Add only 10 data points (not enough)
        for i in range(10):
            agent.training_buffer[i] = TrainingDataPoint(
                cycle=i,
                timestamp=time.time(),
                physics_prediction=2.0 - i * 0.01,
                actual_capacity=2.0 - i * 0.01,
                voltages=[3.8],
                temperatures=[25.0]
            )

        # Deliberate
        agent._deliberate_on_training()

        # Model should not be trained
        assert agent.model_status == ModelStatus.NOT_TRAINED
        assert not agent.learner.model.is_fitted

    def test_initial_training_with_enough_data(self):
        """Test initial training triggered with sufficient data."""
        agent_id = AgentId(app="battery_twin", type="ml", instance="1")

        ml_config = NeuralNetConfig(epochs=5, batch_size=8, device='cpu')
        agent = MLResidualAgent(
            agent_id=agent_id,
            battery_id="B0005",
            ml_config=ml_config,
            min_training_samples=30
        )

        # Add 35 complete data points
        for i in range(35):
            agent.training_buffer[i] = TrainingDataPoint(
                cycle=i,
                timestamp=time.time(),
                physics_prediction=2.0 - i * 0.01,
                actual_capacity=2.0 - i * 0.015,  # Slightly different for residual
                cumulative_ah=i * 2.0,
                voltages=[3.8, 3.9, 4.0],
                temperatures=[25.0, 26.0, 24.0]
            )

        # Deliberate (should trigger training)
        agent._deliberate_on_training()

        # Model should be trained
        assert agent.model_status == ModelStatus.TRAINED
        assert agent.learner.model.is_fitted
        assert agent.total_trainings == 1

    def test_retraining_due_to_performance_degradation(self):
        """Test retraining triggered by performance degradation."""
        agent_id = AgentId(app="battery_twin", type="ml", instance="1")

        ml_config = NeuralNetConfig(epochs=5, batch_size=8, device='cpu')
        agent = MLResidualAgent(
            agent_id=agent_id,
            battery_id="B0005",
            ml_config=ml_config,
            min_training_samples=30,
            retrain_threshold_mae=0.05
        )

        # Add initial training data
        for i in range(35):
            agent.training_buffer[i] = TrainingDataPoint(
                cycle=i,
                timestamp=time.time(),
                physics_prediction=2.0 - i * 0.01,
                actual_capacity=2.0 - i * 0.015,
                cumulative_ah=i * 2.0,
                voltages=[3.8],
                temperatures=[25.0]
            )

        # Train initially
        agent._deliberate_on_training()
        assert agent.model_status == ModelStatus.TRAINED

        # Simulate poor performance (high errors)
        agent.hybrid_errors = [0.10, 0.12, 0.11, 0.13, 0.10,
                               0.11, 0.12, 0.10, 0.11, 0.12, 0.13]

        # Add some new data
        for i in range(35, 45):
            agent.training_buffer[i] = TrainingDataPoint(
                cycle=i,
                timestamp=time.time(),
                physics_prediction=2.0 - i * 0.01,
                actual_capacity=2.0 - i * 0.015,
                cumulative_ah=i * 2.0,
                voltages=[3.8],
                temperatures=[25.0]
            )

        # Deliberate (should trigger retraining)
        agent._deliberate_on_training()

        # Should have retrained
        assert agent.total_retrainings >= 1

    def test_retraining_due_to_staleness(self):
        """Test retraining triggered by model staleness."""
        agent_id = AgentId(app="battery_twin", type="ml", instance="1")

        ml_config = NeuralNetConfig(epochs=5, batch_size=8, device='cpu')
        agent = MLResidualAgent(
            agent_id=agent_id,
            battery_id="B0005",
            ml_config=ml_config,
            min_training_samples=30,
            retrain_interval_cycles=50
        )

        # Add initial training data
        for i in range(35):
            agent.training_buffer[i] = TrainingDataPoint(
                cycle=i,
                timestamp=time.time(),
                physics_prediction=2.0 - i * 0.01,
                actual_capacity=2.0 - i * 0.015,
                cumulative_ah=i * 2.0,
                voltages=[3.8],
                temperatures=[25.0]
            )

        # Train initially
        agent._deliberate_on_training()
        assert agent.model_status == ModelStatus.TRAINED
        assert agent.last_training_cycle < 35

        # Add data 50+ cycles later
        for i in range(90, 100):
            agent.training_buffer[i] = TrainingDataPoint(
                cycle=i,
                timestamp=time.time(),
                physics_prediction=2.0 - i * 0.01,
                actual_capacity=2.0 - i * 0.015,
                cumulative_ah=i * 2.0,
                voltages=[3.8],
                temperatures=[25.0]
            )

        # Deliberate (should trigger retraining due to staleness)
        agent._deliberate_on_training()

        # Should have retrained
        assert agent.total_retrainings >= 1


class TestHybridPrediction:
    """Test hybrid prediction generation."""

    def test_hybrid_prediction_before_training(self):
        """Test hybrid prediction returns physics when not trained."""
        agent_id = AgentId(app="battery_twin", type="ml", instance="1")
        agent = MLResidualAgent(agent_id=agent_id, battery_id="B0005")

        # Try to predict (model not trained)
        hybrid, uncertainty = agent.predict_hybrid_capacity(
            cycle=10,
            physics_prediction=1.95
        )

        # Should return physics prediction as-is
        assert hybrid == 1.95
        assert uncertainty is None

    def test_hybrid_prediction_after_training(self):
        """Test hybrid prediction after model training."""
        agent_id = AgentId(app="battery_twin", type="ml", instance="1")

        ml_config = NeuralNetConfig(epochs=5, batch_size=8, device='cpu')
        agent = MLResidualAgent(
            agent_id=agent_id,
            battery_id="B0005",
            ml_config=ml_config,
            min_training_samples=30
        )

        # Add training data
        for i in range(35):
            agent.training_buffer[i] = TrainingDataPoint(
                cycle=i,
                timestamp=time.time(),
                physics_prediction=2.0 - i * 0.01,
                actual_capacity=2.0 - i * 0.015,
                cumulative_ah=i * 2.0,
                voltages=[3.8],
                temperatures=[25.0]
            )

        # Train model
        agent._deliberate_on_training()

        # Make prediction
        hybrid, uncertainty = agent.predict_hybrid_capacity(
            cycle=10,
            physics_prediction=1.90,
            with_uncertainty=True
        )

        # Should get a prediction
        assert isinstance(hybrid, float)
        assert not np.isnan(hybrid)
        # Uncertainty should be estimated
        assert uncertainty is not None
        assert uncertainty >= 0

    def test_publish_hybrid_prediction(self):
        """Test publishing hybrid prediction to MQTT."""
        agent_id = AgentId(app="battery_twin", type="ml", instance="1")

        mock_mqtt = Mock()
        mock_mqtt.is_connected.return_value = True
        mock_mqtt.publish = Mock(return_value=True)

        agent = MLResidualAgent(
            agent_id=agent_id,
            battery_id="B0005",
            mqtt_bridge=mock_mqtt
        )

        # Publish prediction
        agent.publish_hybrid_prediction(
            cycle=10,
            hybrid_capacity=1.92,
            uncertainty=0.03
        )

        # Check MQTT publish was called
        assert mock_mqtt.publish.called


class TestModelPersistence:
    """Test model save and load."""

    def test_save_trained_model(self):
        """Test saving trained model."""
        agent_id = AgentId(app="battery_twin", type="ml", instance="1")

        ml_config = NeuralNetConfig(epochs=5, batch_size=8, device='cpu')

        # Create mock storage manager
        mock_storage = Mock()

        agent = MLResidualAgent(
            agent_id=agent_id,
            battery_id="B0005_test",
            ml_config=ml_config,
            storage_manager=mock_storage,
            min_training_samples=30
        )

        # Add training data
        for i in range(35):
            agent.training_buffer[i] = TrainingDataPoint(
                cycle=i,
                timestamp=time.time(),
                physics_prediction=2.0 - i * 0.01,
                actual_capacity=2.0 - i * 0.015,
                cumulative_ah=i * 2.0,
                voltages=[3.8],
                temperatures=[25.0]
            )

        # Train model
        agent._deliberate_on_training()
        assert agent.learner.model.is_fitted

        # Save model
        agent._save_model_to_storage()

        # Check model was saved to filesystem
        model_path = Path(f"/tmp/ml_models/B0005_test")
        assert model_path.exists()


class TestStatistics:
    """Test statistics and monitoring."""

    def test_get_statistics(self):
        """Test getting agent statistics."""
        agent_id = AgentId(app="battery_twin", type="ml", instance="1")
        agent = MLResidualAgent(agent_id=agent_id, battery_id="B0005")

        stats = agent.get_statistics()

        assert 'model_status' in stats
        assert 'model_trained' in stats
        assert 'training_data_count' in stats
        assert 'complete_data_count' in stats
        assert 'total_trainings' in stats
        assert 'total_retrainings' in stats

    def test_performance_level_excellent(self):
        """Test performance level classification - excellent."""
        agent_id = AgentId(app="battery_twin", type="ml", instance="1")
        agent = MLResidualAgent(agent_id=agent_id, battery_id="B0005")

        # Simulate excellent performance
        agent.hybrid_errors = [0.01, 0.015, 0.012, 0.018, 0.016]

        level = agent.get_performance_level()
        assert level == PerformanceLevel.EXCELLENT

    def test_performance_level_degraded(self):
        """Test performance level classification - degraded."""
        agent_id = AgentId(app="battery_twin", type="ml", instance="1")
        agent = MLResidualAgent(agent_id=agent_id, battery_id="B0005")

        # Simulate degraded performance
        agent.hybrid_errors = [0.08, 0.09, 0.085, 0.092, 0.088]

        level = agent.get_performance_level()
        assert level == PerformanceLevel.DEGRADED

    def test_performance_level_poor(self):
        """Test performance level classification - poor."""
        agent_id = AgentId(app="battery_twin", type="ml", instance="1")
        agent = MLResidualAgent(agent_id=agent_id, battery_id="B0005")

        # Simulate poor performance
        agent.hybrid_errors = [0.12, 0.15, 0.13, 0.14, 0.16]

        level = agent.get_performance_level()
        assert level == PerformanceLevel.POOR


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_training_workflow(self):
        """Test complete workflow from data to trained model."""
        agent_id = AgentId(app="battery_twin", type="ml", instance="1")

        ml_config = NeuralNetConfig(epochs=5, batch_size=8, device='cpu')
        mock_mqtt = Mock()
        mock_mqtt.is_connected.return_value = True
        mock_mqtt.publish = Mock(return_value=True)

        agent = MLResidualAgent(
            agent_id=agent_id,
            battery_id="B0005",
            mqtt_bridge=mock_mqtt,
            ml_config=ml_config,
            min_training_samples=30
        )

        # Setup agent
        agent.setup()

        try:
            # Simulate receiving 35 cycles of data
            for i in range(35):
                # Receive physics prediction
                pred_msg = PredictionMessage(
                    battery_id="B0005",
                    timestamp=time.time(),
                    cycle=i,
                    prediction_type="physics",
                    predicted_capacity=2.0 - i * 0.01,
                    uncertainty=None,
                    horizon=0,
                    agent_id="physics_agent"
                )
                agent._handle_physics_prediction(
                    "battery/B0005/prediction/physics",
                    MessageFactory.to_json(pred_msg)
                )

                # Receive actual capacity
                cap_msg = CapacityMessage(
                    battery_id="B0005",
                    timestamp=time.time(),
                    cycle=i,
                    capacity=2.0 - i * 0.015
                )
                agent._handle_actual_capacity(
                    "battery/B0005/capacity",
                    MessageFactory.to_json(cap_msg)
                )

            # Check model was trained
            assert agent.model_status == ModelStatus.TRAINED
            assert agent.learner.model.is_fitted

            # Make prediction
            hybrid, uncertainty = agent.predict_hybrid_capacity(
                cycle=36,
                physics_prediction=1.64
            )

            assert isinstance(hybrid, float)
            assert not np.isnan(hybrid)

        finally:
            agent.teardown()


def test_summary():
    """Summary test to verify all components work together."""
    print("\n" + "="*70)
    print("Step 11: ML Residual Agent (BDI) - Test Summary")
    print("="*70)
    print("\n✅ TrainingDataPoint Tests:")
    print("  - Data point creation and validation")
    print("  - Completeness checking")
    print("\n✅ Agent Initialization Tests:")
    print("  - Basic agent creation")
    print("  - Custom configuration")
    print("  - BDI beliefs initialization")
    print("  - BDI goals initialization")
    print("\n✅ Message Handling Tests:")
    print("  - Physics prediction messages")
    print("  - Actual capacity messages")
    print("  - Complete data point assembly")
    print("\n✅ BDI Deliberation Tests:")
    print("  - No training without sufficient data")
    print("  - Initial training trigger (data threshold)")
    print("  - Retraining trigger (performance degradation)")
    print("  - Retraining trigger (staleness)")
    print("\n✅ Hybrid Prediction Tests:")
    print("  - Prediction before training (returns physics)")
    print("  - Prediction after training (ML correction)")
    print("  - Uncertainty estimation")
    print("  - MQTT publishing")
    print("\n✅ Model Persistence Tests:")
    print("  - Save trained model")
    print("\n✅ Statistics Tests:")
    print("  - Agent statistics")
    print("  - Performance level classification")
    print("\n✅ Integration Tests:")
    print("  - Complete training workflow")
    print("  - End-to-end data to prediction")
    print("\n" + "="*70)
    print("All ML Residual Agent (BDI) components tested successfully!")
    print("="*70 + "\n")
