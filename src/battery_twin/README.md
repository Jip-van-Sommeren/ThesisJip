# Battery Digital Twin System

A formal multi-agent system for real-time battery state estimation, health monitoring, and capacity prediction using BDI (Belief-Desire-Intention) and reactive agent architectures.

## Overview

The Battery Digital Twin is a comprehensive system that combines:
- **State Estimation**: Extended Kalman Filter (EKF) for SoC/SoH estimation
- **Health Monitoring**: Degradation tracking and RUL prediction
- **Capacity Prediction**: Hybrid physics-based + ML models
- **Real-time Processing**: MQTT-based agent communication
- **Multi-agent Architecture**: Reactive, BDI, and Hybrid agents

### Key Features

✨ **Accurate State Estimation**
- Extended Kalman Filter with 6-state vector [SoC, SoH, R0, R1, C1, V1]
- Uncertainty quantification with covariance tracking
- Divergence detection and adaptive tuning
- SoC RMSE: ~3%, SoH RMSE: ~2%

🏥 **Intelligent Health Monitoring**
- Capacity fade analysis
- Internal resistance tracking
- Remaining Useful Life (RUL) estimation
- Alert generation for anomalies

🔮 **Hybrid Prediction**
- Physics-based capacity model (battery degradation)
- ML residual correction (PyTorch neural network)
- Online learning with experience replay
- Catastrophic forgetting mitigation

🤖 **Formal Agent Architecture**
- **ReactiveAgent**: Pure stimulus-response, no goals
- **BDIAgent**: Full deliberation with beliefs, desires, intentions
- **HybridAgent**: Layered decision-making

📡 **Real-time Communication**
- MQTT message passing between agents
- QoS levels for reliability
- Topic-based routing
- Async/await architecture

💾 **Multi-backend Storage** (Optional)
- InfluxDB: Time-series metrics
- MongoDB: Agent profiles and configurations
- Neo4j: Hierarchy relationships
- Redis: Fast state caching

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Battery Digital Twin System                   │
│                     (BatteryTwinOrchestrator)                    │
└─────────────────────────────────────────────────────────────────┘
                               │
                               │ MQTT Communication
                               │
    ┌──────────────────────────┴──────────────────────────┐
    │                                                      │
    ▼                                                      ▼
┌─────────────────┐                            ┌──────────────────┐
│  Telemetry      │                            │  State           │
│  Ingestor       │──────────────────────────▶│  Estimator       │
│  (Reactive)     │     cleaned telemetry      │  (BDI + EKF)     │
└─────────────────┘                            └──────────────────┘
                                                        │
                                                        │ state estimates
                                                        │
                                    ┌───────────────────┴─────────────┐
                                    │                                 │
                                    ▼                                 ▼
                          ┌──────────────────┐           ┌──────────────────┐
                          │  Health          │           │  Physics         │
                          │  Monitor         │           │  Model           │
                          │  (BDI)           │           │  (Hybrid Agent)  │
                          └──────────────────┘           └──────────────────┘
                                                                  │
                                                                  │ physics predictions
                                                                  ▼
                                                         ┌──────────────────┐
                                                         │ Orchestrator     │
                                                         │ Hybrid Service   │
                                                         │ (HybridDigitalTwin) │
                                                         └──────────────────┘
                                                                  │
                                                                  │ hybrid predictions + training samples
                                                                  ▼
                                                         ┌──────────────────┐
                                                         │  ML Residual     │
                                                         │  Correction      │
                                                         │  (BDI + ML)      │
                                                         └──────────────────┘
```

### Hybrid Digital Twin Service

The orchestrator embeds the HybridDigitalTwin models (physics + ML residual) directly from `src/battery_twin/hybrid/`. It:

- Subscribes to clean telemetry and physics prediction topics to build consistent cycle summaries.
- Buffers per-cycle averages/durations so both historic data replay and live telemetry share the same feature shape.
- Publishes hybrid predictions to `battery/{id}/prediction/hybrid` and persists training samples in MongoDB (`hybrid_training_samples`) and predictions in InfluxDB.
- Exposes a lightweight API (`train_hybrid_twin`, `predict_hybrid_capacity`) that agents use instead of instantiating duplicate models.

### Agent Descriptions

| Agent | Type | Purpose | Key Features |
|-------|------|---------|--------------|
| **Telemetry Ingestor** | Reactive | Data validation and cleaning | Outlier detection, gap filling, publishing |
| **State Estimator** | BDI | Real-time SoC/SoH estimation | EKF, uncertainty tracking, divergence handling |
| **Health Monitor** | BDI | Battery health assessment | Degradation analysis, RUL estimation, alerts |
| **Physics Model** | Hybrid | Capacity prediction | Physics-based degradation model |
| **Orchestrator Hybrid Service** | Shared | Runs HybridDigitalTwin, publishes hybrid predictions | Buffers telemetry per cycle, persists training samples |
| **ML Residual** | BDI | Hybrid prediction | Neural network correction, online learning |

## Installation

### Prerequisites

- Python 3.10+
- Docker and Docker Compose (for storage backends)
- MQTT broker (Mosquitto recommended)

### Setup

1. **Clone repository and activate virtual environment:**
   ```bash
   cd /path/to/thesis
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start storage backends (optional):**
   ```bash
   docker-compose up -d
   ```
   This starts InfluxDB, MongoDB, Neo4j, Redis, and monitoring tools.

4. **Download NASA battery dataset:**
   - Place `discharge.csv` in `Digital-Twin-in-python/data/raw/`
   - Dataset: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

## Quick Start

### 1. Basic Usage (Default Configuration)

Run the system with default configuration (core monitoring pipeline):

```bash
python3 -m src.battery_twin.orchestrator
```

This starts:
- Telemetry Ingestor
- State Estimator
- Health Monitor

### 2. Full System (All Agents)

Run with all agents enabled:

```bash
python3 -m src.battery_twin.orchestrator --config src/battery_twin/config/full_system.yaml
```

This includes physics model and ML residual agents.

### 3. Custom Configuration

```bash
python3 -m src.battery_twin.orchestrator \
    --config src/battery_twin/config/test.yaml \
    --battery-id B0005 \
    --duration 300 \
    --log-level DEBUG
```

### 4. Hybrid Validation Run

```bash
python3 -m pytest src/battery_twin/tests/test_step17_hybrid_pipeline.py -v
```

This integration test streams synthetic telemetry, physics predictions, and capacity measurements to verify the orchestrator publishes hybrid predictions and stores training samples end-to-end.

### 4. Python API

```python
import asyncio
from src.battery_twin.orchestrator import BatteryTwinConfig, BatteryTwinOrchestrator

async def main():
    # Load configuration
    config = BatteryTwinConfig.from_yaml('src/battery_twin/config/default.yaml')

    # Or create programmatically
    config = BatteryTwinConfig(
        battery_id='B0005',
        enable_health_monitor=True,
        enable_physics_model=True,
        log_level='INFO'
    )

    # Create and run orchestrator
    orchestrator = BatteryTwinOrchestrator(config)
    await orchestrator.initialize()
    await orchestrator.start()

    # Run until Ctrl+C
    await orchestrator.run()

    # Or run for specific duration
    # await orchestrator.run(duration=60.0)

if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration

### Configuration Files

Three example configurations are provided in `src/battery_twin/config/`:

**`default.yaml`** - Core monitoring pipeline:
```yaml
battery_id: "B0005"
enable_telemetry_ingestor: true
enable_state_estimator: true
enable_health_monitor: true
enable_physics_model: false
enable_ml_residual: false
enable_storage: false
log_level: "INFO"
```

**`full_system.yaml`** - All agents enabled:
```yaml
battery_id: "B0005"
enable_telemetry_ingestor: true
enable_state_estimator: true
enable_health_monitor: true
enable_physics_model: true
enable_ml_residual: true
enable_storage: true
log_level: "DEBUG"
```

**`test.yaml`** - Minimal configuration for testing

### Key Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `battery_id` | Battery identifier | "B0005" |
| `mqtt_broker` | MQTT broker host | "localhost" |
| `mqtt_port` | MQTT broker port | 1883 |
| `enable_storage` | Enable persistent storage | false |
| `enable_telemetry_ingestor` | Enable telemetry agent | true |
| `enable_state_estimator` | Enable EKF state estimation | true |
| `enable_health_monitor` | Enable health monitoring | true |
| `enable_physics_model` | Enable physics predictions | false |
| `enable_ml_residual` | Enable ML corrections | false |
| `ekf_initial_soc` | Initial State of Charge | 0.8 |
| `ekf_initial_soh` | Initial State of Health | 1.0 |
| `ekf_capacity_nominal` | Nominal capacity (Ah) | 2.0 |
| `health_eol_threshold` | End-of-life SoH threshold | 0.8 |
| `log_level` | Logging level | "INFO" |

See `src/battery_twin/config/README.md` for complete documentation.

## Testing and Validation

### Run Integration Tests

```bash
# All integration tests
pytest src/battery_twin/tests/test_step15_integration.py -v

# Specific test
pytest src/battery_twin/tests/test_step15_integration.py::test_06_orchestrator_creation -v
```

**Test Coverage**: 30 integration tests (100% passing)
- Configuration management
- Orchestrator initialization
- Agent lifecycle
- Status monitoring
- Error handling
- Integration scenarios

### Run Validation Against NASA Dataset

```bash
python3 -m src.battery_twin.validation.validation_runner \
    --battery-id B0005 \
    --cycles 50 \
    --replay-speed 10.0
```

This validates:
- State estimation accuracy (SoC, SoH)
- Prediction performance (capacity, voltage)
- System reliability and throughput

**Output**: `src/battery_twin/validation/results/validation_report_*.txt`

### Run Performance Benchmark

```bash
python3 -m src.battery_twin.validation.performance_benchmark \
    --duration 60 \
    --rate 10.0
```

This measures:
- Message processing latency (min, max, mean, p95, p99)
- System throughput (messages/second)
- Resource utilization (CPU, memory)

**Output**: `src/battery_twin/validation/results/performance_benchmark_*.txt`

## Data Flow

### 1. Telemetry Ingestion

```
Raw Telemetry (battery/{id}/telemetry/raw)
    │
    ▼
TelemetryIngestorAgent
    │ • Validate measurements
    │ • Detect outliers
    │ • Fill gaps
    ▼
Clean Telemetry (battery/{id}/telemetry/clean)
```

### 2. State Estimation

```
Clean Telemetry
    │
    ▼
StateEstimatorAgent (EKF)
    │ • Predict step
    │ • Update with measurement
    │ • Compute uncertainty
    ▼
State Estimate (battery/{id}/state/estimate)
    └─▶ {soc, soh, r0, r1, c1, v1, uncertainties}
```

### 3. Health Monitoring

```
State Estimate
    │
    ▼
HealthMonitorAgent
    │ • Track degradation trends
    │ • Estimate RUL
    │ • Generate alerts
    ▼
Health Report (battery/{id}/health/report)
    └─▶ {status, risk_level, rul, alerts}
```

### 4. Capacity Prediction (Optional)

```
State Estimate
    │
    ▼
PhysicsModelAgent
    │ • Apply degradation model
    │ • Predict capacity fade
    ▼
Physics Prediction (battery/{id}/prediction/physics)
    │
    ▼
MLResidualAgent
    │ • Compute error
    │ • Apply ML correction
    │ • Update model
    ▼
Hybrid Prediction (battery/{id}/prediction/hybrid)
```

## MQTT Topics

All communication uses MQTT with the following topic structure:

```
battery/{battery_id}/telemetry/raw         # Raw sensor data
battery/{battery_id}/telemetry/clean       # Validated telemetry
battery/{battery_id}/state/estimate        # SoC/SoH estimates
battery/{battery_id}/health/report         # Health status
battery/{battery_id}/health/alert          # Health alerts
battery/{battery_id}/prediction/physics    # Physics-based prediction
battery/{battery_id}/prediction/hybrid     # Hybrid prediction
battery/{battery_id}/command/*             # Agent commands
battery/{battery_id}/status/*              # Agent status
```

### Message Formats

All messages use JSON serialization with Pydantic validation.

**TelemetryMessage**:
```json
{
  "battery_id": "B0005",
  "cycle": 42,
  "timestamp": 1699123456.789,
  "voltage": 3.72,
  "current": 1.48,
  "temperature": 25.3,
  "soc": 0.85,
  "soh": 0.98
}
```

**StateEstimateMessage**:
```json
{
  "battery_id": "B0005",
  "cycle": 42,
  "timestamp": 1699123456.789,
  "soc": 0.847,
  "soh": 0.982,
  "r0": 0.012,
  "r1": 0.008,
  "c1": 1500.0,
  "v1": 0.05,
  "soc_uncertainty": 0.025,
  "soh_uncertainty": 0.018,
  "confidence_level": "HIGH"
}
```

## Performance

Based on benchmarking with NASA battery dataset (B0005):

### State Estimation
- SoC RMSE: **~3%**
- SoH RMSE: **~2%**
- Average latency: **~15 ms**
- Estimation rate: **10-20 Hz**

### System Throughput
- Message processing: **~100 msg/s**
- End-to-end latency (p99): **<50 ms**
- CPU usage (3 agents): **~15%**
- Memory usage: **~200 MB**

### Prediction Accuracy
- Capacity RMSE: **~5%**
- Voltage RMSE: **~3%**
- Hybrid improvement: **20-30%** over physics-only

## Examples

### Example 1: Real-time Monitoring

```python
import asyncio
from src.battery_twin.orchestrator import BatteryTwinConfig, BatteryTwinOrchestrator

async def monitor_battery():
    config = BatteryTwinConfig(
        battery_id='B0005',
        enable_telemetry_ingestor=True,
        enable_state_estimator=True,
        enable_health_monitor=True,
        log_level='INFO'
    )

    orchestrator = BatteryTwinOrchestrator(config)
    await orchestrator.initialize()
    await orchestrator.start()

    # Monitor for 5 minutes
    await orchestrator.run(duration=300)

asyncio.run(monitor_battery())
```

### Example 2: Data Replay from NASA Dataset

```python
import asyncio
from src.battery_twin.data.replay_engine import DataReplayEngine
from src.battery_twin.data.nasa_loader import NASABatteryLoader
from src.battery_twin.communication.mqtt_bridge import MqttBridge, MqttConfig

async def replay_battery_data():
    # Load NASA dataset
    loader = NASABatteryLoader()

    # Connect to MQTT
    mqtt_config = MqttConfig(broker="localhost", port=1883)
    mqtt_bridge = MqttBridge(mqtt_config)
    await mqtt_bridge.connect()

    # Create replay engine
    replay = DataReplayEngine(
        loader=loader,
        mqtt_bridge=mqtt_bridge,
        battery_ids=['B0005'],
        replay_speed=10.0  # 10x speed
    )

    # Replay 100 cycles
    await replay.replay_cycles(start_cycle=0, num_cycles=100)

    await mqtt_bridge.disconnect()

asyncio.run(replay_battery_data())
```

### Example 3: Subscribe to State Estimates

```python
import asyncio
import json
from src.battery_twin.communication.mqtt_bridge import MqttBridge, MqttConfig

async def monitor_state_estimates():
    mqtt_config = MqttConfig(broker="localhost", port=1883)
    mqtt_bridge = MqttBridge(mqtt_config)
    await mqtt_bridge.connect()

    # Subscribe to state estimates
    topic = "battery/B0005/state/estimate"

    def on_estimate(message):
        estimate = json.loads(message)
        print(f"Cycle {estimate['cycle']}: "
              f"SoC={estimate['soc']:.3f}, "
              f"SoH={estimate['soh']:.3f}")

    await mqtt_bridge.subscribe(topic, on_estimate)

    # Listen for 60 seconds
    await asyncio.sleep(60)

    await mqtt_bridge.disconnect()

asyncio.run(monitor_state_estimates())
```

## Troubleshooting

### MQTT Connection Issues

**Problem**: "Failed to connect to MQTT broker"

**Solutions**:
1. Ensure Mosquitto is running: `sudo systemctl status mosquitto`
2. Start Mosquitto: `sudo systemctl start mosquitto`
3. Check broker host/port in configuration
4. Verify firewall rules allow port 1883

### Dataset Not Found

**Problem**: "Dataset file not found: Digital-Twin-in-python/data/raw/discharge.csv"

**Solutions**:
1. Download NASA battery dataset
2. Place `discharge.csv` in correct location
3. Or specify custom path in config:
   ```python
   loader = NASABatteryLoader(dataset_path="/path/to/discharge.csv")
   ```

### Storage Connection Errors

**Problem**: "Failed to connect to InfluxDB/MongoDB"

**Solutions**:
1. Start storage backends: `docker-compose up -d`
2. Check container status: `docker-compose ps`
3. Or disable storage in config: `enable_storage: false`

### High CPU/Memory Usage

**Problem**: System using too many resources

**Solutions**:
1. Disable unused agents (physics_model, ml_residual)
2. Reduce message rate in replay engine
3. Disable storage (`enable_storage: false`)
4. Set log level to WARNING or ERROR

## Development

### Project Structure

```
src/battery_twin/
├── agents/                    # Agent implementations
│   ├── telemetry_ingestor_agent.py
│   ├── state_estimator_agent.py
│   ├── health_monitor_agent.py
│   ├── physics_model_agent.py
│   └── ml_residual_agent.py
├── communication/             # MQTT communication
│   ├── mqtt_bridge.py
│   └── message_schemas.py
├── config/                    # Configuration files
│   ├── default.yaml
│   ├── full_system.yaml
│   ├── test.yaml
│   └── README.md
├── data/                      # Data loading and replay
│   ├── nasa_loader.py
│   ├── replay_engine.py
│   └── data_pipeline.py
├── models/                    # Core models
│   ├── extended_kalman_filter.py
│   ├── battery_degradation_model.py
│   └── neural_network.py
├── storage/                   # Storage backends
│   └── battery_storage_manager.py
├── tests/                     # Integration tests
│   └── test_step15_integration.py
├── validation/                # Validation and benchmarking
│   ├── validation_runner.py
│   └── performance_benchmark.py
├── orchestrator.py            # System orchestrator
└── README.md                  # This file
```

### Running Tests

```bash
# All tests
pytest src/battery_twin/tests/ -v

# Specific module
pytest src/battery_twin/tests/test_step15_integration.py -v

# With coverage
pytest src/battery_twin/tests/ --cov=src.battery_twin --cov-report=html
```

### Adding New Agents

1. Create agent class inheriting from BatteryBDIAgent or BatteryReactiveAgent
2. Implement required methods (perception, decision, action)
3. Subscribe to relevant MQTT topics
4. Add agent creation to orchestrator
5. Add configuration flag for enabling/disabling
6. Write unit and integration tests

## Citation

If you use this system in your research, please cite:

```bibtex
@mastersthesis{battery_digital_twin_2025,
  title={Battery Digital Twin: A Formal Multi-Agent System for Real-time State Estimation and Health Monitoring},
  author={Your Name},
  year={2025},
  school={Your University}
}
```

## License

[Your License Here]

## Contact

For questions or support, please contact [Your Email].

## Acknowledgments

- NASA PCoE Battery Aging Dataset
- Formal multi-agent system architecture from thesis framework
- Extended Kalman Filter implementation references
