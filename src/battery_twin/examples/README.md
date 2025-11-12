# Battery Digital Twin Examples

This directory contains example scripts demonstrating how to use the Battery Digital Twin system.

## Examples Overview

### Example 1: Basic Monitoring (`example_01_basic_monitoring.py`)

**Purpose**: Demonstrates basic battery monitoring with core agents.

**Features**:
- Telemetry ingestion
- State estimation (SoC/SoH)
- Health monitoring

**Agents**: TelemetryIngestor, StateEstimator, HealthMonitor

**Usage**:
```bash
python3 src/battery_twin/examples/example_01_basic_monitoring.py
```

**Duration**: 5 minutes

---

### Example 2: Full System (`example_02_full_system.py`)

**Purpose**: Demonstrates complete system with all agents enabled.

**Features**:
- All monitoring capabilities
- Physics-based capacity prediction
- ML residual correction
- Hybrid predictions

**Agents**: All agents (TelemetryIngestor, StateEstimator, HealthMonitor, PhysicsModel, MLResidual)

**Usage**:
```bash
python3 src/battery_twin/examples/example_02_full_system.py
```

**Duration**: 10 minutes

---

### Example 3: Data Replay (`example_03_data_replay.py`)

**Purpose**: Replay historical NASA battery data through the system.

**Features**:
- NASA dataset loading
- Data replay at 10x speed
- Validates system with real battery cycles
- Performance monitoring

**Prerequisites**: NASA dataset at `Digital-Twin-in-python/data/raw/discharge.csv`

**Usage**:
```bash
python3 src/battery_twin/examples/example_03_data_replay.py
```

**What it does**:
1. Loads 50 cycles from NASA B0005 battery
2. Replays data through the digital twin at 10x speed
3. Collects state estimates and predictions
4. Shows processing statistics

---

### Example 4: Subscribe to Messages (`example_04_subscribe_to_messages.py`)

**Purpose**: Demonstrates MQTT message subscription and real-time monitoring.

**Features**:
- MQTT topic subscription
- Real-time message processing
- Custom message handlers
- Message statistics

**Usage**:
```bash
# Terminal 1: Start the subscriber
python3 src/battery_twin/examples/example_04_subscribe_to_messages.py

# Terminal 2: Start the orchestrator
python3 -m src.battery_twin.orchestrator
```

**Messages Monitored**:
- `battery/B0005/telemetry/clean` - Clean telemetry data
- `battery/B0005/state/estimate` - State estimates (SoC, SoH)
- `battery/B0005/health/report` - Health reports
- `battery/B0005/prediction/hybrid` - Predictions

---

## Prerequisites

### All Examples
- Python 3.10+
- MQTT broker running (Mosquitto)
- Virtual environment activated

### Example 3 Only
- NASA battery dataset downloaded
- Dataset placed at: `Digital-Twin-in-python/data/raw/discharge.csv`

## Setup

1. **Activate virtual environment:**
   ```bash
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate      # Windows
   ```

2. **Ensure MQTT broker is running:**
   ```bash
   sudo systemctl status mosquitto
   # If not running:
   sudo systemctl start mosquitto
   ```

3. **For Example 3, download NASA dataset:**
   - Download from: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
   - Place `discharge.csv` in `Digital-Twin-in-python/data/raw/`

## Running Examples

### Quick Start

Run any example directly:
```bash
python3 src/battery_twin/examples/example_01_basic_monitoring.py
```

### With Custom Configuration

Modify the configuration in the example script:
```python
config = BatteryTwinConfig(
    battery_id='B0005',
    enable_telemetry_ingestor=True,
    enable_state_estimator=True,
    enable_health_monitor=True,
    log_level='INFO'
)
```

### Early Termination

All examples support early termination with `Ctrl+C`.

## Output

### Console Output

Examples produce structured console output showing:
- System initialization
- Agent status
- Real-time messages
- Statistics

### Example Output (Example 1):

```
================================================================================
EXAMPLE 1: Basic Battery Monitoring
================================================================================
Creating configuration...
✓ Configuration created for battery B0005
Creating orchestrator...
Initializing system...
✓ System initialized
Starting agents...
✓ All agents running
System State: RUNNING
Agents Running: 3
  - telemetry_ingestor: RUNNING
  - state_estimator: RUNNING
  - health_monitor: RUNNING
Running system for 5 minutes...
Press Ctrl+C to stop early
...
```

## Common Issues

### MQTT Connection Failed

**Error**: `Failed to connect to MQTT broker`

**Solution**:
```bash
# Check if Mosquitto is running
sudo systemctl status mosquitto

# Start Mosquitto if needed
sudo systemctl start mosquitto
```

### Dataset Not Found (Example 3)

**Error**: `NASA dataset not found!`

**Solution**:
1. Download NASA battery dataset
2. Place `discharge.csv` at: `Digital-Twin-in-python/data/raw/discharge.csv`
3. Or modify the example to use a custom path:
   ```python
   loader = NASABatteryLoader(dataset_path="/path/to/discharge.csv")
   ```

### No Messages Received (Example 4)

**Issue**: Subscriber receives no messages

**Solution**:
1. Ensure the orchestrator is running in another terminal
2. Check MQTT broker is accessible
3. Verify battery_id matches between subscriber and orchestrator

## Next Steps

After running the examples, try:

1. **Modify configurations** to enable/disable different agents
2. **Change battery ID** to test with different NASA batteries (B0006, B0007, etc.)
3. **Adjust replay speed** in Example 3 to test different throughputs
4. **Add custom message handlers** in Example 4 to process specific data

## Advanced Usage

### Running with Storage

Enable persistent storage in examples:
```python
config = BatteryTwinConfig(
    battery_id='B0005',
    enable_storage=True,
    storage_config_path='src/battery_twin/config/storage_config.yaml',
    ...
)
```

**Prerequisite**: Storage backends running in Docker:
```bash
docker-compose up -d
```

### Custom Duration

Modify the run duration:
```python
await orchestrator.run(duration=600.0)  # 10 minutes
```

### Different Log Levels

```python
config = BatteryTwinConfig(
    log_level='DEBUG',  # More detailed output
    # or
    log_level='WARNING',  # Less output
    ...
)
```

## See Also

- Main documentation: `src/battery_twin/README.md`
- Configuration guide: `src/battery_twin/config/README.md`
- Validation tools: `src/battery_twin/validation/`
- Integration tests: `src/battery_twin/tests/test_step15_integration.py`
