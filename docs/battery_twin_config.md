# Battery Digital Twin Configuration

This directory contains configuration files for the Battery Digital Twin system.

## Configuration Files

### `default.yaml`
Default configuration with core monitoring pipeline:
- Telemetry Ingestor
- State Estimator (EKF)
- Health Monitor

Use for: Basic real-time battery monitoring

### `full_system.yaml`
Complete system configuration with all agents:
- All core agents
- Physics Model
- ML Residual Model
- Storage enabled
- Detailed logging

Use for: Full hybrid digital twin with predictions

### `test.yaml`
Minimal configuration for testing:
- Core agents only
- No storage
- Fast metrics intervals
- Debug logging

Use for: Integration testing and development

## Configuration Parameters

### Battery Identification
- `battery_id`: Unique identifier for the battery being monitored

### MQTT Configuration
- `mqtt_broker`: MQTT broker hostname (default: localhost)
- `mqtt_port`: MQTT broker port (default: 1883)
- `mqtt_keepalive`: Connection keepalive in seconds

### Storage Configuration
- `enable_storage`: Enable persistent storage (true/false)
- `storage_config_path`: Path to storage backend configuration

When storage is enabled the orchestrator writes:
- InfluxDB measurement `battery_predictions` with physics/ml/hybrid predictions.
- Mongo collection `hybrid_training_samples` that captures per-cycle training features (temperature, duration, measured capacity, source).

### Agent Enable Flags
- `enable_telemetry_ingestor`: Enable telemetry data ingestion
- `enable_state_estimator`: Enable EKF-based state estimation
- `enable_health_monitor`: Enable health monitoring and alerts
- `enable_physics_model`: Enable physics-based predictions
- `enable_ml_residual`: Enable ML residual corrections

### EKF Configuration
- `ekf_initial_soc`: Initial State of Charge (0.0-1.0)
- `ekf_initial_soh`: Initial State of Health (0.0-1.0)
- `ekf_capacity_nominal`: Nominal battery capacity in Ah

### Health Monitoring Configuration
- `health_initial_soh`: Initial SoH for health baseline
- `health_initial_r0`: Initial internal resistance in Ohms
- `health_eol_threshold`: End-of-life SoH threshold (default: 0.8)

### Logging Configuration
- `log_level`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `log_file`: Optional log file path (null for console only)

### Performance Monitoring
- `enable_metrics`: Enable metrics logging
- `metrics_interval`: Metrics reporting interval in seconds

### Hybrid Service
- The orchestrator always spins up a shared HybridDigitalTwin when TensorFlow is available. No extra flags are required, but hybrid predictions only publish when the service has been trained via telemetry/capacity replay.
- Telemetry buffering and training sample storage use the same configuration as above; ensure MongoDB is enabled if you want to persist `hybrid_training_samples`.

## Usage

### Command Line
```bash
# Use default configuration
python3 -m src.battery_twin.orchestrator

# Use specific configuration
python3 -m src.battery_twin.orchestrator --config src/battery_twin/config/full_system.yaml

# Override specific parameters
python3 -m src.battery_twin.orchestrator --config default.yaml --battery-id B0006 --log-level DEBUG
```

### Python API
```python
from src.battery_twin.orchestrator import BatteryTwinConfig, BatteryTwinOrchestrator

# Load from YAML
config = BatteryTwinConfig.from_yaml('src/battery_twin/config/default.yaml')

# Or create programmatically
config = BatteryTwinConfig(
    battery_id='B0005',
    enable_health_monitor=True,
    log_level='INFO'
)

# Create and run orchestrator
orchestrator = BatteryTwinOrchestrator(config)
await orchestrator.initialize()
await orchestrator.start()
await orchestrator.run()
```

## Adding Custom Configurations

1. Create a new YAML file in this directory
2. Copy from an existing template
3. Modify parameters as needed
4. Reference the file with `--config` flag

Example:
```yaml
battery_id: "MY_BATTERY"
mqtt_broker: "mqtt.example.com"
enable_storage: true
log_level: "INFO"
```
