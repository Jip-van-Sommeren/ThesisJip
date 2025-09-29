# Communication Protocol Benchmarks

This directory contains a comprehensive benchmarking suite for comparing the performance of 4 communication protocols: **REST**, **gRPC**, **MQTT**, and **Kafka**.

## Overview

The benchmark suite provides standardized performance testing across all protocols with identical test scenarios, enabling fair comparison of:
- **Latency metrics**: Average, P95, P99 response times
- **Throughput metrics**: Messages per second, peak throughput
- **Reliability metrics**: Success rates, failure handling
- **Resource usage**: CPU and memory consumption
- **Scalability**: Performance under different loads and topologies

## Architecture

### Core Components

1. **Protocol-Specific Scenarios** (`*_benchmark_scenarios.py`)
   - `rest_benchmark_scenarios.py` - REST API benchmarks
   - `grpc_benchmark_scenarios.py` - gRPC RPC benchmarks
   - `mqtt_benchmark_scenarios.py` - MQTT pub/sub benchmarks
   - `kafka_benchmark_scenarios.py` - Kafka streaming benchmarks

2. **Unified Runner** (`benchmark_runner.py`)
   - Cross-protocol benchmark execution
   - Simple and extensive test modes
   - Structured output generation

3. **Analysis Tools** (`benchmark_analysis.py`)
   - Visualization generation
   - Performance comparison charts
   - Statistical analysis

4. **Testing Tools** (`test_benchmark_consistency.py`)
   - Consistency verification across protocols
   - Integration testing

### Standardized Test Scenarios

All protocols implement these 4 identical scenarios:

1. **Point-to-Point Latency**: Basic message latency between two agents
2. **Broadcast Throughput**: One-to-many message broadcasting performance
3. **Concurrent Messaging**: Multi-agent concurrent communication
4. **Scalability Stress**: High-load performance testing

## Quick Start

### Simple Benchmarks (Quick)
```bash
# Run basic benchmarks for all protocols
python benchmark_runner.py --simple

# Run specific protocols only
python benchmark_runner.py --simple --protocols rest grpc

# Run specific scenarios only
python benchmark_runner.py --simple --scenarios point_to_point_latency concurrent_messaging
```

### Extensive Benchmarks (Comprehensive)
```bash
# Full comprehensive benchmarks
python benchmark_runner.py --extensive

# Extensive benchmarks for specific protocols
python benchmark_runner.py --extensive --protocols mqtt kafka
```

### Generate Analysis and Visualizations
```bash
# Generate all plots and analysis
python benchmark_analysis.py

# Generate specific visualization
python benchmark_analysis.py --plot latency
python benchmark_analysis.py --plot throughput
python benchmark_analysis.py --plot radar
```

### Test Benchmark Consistency
```bash
# Verify all protocols implement consistent benchmarks
python test_benchmark_consistency.py
```

## Detailed Usage

### Benchmark Runner Options

```bash
python benchmark_runner.py [OPTIONS]

Options:
  --protocols PROTOCOL [PROTOCOL ...]
                        Protocols to benchmark: rest, grpc, mqtt, kafka
                        (default: all)

  --scenarios SCENARIO [SCENARIO ...]
                        Scenarios to run: point_to_point_latency,
                        broadcast_throughput, concurrent_messaging,
                        scalability_stress (default: all)

  --simple              Run simple benchmarks (quick, basic parameters)
  --extensive           Run extensive benchmarks (comprehensive, multiple configs)
  --output-dir DIR      Output directory for results (default: results)
  --no-csv              Skip CSV export
  --no-json             Skip JSON export
```

### Analysis Tool Options

```bash
python benchmark_analysis.py [OPTIONS]

Options:
  --results-file FILE   Specific benchmark results JSON file to analyze
  --results-dir DIR     Directory containing results (default: results)
  --plot TYPE           Plot type: latency, throughput, resources, radar,
                        ranking, all (default: all)
```

## Benchmark Configurations

### Simple Mode
- **Agent counts**: 5 agents
- **Topologies**: Fully connected only
- **Duration**: 3-5 seconds per test
- **Messages**: 30-50 messages per scenario
- **Total time**: ~5-10 minutes for all protocols

### Extensive Mode
- **Agent counts**: 3, 5, 8, 12 agents
- **Topologies**: Fully connected, star, ring, chain
- **Duration**: 5-15 seconds per test
- **Messages**: 50-100 messages per scenario
- **Total time**: ~30-60 minutes for all protocols

## Output Formats

### JSON Results
- `benchmark_results_YYYYMMDD_HHMMSS.json` - Complete results with metadata
- `comparison_summary_YYYYMMDD_HHMMSS.json` - Cross-protocol comparison data

### CSV Exports (for plotting tools)
- `latency_comparison_YYYYMMDD_HHMMSS.csv` - Latency metrics by protocol/scenario
- `throughput_comparison_YYYYMMDD_HHMMSS.csv` - Throughput and success rates
- `resource_usage_YYYYMMDD_HHMMSS.csv` - CPU and memory usage data

### Visualizations
- `latency_comparison.png` - Average and P95 latency charts
- `throughput_comparison.png` - Throughput and success rate charts
- `resource_usage.png` - CPU and memory usage charts
- `performance_radar.png` - Multi-dimensional performance comparison
- `protocol_ranking.png` - Performance ranking table
- `benchmark_summary_report.txt` - Text summary with recommendations

## Metrics Explained

### Latency Metrics
- **Average Latency**: Mean message round-trip time
- **P95 Latency**: 95th percentile (95% of messages faster than this)
- **P99 Latency**: 99th percentile (99% of messages faster than this)

### Throughput Metrics
- **Messages/Second**: Average message processing rate
- **Peak Throughput**: Maximum observed throughput
- **Success Rate**: Percentage of successfully delivered messages

### Resource Metrics
- **CPU Usage**: Percentage of CPU utilization during test
- **Memory Usage**: Memory consumption in MB
- **Delivery Failures**: Number of failed message deliveries
- **Timeout Failures**: Number of communication timeouts

### Scalability Metrics
- **Agent Count**: Number of communicating agents
- **Topology Density**: Ratio of active links to possible links
- **Total Messages**: Total messages processed during test

## Dependencies

### Required Python Packages
```bash
pip install matplotlib seaborn pandas numpy
```

### Protocol Dependencies
- **REST**: `requests`, `flask`
- **gRPC**: `grpcio`, `grpcio-tools`
- **MQTT**: `paho-mqtt`
- **Kafka**: `kafka-python`

### System Dependencies
- **MQTT**: Requires MQTT broker (e.g., Mosquitto)
- **Kafka**: Requires Kafka server and Zookeeper

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure you're in the correct directory
   cd /path/to/thesis/src
   python -m benchmarks.benchmark_runner --simple
   ```

2. **Protocol Connection Failures**
   - **MQTT**: Ensure broker is running on localhost:1883
   - **Kafka**: Ensure Kafka is running on localhost:9092
   - **gRPC**: Ports 50051+ should be available
   - **REST**: Ports 5000+ should be available

3. **Memory Issues with Extensive Benchmarks**
   ```bash
   # Run protocols individually
   python benchmark_runner.py --extensive --protocols rest
   python benchmark_runner.py --extensive --protocols grpc
   ```

4. **Permission Errors**
   ```bash
   # Ensure results directory is writable
   mkdir -p results
   chmod 755 results
   ```

### Debug Mode
```bash
# Test benchmark consistency first
python test_benchmark_consistency.py

# Run individual protocol benchmarks
python rest_benchmark_scenarios.py
python grpc_benchmark_scenarios.py
python mqtt_benchmark_scenarios.py
python kafka_benchmark_scenarios.py
```

## Example Workflow

```bash
# 1. Test consistency
python test_benchmark_consistency.py

# 2. Run simple benchmarks
python benchmark_runner.py --simple

# 3. Generate analysis
python benchmark_analysis.py

# 4. Run extensive benchmarks (optional)
python benchmark_runner.py --extensive

# 5. Generate comprehensive analysis
python benchmark_analysis.py
```

## Integration with Research

### For Thesis Analysis
The benchmark results provide quantitative data for:
- Protocol performance comparison tables
- Scalability analysis graphs
- Resource utilization studies
- Reliability assessment metrics

### Data Export for External Tools
- **CSV files**: Import into Excel, R, or Python for further analysis
- **JSON files**: Process with custom analysis scripts
- **PNG plots**: Include directly in research papers

## Contributing

When adding new protocols or scenarios:

1. Follow the standardized scenario structure
2. Implement all 4 required test scenarios
3. Use the same parameter names and types
4. Include proper setup/teardown functions
5. Run consistency tests to verify integration

## License

This benchmark suite is part of the thesis research project and follows the same license as the main project.