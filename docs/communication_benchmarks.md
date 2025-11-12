# Communication Benchmarks: REST, gRPC, MQTT, Kafka

This document explains the communication benchmark suite used to evaluate different transport protocols (REST, gRPC, MQTT, Kafka) under a consistent set of scenarios and metrics. It covers what is measured, how each protocol’s scenarios are implemented, how to run them, and how to customize and compare results.

- Core runner: `src/benchmarks/benchmark_runner.py`
- Protocol scenarios:
  - REST: `src/benchmarks/rest_benchmark_scenarios.py`
  - gRPC: `src/benchmarks/grpc_benchmark_scenarios.py`
  - MQTT: `src/benchmarks/mqtt_benchmark_scenarios.py`
  - Kafka: `src/benchmarks/kafka_benchmark_scenarios.py`


## Goals and Design

The suite provides apples-to-apples comparisons across four protocols by:

- Reusing the same high-level benchmark scenarios for all protocols.
- Normalizing configuration via a single runner and a shared `CommunicationBenchmark` harness.
- Measuring consistent metrics (latency, throughput, reliability, resource usage) for each run.
- Producing JSON and CSV outputs for cross‑protocol analysis and plotting.

Implementation anchors:

- Scenario harness and metrics: `src/benchmarks/communication_benchmark.py`
- Topology patterns and configuration: `src/communication/communication_config.py`
- Protocol-specific environments and agents under `src/communication/{rest,grpc,mqtt,kafka}/`


## Benchmarked Scenarios (common across protocols)

Each protocol module defines the same four scenarios with protocol-specific plumbing underneath. Scenario names are stable and are referenced by the runner and YAML configs.

1) point_to_point_latency
- Two agents (sender → receiver) exchange messages to measure latency and reliability.
- Params: `agent_count` (2+), `message_count`, `payload_size_bytes`, `latency_mode`.
- Output: latency distribution and throughput under low contention.

2) broadcast_throughput
- One sender broadcasts to all other agents and measures broadcast throughput and completion.
- Params: `agent_count`, `message_count`, `payload_size_bytes`, `latency_mode`.
- Output: throughput and reliability as fan-out increases.

3) concurrent_messaging
- Multiple senders concurrently send to valid targets per the configured topology.
- Params: `agent_count`, `messages_per_agent`, `payload_size_bytes`, `latency_mode`, optional `concurrent_senders` (sweep via runner’s concurrency matrix).
- Output: latency, throughput, and failure behavior under contention.

4) scalability_stress
- All agents send at high rate for a fixed duration.
- Params: `agent_count`, `stress_duration`, `payload_size_bytes`, `latency_mode`.
- Output: sustained throughput, resource usage, and error rates under load.

The runner provides “simple” defaults and an “extensive” mode to sweep agent counts and topologies, plus a concurrency matrix for concurrent messaging.


## Latency Modes

All protocols support three latency measurement modes which can be selected runner-wide (`--latency-mode`) or per-scenario via overrides:

- send_only: Times the send/publish call on the sender side. Appropriate for strictly asynchronous transports.
- end_to_end: Uses the protocol’s default semantics to include delivery overhead inherent to the stack (e.g., HTTP/2 stream completion, gRPC unary roundtrip, broker acks for some settings). No explicit application‑level ACK is awaited.
- app_ack: Measures a full round trip including an application-level ACK sent by the receiver(s). Each protocol scenario implements a lightweight ACK loop using a designated ACK message type so the sender can stop timing only after receiving matching ACK(s).

Notes:
- For REST/gRPC, `end_to_end` generally reflects synchronous request completion semantics of the underlying transport; for MQTT/Kafka, effective behavior depends on QoS/acks settings.
- `app_ack` provides a strict, protocol‑agnostic round trip including receiver processing, making it the fairest for cross‑protocol comparisons when feasible.


## Metrics Collected

The harness captures a consistent set of metrics across scenarios (see `PerformanceMetrics` in `src/benchmarks/communication_benchmark.py`):

- Latency: avg, min, max, std, p50, p95, p99, p99.9, jitter; down‑sampled latency samples for plotting.
- Throughput: average and 1s‑window peak; per‑second history.
- Reliability: success rate, delivery failures, timeout failures; ordering violations/duplicates (when tracked).
- Resources: process CPU% and RSS memory MB sampled during the run.
- Test info: agent count, total messages, test duration, topology density, payload size, latency mode.

The runner then aggregates comparable fields into simple JSON/CSV summaries per scenario across protocols (see “Outputs”).


## Topology Patterns

Scenarios can run over different communication topologies configured via `TopologyPattern` (see `src/communication/communication_config.py`). Built‑ins include: `fully_connected`, `star`, `ring`, `chain` (plus others in the module). The setup phase builds the topology and installs the directed links (sender → receiver) into each protocol’s environment. A simple density statistic is reported to contextualize results.


## Protocol Modules: What Each Does

All protocol modules follow the same structure: create agents and an environment, build a topology, run the same four scenarios, and implement receiver ACK loops for `app_ack` mode.

### REST (`src/benchmarks/rest_benchmark_scenarios.py`)

- Environment/agents: `RestCommunicationEnvironment`, `ExtendedRestCommunicatingAgent`.
- Transport variants: `transport_mode` controls HTTP/1.1 vs HTTP/2 (`http1`/`http2`).
- ACKs: Uses `MessageType.ACK` for `app_ack` mode. A background receiver thread replies with ACKs; the sender waits for matching ACK IDs.
- Setup/teardown: Starts the REST service, registers agents, applies topology links, then stops the service and clears resources.

Key parameters:
- `transport_mode`: `http1` or `http2` (set by the runner’s REST variants).
- `payload_size_bytes`: controls message body size via `generate_payload` patterns.

### gRPC (`src/benchmarks/grpc_benchmark_scenarios.py`)

- Environment/agents: `GrpcCommunicationEnvironment`, `ExtendedGrpcCommunicatingAgent`.
- Transport variants: `grpc_mode` controls unary vs streaming (`unary`/`streaming`).
- ACKs: Uses `GrpcMessageType.ACK` in `app_ack` mode; receiver threads emit ACKs; senders await matching IDs.
- Setup/teardown: Starts the gRPC server, registers agents and mailboxes, applies topology links; then closes agents and stops service.

Key parameters:
- `grpc_mode`: `unary` or `streaming` (set by the runner’s gRPC variants).
- `payload_size_bytes`.

### MQTT (`src/benchmarks/mqtt_benchmark_scenarios.py`)

- Environment/agents: `MqttCommunicationEnvironment` with `env.create_agent(...)`.
- Broker management: Includes `is_mqtt_running`, `start_mqtt_docker`, `stop_mqtt_docker`. The runner will ensure the broker is up before timing; the module can self‑start Docker if run directly.
- QoS variants: `mqtt_qos` parameter (0/1/2) is used to match different delivery guarantees. The runner maps variants `qos0`, `qos1`, `qos2` to these integers.
- ACKs: Uses `MessageType.ACK` at the application level for `app_ack` mode.
- Setup/teardown: Starts the environment (against `localhost:1883`), registers agents, applies topology; then closes all agents and stops the service.

Key parameters:
- `mqtt_qos`: 0/1/2 (set by the runner’s MQTT variants).
- `payload_size_bytes`.

### Kafka (`src/benchmarks/kafka_benchmark_scenarios.py`)

- Environment/agents: `KafkaCommunicationEnvironment` with `env.create_agent(...)` and `env.setup()/teardown()`.
- Broker management: `is_kafka_running`, `start_kafka_docker`, `stop_kafka_docker`. The runner ensures Kafka is up before timing; the module can manage Docker if run directly.
- Acks variants: Producer `acks` behavior is configurable: `0`, `1`, or `all` (also accepts `-1`). The runner maps variants `acks0`, `acks1`, `acksall` to those values.
- ACKs: Uses `KafkaMessageType.ACK` at the application level for `app_ack` mode.
- Setup/teardown: Builds topology and links in the Kafka environment, then tears down cleanly.

Key parameters:
- `kafka_acks`: one of `0`, `1`, `all`/`-1` (set by the runner’s Kafka variants).
- `payload_size_bytes` and optional Kafka‑specific settings (e.g., compression) via YAML.


## The Benchmark Harness

`CommunicationBenchmark` (in `src/benchmarks/communication_benchmark.py`) provides:

- Scenario registry via `BenchmarkScenario` objects with `setup`, `test`, `teardown` callbacks.
- High‑precision `LatencyTracker` and 1s‑window `ThroughputTracker`.
- Lightweight `ResourceMonitor` sampling process CPU% and memory.
- Single‑trial and multi‑trial execution paths and helper exporters/summary printers.

Each scenario’s `test` function receives a reference to the harness to record timing (`latency_tracker.start/end_message_timing`) and throughput samples.


## Runner Orchestration and Variants

`src/benchmarks/benchmark_runner.py` is the unified entry point. It:

- Ensures brokers (MQTT/Kafka) are running before measurements (`_ensure_brokers_running`).
- Creates protocol‑specific benchmark factories: `rest`, `grpc`, `mqtt`, `kafka`.
- Iterates configured protocol variants and scenarios, gathering comparable metrics.
- Optionally sweeps topologies in extensive mode and runs a concurrency matrix for `concurrent_messaging`.
- Exports per‑protocol results and cross‑protocol CSV/JSON summaries, then prints a human‑readable summary.

Protocol variants and mapping (default set, all overridable):

- REST: `http1`, `http2` → `transport_mode`.
- gRPC: `unary`, `streaming` → `grpc_mode`.
- MQTT: `qos0`, `qos1`, `qos2` → `mqtt_qos` 0/1/2.
- Kafka: `acks0`, `acks1`, `acksall` → `kafka_acks` 0/1/"all" (also accepts `-1`).

Concurrency matrix:
- For `concurrent_messaging`, the runner can sweep `concurrent_senders` (default `[1, 4, 16, 64]`, per‑variant overrideable) to characterize throughput/latency scaling with active senders.


## Defaults: Simple vs. Extensive

Simple mode (quick sanity runs):
- Agent counts: typically `5` (or `2` for point‑to‑point).
- Scenario defaults:
  - `point_to_point_latency`: `message_count=50`.
  - `broadcast_throughput`: `message_count=30`.
  - `concurrent_messaging`: `messages_per_agent=100`.
  - `scalability_stress`: `stress_duration≈3.0s`.

Extensive mode (deeper sweeps):
- Agent counts: e.g., `[3, 5, 8]` by default (or custom via CLI/YAML).
- Topologies: full sweep across `fully_connected`, `star`, `ring`, `chain`.
- Additional concurrency matrix per variant for `concurrent_messaging`.

All of these can be overridden programmatically or via YAML (see below).


## Outputs

For each full run, the runner writes to `results/` (configurable via `--output-dir`). Files include a timestamp and the latency mode in their names.

- JSON (full results): `benchmark_results_{latencyMode}_{timestamp}.json`
- JSON (summary across protocols): `comparison_summary_{latencyMode}_{timestamp}.json`
- CSV (latency): `latency_comparison_{latencyMode}_{timestamp}.csv`
- CSV (throughput): `throughput_comparison_{latencyMode}_{timestamp}.csv`
- CSV (resources): `resource_usage_{latencyMode}_{timestamp}.csv`

Each CSV row includes protocol, scenario, latency mode, and the relevant metric. Protocol variants are labeled as `protocol::variant` (e.g., `grpc::streaming`).


## Results Overview

You ran the extensive benchmark suite for all three latency modes; the artifacts live here:

- send_only: `src/results/all_benchmarks_extensive_send_only`
- end_to_end: `src/results/all_benchmarks_extensive_end_to_end`
- app_ack: `src/results/all_benchmarks_extensive_app_ack`

What to look at:
- JSON: `benchmark_results_*.json` for full per‑protocol, per‑scenario metrics; `comparison_summary_*.json` for cross‑protocol aggregates.
- CSV: `latency_comparison_*.csv`, `throughput_comparison_*.csv`, `resource_usage_*.csv` for quick spreadsheet comparisons.
- Plots: `latency_cdf.png`, `p99_vs_concurrency.png`, `throughput_vs_payload.png`, `topology_comparison.png`, `performance_radar.png`, `protocol_ranking.png` (some plots only exist for certain runs).

High‑level takeaways (mode‑dependent):
- send_only: Measures publish/send cost only, so it typically shows the lowest latencies and highest throughput. Useful to isolate serialization and client overhead without delivery confirmation.
- end_to_end: Includes protocol overhead (e.g., request/response completion, stream flush), so latency and throughput reflect transport semantics more fully than send_only.
- app_ack: Adds an application‑level ACK roundtrip, so it usually increases latency and reduces throughput relative to the other modes while providing the fairest protocol‑agnostic comparison of end‑to‑end behavior.

Typical protocol patterns to verify against your plots/CSVs:
- REST: `http2` commonly outperforms `http1` under concurrency due to multiplexing, with smaller latency tails and better sustained throughput.
- gRPC: `streaming` generally amortizes per‑message overhead better than `unary` at higher concurrency levels.
- MQTT/Kafka: QoS/acks settings drive reliability/latency/throughput trade‑offs (e.g., higher QoS or `acks=all` improves delivery guarantees at the cost of latency and throughput). Compare `latency_cdf.png` and `p99_vs_concurrency.png` across variants.

Use the protocol ranking and radar plots to summarize trade‑offs, and the resource usage CSV/plots to keep results in context with CPU/RSS footprint.


### Representative Results (Concurrent Messaging)

The tables below summarize the concurrent_messaging scenario using average latency and throughput with success rate, extracted from the CSVs in each folder.

Send-only mode (`src/results/all_benchmarks_extensive_send_only`):

| Protocol Variant | Avg Latency (ms) | Avg Throughput (msg/s) | Success Rate (%) |
| --- | ---: | ---: | ---: |
| rest/http1 | 2.68 | 243.0 | 100.0 |
| rest/http2 | 2.38 | 246.2 | 100.0 |
| grpc/unary | 0.96 | 260.6 | 100.0 |
| grpc/streaming | 1.12 | 269.1 | 100.0 |
| mqtt/qos0 | 0.24 | 278.9 | 100.0 |
| mqtt/qos1 | 0.26 | 263.6 | 100.0 |
| mqtt/qos2 | 0.26 | 277.6 | 100.0 |
| kafka/acks0 | 1.17 | 245.6 | 100.0 |
| kafka/acks1 | 0.99 | 245.0 | 100.0 |
| kafka/acksall | 1.50 | 253.5 | 100.0 |

End-to-end mode (`src/results/all_benchmarks_extensive_end_to_end`):

| Protocol Variant | Avg Latency (ms) | Avg Throughput (msg/s) | Success Rate (%) |
| --- | ---: | ---: | ---: |
| rest/http1 | 9.51 | 484.5 | 100.0 |
| rest/http2 | 10.31 | 472.6 | 100.0 |
| grpc/unary | 0.82 | 826.7 | 100.0 |
| grpc/streaming | 1.52 | 784.5 | 100.0 |
| mqtt/qos0 | 0.17 | 856.5 | 100.0 |
| mqtt/qos1 | 0.18 | 835.3 | 100.0 |
| mqtt/qos2 | 0.19 | 858.4 | 100.0 |
| kafka/acks0 | 5.88 | 593.4 | 100.0 |
| kafka/acks1 | 6.99 | 558.7 | 100.0 |
| kafka/acksall | 6.81 | 555.5 | 100.0 |

App-ack mode (`src/results/all_benchmarks_extensive_app_ack`):

| Protocol Variant | Avg Latency (ms) | Avg Throughput (msg/s) | Success Rate (%) |
| --- | ---: | ---: | ---: |
| rest/http1 | 11.37 | 2.2 | 48.0 |
| rest/http2 | 7.71 | 2.1 | 48.4 |
| grpc/unary | 1.97 | 2.6 | 52.9 |
| grpc/streaming | 2.41 | 2.5 | 52.2 |
| mqtt/qos0 | 1.19 | 2.3 | 51.6 |
| mqtt/qos1 | 1.23 | 2.3 | 52.4 |
| mqtt/qos2 | 1.20 | 2.2 | 51.6 |
| kafka/acks0 | 70.51 | -0.0 | -0.8 |
| kafka/acks1 | 30.96 | 2.2 | 50.5 |
| kafka/acksall | 29.32 | 2.1 | 50.3 |

Note: The markedly lower throughput and success rates in app_ack reflect the stricter round‑trip requirement across all senders and receivers; brokered transports are sensitive to ACK fan‑in and consumer processing behavior under this mode.

## Running the Benchmarks

From the repository root:

Basic, all protocols, simple mode:

```
python3 src/benchmarks/benchmark_runner.py --simple
```

Run only communication benchmarks (skip hierarchy) and choose scenarios:

```
python3 src/benchmarks/benchmark_runner.py \
  --communication-only --simple \
  --scenarios point_to_point_latency concurrent_messaging
```

Extensive run for REST+gRPC with alternate latency mode:

```
python3 src/benchmarks/benchmark_runner.py \
  --communication-only --extensive --latency-mode app_ack \
  --protocols rest grpc
```

Override protocol variants via CLI:

```
python3 src/benchmarks/benchmark_runner.py \
  --communication-only --simple \
  --protocol-variants "rest=http1,http2" "grpc=unary,streaming" \
                      "mqtt=qos0,qos1" "kafka=acks1,acksall"
```


## YAML Configuration

Instead of many flags, use a YAML file to describe a matrix of variants, parameters, and per‑scenario overrides. Example files:

- `src/benchmarks/example_configs/fast_benchmark.yaml`
- `src/benchmarks/example_configs/extensive_benchmarks.yml`
- `src/benchmarks/example_configs/all_benchmarks_concurrency.yml`

Typical structure:

```yaml
mode: simple | extensive
latency_mode: send_only | end_to_end | app_ack
output_dir: results/...
scenarios: [point_to_point_latency, broadcast_throughput, concurrent_messaging, scalability_stress]
agent_counts: [5, 10, 15]  # optional

protocols:
  rest:
    concurrency_levels: [1, 4, 16]         # optional
    variants:
      http1:
        parameters:
          payload_size_bytes: 512          # protocol or scenario params
        scenarios:
          point_to_point_latency:
            message_count: 200             # per‑scenario overrides
  grpc:
    variants:
      streaming:
        parameters:
          grpc_mode: streaming
        scenarios:
          concurrent_messaging:
            messages_per_agent: 400
  mqtt:
    variants:
      qos1:
        parameters:
          mqtt_qos: 1
  kafka:
    variants:
      acksall:
        parameters:
          kafka_acks: all
          compression_type: lz4
```

Run with a YAML file:

```
python3 src/benchmarks/benchmark_runner.py --config-file src/benchmarks/example_configs/fast_benchmark.yaml
```

CLI flags still act as overrides when provided (e.g., `--protocols rest kafka`).


## Practical Notes and Fairness Considerations

- Brokers: The runner proactively ensures MQTT/Kafka brokers are up before starting the clock. If Docker is unavailable, start brokers manually on the default ports (`1883` for MQTT, `9092` for Kafka).
- QoS/Acks parity: For fair comparisons across async brokered transports, prefer MQTT `qos1` and Kafka `acks=1`; sweeping QoS/acks is supported to explore tradeoffs.
- Payload control: Use `payload_size_bytes` to examine bandwidth and serialization overhead effects.
- Topology realism: Use `star`/`ring`/`chain` to emulate constrained communication patterns; density is reported for context.
- Latency modes: Use `app_ack` when you need strict, protocol‑agnostic round‑trip measurements that include receiver processing; use `send_only` to isolate publish cost; rely on `end_to_end` when the transport’s synchronous semantics appropriately reflect delivery.


## Extending or Adding a Protocol

To add a new transport, mirror the existing modules:

1) Create `create_<proto>_benchmark_scenarios(latency_mode)` factory returning a `CommunicationBenchmark` with the four scenarios wired to your environment/agents.
2) Implement setup/teardown to start/stop services and apply topology links.
3) Implement `app_ack` receiver loops using a dedicated ACK message type; ensure senders wait for matching ACK IDs in that mode.
4) Register a factory in the runner and define sensible variant mappings (e.g., for transport modes/QoS/acks).


## File Map and References

- Runner orchestration and outputs: `src/benchmarks/benchmark_runner.py`
- Harness and metrics: `src/benchmarks/communication_benchmark.py`
- REST scenarios: `src/benchmarks/rest_benchmark_scenarios.py`
- gRPC scenarios: `src/benchmarks/grpc_benchmark_scenarios.py`
- MQTT scenarios + Docker helpers: `src/benchmarks/mqtt_benchmark_scenarios.py`
- Kafka scenarios + Docker helpers: `src/benchmarks/kafka_benchmark_scenarios.py`
- Topology patterns and configuration: `src/communication/communication_config.py`
