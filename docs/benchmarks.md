# Benchmarks

This project has two benchmark suites with different goals:

- **Communication benchmarks** measure real network performance across protocols (REST, gRPC, MQTT, Kafka).
- **Hierarchy benchmarks** measure coordination/organization strategies in a simulated environment (no network).

The separation is intentional: protocol benchmarks isolate transport costs and reliability, while hierarchy benchmarks isolate organizational trade-offs. Together they answer **which protocol** is best for a message pattern and **which coordination structure** scales best independent of the network.

---

## Communication benchmarks (real network)

### Why we run them
Multi-agent systems depend on message latency, throughput, and reliability. These benchmarks quantify how different protocols behave under representative workloads (point-to-point RPC, broadcast, concurrent peer traffic, and sustained stress). They provide apples-to-apples comparisons across protocols and highlight where protocol semantics (QoS, broker acks, request/response) change performance.

### Protocols and variants
Each protocol is exercised under a small set of variants to expose trade-offs:

- **REST**: HTTP/1.1 vs HTTP/2 (`http1`, `http2`).
- **gRPC**: unary vs streaming (`unary`, `streaming`).
- **MQTT**: QoS levels 0/1/2 (`qos0`, `qos1`, `qos2`).
- **Kafka**: producer acknowledgments 0/1/all (`acks0`, `acks1`, `acksall`).

Infrastructure is launched per protocol as needed:

- REST and gRPC start local servers.
- MQTT and Kafka can start Docker containers (`mqtt-benchmark`, `kafka-benchmark`).

### Latency modes
Three measurement modes are supported; each changes what "latency" means:

- **`send_only`**: time from sender invocation until the transport's default completion point.
- **`end_to_end`**: transport-level delivery confirmation. For MQTT this still measures publish completion (broker handoff), not receiver processing.
- **`app_ack`**: full application-level round trip (send -> receiver processing -> ACK -> sender receives ACK). This is the strictest, protocol-agnostic end-to-end measure.

### Scenarios (what we benchmark)
All protocols implement the same scenarios via `src/benchmarks/*_benchmark_scenarios.py`.

1) **Point-to-point latency** (`point_to_point_latency`)
   - **What**: sequential request/response between two agents.
   - **Why**: isolates per-message latency and tail behavior.

2) **Broadcast throughput** (`broadcast_throughput`)
   - **What**: one sender broadcasts to all other agents.
   - **Why**: models dissemination and notification patterns.
   - **Important**: there is a fixed **20 ms inter-broadcast pause** (`time.sleep(0.02)`), which caps broadcast rate at ~50 broadcasts/s. This is a deliberate rate limiter; throughput comparisons should be interpreted with this ceiling in mind.

3) **Concurrent messaging** (`concurrent_messaging`)
   - **What**: many agents send to random peers in parallel.
   - **Why**: stresses connection pooling, contention, and queue handling.
   - **Concurrency matrix**: additional sweep of concurrent senders (default **1, 4, 16, 64**). The runner ensures `agent_count >= concurrent_senders`.

4) **Scalability stress** (`scalability_stress`)
   - **What**: sustained high-rate sending for a fixed duration.
   - **Why**: measures degradation near saturation and recovery.

5) **Topology comparison** (`topology_comparison`)
   - **What**: runs `concurrent_messaging` over **Fully Connected, Star, Ring, Chain** topologies.
   - **Why**: shows how routing structure interacts with protocol behavior.

### Default parameters and overrides
The runner provides defaults (see `src/benchmarks/benchmark_runner.py`). These are often overridden via YAML for thesis runs.

**Runner defaults (simple vs extensive):**

- **Simple mode**
  - `point_to_point_latency`: 2 agents, 50 messages
  - `broadcast_throughput`: 5 agents, 30 broadcasts, 20 ms spacing
  - `concurrent_messaging`: 4 agents, 100 messages/agent
  - `scalability_stress`: 5 agents, 3 s

- **Extensive mode**
  - Agent counts are swept (default `[3, 5, 8]` unless overridden)
  - `point_to_point_latency`: 100 messages
  - `broadcast_throughput`: 50 broadcasts, 20 ms spacing
  - `concurrent_messaging`: 100 messages/agent
  - `scalability_stress`: 5 s

**Thesis runs** (see `sections/sec_experiments.tex`) override agent counts to larger sweeps (e.g., 5-20 agents) via YAML or CLI overrides.

### Metrics collected
The communication harness records a consistent set of metrics across scenarios:

- **Latency**: avg/min/max/std, p50/p95/p99/p99.9, jitter; sampled latencies for plots.
- **Throughput**: average and peak messages/s (1 s sliding window).
- **Reliability**: success rate, delivery failures, timeout failures, ordering violations, duplicate messages.
- **Resources**: process CPU% and RSS memory (via `psutil`).
- **Context**: agent count, topology density, total messages, payload size, test duration, latency mode.

Hardware/OS metadata is embedded in `benchmark_metadata.hardware` (CPU model, cores, memory, OS, Python version).

### Outputs and plots
Results are written to the configured output directory (default `results/`). In this repo the thesis artifacts live under `src/results/`.

Communication analysis (`src/benchmarks/benchmark_analysis.py`) produces:

- `latency_comparison.png`
- `throughput_comparison.png`
- `resource_usage.png`
- `latency_cdf.png`
- `p99_vs_concurrency.png`
- `throughput_vs_payload.png` (appears when multiple payload sizes were run)
- `performance_radar.png`
- `protocol_ranking.png`
- `topology_comparison.png`
- `scalability_analysis.png`
- `benchmark_summary_report.txt`

---

## Hierarchy benchmarks (simulated coordination)

### Why we run them
Protocol benchmarks answer *how fast* messages move. Hierarchy benchmarks answer *how agents should be organized* to coordinate tasks efficiently. They model coordination logic without network effects, so organizational trade-offs are visible without transport noise.

### Strategies evaluated
Implemented in `src/benchmarks/hierarchy_strategies.py`:

- **Tree**: manager-worker hierarchy with delegated tasks.
- **Peer-to-peer**: fully distributed coordination and consensus.
- **Hybrid**: manager assigns tasks; workers coordinate laterally.

### Environments
Defined in `src/benchmarks/hierarchy_environments.py`:

- **Resource allocation**: tasks compete for limited resources.
- **Task distribution**: continuous task stream assigned to agents.
- **Collaborative problem solving**: tasks with dependencies.
- **Fault recovery** (optional): random failures and recovery.
- **Scalability** (optional): performance scaling with agent count.

### Coordination record semantics (important)
Hierarchy benchmarks do **not** transmit real network messages. "Messages" are **coordination records** created for bookkeeping only:

- Created and counted in memory.
- Not transmitted or delivered.
- Used to compare coordination overhead, not bandwidth.

### Metrics collected
Key metrics include:

- **Success rate** and **normalized return**.
- **Makespan** (steps to success) and **action efficiency** (primitive actions/task).
- **Manager utilization** = manager actions per 100 environment steps.
  - Can exceed 100 when multiple managers act in the same step.
  - Now counted **only on planning steps**, so higher planning frequency lowers utilization.
- **Delegation success rate** and **preemption rate**.
- **Coordination latency** (steps between delegation and first action).
- **Coordination records/episode** and **estimated bytes/step** (proxy only).

### Ablation studies
Single-parameter sweeps are run for the **Tree** strategy in **task_distribution** (8 agents, 5 episodes). Baseline values:

- hierarchy depth = 2
- planning frequency = 1
- coordination record limit = None

The results are summarized in LaTeX (see `sections/sec_experiments.tex`, Table `tab:hierarchy-ablation`) and plotted in `src/results/hierarchy/hierarchy_ablation.png`.

### Outputs and plots
Hierarchy analysis (`src/benchmarks/hierarchy_analysis.py`) generates:

- `hierarchy_success_rates.png`
- `hierarchy_scalability.png`
- `hierarchy_overhead.png`
- `hierarchy_communication.png` (coordination record costs)
- `hierarchy_strategy_radar.png`
- `hierarchy_ablation.png`
- `hierarchy_benchmark_report.txt`

---

## Reproducibility notes

- **Hardware metadata** is captured automatically in `benchmark_metadata.hardware`.
- **Parameter overrides** are supported via YAML (`src/benchmarks/config_loader.py`). This is how thesis sweeps (e.g., 5-20 agents) are run when defaults are smaller.
- **Results location** is controlled by `--output-dir` (default `results/`). Thesis figures in LaTeX reference the `results/...` paths (currently stored under `src/results/`).

---

## Quick pointers

- Communication runner: `src/benchmarks/benchmark_runner.py`
- Communication analysis: `src/benchmarks/benchmark_analysis.py`
- Hierarchy scenarios: `src/benchmarks/hierarchy_benchmark_scenarios.py`
- Hierarchy analysis: `src/benchmarks/hierarchy_analysis.py`
