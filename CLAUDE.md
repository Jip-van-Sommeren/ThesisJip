# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a research thesis implementation of a formal multi-agent system with BDI (Belief-Desire-Intention) and reactive agent architectures. The system implements hierarchical agent organizations with multiple communication protocols (REST, gRPC, MQTT, Kafka) and comprehensive benchmarking capabilities.

## Core Architecture

### Agent Types (src/)

The system implements three agent architectures based on formal definitions:

1. **ReactiveAgent** (`reactive_agent.py`): Minimal state, no goals, direct stimulus-response via rules
   - Goals = ∅ (empty set by design)
   - Only reactive rules, no deliberation
   - Fast response for anomaly detection and fail-safe mechanisms

2. **BDIAgent** (`bdi_agent.py`): Belief-Desire-Intention architecture
   - Rich belief state (internal + external beliefs with history)
   - Goals → Desires → Intentions (filtered subset with plans)
   - Deliberative reasoning with plan library
   - Best for coordination, optimization, scheduling

3. **HybridAgent** (`hybrid_agent.py`): Combines reactive and BDI
   - Layered decision making (reactive rules override deliberation)
   - Balances fast response with planning

All agents inherit from **AbstractAgent** (`abstract_agent.py`) which defines the formal architecture:
```
A = (Id, State, Goal, Perception, Action, Decision)
```

### Hierarchy System (src/hierarchy.py)

Implements organizational structures with formal supervisor/subordinate relationships:
- `Sup: A → A ∪ {⊥}` - supervisor function
- `Sub: A → 2^A` - subordinates function
- Tree hierarchy with roles, groups, and message passing
- Organizational capabilities via `OrganizationalMixin`
- Commands, reports, escalations, delegations

Key classes:
- `HierarchyManager`: Manages tree structure and relationships
- `OrganizationalPosition`: Agent's position (roles, supervisor, subordinates, level)
- `HierarchicalReactiveAgent`, `HierarchicalBDIAgent`, `HierarchicalHybridAgent`

### Communication System (src/communication/)

Protocol-agnostic base model (`base_communication.py`):
- **Message Space (M)**: Message types (INFORM, REQUEST, REPLY, BROADCAST, ERROR, ACK)
- **Mailbox (MB_i)**: Thread-safe message buffers per agent
- **Topology (Comm ⊆ A × A)**: Communication links between agents
- **LatencyMode**: Three measurement modes (SEND_ONLY, END_TO_END, APP_ACK)

Protocol implementations:
- **REST** (`rest/`): HTTP/1.1 and HTTP/2, Flask-based
- **gRPC** (`grpc/`): Unary and streaming RPC, protobuf-based
- **MQTT** (`mqtt/`): QoS 0/1/2 variants, pub-sub pattern
- **Kafka** (`kafka/`): acks=0/1/all variants, topic-based

Each protocol has:
- `*_communication.py`: Core protocol implementation
- `*_communication_agent.py`: Agent wrapper with mailbox
- `*_benchmark_scenarios.py`: Protocol-specific benchmarks

### Storage System (src/storage/)

Multi-backend storage manager for persistence:
- **Time Series** (InfluxDB): Metrics, beliefs, actions over time
- **Document Store** (MongoDB): Agent profiles, configurations
- **Graph Store** (Neo4j): Hierarchy relationships, agent graphs
- **Cache** (Redis): Fast state access, temporary data

`storage_manager.py` provides unified interface with:
- Batch processing for bulk operations
- Health monitoring of all backends
- Automatic persistence of beliefs, goals, actions
- Retention policies and cleanup

## Running the System

### Prerequisites

1. **Start storage backends** (required for agent storage features):
   ```bash
   docker-compose up -d
   ```
   This starts InfluxDB, MongoDB, Neo4j, Redis, and monitoring tools (Grafana, Mongo Express, Redis Commander).

2. **Activate Python environment**:
   ```bash
   source venv/bin/activate  # or activate.bat on Windows
   ```

### Running Benchmarks

The main benchmark runner is `src/benchmarks/benchmark_runner.py`.

**Simple benchmarks** (quick, basic parameters):
```bash
python3 src/benchmarks/benchmark_runner.py --simple
```

**Extensive benchmarks** (comprehensive, multiple configs):
```bash
python3 src/benchmarks/benchmark_runner.py --extensive
```

**Protocol-specific benchmarks**:
```bash
# Only test specific protocols
python3 src/benchmarks/benchmark_runner.py --simple --protocols rest grpc

# Only communication benchmarks (skip hierarchy)
python3 src/benchmarks/benchmark_runner.py --communication-only --simple

# Only hierarchy strategy benchmarks
python3 src/benchmarks/benchmark_runner.py --hierarchy-only
```

**Latency measurement modes**:
```bash
# Send-only latency (publish/send time only)
python3 src/benchmarks/benchmark_runner.py --simple --latency-mode send_only

# End-to-end latency (protocol-level confirmation)
python3 src/benchmarks/benchmark_runner.py --simple --latency-mode end_to_end

# Application-level acknowledgment (full round-trip)
python3 src/benchmarks/benchmark_runner.py --simple --latency-mode app_ack
```

**Using YAML configuration**:
```bash
python3 src/benchmarks/benchmark_runner.py --config-file src/benchmarks/example_configs/fast_benchmark.yaml
```

**Hierarchy ablation study**:
```bash
python3 src/benchmarks/benchmark_runner.py --hierarchy-only --extensive --hierarchy-ablation
```

### Running Tests

```bash
# Run all tests
pytest test/

# Run specific test
pytest test/test_hierarchy_metrics.py

# Run with verbose output
pytest -v test/
```

### gRPC Protocol

When modifying the gRPC protocol definition:

1. Edit `src/communication/grpc/communication.proto`
2. Regenerate Python code:
   ```bash
   python -m grpc_tools.protoc -I src/communication/grpc \
       --python_out=src/communication/grpc \
       --grpc_python_out=src/communication/grpc \
       src/communication/grpc/communication.proto
   ```

## Development Patterns

### Creating New Agents

Use the factory pattern for hierarchical agents:
```python
from hierarchy import OrganizationalAgentFactory, HierarchyManager
from role import RoleManager
from group import GroupManager

role_manager = RoleManager()
group_manager = GroupManager()
hierarchy_manager = HierarchyManager(role_manager, group_manager)
factory = OrganizationalAgentFactory(hierarchy_manager)

agent = factory.create_agent(
    agent_type="bdi",  # or "reactive", "hybrid"
    agent_id=AgentId(app="myapp", type="worker", instance="001"),
    observable_properties={"temperature", "pressure"},
    roles={"operator", "monitor"},
    supervisor="supervisor_agent_id",  # optional
    groups={"production_team"}  # optional
)
```

### Adding Communication to Agents

Wrap agents with protocol-specific communication:
```python
from communication.grpc.grpc_communication_agent import GrpcCommunicatingAgent

# Wrap any agent type
comm_agent = GrpcCommunicatingAgent(
    agent=my_agent,
    server_address="localhost:50051"
)
comm_agent.start()

# Send messages
comm_agent.send_message(receiver_id, MessageType.INFORM, content)

# Process incoming messages
messages = comm_agent.receive_messages()
```

### Benchmark Development

To add new benchmark scenarios:

1. Add scenario method to `communication_benchmark.py`:
   ```python
   def run_my_scenario(self, **kwargs) -> BenchmarkResult:
       # Setup
       # Run test
       # Collect metrics
       return BenchmarkResult(...)
   ```

2. Register scenario in protocol-specific files (`*_benchmark_scenarios.py`):
   ```python
   benchmark = CommunicationBenchmark(...)
   benchmark.scenarios["my_scenario"] = benchmark.run_my_scenario
   ```

3. Add to scenario choices in `benchmark_runner.py` argparse

### Storage Integration

Enable automatic persistence for agents:
```python
from config.storage_config import StorageConfig
from storage.storage_manager import MultiAgentStorageManager

config = StorageConfig()
storage = MultiAgentStorageManager(config)
storage.connect_all()

# Initialize agent storage
storage.initialize_agent_storage(
    agent_id=str(agent.id),
    agent_type=agent.get_agent_type(),
    roles=agent.assigned_roles,
    groups=set()
)

# Auto-recorded if config.auto_persist_beliefs = True
agent.state.update_belief("sensor_temp", "25.3", confidence=0.95)

# Cleanup when done
storage.disconnect_all()
```

## Key Concepts

### Formal Definitions

The code follows formal definitions from the thesis:

- **Agent**: `A = (Id, State, Goal, Perception, Action, Decision)`
- **State**: `State_A = ⟨B^int_A, B^ext_A⟩` (internal and external beliefs)
- **Communication**: `send(i, j, m)` where `(i,j) ∈ Comm` and `m ∈ M`
- **Mailbox**: `MB_i` as multiset/queue of messages
- **Topology**: `Comm ⊆ A × A` defining valid communication links
- **Hierarchy**: `Sup: A → A ∪ {⊥}` and `Sub: A → 2^A`

### Design Principles

1. **Protocol Agnostic**: Base abstractions (`Message`, `Mailbox`, `CommunicationTopology`) work with any protocol
2. **Formal Correctness**: Implementation follows mathematical definitions
3. **Composability**: Agents can be wrapped with communication, hierarchy, and storage layers
4. **Fair Benchmarking**: Three latency modes for protocol-agnostic comparison
5. **Type Safety**: Use of dataclasses, enums, and type hints throughout

## Results and Analysis

Benchmark results are saved to `results/` (or specified `--output-dir`):
- **JSON**: Full results with metadata (`benchmark_results_*.json`)
- **CSV**: Latency, throughput, resource usage for plotting
- **Hierarchy**: Separate `hierarchy/` subdirectory for strategy benchmarks

Result files include latency mode suffix (e.g., `_end_to_end_`, `_app_ack_`).

## Important Notes

- **Reactive agents cannot have goals** - calling `add_goal()` raises `NotImplementedError`
- **Docker services must run** for storage features (but not required for benchmarks)
- **MQTT/Kafka benchmarks auto-start brokers** in Docker if not running
- **gRPC requires protoc** to regenerate files after `.proto` changes
- **Latency modes are not comparable** - use same mode for all protocols in a comparison
- **Batch processing** in storage manager prevents overwhelming backends with high-frequency updates
