# Hierarchy Strategy Benchmarks

Comprehensive benchmarking system for comparing three hierarchy strategies in multi-agent systems:

1. **Tree Hierarchy**: Traditional manager-worker with centralized control
2. **Peer-to-Peer**: Fully distributed coordination with consensus
3. **Hybrid**: Tree for task allocation + peer communication for execution

## Metrics Tracked

### Task Effectiveness

- **Success rate**: Fraction of episodes reaching the goal
- **Normalized return**: (R - R_min) / (R_max - R_min) per task
- Macro-averaged across tasks

### Time and Resource Efficiency

- **Makespan**: Average steps-to-success for successful episodes
- **Action efficiency**: Primitive actions per task
- **Compute overhead**: Wall-clock per episode, broken down by manager vs. worker time

### Hierarchy Overhead

- **Manager utilization**: Manager actions per 100 environment steps
- **Delegation success rate**: Fraction of subgoals completed without intervention
- **Preemption count**: Number of times a subtask is aborted or reassigned

### Communication Cost

- **Messages per episode**: Total hierarchical messages
- **Bytes per step**: Average communication payload
- **Coordination latency**: Steps between subgoal issue and first worker action

### Generalization and Scalability

- **Team-size scaling**: Performance when scaling from N to 2N agents
- **Domain shift**: Train/test split across different task configurations

### Robustness

- **Fault injection**: Random agent failures with probability p
- **Recovery time**: Steps to recover from failures
- **Performance degradation**: Success rate drop under faults

## Usage

### Quick Test (3 episodes, 1 environment, 1 agent count)

```bash
python benchmarks/hierarchy_benchmark_scenarios.py
```

### Full Benchmark Suite

```bash
python benchmarks/hierarchy_benchmark_scenarios.py --full
```

### Via Unified Benchmark Runner

#### Basic Hierarchy Benchmarks

```bash
python benchmarks/benchmark_runner.py --hierarchy
```

#### Specific Strategies

```bash
python benchmarks/benchmark_runner.py --hierarchy --hierarchy-types tree hybrid
```

#### Specific Environments

```bash
python benchmarks/benchmark_runner.py --hierarchy \
    --hierarchy-environments task_distribution resource_allocation fault_recovery
```

#### Full Comparison with Ablation Study

```bash
python benchmarks/benchmark_runner.py --hierarchy --extensive --hierarchy-ablation
```

#### Custom Configuration

```bash
python benchmarks/benchmark_runner.py --hierarchy \
    --hierarchy-types tree peer_to_peer hybrid \
    --hierarchy-environments task_distribution collaborative scalability \
    --hierarchy-episodes 20 \
    --hierarchy-ablation
```

## Command-Line Options

### `--hierarchy`

Enable hierarchy strategy benchmarks

### `--hierarchy-types [tree|peer_to_peer|hybrid]`

Which strategies to test (default: all three)

### `--hierarchy-environments [ENV...]`

Available environments:

- `task_distribution`: Efficient task assignment and completion
- `resource_allocation`: Coordinate limited resource usage
- `collaborative`: Multi-step tasks with dependencies
- `fault_recovery`: Handle agent failures during execution
- `scalability`: Test performance at different team sizes

### `--hierarchy-episodes N`

Number of episodes per benchmark (default: 10)

### `--hierarchy-ablation`

Run ablation study testing:

- Hierarchy depth: 1, 2, 3 levels
- Planning frequency: Manager acts every 1, 5, 10 steps
- Communication limits: Unlimited, 50, 100 messages per episode

### `--extensive`

More comprehensive testing:

- Agent counts: [3, 5, 8, 12] instead of [5, 8]
- All environments
- More episodes

## Environments

### Task Distribution

- **Goal**: Complete maximum tasks with minimum makespan
- **Success**: All tasks completed efficiently
- **Metrics**: Task completion rate, makespan, coordination overhead

### Resource Allocation

- **Goal**: Maximize resource utilization while completing all tasks
- **Success**: All tasks done within resource constraints
- **Metrics**: Utilization efficiency, constraint violations

### Collaborative Problem Solving

- **Goal**: Solve multi-step problems with dependencies
- **Success**: All sub-problems solved in correct order
- **Metrics**: Dependency satisfaction, completion order

### Fault Recovery

- **Goal**: Complete tasks despite random agent failures
- **Success**: >80% task completion despite faults
- **Metrics**: Recovery time, performance degradation

### Scalability

- **Goal**: Maintain efficiency as team size increases
- **Success**: Performance scales sub-linearly
- **Metrics**: Coordination overhead vs. team size

## Output Files

All results are saved to `results/hierarchy/`:

### `hierarchy_results_TIMESTAMP.json`

Complete benchmark results including:

- Configuration for each run
- All metric values
- Execution times
- Episode-level data

### `hierarchy_metrics_TIMESTAMP.csv`

CSV format for easy plotting:

```
Strategy,Environment,Agents,SuccessRate,NormReturn,Makespan,ActionEfficiency,ManagerUtil,DelegationSuccess,MessagesPerEp,BytesPerStep,CoordLatency
tree,task_distribution,5,0.9,0.85,45.2,12.3,35.6,0.92,28.5,150.2,0.0045
...
```

### `ablation_study_TIMESTAMP.json`

Ablation study results showing impact of:

- Hierarchy depth
- Planning frequency
- Communication limits
- Static vs. dynamic role assignment

## Programmatic Usage

```python
from benchmarks.hierarchy_benchmark_scenarios import (
    HierarchyComparisonBenchmark,
    BenchmarkConfiguration,
    HierarchyType
)

# Create benchmark runner
benchmark = HierarchyComparisonBenchmark(output_dir="results/hierarchy")

# Run comparison
benchmark.run_comparison(
    hierarchy_types=[HierarchyType.TREE, HierarchyType.PEER_TO_PEER, HierarchyType.HYBRID],
    environment_types=["task_distribution", "collaborative"],
    agent_counts=[5, 8, 12],
    num_episodes=10
)

# Run ablation study
from benchmarks.hierarchy_strategies import HierarchyType

base_config = BenchmarkConfiguration(
    hierarchy_type=HierarchyType.TREE,
    num_agents=8,
    environment_type="task_distribution",
    num_episodes=5
)

benchmark.run_ablation_study(
    base_config=base_config,
    ablation_params={
        "hierarchy_depth": [1, 2, 3],
        "planning_frequency": [1, 5, 10],
        "communication_limit": [None, 50, 100]
    }
)
```

## Visualization and Analysis

After running benchmarks, generate comprehensive visualizations:

```bash
# Analyze latest results
python benchmarks/hierarchy_analysis.py

# Analyze specific results file
python benchmarks/hierarchy_analysis.py --results-file results/hierarchy/hierarchy_results_20250101_120000.json

# Generate specific plot types
python benchmarks/hierarchy_analysis.py --plot scalability
python benchmarks/hierarchy_analysis.py --plot radar
python benchmarks/hierarchy_analysis.py --plot ablation
```

### Available Visualizations

1. **Success Rate Comparison** (`hierarchy_success_rates.png`)
   - Success rate by environment
   - Normalized return by environment

2. **Scalability Analysis** (`hierarchy_scalability.png`)
   - Success rate vs team size
   - Makespan vs team size
   - Communication overhead scaling
   - Manager utilization trends

3. **Hierarchy Overhead Analysis** (`hierarchy_overhead.png`)
   - Manager utilization by strategy
   - Delegation success rates
   - Compute time distribution
   - Task preemption rates

4. **Communication Cost Analysis** (`hierarchy_communication.png`)
   - Messages per episode
   - Communication bandwidth
   - Coordination latency

5. **Strategy Radar Chart** (`hierarchy_strategy_radar.png`)
   - Multi-dimensional comparison across all metrics

6. **Ablation Study Plots** (`hierarchy_ablation.png`)
   - Impact of hierarchy depth, planning frequency, communication limits

7. **Text Report** (`hierarchy_benchmark_report.txt`)
   - Comprehensive summary with recommendations

## Architecture

```
benchmarks/
├── hierarchy_metrics.py                # Metric tracking infrastructure
├── hierarchy_strategies.py             # Three hierarchy implementations
├── hierarchy_environments.py           # Task environments
├── hierarchy_benchmark_scenarios.py    # Main benchmark orchestration
├── hierarchy_analysis.py               # Analysis & visualization tools
├── benchmark_runner.py                 # Unified CLI entry point
└── HIERARCHY_BENCHMARKS.md            # This documentation
```

### Key Classes

**HierarchyBenchmarkTracker**: Aggregates all metric trackers

- TaskEffectivenessTracker
- HierarchyOverheadTracker
- CommunicationCostTracker
- ResourceEfficiencyTracker
- RobustnessTracker

**HierarchyStrategy** (ABC):

- TreeHierarchy
- PeerToPeerHierarchy
- HybridHierarchy

**HierarchyEnvironment** (ABC):

- ResourceAllocationEnvironment
- TaskDistributionEnvironment
- CollaborativeProblemSolving
- FaultRecoveryEnvironment
- ScalabilityTestEnvironment

## Example Results

### Comparison Output

```
HIERARCHY STRATEGY COMPARISON REPORT
======================================================================

TASK_DISTRIBUTION - 5 agents:
----------------------------------------------------------------------
  tree            | Success:  90.0% | Return: 0.850 | Makespan:   45.2 | Messages:   28.5
  hybrid          | Success:  85.0% | Return: 0.820 | Makespan:   48.1 | Messages:   42.3
  peer_to_peer    | Success:  80.0% | Return: 0.780 | Makespan:   52.7 | Messages:   67.8

RESOURCE_ALLOCATION - 8 agents:
----------------------------------------------------------------------
  hybrid          | Success:  95.0% | Return: 0.920 | Makespan:   38.5 | Messages:   51.2
  tree            | Success:  90.0% | Return: 0.885 | Makespan:   42.3 | Messages:   35.7
  peer_to_peer    | Success:  85.0% | Return: 0.845 | Makespan:   49.1 | Messages:   89.4
```

### Ablation Study Output

Shows how performance varies with:

- **Depth 1 (flat)**: Lower overhead, but limited coordination
- **Depth 2**: Balanced performance
- **Depth 3**: Higher overhead, better for large teams
- **Planning freq 1**: Responsive but high overhead
- **Planning freq 10**: Lower overhead but slower adaptation

## Extending

### Add New Environment

```python
from benchmarks.hierarchy_environments import HierarchyEnvironment

class MyCustomEnvironment(HierarchyEnvironment):
    def reset(self):
        # Initialize environment state
        pass

    def generate_tasks(self, num_tasks):
        # Create tasks
        pass

    def step(self, hierarchy):
        # Execute one step
        return state, reward, done

    def check_success(self):
        # Check if goal achieved
        return True/False
```

### Add New Strategy

```python
from benchmarks.hierarchy_strategies import HierarchyStrategy

class MyCustomStrategy(HierarchyStrategy):
    def initialize_hierarchy(self):
        # Setup agent structure
        pass

    def allocate_task(self, task):
        # Assign task to agent(s)
        pass

    def process_step(self):
        # Execute coordination logic
        pass

    def handle_agent_failure(self, agent_id):
        # Handle failures
        pass
```

## Citation

If you use these benchmarks in your research, please cite:

```bibtex
@misc{hierarchy_benchmarks_2025,
  title={Hierarchy Strategy Benchmarks for Multi-Agent Systems},
  author={Your Name},
  year={2025},
  note={Comprehensive benchmarking framework for comparing tree, peer-to-peer, and hybrid coordination strategies}
}
```
