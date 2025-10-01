# Hierarchy Benchmark Analysis Tool

Comprehensive analysis and visualization tool for hierarchy strategy benchmark results.

## Quick Start

```bash
# Generate all visualizations from latest results
python benchmarks/hierarchy_analysis.py

# Analyze specific results file
python benchmarks/hierarchy_analysis.py \
    --results-file results/hierarchy/hierarchy_results_20250101_120000.json

# Generate specific visualization
python benchmarks/hierarchy_analysis.py --plot radar
```

## Generated Visualizations

### 1. Success Rate Comparison (`hierarchy_success_rates.png`)
**What it shows:**
- Success rate by environment (grouped bar chart)
- Normalized return by environment (grouped bar chart)
- Side-by-side comparison of all three strategies

**Use for:**
- Quick comparison of task effectiveness
- Identifying which strategy performs best in each environment
- Understanding return vs success rate tradeoffs

### 2. Scalability Analysis (`hierarchy_scalability.png`)
**What it shows:**
- 4-panel plot showing performance vs team size:
  - Success rate scaling
  - Makespan (time-to-completion) scaling
  - Communication overhead (messages/episode)
  - Manager utilization (for hierarchical strategies)

**Use for:**
- Understanding how strategies scale with team size
- Identifying inflection points where performance degrades
- Comparing coordination overhead growth rates

### 3. Hierarchy Overhead Analysis (`hierarchy_overhead.png`)
**What it shows:**
- 4-panel comparison of hierarchy-specific metrics:
  - Manager utilization by environment
  - Delegation success rates
  - Compute time distribution (manager vs worker)
  - Task preemption rates

**Use for:**
- Understanding coordination costs
- Identifying delegation bottlenecks
- Comparing manager efficiency across strategies

### 4. Communication Cost Analysis (`hierarchy_communication.png`)
**What it shows:**
- 3-panel analysis of communication metrics:
  - Messages per episode
  - Bandwidth usage (bytes/step)
  - Coordination latency

**Use for:**
- Comparing communication efficiency
- Understanding bandwidth requirements
- Identifying coordination delays

### 5. Strategy Radar Chart (`hierarchy_strategy_radar.png`)
**What it shows:**
- Multi-dimensional radar plot comparing strategies on:
  - Task success
  - Time efficiency
  - Action efficiency
  - Low communication overhead
  - Delegation quality

**Use for:**
- Holistic strategy comparison
- Identifying strengths and weaknesses
- Publication-ready comparison visualization

### 6. Ablation Study Plots (`hierarchy_ablation.png`)
**What it shows:**
- 4-panel analysis of ablation parameters:
  - Impact of hierarchy depth (1, 2, 3 levels)
  - Planning frequency effects (1, 5, 10 steps)
  - Communication limit constraints
  - Statistical summaries

**Use for:**
- Understanding parameter sensitivity
- Optimizing hierarchy configuration
- Justifying design choices

### 7. Text Report (`hierarchy_benchmark_report.txt`)
**What it contains:**
- Executive summary with key metrics
- Strategy comparison by environment
- Best performing strategy by metric
- Recommendations for different use cases

**Use for:**
- Quick reference
- Including in thesis/paper
- Sharing with non-technical stakeholders

## Command-Line Options

```
--results-file FILE    Specific JSON file to analyze
--results-dir DIR      Directory containing results (default: results/hierarchy)
--plot TYPE            Type of visualization to generate
```

Plot types:
- `success` - Success rate comparison only
- `scalability` - Scalability analysis only
- `overhead` - Hierarchy overhead only
- `communication` - Communication cost only
- `radar` - Strategy radar chart only
- `ablation` - Ablation study plots only
- `all` - Generate all visualizations (default)

## Programmatic Usage

```python
from benchmarks.hierarchy_analysis import HierarchyBenchmarkAnalyzer

# Create analyzer
analyzer = HierarchyBenchmarkAnalyzer(
    results_file="results/hierarchy/hierarchy_results_20250101_120000.json"
)

# Generate specific plots
analyzer.create_success_rate_comparison()
analyzer.create_scalability_analysis()
analyzer.create_strategy_radar_chart()

# Or generate all
analyzer.generate_all_visualizations()
```

## Output Format

All visualizations are saved as:
- High-resolution PNG files (300 DPI)
- Publication-ready formatting
- Consistent color scheme across plots
- Clear labels and legends

Files are saved to the same directory as the results:
```
results/hierarchy/
├── hierarchy_results_TIMESTAMP.json
├── hierarchy_metrics_TIMESTAMP.csv
├── hierarchy_success_rates.png
├── hierarchy_scalability.png
├── hierarchy_overhead.png
├── hierarchy_communication.png
├── hierarchy_strategy_radar.png
├── hierarchy_ablation.png
└── hierarchy_benchmark_report.txt
```

## Integration with Benchmark Runner

The analysis tool integrates seamlessly with the benchmark runner:

```bash
# Run benchmarks
python benchmarks/benchmark_runner.py --hierarchy --extensive

# Analyze results (automatically finds latest)
python benchmarks/hierarchy_analysis.py

# Or combine in one workflow
python benchmarks/benchmark_runner.py --hierarchy && \
python benchmarks/hierarchy_analysis.py
```

## Requirements

- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- pandas >= 1.3.0
- numpy >= 1.21.0

All dependencies are included in the main project requirements.

## Customization

To customize plots, edit `hierarchy_analysis.py`:

```python
# Change color scheme
colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]  # Red, teal, blue

# Adjust figure sizes
fig, ax = plt.subplots(figsize=(16, 12))  # Width x Height in inches

# Modify DPI for different quality
plt.savefig(output_path, dpi=300)  # 300 for print, 150 for web
```

## Troubleshooting

**No results found:**
```bash
# Check if benchmarks have been run
ls results/hierarchy/

# Run benchmarks first
python benchmarks/benchmark_runner.py --hierarchy
```

**Missing dependencies:**
```bash
pip install matplotlib seaborn pandas numpy
```

**Plot not displaying:**
- Plots are saved to files, not displayed interactively
- Check the console output for file locations
- Open PNG files with image viewer

## Tips for Best Results

1. **Run extensive benchmarks** for more data points:
   ```bash
   python benchmarks/benchmark_runner.py --hierarchy --extensive
   ```

2. **Include ablation study** for parameter analysis:
   ```bash
   python benchmarks/benchmark_runner.py --hierarchy --hierarchy-ablation
   ```

3. **Test multiple agent counts** for scalability insights:
   - Simple mode: [5, 8]
   - Extensive mode: [3, 5, 8, 12]

4. **Run multiple episodes** for statistical significance:
   ```bash
   python benchmarks/benchmark_runner.py --hierarchy --hierarchy-episodes 20
   ```

## Example Workflow

Complete workflow for comprehensive analysis:

```bash
# 1. Run full benchmark suite
python benchmarks/benchmark_runner.py \
    --hierarchy \
    --extensive \
    --hierarchy-ablation \
    --hierarchy-episodes 20

# 2. Generate all visualizations
python benchmarks/hierarchy_analysis.py --plot all

# 3. View results
ls results/hierarchy/*.png
cat results/hierarchy/hierarchy_benchmark_report.txt
```

## Citation

If you use this analysis tool in your research, please cite:

```bibtex
@misc{hierarchy_analysis_2025,
  title={Hierarchy Benchmark Analysis Tool},
  author={Your Name},
  year={2025},
  note={Visualization and analysis toolkit for multi-agent hierarchy strategies}
}
```
