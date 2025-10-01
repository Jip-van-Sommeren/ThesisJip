#!/usr/bin/env python3
"""
Benchmark Analysis and Visualization Utilities
Provides tools for analyzing benchmark results and generating visualizations
for cross-protocol performance comparison.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import glob
import os
from typing import List, Optional


class BenchmarkAnalyzer:
    """Analyzer for communication protocol benchmark results."""

    def __init__(
        self, results_file: Optional[str] = None, results_dir: str = "results"
    ):
        self.results_dir = results_dir
        self.data = None
        self.comparison_data = None

        if results_file:
            self.load_results(results_file)
        else:
            # Load most recent results
            self.load_latest_results()

    def load_results(self, results_file: str):
        """Load benchmark results from JSON file."""
        try:
            with open(results_file, "r") as f:
                self.data = json.load(f)

            self.comparison_data = self.data.get(
                "cross_protocol_comparison", {}
            )
            print(f"Loaded results from {results_file}")

        except Exception as e:
            print(f"Error loading results: {e}")
            self.data = None
            self.comparison_data = None

    def load_latest_results(self):
        """Load the most recent benchmark results."""
        pattern = os.path.join(self.results_dir, "benchmark_results_*.json")
        files = glob.glob(pattern)

        if not files:
            print(f"No benchmark results found in {self.results_dir}")
            return

        # Get most recent file
        latest_file = max(files, key=os.path.getctime)
        self.load_results(latest_file)

    def get_available_protocols(self) -> List[str]:
        """Get list of protocols in the results."""
        if not self.data:
            return []

        return list(self.data.get("protocol_results", {}).keys())

    def get_available_scenarios(self) -> List[str]:
        """Get list of scenarios in the results."""
        if not self.comparison_data:
            return []

        return list(self.comparison_data.keys())

    def create_latency_comparison_plot(
        self, output_file: str = "latency_comparison.png"
    ):
        """Create latency comparison plot across protocols and scenarios."""
        if not self.comparison_data:
            print("No comparison data available")
            return

        # Prepare data for plotting
        protocols = []
        scenarios = []
        avg_latencies = []
        p95_latencies = []

        for scenario, protocol_data in self.comparison_data.items():
            for protocol, metrics in protocol_data.items():
                protocols.append(protocol.upper())
                scenarios.append(scenario.replace("_", " ").title())
                avg_latencies.append(metrics.get("avg_latency_ms", 0))
                p95_latencies.append(metrics.get("p95_latency_ms", 0))

        df = pd.DataFrame(
            {
                "Protocol": protocols,
                "Scenario": scenarios,
                "Avg_Latency_ms": avg_latencies,
                "P95_Latency_ms": p95_latencies,
            }
        )

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Average latency plot
        sns.barplot(
            data=df, x="Scenario", y="Avg_Latency_ms", hue="Protocol", ax=ax1
        )
        ax1.set_title(
            "Average Message Latency by Protocol",
            fontsize=14,
            fontweight="bold",
        )
        ax1.set_ylabel("Average Latency (ms)")
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(True, alpha=0.3)

        # P95 latency plot
        sns.barplot(
            data=df, x="Scenario", y="P95_Latency_ms", hue="Protocol", ax=ax2
        )
        ax2.set_title(
            "95th Percentile Message Latency by Protocol",
            fontsize=14,
            fontweight="bold",
        )
        ax2.set_ylabel("P95 Latency (ms)")
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = os.path.join(self.results_dir, output_file)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Latency comparison plot saved to {output_path}")
        plt.close()

    def create_throughput_comparison_plot(
        self, output_file: str = "throughput_comparison.png"
    ):
        """Create throughput comparison plot across protocols and scenarios."""
        if not self.comparison_data:
            print("No comparison data available")
            return

        # Prepare data for plotting
        protocols = []
        scenarios = []
        throughputs = []
        success_rates = []

        for scenario, protocol_data in self.comparison_data.items():
            for protocol, metrics in protocol_data.items():
                protocols.append(protocol.upper())
                scenarios.append(scenario.replace("_", " ").title())
                throughputs.append(metrics.get("throughput_msg_per_sec", 0))
                success_rates.append(metrics.get("success_rate_percent", 0))

        df = pd.DataFrame(
            {
                "Protocol": protocols,
                "Scenario": scenarios,
                "Throughput_msg_per_sec": throughputs,
                "Success_Rate_percent": success_rates,
            }
        )

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Throughput plot
        sns.barplot(
            data=df,
            x="Scenario",
            y="Throughput_msg_per_sec",
            hue="Protocol",
            ax=ax1,
        )
        ax1.set_title(
            "Message Throughput by Protocol", fontsize=14, fontweight="bold"
        )
        ax1.set_ylabel("Throughput (messages/sec)")
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(True, alpha=0.3)

        # Success rate plot
        sns.barplot(
            data=df,
            x="Scenario",
            y="Success_Rate_percent",
            hue="Protocol",
            ax=ax2,
        )
        ax2.set_title(
            "Success Rate by Protocol", fontsize=14, fontweight="bold"
        )
        ax2.set_ylabel("Success Rate (%)")
        ax2.set_ylim(0, 105)
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = os.path.join(self.results_dir, output_file)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Throughput comparison plot saved to {output_path}")
        plt.close()

    def create_resource_usage_plot(
        self, output_file: str = "resource_usage.png"
    ):
        """Create resource usage comparison plot."""
        if not self.comparison_data:
            print("No comparison data available")
            return

        # Prepare data for plotting
        protocols = []
        scenarios = []
        cpu_usage = []
        memory_usage = []

        for scenario, protocol_data in self.comparison_data.items():
            for protocol, metrics in protocol_data.items():
                protocols.append(protocol.upper())
                scenarios.append(scenario.replace("_", " ").title())
                cpu_usage.append(metrics.get("cpu_usage_percent", 0))
                memory_usage.append(metrics.get("memory_usage_mb", 0))

        df = pd.DataFrame(
            {
                "Protocol": protocols,
                "Scenario": scenarios,
                "CPU_Usage_percent": cpu_usage,
                "Memory_Usage_mb": memory_usage,
            }
        )

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # CPU usage plot
        sns.barplot(
            data=df,
            x="Scenario",
            y="CPU_Usage_percent",
            hue="Protocol",
            ax=ax1,
        )
        ax1.set_title("CPU Usage by Protocol", fontsize=14, fontweight="bold")
        ax1.set_ylabel("CPU Usage (%)")
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(True, alpha=0.3)

        # Memory usage plot
        sns.barplot(
            data=df, x="Scenario", y="Memory_Usage_mb", hue="Protocol", ax=ax2
        )
        ax2.set_title(
            "Memory Usage by Protocol", fontsize=14, fontweight="bold"
        )
        ax2.set_ylabel("Memory Usage (MB)")
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = os.path.join(self.results_dir, output_file)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Resource usage plot saved to {output_path}")
        plt.close()

    def create_performance_radar_chart(
        self, output_file: str = "performance_radar.png"
    ):
        """Create radar chart comparing overall protocol performance."""
        if not self.comparison_data:
            print("No comparison data available")
            return

        protocols = self.get_available_protocols()
        if not protocols:
            return

        # Aggregate metrics across scenarios for each protocol
        protocol_metrics = {}

        for protocol in protocols:
            avg_latency = []
            avg_throughput = []
            avg_success_rate = []
            avg_cpu = []
            avg_memory = []

            for scenario, protocol_data in self.comparison_data.items():
                if protocol in protocol_data:
                    metrics = protocol_data[protocol]
                    avg_latency.append(metrics.get("avg_latency_ms", 0))
                    avg_throughput.append(
                        metrics.get("throughput_msg_per_sec", 0)
                    )
                    avg_success_rate.append(
                        metrics.get("success_rate_percent", 0)
                    )
                    avg_cpu.append(metrics.get("cpu_usage_percent", 0))
                    avg_memory.append(metrics.get("memory_usage_mb", 0))

            protocol_metrics[protocol] = {
                "Low Latency": 100
                - (np.mean(avg_latency) if avg_latency else 0),  # Inverted
                "High Throughput": (
                    np.mean(avg_throughput) if avg_throughput else 0
                ),
                "Reliability": (
                    np.mean(avg_success_rate) if avg_success_rate else 0
                ),
                "Low CPU Usage": 100
                - (np.mean(avg_cpu) if avg_cpu else 0),  # Inverted
                "Low Memory Usage": 100
                - (np.mean(avg_memory) if avg_memory else 0),  # Inverted
            }

        # Create radar chart
        categories = list(protocol_metrics[protocols[0]].keys())
        num_categories = len(categories)

        # Compute angle for each axis
        angles = [
            n / float(num_categories) * 2 * np.pi
            for n in range(num_categories)
        ]
        angles += angles[:1]  # Complete the circle

        fig, ax = plt.subplots(
            figsize=(10, 10), subplot_kw=dict(projection="polar")
        )

        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]

        for i, protocol in enumerate(protocols):
            values = list(protocol_metrics[protocol].values())
            values += values[:1]  # Complete the circle

            ax.plot(
                angles,
                values,
                "o-",
                linewidth=2,
                label=protocol.upper(),
                color=colors[i % len(colors)],
            )
            ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])

        # Customize the chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.set_title(
            "Protocol Performance Comparison\n(Higher values are better)",
            size=16,
            fontweight="bold",
            pad=20,
        )
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)

        output_path = os.path.join(self.results_dir, output_file)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Performance radar chart saved to {output_path}")
        plt.close()

    def create_protocol_ranking_table(
        self, output_file: str = "protocol_ranking.png"
    ):
        """Create a table showing protocol rankings by metric."""
        if not self.comparison_data:
            print("No comparison data available")
            return

        protocols = self.get_available_protocols()
        scenarios = self.get_available_scenarios()

        if not protocols or not scenarios:
            return

        # Calculate rankings for each metric
        rankings = {}

        for scenario in scenarios:
            scenario_data = self.comparison_data[scenario]

            # Rank by latency (lower is better)
            latency_ranking = sorted(
                scenario_data.items(),
                key=lambda x: x[1].get("avg_latency_ms", float("inf")),
            )

            # Rank by throughput (higher is better)
            throughput_ranking = sorted(
                scenario_data.items(),
                key=lambda x: x[1].get("throughput_msg_per_sec", 0),
                reverse=True,
            )

            # Rank by success rate (higher is better)
            reliability_ranking = sorted(
                scenario_data.items(),
                key=lambda x: x[1].get("success_rate_percent", 0),
                reverse=True,
            )

            for i, (protocol, _) in enumerate(latency_ranking):
                rankings[f"{protocol}_latency_{scenario}"] = i + 1

            for i, (protocol, _) in enumerate(throughput_ranking):
                rankings[f"{protocol}_throughput_{scenario}"] = i + 1

            for i, (protocol, _) in enumerate(reliability_ranking):
                rankings[f"{protocol}_reliability_{scenario}"] = i + 1

        # Create summary table
        summary_data = []
        for protocol in protocols:
            row = {"Protocol": protocol.upper()}

            # Average rankings across scenarios
            latency_ranks = [
                rankings.get(f"{protocol}_latency_{s}", 5) for s in scenarios
            ]
            throughput_ranks = [
                rankings.get(f"{protocol}_throughput_{s}", 5)
                for s in scenarios
            ]
            reliability_ranks = [
                rankings.get(f"{protocol}_reliability_{s}", 5)
                for s in scenarios
            ]

            row["Avg Latency Rank"] = f"{np.mean(latency_ranks):.1f}"
            row["Avg Throughput Rank"] = f"{np.mean(throughput_ranks):.1f}"
            row["Avg Reliability Rank"] = f"{np.mean(reliability_ranks):.1f}"
            ss = latency_ranks + throughput_ranks + reliability_ranks
            row["Overall Score"] = f"{np.mean(ss):.1f}"

            summary_data.append(row)

        # Sort by overall score
        summary_data.sort(key=lambda x: float(x["Overall Score"]))

        # Create table visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis("tight")
        ax.axis("off")

        df = pd.DataFrame(summary_data)
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc="center",
            loc="center",
        )

        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)

        # Color code the table
        for i in range(len(df)):
            for j in range(len(df.columns)):
                if j == 0:  # Protocol column
                    table[(i + 1, j)].set_facecolor("#E8F4FD")
                elif "Rank" in df.columns[j]:
                    rank_val = float(df.iloc[i, j])
                    if rank_val <= 1.5:
                        table[(i + 1, j)].set_facecolor(
                            "#90EE90"
                        )  # Light green
                    elif rank_val <= 2.5:
                        table[(i + 1, j)].set_facecolor(
                            "#FFFFE0"
                        )  # Light yellow
                    else:
                        table[(i + 1, j)].set_facecolor("#FFB6C1")  # Light red

        plt.title(
            "Communication Protocol Performance Rankings\n\
                Lower ranks are better)",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )

        output_path = os.path.join(self.results_dir, output_file)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Protocol ranking table saved to {output_path}")
        plt.close()

    def generate_summary_report(
        self, output_file: str = "benchmark_summary_report.txt"
    ):
        """Generate a text summary report of benchmark results."""
        if not self.data:
            print("No data available for summary report")
            return

        output_path = os.path.join(self.results_dir, output_file)

        with open(output_path, "w") as f:
            f.write("COMMUNICATION PROTOCOL BENCHMARK SUMMARY REPORT\n")
            f.write("=" * 60 + "\n\n")

            # Metadata
            metadata = self.data.get("benchmark_metadata", {})
            f.write(
                f"Benchmark Date: {metadata.get('timestamp', 'Unknown')}\n"
            )
            f.write(
                f"Total Duration: {metadata.get('total_duration_sec', 0):.1f}\
                    seconds\n"
            )
            f.write(f"Mode: {metadata.get('mode', 'Unknown')}\n")
            f.write(
                f"Protocols Tested:\
                    {', '.join(metadata.get('protocols_tested', []))}\n"
            )
            f.write(
                f"Scenarios Tested:\
                    {', '.join(metadata.get('scenarios_tested', []))}\n\n"
            )

            # Protocol comparisons
            f.write("CROSS-PROTOCOL COMPARISON\n")
            f.write("-" * 30 + "\n\n")

            for scenario, protocol_data in self.comparison_data.items():
                f.write(f"Scenario: {scenario.replace('_', ' ').title()}\n")

                # Sort protocols by latency for this scenario
                sorted_protocols = sorted(
                    protocol_data.items(),
                    key=lambda x: x[1].get("avg_latency_ms", float("inf")),
                )

                for protocol, metrics in sorted_protocols:
                    f.write(f"  {protocol.upper():>8}: ")
                    f.write(
                        f"Latency {metrics.get('avg_latency_ms', 0):6.1f}ms | "
                    )
                    f.write(
                        f"Throughput\
                            {metrics.get('throughput_msg_per_sec', 0):6.1f}\
                                msg/s | "
                    )
                    f.write(
                        f"Success\
                            {metrics.get('success_rate_percent', 0):5.1f}% | "
                    )
                    f.write(
                        f"CPU {metrics.get('cpu_usage_percent', 0):4.1f}% | "
                    )
                    f.write(
                        f"Memory {metrics.get('memory_usage_mb', 0):5.1f}MB\n"
                    )

                f.write("\n")

            # Best performing protocol per metric
            f.write("BEST PERFORMING PROTOCOLS BY METRIC\n")
            f.write("-" * 40 + "\n\n")

            all_protocols = self.get_available_protocols()

            # Find best overall performers
            best_latency = min(
                all_protocols,
                key=lambda p: np.mean(
                    [
                        self.comparison_data[s][p].get(
                            "avg_latency_ms", float("inf")
                        )
                        for s in self.get_available_scenarios()
                        if p in self.comparison_data[s]
                    ]
                ),
            )

            best_throughput = max(
                all_protocols,
                key=lambda p: np.mean(
                    [
                        self.comparison_data[s][p].get(
                            "throughput_msg_per_sec", 0
                        )
                        for s in self.get_available_scenarios()
                        if p in self.comparison_data[s]
                    ]
                ),
            )

            best_reliability = max(
                all_protocols,
                key=lambda p: np.mean(
                    [
                        self.comparison_data[s][p].get(
                            "success_rate_percent", 0
                        )
                        for s in self.get_available_scenarios()
                        if p in self.comparison_data[s]
                    ]
                ),
            )

            f.write(f"Lowest Latency:     {best_latency.upper()}\n")
            f.write(f"Highest Throughput: {best_throughput.upper()}\n")
            f.write(f"Most Reliable:      {best_reliability.upper()}\n\n")

            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 15 + "\n\n")

            if best_latency == best_throughput == best_reliability:
                f.write(
                    f"• {best_latency.upper()} is \
                        the best across all metrics.\n"
                )
            else:
                f.write(
                    f"• For low-latency applications: Use\
                        {best_latency.upper()}\n"
                )
                f.write(
                    f"• For high-throughput applications: Use\
                        {best_throughput.upper()}\n"
                )
                f.write(
                    f"• For mission-critical reliability: Use\
                        {best_reliability.upper()}\n"
                )

            f.write("\n" + "=" * 60 + "\n")

        print(f"Summary report saved to {output_path}")

    def generate_all_visualizations(self):
        """Generate all available visualizations."""
        print("\nGenerating benchmark visualizations...")

        self.create_latency_comparison_plot()
        self.create_throughput_comparison_plot()
        self.create_resource_usage_plot()
        self.create_performance_radar_chart()
        self.create_protocol_ranking_table()
        self.generate_summary_report()

        print("\n✅ All visualizations generated successfully!")


def main():
    """Main entry point for benchmark analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze and visualize communication protocol benchmarks"
    )

    parser.add_argument(
        "--results-file",
        help="Specific benchmark results JSON file to analyze",
    )

    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing benchmark results (default: results)",
    )

    parser.add_argument(
        "--plot",
        choices=[
            "latency",
            "throughput",
            "resources",
            "radar",
            "ranking",
            "all",
        ],
        default="all",
        help="Type of plot to generate (default: all)",
    )

    args = parser.parse_args()

    # Create analyzer
    analyzer = BenchmarkAnalyzer(args.results_file, args.results_dir)

    if not analyzer.data:
        print("No benchmark data loaded. Please run benchmarks first.")
        return

    # Generate requested visualizations
    if args.plot == "all":
        analyzer.generate_all_visualizations()
    elif args.plot == "latency":
        analyzer.create_latency_comparison_plot()
    elif args.plot == "throughput":
        analyzer.create_throughput_comparison_plot()
    elif args.plot == "resources":
        analyzer.create_resource_usage_plot()
    elif args.plot == "radar":
        analyzer.create_performance_radar_chart()
    elif args.plot == "ranking":
        analyzer.create_protocol_ranking_table()


if __name__ == "__main__":
    main()
