#!/usr/bin/env python3
"""
Benchmark Analysis and Visualization Utilities
Provides tools for analyzing benchmark results and generating visualizations
for cross-protocol performance comparison.
"""

import json
import math
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
        self.latency_mode = None  # Track latency mode from results

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
            # Extract latency mode from metadata
            metadata = self.data.get("benchmark_metadata", {})
            self.latency_mode = metadata.get("latency_mode", "unknown")
            print(f"Loaded results from {results_file}")
            print(f"Latency mode: {self.latency_mode}")

        except Exception as e:
            print(f"Error loading results: {e}")
            self.data = None
            self.comparison_data = None
            self.latency_mode = None

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

    def _format_variant_label(self, protocol: str, variant: str) -> str:
        protocol_display = protocol.upper()
        variant = variant or "default"
        if variant.lower() in {"default", protocol.lower()}:
            return protocol_display
        return f"{protocol_display} / {variant}"

    def _iter_scenario_metrics(self):
        if not self.data:
            return

        proto_results = self.data.get("protocol_results", {})

        for protocol, proto_data in proto_results.items():
            variant_map = proto_data.get("variants")
            if not variant_map:
                variant_map = {"default": proto_data}

            for variant, variant_data in variant_map.items():
                scenarios = variant_data.get("scenarios", {})
                for scenario_name, scenario_metrics in scenarios.items():
                    if not isinstance(scenario_metrics, dict):
                        continue
                    if "avg_latency_ms" in scenario_metrics:
                        yield (
                            protocol,
                            variant,
                            scenario_name,
                            scenario_metrics,
                            variant_data.get("metadata", {}),
                        )
                    else:
                        for label, metrics in scenario_metrics.items():
                            if (
                                isinstance(metrics, dict)
                                and "avg_latency_ms" in metrics
                            ):
                                scenario_label = f"{scenario_name}:{label}"
                                yield (
                                    protocol,
                                    variant,
                                    scenario_label,
                                    metrics,
                                    variant_data.get("metadata", {}),
                                )

    def _iter_concurrency_metrics(self):
        if not self.data:
            return

        proto_results = self.data.get("protocol_results", {})

        for protocol, proto_data in proto_results.items():
            variant_map = proto_data.get("variants")
            if not variant_map:
                variant_map = {"default": proto_data}

            for variant, variant_data in variant_map.items():
                concurrency_matrix = variant_data.get("concurrency_matrix", {})
                metadata = variant_data.get("metadata", {})

                for concurrency_level, metrics in concurrency_matrix.items():
                    if not isinstance(metrics, dict):
                        continue
                    yield (
                        protocol,
                        variant,
                        concurrency_level,
                        metrics,
                        metadata,
                    )

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
                avg_lat = metrics.get("avg_latency_ms", 0)
                p95_lat = metrics.get("p95_latency_ms", 0)

                if avg_lat == 0 and p95_lat == 0:
                    continue

                protocols.append(protocol.upper())
                scenarios.append(scenario.replace("_", " ").title())
                avg_latencies.append(avg_lat)
                p95_latencies.append(p95_lat)

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
        mode_label = (
            f" ({self.latency_mode.replace('_', ' ').title()})"
            if self.latency_mode
            else ""
        )
        ax1.set_title(
            f"Average Message Latency by Protocol{mode_label}",
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
            f"95th Percentile Message Latency by Protocol{mode_label}",
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

        mode_label = (
            f" ({self.latency_mode.replace('_', ' ').title()})"
            if self.latency_mode
            else ""
        )

        # Throughput plot
        sns.barplot(
            data=df,
            x="Scenario",
            y="Throughput_msg_per_sec",
            hue="Protocol",
            ax=ax1,
        )
        ax1.set_title(
            f"Message Throughput by Protocol{mode_label}",
            fontsize=14,
            fontweight="bold",
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
            f"Success Rate by Protocol{mode_label}",
            fontsize=14,
            fontweight="bold",
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

        mode_label = (
            f" ({self.latency_mode.replace('_', ' ').title()})"
            if self.latency_mode
            else ""
        )

        # CPU usage plot
        sns.barplot(
            data=df,
            x="Scenario",
            y="CPU_Usage_percent",
            hue="Protocol",
            ax=ax1,
        )
        ax1.set_title(
            f"CPU Usage by Protocol{mode_label}",
            fontsize=14,
            fontweight="bold",
        )
        ax1.set_ylabel("CPU Usage (%)")
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(True, alpha=0.3)

        # Memory usage plot
        sns.barplot(
            data=df, x="Scenario", y="Memory_Usage_mb", hue="Protocol", ax=ax2
        )
        ax2.set_title(
            f"Memory Usage by Protocol{mode_label}",
            fontsize=14,
            fontweight="bold",
        )
        ax2.set_ylabel("Memory Usage (MB)")
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = os.path.join(self.results_dir, output_file)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Resource usage plot saved to {output_path}")
        plt.close()

    def create_latency_cdf_plot(
        self,
        scenarios: Optional[List[str]] = None,
        output_file: str = "latency_cdf.png",
    ):
        """Create CDF plots of latency samples for selected scenarios."""

        records = []
        scenario_filter = set(scenarios) if scenarios else None

        for (
            protocol,
            variant,
            scenario_name,
            metrics,
            _,
        ) in self._iter_scenario_metrics():
            if scenario_filter and scenario_name not in scenario_filter:
                continue

            samples = metrics.get("latency_samples_ms") or []
            if not samples:
                continue

            label = self._format_variant_label(protocol, variant)
            scenario_display = scenario_name.replace("_", " ").title()
            records.extend(
                {
                    "Latency_ms": sample,
                    "Variant": label,
                    "Scenario": scenario_display,
                }
                for sample in samples
            )

        if not records:
            print("No latency sample data available for CDF plot")
            return

        df = pd.DataFrame(records)
        df = df.sort_values("Latency_ms")

        scenarios_order = df["Scenario"].unique()
        num_scenarios = len(scenarios_order)
        cols = min(3, num_scenarios)
        rows = math.ceil(num_scenarios / cols)

        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(cols * 5, rows * 4),
            squeeze=False,
        )

        for idx, scenario in enumerate(scenarios_order):
            r, c = divmod(idx, cols)
            ax = axes[r][c]
            subset = df[df["Scenario"] == scenario]
            sns.ecdfplot(data=subset, x="Latency_ms", hue="Variant", ax=ax)
            ax.set_title(f"{scenario}")
            ax.set_xlabel("Latency (ms)")
            ax.set_ylabel("CDF")
            ax.grid(True, alpha=0.3)
            ax.legend_.remove()

        # Hide unused subplots
        total_axes = rows * cols
        for idx in range(num_scenarios, total_axes):
            r, c = divmod(idx, cols)
            axes[r][c].axis("off")

        unique_variants = sorted(df["Variant"].unique())
        handles = []
        labels = []
        palette = sns.color_palette()
        for idx, variant in enumerate(unique_variants):
            handle = plt.Line2D(
                [0],
                [0],
                color=palette[idx % len(palette)],
                label=variant,
            )
            handles.append(handle)
            labels.append(variant)
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                ncol=min(4, len(labels)),
            )
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        output_path = os.path.join(self.results_dir, output_file)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Latency CDF plot saved to {output_path}")
        plt.close()

    def create_p99_vs_concurrency_plot(
        self, output_file: str = "p99_vs_concurrency.png"
    ):
        """Plot p99 latency against concurrency levels for each variant."""

        records = []
        for (
            protocol,
            variant,
            concurrency_level,
            metrics,
            _,
        ) in self._iter_concurrency_metrics():
            try:
                concurrency = int(concurrency_level)
            except (TypeError, ValueError):
                continue

            p99 = metrics.get("p99_latency_ms")
            if p99 is None:
                continue

            label = self._format_variant_label(protocol, variant)
            records.append(
                {
                    "Concurrency": concurrency,
                    "P99_Latency_ms": p99,
                    "Variant": label,
                }
            )

        if not records:
            print("No concurrency matrix data available for p99 plot")
            return

        df = pd.DataFrame(records)
        df = df.sort_values(["Variant", "Concurrency"])

        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=df,
            x="Concurrency",
            y="P99_Latency_ms",
            hue="Variant",
            marker="o",
        )
        plt.title("P99 Latency vs Concurrent Senders")
        plt.ylabel("P99 Latency (ms)")
        plt.grid(True, alpha=0.3)
        plt.xscale("log", base=2)
        plt.tight_layout()

        output_path = os.path.join(self.results_dir, output_file)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"P99 vs concurrency plot saved to {output_path}")
        plt.close()

    def create_throughput_vs_payload_plot(
        self,
        scenarios: Optional[List[str]] = None,
        output_file: str = "throughput_vs_payload.png",
    ):
        """Plot throughput as a function of payload size."""

        records = []
        scenario_filter = set(scenarios) if scenarios else None

        for (
            protocol,
            variant,
            scenario_name,
            metrics,
            _,
        ) in self._iter_scenario_metrics():
            if scenario_filter and scenario_name not in scenario_filter:
                continue

            payload = metrics.get("payload_size_bytes")
            throughput = metrics.get("throughput_msg_per_sec")
            if payload is None or throughput is None:
                continue

            label = self._format_variant_label(protocol, variant)
            scenario_display = scenario_name.replace("_", " ").title()
            records.append(
                {
                    "Payload_Bytes": payload,
                    "Throughput_msg_per_sec": throughput,
                    "Variant": label,
                    "Scenario": scenario_display,
                }
            )

        if not records:
            print("No throughput/payload data available for plotting")
            return

        df = pd.DataFrame(records)
        df = df.groupby(
            ["Variant", "Scenario", "Payload_Bytes"], as_index=False
        ).agg({"Throughput_msg_per_sec": "mean"})

        scenarios_order = df["Scenario"].unique()
        num_scenarios = len(scenarios_order)
        cols = min(3, num_scenarios)
        rows = math.ceil(num_scenarios / cols)

        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(cols * 5, rows * 4),
            squeeze=False,
        )

        for idx, scenario in enumerate(scenarios_order):
            r, c = divmod(idx, cols)
            ax = axes[r][c]
            subset = df[df["Scenario"] == scenario]
            sns.lineplot(
                data=subset,
                x="Payload_Bytes",
                y="Throughput_msg_per_sec",
                hue="Variant",
                marker="o",
                ax=ax,
            )
            ax.set_title(scenario)
            ax.set_xlabel("Payload Size (bytes)")
            ax.set_ylabel("Throughput (msg/s)")
            ax.grid(True, alpha=0.3)
            ax.legend_.remove()

        total_axes = rows * cols
        for idx in range(num_scenarios, total_axes):
            r, c = divmod(idx, cols)
            axes[r][c].axis("off")

        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="upper center", ncol=min(4, len(labels))
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        output_path = os.path.join(self.results_dir, output_file)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Throughput vs payload plot saved to {output_path}")
        plt.close()

    def create_performance_radar_chart(
        self, output_file: str = "performance_radar.png"
    ):
        """Create radar chart comparing overall protocol performance."""
        if not self.comparison_data:
            print("No comparison data available")
            return

        base_protocols = {
            protocol.lower(): protocol
            for protocol in self.get_available_protocols()
        }
        if not base_protocols:
            return

        # Aggregate metrics across scenarios for each protocol
        protocol_metrics = {
            canonical: {} for canonical in base_protocols.values()
        }

        for protocol_key, canonical_name in base_protocols.items():
            avg_latency = []
            avg_throughput = []
            avg_success_rate = []
            avg_cpu = []
            avg_memory = []

            for scenario, protocol_data in self.comparison_data.items():
                for variant_key, metrics in protocol_data.items():
                    variant_proto = variant_key.split("::")[0].lower()
                    if variant_proto != protocol_key:
                        continue

                    lat = metrics.get("avg_latency_ms", 0)
                    if lat > 0:
                        avg_latency.append(lat)

                    avg_throughput.append(
                        metrics.get("throughput_msg_per_sec", 0)
                    )
                    avg_success_rate.append(
                        metrics.get("success_rate_percent", 0)
                    )
                    avg_cpu.append(metrics.get("cpu_usage_percent", 0))
                    avg_memory.append(metrics.get("memory_usage_mb", 0))

            protocol_metrics[canonical_name] = {
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
        protocols = list(protocol_metrics.keys())
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

            hardware = metadata.get("hardware", {})
            if hardware:
                os_info = hardware.get("os", {})
                cpu_info = hardware.get("cpu", {})
                memory_info = hardware.get("memory", {})

                f.write("Hardware:\n")
                system_label = " ".join(
                    part
                    for part in [
                        os_info.get("system", ""),
                        os_info.get("release", ""),
                    ]
                    if part
                )
                if os_info.get("machine"):
                    system_label = f"{system_label} ({os_info['machine']})"
                if system_label:
                    f.write(f"  System: {system_label}\n")
                if os_info.get("node"):
                    f.write(f"  Node: {os_info['node']}\n")

                if cpu_info:
                    cpu_line = cpu_info.get("model", "unknown")
                    physical = cpu_info.get("physical_cores")
                    logical = cpu_info.get("logical_cores")
                    if physical or logical:
                        cpu_line += f" ({physical}C/{logical}T)"
                    f.write(f"  CPU: {cpu_line}\n")
                    if cpu_info.get("max_mhz"):
                        f.write(f"  CPU Max MHz: {cpu_info['max_mhz']:.0f}\n")

                if memory_info.get("total_gb"):
                    f.write(f"  Memory: {memory_info['total_gb']:.2f} GB\n")
                if hardware.get("python_version"):
                    f.write(f"  Python: {hardware['python_version']}\n")

                f.write("\n")

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

                    avg_lat = metrics.get("avg_latency_ms", 0)
                    if avg_lat == 0 and scenario == "broadcast_throughput":
                        f.write("(Skipped in end_to_end mode) | ")
                    else:
                        f.write(f"Latency {avg_lat:6.1f}ms | ")

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
            # Helper function to get metrics for a protocol across all variants
            def get_protocol_metrics(
                protocol: str, scenario: str, metric_key: str, default_value
            ):
                """Get all metrics for a protocol (including variants) in a scenario."""
                metrics = []
                protocol_lower = protocol.lower()
                scenario_data = self.comparison_data.get(scenario, {})

                for variant_key, variant_metrics in scenario_data.items():
                    # Extract base protocol from variant key (e.g., "rest::http1" -> "rest")
                    base_proto = variant_key.split("::")[0].lower()
                    if base_proto == protocol_lower:
                        value = variant_metrics.get(metric_key, default_value)
                        if value != default_value:  # Skip default values
                            metrics.append(value)

                return metrics

            best_latency = min(
                all_protocols,
                key=lambda p: np.mean(
                    [
                        metric
                        for s in self.get_available_scenarios()
                        for metric in get_protocol_metrics(
                            p, s, "avg_latency_ms", float("inf")
                        )
                    ]
                    or [float("inf")]  # Fallback if no metrics found
                ),
            )

            best_throughput = max(
                all_protocols,
                key=lambda p: np.mean(
                    [
                        metric
                        for s in self.get_available_scenarios()
                        for metric in get_protocol_metrics(
                            p, s, "throughput_msg_per_sec", 0
                        )
                    ]
                    or [0]  # Fallback if no metrics found
                ),
            )

            best_reliability = max(
                all_protocols,
                key=lambda p: np.mean(
                    [
                        metric
                        for s in self.get_available_scenarios()
                        for metric in get_protocol_metrics(
                            p, s, "success_rate_percent", 0
                        )
                    ]
                    or [0]  # Fallback if no metrics found
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

    def create_topology_comparison_plot(
        self, output_file: str = "topology_comparison.png"
    ):
        """Create topology comparison plot from individual protocol results."""
        if not self.data:
            print("No data available")
            return

        # Extract topology scenarios from protocol results
        topology_data = []
        topologies = ["fully_connected", "star", "ring", "chain"]

        for protocol, protocol_data in self.data.get(
            "protocol_results", {}
        ).items():
            variant_map = protocol_data.get("variants", {})
            if not variant_map:
                continue

            multiple_variants = len(variant_map) > 1

            for variant_name, variant_data in variant_map.items():
                label = (
                    f"{protocol.upper()} / {variant_name}"
                    if multiple_variants
                    else protocol.upper()
                )
                topology_comp = variant_data.get("topology_comparison", {})

                for topo, metrics in topology_comp.items():
                    if topo not in topologies:
                        continue
                    topology_data.append(
                        {
                            "Protocol": label,
                            "Topology": topo.replace("_", " ").title(),
                            "Avg_Latency_ms": metrics.get("avg_latency_ms", 0),
                            "Throughput_msg_per_sec": metrics.get(
                                "throughput_msg_per_sec", 0
                            ),
                            "Success_Rate_percent": metrics.get(
                                "success_rate_percent", 0
                            ),
                        }
                    )

        if not topology_data:
            print("No topology comparison data found")
            return

        df = pd.DataFrame(topology_data)

        # Create 1x3 subplot
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        # Latency by topology
        sns.barplot(
            data=df, x="Topology", y="Avg_Latency_ms", hue="Protocol", ax=ax1
        )
        ax1.set_title(
            "Latency by Topology Pattern", fontsize=12, fontweight="bold"
        )
        ax1.set_ylabel("Average Latency (ms)")
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(True, alpha=0.3, axis="y")

        # Throughput by topology
        sns.barplot(
            data=df,
            x="Topology",
            y="Throughput_msg_per_sec",
            hue="Protocol",
            ax=ax2,
        )
        ax2.set_title(
            "Throughput by Topology Pattern", fontsize=12, fontweight="bold"
        )
        ax2.set_ylabel("Throughput (msg/s)")
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(True, alpha=0.3, axis="y")

        # Success rate by topology
        sns.barplot(
            data=df,
            x="Topology",
            y="Success_Rate_percent",
            hue="Protocol",
            ax=ax3,
        )
        ax3.set_title(
            "Success Rate by Topology Pattern", fontsize=12, fontweight="bold"
        )
        ax3.set_ylabel("Success Rate (%)")
        ax3.set_ylim(0, 105)
        ax3.tick_params(axis="x", rotation=45)
        ax3.grid(True, alpha=0.3, axis="y")

        plt.suptitle(
            "Topology Pattern Performance Comparison",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()

        output_path = os.path.join(self.results_dir, output_file)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Topology comparison plot saved to {output_path}")
        plt.close()

    def create_scalability_analysis_plot(
        self, output_file: str = "scalability_analysis.png"
    ):
        """Create scalability analysis plot showing
        performance vs agent count"""
        if not self.data:
            print("No data available")
            return

        scalability_data = []

        for protocol, protocol_data in self.data.get(
            "protocol_results", {}
        ).items():
            variant_map = protocol_data.get("variants", {})
            if not variant_map:
                continue

            multiple_variants = len(variant_map) > 1

            for variant_name, variant_data in variant_map.items():
                label = (
                    f"{protocol.upper()} / {variant_name}"
                    if multiple_variants
                    else protocol.upper()
                )
                scalability_stress = variant_data.get("scenarios", {}).get(
                    "scalability_stress", {}
                )

                for agent_key, metrics in scalability_stress.items():
                    try:
                        count = int(agent_key.split("_")[0])
                    except (ValueError, IndexError):
                        continue

                    scalability_data.append(
                        {
                            "Protocol": label,
                            "Agent_Count": count,
                            "Avg_Latency_ms": metrics.get("avg_latency_ms", 0),
                            "Throughput_msg_per_sec": metrics.get(
                                "throughput_msg_per_sec", 0
                            ),
                            "Success_Rate_percent": metrics.get(
                                "success_rate_percent", 0
                            ),
                            "CPU_Usage_percent": metrics.get(
                                "cpu_usage_percent", 0
                            ),
                            "Memory_Usage_mb": metrics.get(
                                "memory_usage_mb", 0
                            ),
                        }
                    )

        if not scalability_data:
            print("No scalability analysis data found")
            return

        df = pd.DataFrame(scalability_data)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        for protocol in df["Protocol"].unique():
            protocol_df = df[df["Protocol"] == protocol]
            axes[0, 0].plot(
                protocol_df["Agent_Count"],
                protocol_df["Avg_Latency_ms"],
                marker="o",
                linewidth=2,
                label=protocol,
            )
        axes[0, 0].set_title(
            "Latency vs Team Size", fontsize=12, fontweight="bold"
        )
        axes[0, 0].set_xlabel("Number of Agents")
        axes[0, 0].set_ylabel("Average Latency (ms)")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Throughput vs Agent Count
        for protocol in df["Protocol"].unique():
            protocol_df = df[df["Protocol"] == protocol]
            axes[0, 1].plot(
                protocol_df["Agent_Count"],
                protocol_df["Throughput_msg_per_sec"],
                marker="s",
                linewidth=2,
                label=protocol,
            )
        axes[0, 1].set_title(
            "Throughput vs Team Size", fontsize=12, fontweight="bold"
        )
        axes[0, 1].set_xlabel("Number of Agents")
        axes[0, 1].set_ylabel("Throughput (msg/s)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: CPU Usage vs Agent Count
        for protocol in df["Protocol"].unique():
            protocol_df = df[df["Protocol"] == protocol]
            axes[1, 0].plot(
                protocol_df["Agent_Count"],
                protocol_df["CPU_Usage_percent"],
                marker="^",
                linewidth=2,
                label=protocol,
            )
        axes[1, 0].set_title(
            "CPU Usage vs Team Size", fontsize=12, fontweight="bold"
        )
        axes[1, 0].set_xlabel("Number of Agents")
        axes[1, 0].set_ylabel("CPU Usage (%)")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Memory Usage vs Agent Count
        for protocol in df["Protocol"].unique():
            protocol_df = df[df["Protocol"] == protocol]
            axes[1, 1].plot(
                protocol_df["Agent_Count"],
                protocol_df["Memory_Usage_mb"],
                marker="d",
                linewidth=2,
                label=protocol,
            )
        axes[1, 1].set_title(
            "Memory Usage vs Team Size", fontsize=12, fontweight="bold"
        )
        axes[1, 1].set_xlabel("Number of Agents")
        axes[1, 1].set_ylabel("Memory Usage (MB)")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(
            "Protocol Scalability Analysis",
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )
        plt.tight_layout()

        output_path = os.path.join(self.results_dir, output_file)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Scalability analysis plot saved to {output_path}")
        plt.close()

    def generate_all_visualizations(self):
        """Generate all available visualizations."""
        print("\nGenerating benchmark visualizations...")

        self.create_latency_comparison_plot()
        self.create_throughput_comparison_plot()
        self.create_resource_usage_plot()
        self.create_latency_cdf_plot()
        self.create_p99_vs_concurrency_plot()
        self.create_throughput_vs_payload_plot()
        self.create_performance_radar_chart()
        self.create_protocol_ranking_table()
        self.create_topology_comparison_plot()
        self.create_scalability_analysis_plot()
        self.generate_summary_report()

        print("\nAll visualizations generated successfully!")


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
            "topology",
            "scalability",
            "latency_cdf",
            "p99_concurrency",
            "throughput_payload",
            "all",
        ],
        default="all",
        help="Type of plot to generate (default: all)",
    )

    parser.add_argument(
        "--cdf-scenarios",
        nargs="+",
        help="Specific scenarios to include in latency CDF plots",
    )

    parser.add_argument(
        "--payload-scenarios",
        nargs="+",
        help="Scenarios to include in throughput vs payload plots",
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
    elif args.plot == "topology":
        analyzer.create_topology_comparison_plot()
    elif args.plot == "scalability":
        analyzer.create_scalability_analysis_plot()
    elif args.plot == "latency_cdf":
        analyzer.create_latency_cdf_plot(scenarios=args.cdf_scenarios)
    elif args.plot == "p99_concurrency":
        analyzer.create_p99_vs_concurrency_plot()
    elif args.plot == "throughput_payload":
        analyzer.create_throughput_vs_payload_plot(
            scenarios=args.payload_scenarios
        )


if __name__ == "__main__":
    main()
