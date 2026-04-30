#!/usr/bin/env python3
"""
Distributed Benchmark Analysis and Visualization.

Reads results produced by distributed_runner.py and generates
plots tailored to the distributed result format (multi-host,
multi-trial with confidence intervals, agent-count scaling).

Usage:
    python3 -m benchmarks.distributed.distributed_analysis \
        --results-file results/distributed_benchmarks/distributed_results_*.json
    python3 -m benchmarks.distributed.distributed_analysis \
        --results-dir results/distributed_benchmarks
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class DistributedBenchmarkAnalyzer:
    """Analyzer for distributed benchmark results."""

    def __init__(
        self,
        results_file: Optional[str] = None,
        results_dir: str = "results/distributed_benchmarks",
    ):
        self.results_dir = results_dir
        self.data: Optional[Dict[str, Any]] = None
        self.latency_mode: Optional[str] = None

        if results_file:
            self.load_results(results_file)
        else:
            self.load_latest_results()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_results(self, results_file: str):
        """Load distributed benchmark results from JSON file."""
        with open(results_file, "r") as f:
            self.data = json.load(f)

        metadata = self.data.get("benchmark_metadata", {})
        self.latency_mode = metadata.get("latency_mode", "unknown")
        print(f"Loaded results from {results_file}")
        print(f"  Execution mode: {metadata.get('execution_mode')}")
        print(f"  Latency mode: {self.latency_mode}")
        print(f"  Protocols: {metadata.get('protocols_tested')}")
        print(f"  Scenarios: {metadata.get('scenarios_tested')}")
        hosts = metadata.get("hosts", {})
        print(f"  Hosts: {len(hosts)} ({', '.join(hosts.keys())})")

    def load_latest_results(self):
        """Load the most recent distributed results file."""
        pattern = os.path.join(self.results_dir, "distributed_results_*.json")
        files = glob.glob(pattern)
        if not files:
            print(f"No distributed results found in {self.results_dir}")
            return
        latest = max(files, key=os.path.getctime)
        self.load_results(latest)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _results(self) -> Dict[str, Any]:
        """Return the results dict or empty."""
        if not self.data:
            return {}
        return self.data.get("results", {})

    def _mode_label(self) -> str:
        if self.latency_mode and self.latency_mode != "unknown":
            return f" ({self.latency_mode.replace('_', ' ').title()})"
        return ""

    def _format_variant_label(self, protocol: str, variant: str) -> str:
        protocol_display = protocol.upper()
        if variant in ("default", protocol):
            return protocol_display
        return f"{protocol_display} / {variant}"

    def _iter_results(self):
        """Yield (key, protocol, variant, scenario, agent_count, entry)
        for every result entry."""
        for key, entry in self._results().items():
            yield (
                key,
                entry["protocol"],
                entry["variant"],
                entry["scenario"],
                entry["agent_count"],
                entry,
            )

    def _build_dataframe(self) -> pd.DataFrame:
        """Build a flat DataFrame from all result entries."""
        rows = []
        for (
            key,
            protocol,
            variant,
            scenario,
            agent_count,
            entry,
        ) in self._iter_results():
            agg = entry.get("aggregated", {})
            label = self._format_variant_label(protocol, variant)
            scenario_display = scenario.replace("_", " ").title()

            row = {
                "key": key,
                "protocol": protocol,
                "variant": variant,
                "label": label,
                "scenario": scenario,
                "scenario_display": scenario_display,
                "agent_count": agent_count,
                # Latency
                "latency_avg_ms": agg.get("latency_avg_ms", 0),
                "latency_avg_ci_lo": agg.get("latency_avg", {}).get(
                    "ci_lower", 0
                ),
                "latency_avg_ci_hi": agg.get("latency_avg", {}).get(
                    "ci_upper", 0
                ),
                "latency_p95_ms": agg.get("latency_p95", {}).get("mean", 0),
                "latency_p95_ci_lo": agg.get("latency_p95", {}).get(
                    "ci_lower", 0
                ),
                "latency_p95_ci_hi": agg.get("latency_p95", {}).get(
                    "ci_upper", 0
                ),
                "latency_p99_ms": agg.get("latency_p99", {}).get("mean", 0),
                "latency_p99_ci_lo": agg.get("latency_p99", {}).get(
                    "ci_lower", 0
                ),
                "latency_p99_ci_hi": agg.get("latency_p99", {}).get(
                    "ci_upper", 0
                ),
                # Throughput
                "throughput_avg": agg.get("throughput_avg", 0),
                "throughput_ci_lo": agg.get("throughput", {}).get(
                    "ci_lower", 0
                ),
                "throughput_ci_hi": agg.get("throughput", {}).get(
                    "ci_upper", 0
                ),
                # Reliability
                "success_rate": agg.get("success_rate", {}).get("mean", 0),
                "success_rate_ci_lo": agg.get("success_rate", {}).get(
                    "ci_lower", 0
                ),
                "success_rate_ci_hi": agg.get("success_rate", {}).get(
                    "ci_upper", 0
                ),
                "num_trials": agg.get("num_trials", 0),
            }

            # Per-trial resource averages
            trials = entry.get("trials", [])
            cpu_vals = [
                t.get("cpu_avg", 0) for t in trials if t.get("cpu_avg")
            ]
            mem_vals = [
                t.get("memory_avg_mb", 0)
                for t in trials
                if t.get("memory_avg_mb")
            ]
            row["cpu_avg"] = statistics.mean(cpu_vals) if cpu_vals else 0
            row["memory_avg_mb"] = statistics.mean(mem_vals) if mem_vals else 0

            rows.append(row)

        return pd.DataFrame(rows)

    def _collect_latency_samples(
        self,
        scenario_filter: Optional[set] = None,
    ) -> pd.DataFrame:
        """Collect raw latency samples across all trials into a DataFrame."""
        records = []
        for (
            key,
            protocol,
            variant,
            scenario,
            agent_count,
            entry,
        ) in self._iter_results():
            if scenario_filter and scenario not in scenario_filter:
                continue
            label = self._format_variant_label(protocol, variant)
            scenario_display = scenario.replace("_", " ").title()
            for trial in entry.get("trials", []):
                for sample in trial.get("latency_samples", []):
                    # Samples are in seconds from the worker
                    records.append(
                        {
                            "latency_ms": sample * 1000,
                            "label": label,
                            "scenario": scenario_display,
                            "agent_count": agent_count,
                        }
                    )
        return pd.DataFrame(records)

    def _savefig(self, fig, filename: str):
        os.makedirs(self.results_dir, exist_ok=True)
        path = os.path.join(self.results_dir, filename)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def create_latency_comparison(
        self, output_file: str = "dist_latency_comparison.png"
    ):
        """Bar chart of avg and p95 latency per protocol/scenario, with CI."""
        df = self._build_dataframe()
        if df.empty:
            print("No data for latency comparison")
            return

        # Use the largest agent_count so every protocol is represented
        max_ac = df["agent_count"].max()
        df = df[df["agent_count"] == max_ac].copy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # --- Average latency ---
        order = sorted(df["scenario_display"].unique())
        hue_order = sorted(df["label"].unique())
        sns.barplot(
            data=df,
            x="scenario_display",
            y="latency_avg_ms",
            hue="label",
            order=order,
            hue_order=hue_order,
            ax=ax1,
        )
        # Error bars
        for container_idx, lbl in enumerate(hue_order):
            sub = (
                df[df["label"] == lbl]
                .set_index("scenario_display")
                .reindex(order)
            )
            bars = ax1.containers[container_idx]
            yerr_lo = (
                (sub["latency_avg_ms"] - sub["latency_avg_ci_lo"])
                .clip(lower=0)
                .values
            )
            yerr_hi = (
                (sub["latency_avg_ci_hi"] - sub["latency_avg_ms"])
                .clip(lower=0)
                .values
            )
            ax1.errorbar(
                [b.get_x() + b.get_width() / 2 for b in bars],
                sub["latency_avg_ms"].values,
                yerr=[yerr_lo, yerr_hi],
                fmt="none",
                ecolor="black",
                capsize=3,
            )

        ax1.set_title(
            f"Average Latency{self._mode_label()}",
            fontsize=13,
            fontweight="bold",
        )
        ax1.set_ylabel("Latency (ms)")
        ax1.set_xlabel("")
        ax1.tick_params(axis="x", rotation=30)
        ax1.grid(True, alpha=0.3, axis="y")

        # --- P95 latency ---
        sns.barplot(
            data=df,
            x="scenario_display",
            y="latency_p95_ms",
            hue="label",
            order=order,
            hue_order=hue_order,
            ax=ax2,
        )
        for container_idx, lbl in enumerate(hue_order):
            sub = (
                df[df["label"] == lbl]
                .set_index("scenario_display")
                .reindex(order)
            )
            bars = ax2.containers[container_idx]
            yerr_lo = (
                (sub["latency_p95_ms"] - sub["latency_p95_ci_lo"])
                .clip(lower=0)
                .values
            )
            yerr_hi = (
                (sub["latency_p95_ci_hi"] - sub["latency_p95_ms"])
                .clip(lower=0)
                .values
            )
            ax2.errorbar(
                [b.get_x() + b.get_width() / 2 for b in bars],
                sub["latency_p95_ms"].values,
                yerr=[yerr_lo, yerr_hi],
                fmt="none",
                ecolor="black",
                capsize=3,
            )

        ax2.set_title(
            f"P95 Latency{self._mode_label()}", fontsize=13, fontweight="bold"
        )
        ax2.set_ylabel("Latency (ms)")
        ax2.set_xlabel("")
        ax2.tick_params(axis="x", rotation=30)
        ax2.grid(True, alpha=0.3, axis="y")

        fig.suptitle(
            f"Distributed Latency Comparison ({max_ac} agents)",
            fontsize=15,
            fontweight="bold",
        )
        fig.tight_layout()
        self._savefig(fig, output_file)

    def create_throughput_comparison(
        self, output_file: str = "dist_throughput_comparison.png"
    ):
        """Bar chart of throughput and success rate per protocol/scenario."""
        df = self._build_dataframe()
        if df.empty:
            print("No data for throughput comparison")
            return

        max_ac = df["agent_count"].max()
        df = df[df["agent_count"] == max_ac].copy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        order = sorted(df["scenario_display"].unique())
        hue_order = sorted(df["label"].unique())

        # Throughput
        sns.barplot(
            data=df,
            x="scenario_display",
            y="throughput_avg",
            hue="label",
            order=order,
            hue_order=hue_order,
            ax=ax1,
        )
        for container_idx, lbl in enumerate(hue_order):
            sub = (
                df[df["label"] == lbl]
                .set_index("scenario_display")
                .reindex(order)
            )
            bars = ax1.containers[container_idx]
            yerr_lo = (
                (sub["throughput_avg"] - sub["throughput_ci_lo"])
                .clip(lower=0)
                .values
            )
            yerr_hi = (
                (sub["throughput_ci_hi"] - sub["throughput_avg"])
                .clip(lower=0)
                .values
            )
            ax1.errorbar(
                [b.get_x() + b.get_width() / 2 for b in bars],
                sub["throughput_avg"].values,
                yerr=[yerr_lo, yerr_hi],
                fmt="none",
                ecolor="black",
                capsize=3,
            )

        ax1.set_title(
            f"Throughput{self._mode_label()}", fontsize=13, fontweight="bold"
        )
        ax1.set_ylabel("Messages / sec")
        ax1.set_xlabel("")
        ax1.tick_params(axis="x", rotation=30)
        ax1.grid(True, alpha=0.3, axis="y")

        # Success rate
        sns.barplot(
            data=df,
            x="scenario_display",
            y="success_rate",
            hue="label",
            order=order,
            hue_order=hue_order,
            ax=ax2,
        )
        ax2.set_title(
            f"Success Rate{self._mode_label()}", fontsize=13, fontweight="bold"
        )
        ax2.set_ylabel("Success Rate (%)")
        ax2.set_xlabel("")
        ax2.set_ylim(0, 105)
        ax2.tick_params(axis="x", rotation=30)
        ax2.grid(True, alpha=0.3, axis="y")

        fig.suptitle(
            f"Distributed Throughput Comparison ({max_ac} agents)",
            fontsize=15,
            fontweight="bold",
        )
        fig.tight_layout()
        self._savefig(fig, output_file)

    def create_scalability_plot(
        self, output_file: str = "dist_scalability.png"
    ):
        """Line plots of latency, throughput, CPU, and memory vs agent count."""
        df = self._build_dataframe()
        if df.empty:
            print("No data for scalability plot")
            return

        scenarios = sorted(df["scenario"].unique())
        num_scenarios = len(scenarios)

        fig, axes = plt.subplots(
            num_scenarios,
            2,
            figsize=(14, 5 * num_scenarios),
            squeeze=False,
        )

        for row_idx, scenario in enumerate(scenarios):
            sdf = df[df["scenario"] == scenario].sort_values("agent_count")
            scenario_display = scenario.replace("_", " ").title()

            # Latency vs agent count
            ax_lat = axes[row_idx][0]
            for label in sorted(sdf["label"].unique()):
                sub = sdf[sdf["label"] == label]
                ax_lat.errorbar(
                    sub["agent_count"],
                    sub["latency_avg_ms"],
                    yerr=[
                        (
                            sub["latency_avg_ms"] - sub["latency_avg_ci_lo"]
                        ).clip(lower=0),
                        (
                            sub["latency_avg_ci_hi"] - sub["latency_avg_ms"]
                        ).clip(lower=0),
                    ],
                    marker="o",
                    linewidth=2,
                    capsize=4,
                    label=label,
                )
            ax_lat.set_title(
                f"{scenario_display} — Latency", fontweight="bold"
            )
            ax_lat.set_xlabel("Agent Count")
            ax_lat.set_ylabel("Avg Latency (ms)")
            ax_lat.legend()
            ax_lat.grid(True, alpha=0.3)

            # Throughput vs agent count
            ax_tp = axes[row_idx][1]
            for label in sorted(sdf["label"].unique()):
                sub = sdf[sdf["label"] == label]
                ax_tp.errorbar(
                    sub["agent_count"],
                    sub["throughput_avg"],
                    yerr=[
                        (sub["throughput_avg"] - sub["throughput_ci_lo"]).clip(
                            lower=0
                        ),
                        (sub["throughput_ci_hi"] - sub["throughput_avg"]).clip(
                            lower=0
                        ),
                    ],
                    marker="s",
                    linewidth=2,
                    capsize=4,
                    label=label,
                )
            ax_tp.set_title(
                f"{scenario_display} — Throughput", fontweight="bold"
            )
            ax_tp.set_xlabel("Agent Count")
            ax_tp.set_ylabel("Messages / sec")
            ax_tp.legend()
            ax_tp.grid(True, alpha=0.3)

        fig.suptitle(
            f"Scalability Analysis{self._mode_label()}",
            fontsize=15,
            fontweight="bold",
        )
        fig.tight_layout()
        self._savefig(fig, output_file)

    def create_resource_usage_plot(
        self, output_file: str = "dist_resource_usage.png"
    ):
        """Bar chart of CPU and memory usage per protocol."""
        df = self._build_dataframe()
        if df.empty:
            print("No data for resource usage plot")
            return

        max_ac = df["agent_count"].max()
        df = df[df["agent_count"] == max_ac].copy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        order = sorted(df["scenario_display"].unique())
        hue_order = sorted(df["label"].unique())

        sns.barplot(
            data=df,
            x="scenario_display",
            y="cpu_avg",
            hue="label",
            order=order,
            hue_order=hue_order,
            ax=ax1,
        )
        ax1.set_title("CPU Usage", fontsize=13, fontweight="bold")
        ax1.set_ylabel("CPU (%)")
        ax1.set_xlabel("")
        ax1.tick_params(axis="x", rotation=30)
        ax1.grid(True, alpha=0.3, axis="y")

        sns.barplot(
            data=df,
            x="scenario_display",
            y="memory_avg_mb",
            hue="label",
            order=order,
            hue_order=hue_order,
            ax=ax2,
        )
        ax2.set_title("Memory Usage", fontsize=13, fontweight="bold")
        ax2.set_ylabel("Memory (MB)")
        ax2.set_xlabel("")
        ax2.tick_params(axis="x", rotation=30)
        ax2.grid(True, alpha=0.3, axis="y")

        fig.suptitle(
            f"Resource Usage ({max_ac} agents){self._mode_label()}",
            fontsize=15,
            fontweight="bold",
        )
        fig.tight_layout()
        self._savefig(fig, output_file)

    def create_latency_cdf_plot(
        self,
        scenarios: Optional[List[str]] = None,
        output_file: str = "dist_latency_cdf.png",
    ):
        """CDF of raw latency samples, one subplot per scenario."""
        scenario_filter = set(scenarios) if scenarios else None
        df = self._collect_latency_samples(scenario_filter)
        if df.empty:
            print("No latency samples available for CDF plot")
            return

        scenario_order = sorted(df["scenario"].unique())
        num = len(scenario_order)
        cols = min(3, num)
        rows = math.ceil(num / cols)

        fig, axes = plt.subplots(
            rows, cols, figsize=(cols * 5, rows * 4), squeeze=False
        )

        for idx, scenario in enumerate(scenario_order):
            r, c = divmod(idx, cols)
            ax = axes[r][c]
            subset = df[df["scenario"] == scenario]
            sns.ecdfplot(data=subset, x="latency_ms", hue="label", ax=ax)
            ax.set_title(scenario)
            ax.set_xlabel("Latency (ms)")
            ax.set_ylabel("CDF")
            ax.grid(True, alpha=0.3)
            if ax.get_legend():
                ax.get_legend().remove()

        # Hide unused subplots
        for idx in range(num, rows * cols):
            r, c = divmod(idx, cols)
            axes[r][c].axis("off")

        # Shared legend
        unique_labels = sorted(df["label"].unique())
        palette = sns.color_palette()
        handles = [
            plt.Line2D([0], [0], color=palette[i % len(palette)], label=lbl)
            for i, lbl in enumerate(unique_labels)
        ]
        fig.legend(
            handles,
            unique_labels,
            loc="upper center",
            ncol=min(4, len(unique_labels)),
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        self._savefig(fig, output_file)

    def create_performance_radar(
        self, output_file: str = "dist_performance_radar.png"
    ):
        """Radar chart comparing protocols across normalised metrics."""
        df = self._build_dataframe()
        if df.empty:
            print("No data for radar chart")
            return

        max_ac = df["agent_count"].max()
        df = df[df["agent_count"] == max_ac].copy()

        # Aggregate per protocol label
        proto_stats: Dict[str, Dict[str, float]] = {}
        for label in sorted(df["label"].unique()):
            sub = df[df["label"] == label]
            proto_stats[label] = {
                "avg_latency": sub["latency_avg_ms"].mean(),
                "throughput": sub["throughput_avg"].mean(),
                "success_rate": sub["success_rate"].mean(),
                "cpu": sub["cpu_avg"].mean(),
                "memory": sub["memory_avg_mb"].mean(),
            }

        # Normalise: invert latency/cpu/memory so higher = better
        max_lat = max(s["avg_latency"] for s in proto_stats.values()) or 1
        max_tp = max(s["throughput"] for s in proto_stats.values()) or 1
        max_cpu = max(s["cpu"] for s in proto_stats.values()) or 1
        max_mem = max(s["memory"] for s in proto_stats.values()) or 1

        categories = [
            "Low Latency",
            "High Throughput",
            "Reliability",
            "Low CPU",
            "Low Memory",
        ]
        num_cat = len(categories)
        angles = [n / num_cat * 2 * np.pi for n in range(num_cat)]
        angles += angles[:1]

        fig, ax = plt.subplots(
            figsize=(9, 9), subplot_kw=dict(projection="polar")
        )
        colors = [
            "#FF6B6B",
            "#4ECDC4",
            "#45B7D1",
            "#96CEB4",
            "#FFEAA7",
            "#DDA0DD",
        ]

        for i, (label, stats) in enumerate(proto_stats.items()):
            values = [
                (1 - stats["avg_latency"] / max_lat) * 100 if max_lat else 0,
                (stats["throughput"] / max_tp) * 100 if max_tp else 0,
                stats["success_rate"],
                (1 - stats["cpu"] / max_cpu) * 100 if max_cpu else 0,
                (1 - stats["memory"] / max_mem) * 100 if max_mem else 0,
            ]
            values += values[:1]
            color = colors[i % len(colors)]
            ax.plot(
                angles, values, "o-", linewidth=2, label=label, color=color
            )
            ax.fill(angles, values, alpha=0.2, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.set_title(
            "Protocol Performance Comparison\n(higher = better)",
            size=14,
            fontweight="bold",
            pad=20,
        )
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        self._savefig(fig, output_file)

    def create_protocol_ranking_table(
        self, output_file: str = "dist_protocol_ranking.png"
    ):
        """Table visualisation ranking protocols by metric."""
        df = self._build_dataframe()
        if df.empty:
            print("No data for ranking table")
            return

        max_ac = df["agent_count"].max()
        df = df[df["agent_count"] == max_ac].copy()

        labels = sorted(df["label"].unique())
        scenarios = sorted(df["scenario_display"].unique())

        rankings: Dict[str, Dict[str, float]] = {l: {} for l in labels}

        for scenario in scenarios:
            sdf = df[df["scenario_display"] == scenario]
            # Latency rank (lower is better)
            for rank, (_, row) in enumerate(
                sdf.sort_values("latency_avg_ms").iterrows(), 1
            ):
                rankings[row["label"]][f"lat_{scenario}"] = rank
            # Throughput rank (higher is better)
            for rank, (_, row) in enumerate(
                sdf.sort_values("throughput_avg", ascending=False).iterrows(),
                1,
            ):
                rankings[row["label"]][f"tp_{scenario}"] = rank
            # Reliability rank
            for rank, (_, row) in enumerate(
                sdf.sort_values("success_rate", ascending=False).iterrows(), 1
            ):
                rankings[row["label"]][f"rel_{scenario}"] = rank

        summary_rows = []
        for label in labels:
            r = rankings[label]
            lat_ranks = [v for k, v in r.items() if k.startswith("lat_")]
            tp_ranks = [v for k, v in r.items() if k.startswith("tp_")]
            rel_ranks = [v for k, v in r.items() if k.startswith("rel_")]
            all_ranks = lat_ranks + tp_ranks + rel_ranks
            summary_rows.append(
                {
                    "Protocol": label,
                    "Avg Latency Rank": (
                        f"{np.mean(lat_ranks):.1f}" if lat_ranks else "-"
                    ),
                    "Avg Throughput Rank": (
                        f"{np.mean(tp_ranks):.1f}" if tp_ranks else "-"
                    ),
                    "Avg Reliability Rank": (
                        f"{np.mean(rel_ranks):.1f}" if rel_ranks else "-"
                    ),
                    "Overall Score": (
                        f"{np.mean(all_ranks):.1f}" if all_ranks else "-"
                    ),
                }
            )

        summary_rows.sort(
            key=lambda x: (
                float(x["Overall Score"]) if x["Overall Score"] != "-" else 99
            )
        )
        tdf = pd.DataFrame(summary_rows)

        fig, ax = plt.subplots(figsize=(12, max(3, 1 + len(summary_rows))))
        ax.axis("off")
        table = ax.table(
            cellText=tdf.values,
            colLabels=tdf.columns,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.5)

        for i in range(len(tdf)):
            for j in range(len(tdf.columns)):
                if j == 0:
                    table[(i + 1, j)].set_facecolor("#E8F4FD")
                elif "Rank" in tdf.columns[j] or "Score" in tdf.columns[j]:
                    try:
                        val = float(tdf.iloc[i, j])
                    except ValueError:
                        continue
                    if val <= 1.5:
                        table[(i + 1, j)].set_facecolor("#90EE90")
                    elif val <= 2.5:
                        table[(i + 1, j)].set_facecolor("#FFFFE0")
                    else:
                        table[(i + 1, j)].set_facecolor("#FFB6C1")

        ax.set_title(
            f"Protocol Rankings ({max_ac} agents) — lower is better",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        fig.tight_layout()
        self._savefig(fig, output_file)

    def generate_summary_report(
        self, output_file: str = "dist_benchmark_summary.txt"
    ):
        """Generate a text summary of the distributed benchmark."""
        if not self.data:
            print("No data for summary report")
            return

        os.makedirs(self.results_dir, exist_ok=True)
        path = os.path.join(self.results_dir, output_file)

        metadata = self.data.get("benchmark_metadata", {})

        with open(path, "w") as f:
            f.write("DISTRIBUTED BENCHMARK SUMMARY REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Timestamp:    {metadata.get('timestamp', '?')}\n")
            f.write(
                f"Duration:     {metadata.get('total_duration_sec', 0):.1f}s\n"
            )
            f.write(f"Latency mode: {metadata.get('latency_mode', '?')}\n")
            f.write(f"Trials:       {metadata.get('num_trials', '?')}\n")
            f.write(
                f"Protocols:    {', '.join(metadata.get('protocols_tested', []))}\n"
            )
            f.write(
                f"Scenarios:    {', '.join(metadata.get('scenarios_tested', []))}\n\n"
            )

            hosts = metadata.get("hosts", {})
            f.write(f"Hosts ({len(hosts)}):\n")
            for name, info in hosts.items():
                f.write(
                    f"  {name:>10}: {info.get('ip', '?')} ({info.get('role', '?')})\n"
                )

            offsets = metadata.get("clock_offsets_ms", {})
            if offsets:
                f.write("\nClock offsets (ms):\n")
                for name, off in offsets.items():
                    f.write(f"  {name:>10}: {off:+.2f}\n")

            f.write("\n" + "-" * 60 + "\n")
            f.write("RESULTS\n")
            f.write("-" * 60 + "\n\n")

            for (
                key,
                protocol,
                variant,
                scenario,
                agent_count,
                entry,
            ) in self._iter_results():
                agg = entry.get("aggregated", {})
                label = self._format_variant_label(protocol, variant)
                f.write(
                    f"{label} / {scenario.replace('_', ' ').title()} / {agent_count} agents\n"
                )

                lat_avg = agg.get("latency_avg_ms", 0)
                lat_p95 = agg.get("latency_p95", {}).get("mean", 0)
                lat_p99 = agg.get("latency_p99", {}).get("mean", 0)
                tp = agg.get("throughput_avg", 0)
                sr = agg.get("success_rate", {}).get("mean", 0)

                f.write(
                    f"  Latency  avg={lat_avg:7.2f}ms  p95={lat_p95:7.2f}ms  p99={lat_p99:7.2f}ms\n"
                )
                f.write(
                    f"  Throughput {tp:7.1f} msg/s  |  Success {sr:5.1f}%\n"
                )

                # CI info
                lat_ci = agg.get("latency_avg", {})
                tp_ci = agg.get("throughput", {})
                if lat_ci.get("ci_lower") is not None:
                    f.write(
                        f"  CI(95%) latency [{lat_ci.get('ci_lower', 0):.2f}, "
                        f"{lat_ci.get('ci_upper', 0):.2f}]  "
                        f"throughput [{tp_ci.get('ci_lower', 0):.1f}, "
                        f"{tp_ci.get('ci_upper', 0):.1f}]\n"
                    )
                f.write("\n")

            # Best performers
            df = self._build_dataframe()
            if not df.empty:
                max_ac = df["agent_count"].max()
                sub = df[df["agent_count"] == max_ac]
                if not sub.empty:
                    f.write("-" * 60 + "\n")
                    f.write(f"BEST PERFORMERS ({max_ac} agents)\n")
                    f.write("-" * 60 + "\n\n")

                    best_lat = sub.loc[sub["latency_avg_ms"].idxmin()]
                    best_tp = sub.loc[sub["throughput_avg"].idxmax()]
                    best_rel = sub.loc[sub["success_rate"].idxmax()]

                    f.write(
                        f"  Lowest latency:     {best_lat['label']} ({best_lat['latency_avg_ms']:.2f}ms)\n"
                    )
                    f.write(
                        f"  Highest throughput: {best_tp['label']} ({best_tp['throughput_avg']:.1f} msg/s)\n"
                    )
                    f.write(
                        f"  Most reliable:      {best_rel['label']} ({best_rel['success_rate']:.1f}%)\n"
                    )

            f.write("\n" + "=" * 60 + "\n")

        print(f"  Summary report saved to {path}")

    # ------------------------------------------------------------------
    # Generate all
    # ------------------------------------------------------------------

    def generate_all_visualizations(self):
        """Generate all available plots and the summary report."""
        if not self.data:
            print("No data loaded.")
            return

        print("\nGenerating distributed benchmark visualizations...")
        self.create_latency_comparison()
        self.create_throughput_comparison()
        self.create_scalability_plot()
        self.create_resource_usage_plot()
        self.create_latency_cdf_plot()
        self.create_performance_radar()
        self.create_protocol_ranking_table()
        self.generate_summary_report()
        print("\nAll visualizations generated!")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze distributed benchmark results"
    )
    parser.add_argument(
        "--results-file",
        help="Path to a specific distributed results JSON file",
    )
    parser.add_argument(
        "--results-dir",
        default="results/distributed_benchmarks",
        help="Directory containing distributed results (default: results/distributed_benchmarks)",
    )
    parser.add_argument(
        "--plot",
        choices=[
            "latency",
            "throughput",
            "scalability",
            "resources",
            "cdf",
            "radar",
            "ranking",
            "summary",
            "all",
        ],
        default="all",
        help="Which visualisation to generate (default: all)",
    )
    parser.add_argument(
        "--cdf-scenarios",
        nargs="+",
        help="Restrict CDF plot to these scenarios",
    )
    args = parser.parse_args()

    analyzer = DistributedBenchmarkAnalyzer(
        args.results_file, args.results_dir
    )
    if not analyzer.data:
        print("No data loaded. Run distributed benchmarks first.")
        return

    plot_map = {
        "latency": analyzer.create_latency_comparison,
        "throughput": analyzer.create_throughput_comparison,
        "scalability": analyzer.create_scalability_plot,
        "resources": analyzer.create_resource_usage_plot,
        "cdf": lambda: analyzer.create_latency_cdf_plot(
            scenarios=args.cdf_scenarios
        ),
        "radar": analyzer.create_performance_radar,
        "ranking": analyzer.create_protocol_ranking_table,
        "summary": analyzer.generate_summary_report,
    }

    if args.plot == "all":
        analyzer.generate_all_visualizations()
    else:
        plot_map[args.plot]()


if __name__ == "__main__":
    main()
