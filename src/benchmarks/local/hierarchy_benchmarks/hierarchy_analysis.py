#!/usr/bin/env python3
"""
Hierarchy Benchmark Analysis and Visualization
Provides tools for analyzing hierarchy strategy benchmark results and generating
comprehensive visualizations for comparing tree, peer-to-peer, and hybrid strategies.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import glob
import os
from typing import List, Optional, Dict, Any


class HierarchyBenchmarkAnalyzer:
    """Analyzer for hierarchy strategy benchmark results."""

    def __init__(
        self,
        results_file: Optional[str] = None,
        results_dir: str = "results/hierarchy",
    ):
        self.results_dir = results_dir
        self.data = None
        self.benchmarks = []
        self.strategies = []
        self.environments = []
        self.agent_counts = []

        if results_file:
            self.load_results(results_file)
        else:
            self.load_latest_results()

    def load_results(self, results_file: str):
        """Load hierarchy benchmark results from JSON file."""
        try:
            with open(results_file, "r") as f:
                self.data = json.load(f)

            self.benchmarks = self.data.get("benchmarks", [])

            # Extract unique values
            self.strategies = list(
                set(
                    b["configuration"]["hierarchy_type"]
                    for b in self.benchmarks
                )
            )
            self.environments = list(
                set(
                    b["configuration"]["environment_type"]
                    for b in self.benchmarks
                )
            )
            self.agent_counts = sorted(
                list(
                    set(
                        b["configuration"]["num_agents"]
                        for b in self.benchmarks
                    )
                )
            )

            print(f"✓ Loaded results from {results_file}")
            print(f"  Strategies: {self.strategies}")
            print(f"  Environments: {self.environments}")
            print(f"  Agent counts: {self.agent_counts}")

        except Exception as e:
            print(f"✗ Error loading results: {e}")
            self.data = None
            self.benchmarks = []

    def load_latest_results(self):
        """Load the most recent hierarchy benchmark results."""
        pattern = os.path.join(self.results_dir, "hierarchy_results_*.json")
        files = glob.glob(pattern)

        if not files:
            print(
                f"No hierarchy benchmark results found in {self.results_dir}"
            )
            return

        latest_file = max(files, key=os.path.getctime)
        self.load_results(latest_file)

    @staticmethod
    def _metric_mean(entry: Dict[str, Any], metric_name: str) -> float:
        """Return trial-aggregated metric mean if available, else legacy metric."""
        summary = entry.get("trial_metric_summary", {}).get(metric_name, {})
        if isinstance(summary, dict) and "mean" in summary:
            return float(summary.get("mean", 0.0))
        metrics = entry.get("metrics", {})
        return float(metrics.get(metric_name, 0.0))

    @staticmethod
    def _metric_ci_half_width(
        entry: Dict[str, Any], metric_name: str
    ) -> float:
        """Return 95% CI half-width for a metric, falling back to 0."""
        summary = entry.get("trial_metric_summary", {}).get(metric_name, {})
        if not isinstance(summary, dict):
            return 0.0
        try:
            lower = float(summary.get("ci95_lower"))
            upper = float(summary.get("ci95_upper"))
            return max((upper - lower) / 2.0, 0.0)
        except (TypeError, ValueError):
            return 0.0

    def create_success_rate_comparison(
        self, output_file: str = "hierarchy_success_rates.png"
    ):
        """Create success rate comparison across strategies and environments."""
        if not self.benchmarks:
            print("No benchmark data available")
            return

        # Prepare data
        data_rows = []
        for benchmark in self.benchmarks:
            config = benchmark["configuration"]
            data_rows.append(
                {
                    "Strategy": config["hierarchy_type"]
                    .replace("_", " ")
                    .title(),
                    "Environment": config["environment_type"]
                    .replace("_", " ")
                    .title(),
                    "Agents": config["num_agents"],
                    "Success Rate": self._metric_mean(
                        benchmark, "success_rate"
                    )
                    * 100,
                    "Success Rate CI": self._metric_ci_half_width(
                        benchmark, "success_rate"
                    )
                    * 100,
                    "Normalized Return": self._metric_mean(
                        benchmark, "normalized_return_mean"
                    ),
                    "Normalized Return CI": self._metric_ci_half_width(
                        benchmark, "normalized_return_mean"
                    ),
                }
            )

        df = pd.DataFrame(data_rows)

        def _aggregate_error(series: pd.Series) -> float:
            values = np.asarray(series, dtype=float)
            if values.size == 0:
                return 0.0
            return float(np.sqrt(np.mean(np.square(values))))

        def _plot_grouped_bars(
            ax,
            summary_df: pd.DataFrame,
            y_label: str,
            title: str,
            y_min: float,
            y_max: float,
        ):
            envs = sorted(summary_df["Environment"].unique().tolist())
            strategies = sorted(summary_df["Strategy"].unique().tolist())
            x = np.arange(len(envs))
            width = 0.8 / max(len(strategies), 1)

            for idx, strategy in enumerate(strategies):
                subset = summary_df[
                    summary_df["Strategy"] == strategy
                ].set_index("Environment")
                means = [
                    float(subset["mean"].get(env, np.nan)) for env in envs
                ]
                errors = [float(subset["ci"].get(env, 0.0)) for env in envs]
                offset = (idx - (len(strategies) - 1) / 2.0) * width
                ax.bar(
                    x + offset,
                    means,
                    width=width,
                    yerr=errors,
                    capsize=4,
                    label=strategy,
                )

            ax.set_xticks(x)
            ax.set_xticklabels(envs, rotation=45, ha="right")
            ax.set_ylabel(y_label)
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.set_ylim(y_min, y_max)
            ax.legend(title="Strategy")
            ax.grid(True, alpha=0.3, axis="y")

        success_summary = df.groupby(
            ["Environment", "Strategy"], as_index=False
        ).agg(
            mean=("Success Rate", "mean"),
            ci=("Success Rate CI", _aggregate_error),
        )

        return_summary = df.groupby(
            ["Environment", "Strategy"], as_index=False
        ).agg(
            mean=("Normalized Return", "mean"),
            ci=("Normalized Return CI", _aggregate_error),
        )

        # Create grouped bar plots with CI error bars.
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        _plot_grouped_bars(
            ax=ax1,
            summary_df=success_summary,
            y_label="Success Rate (%)",
            title="Success Rate by Environment",
            y_min=0.0,
            y_max=105.0,
        )
        _plot_grouped_bars(
            ax=ax2,
            summary_df=return_summary,
            y_label="Normalized Return",
            title="Normalized Return by Environment",
            y_min=0.0,
            y_max=1.1,
        )

        plt.tight_layout()
        output_path = os.path.join(self.results_dir, output_file)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Success rate comparison saved to {output_path}")
        plt.close()

    def create_scalability_analysis(
        self, output_file: str = "hierarchy_scalability.png"
    ):
        """Create scalability analysis showing performance vs team size."""
        if not self.benchmarks:
            print("No benchmark data available")
            return

        # Prepare data
        data_rows = []
        for benchmark in self.benchmarks:
            config = benchmark["configuration"]
            data_rows.append(
                {
                    "Strategy": config["hierarchy_type"]
                    .replace("_", " ")
                    .title(),
                    "Environment": config["environment_type"]
                    .replace("_", " ")
                    .title(),
                    "Team Size": config["num_agents"],
                    "Success Rate": self._metric_mean(
                        benchmark, "success_rate"
                    )
                    * 100,
                    "Success Rate CI": self._metric_ci_half_width(
                        benchmark, "success_rate"
                    )
                    * 100,
                    "Makespan": self._metric_mean(benchmark, "makespan_mean"),
                    "Makespan CI": self._metric_ci_half_width(
                        benchmark, "makespan_mean"
                    ),
                    "Coordination Records/Episode": self._metric_mean(
                        benchmark, "messages_per_episode"
                    ),
                    "Coordination Records/Episode CI": self._metric_ci_half_width(
                        benchmark, "messages_per_episode"
                    ),
                    "Manager Utilization": self._metric_mean(
                        benchmark, "manager_utilization"
                    ),
                    "Manager Utilization CI": self._metric_ci_half_width(
                        benchmark, "manager_utilization"
                    ),
                }
            )

        df = pd.DataFrame(data_rows)

        def _aggregate_error(series: pd.Series) -> float:
            values = np.asarray(series, dtype=float)
            if values.size == 0:
                return 0.0
            return float(np.sqrt(np.mean(np.square(values))))

        # Create 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Success rate vs team size
        for strategy in self.strategies:
            strategy_df = df[
                df["Strategy"] == strategy.replace("_", " ").title()
            ]
            strategy_summary = strategy_df.groupby(
                "Team Size", as_index=False
            ).agg(
                mean=("Success Rate", "mean"),
                ci=("Success Rate CI", _aggregate_error),
            )
            axes[0, 0].errorbar(
                strategy_summary["Team Size"].to_numpy(),
                strategy_summary["mean"].to_numpy(),
                yerr=strategy_summary["ci"].to_numpy(),
                marker="o",
                linewidth=2,
                capsize=4,
                label=strategy.replace("_", " ").title(),
            )

        axes[0, 0].set_title(
            "Success Rate vs Team Size", fontsize=12, fontweight="bold"
        )
        axes[0, 0].set_xlabel("Team Size (agents)")
        axes[0, 0].set_ylabel("Success Rate (%)")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Makespan vs team size
        for strategy in self.strategies:
            strategy_df = df[
                df["Strategy"] == strategy.replace("_", " ").title()
            ]
            strategy_summary = strategy_df.groupby(
                "Team Size", as_index=False
            ).agg(
                mean=("Makespan", "mean"),
                ci=("Makespan CI", _aggregate_error),
            )
            axes[0, 1].errorbar(
                strategy_summary["Team Size"].to_numpy(),
                strategy_summary["mean"].to_numpy(),
                yerr=strategy_summary["ci"].to_numpy(),
                marker="s",
                linewidth=2,
                capsize=4,
                label=strategy.replace("_", " ").title(),
            )

        axes[0, 1].set_title(
            "Makespan vs Team Size", fontsize=12, fontweight="bold"
        )
        axes[0, 1].set_xlabel("Team Size (agents)")
        axes[0, 1].set_ylabel("Steps to Success")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Coordination record overhead vs team size
        for strategy in self.strategies:
            strategy_df = df[
                df["Strategy"] == strategy.replace("_", " ").title()
            ]
            strategy_summary = strategy_df.groupby(
                "Team Size", as_index=False
            ).agg(
                mean=("Coordination Records/Episode", "mean"),
                ci=(
                    "Coordination Records/Episode CI",
                    _aggregate_error,
                ),
            )
            axes[1, 0].errorbar(
                strategy_summary["Team Size"].to_numpy(),
                strategy_summary["mean"].to_numpy(),
                yerr=strategy_summary["ci"].to_numpy(),
                marker="^",
                linewidth=2,
                capsize=4,
                label=strategy.replace("_", " ").title(),
            )

        axes[1, 0].set_title(
            "Coordination Record Overhead vs Team Size",
            fontsize=12,
            fontweight="bold",
        )
        axes[1, 0].set_xlabel("Team Size (agents)")
        axes[1, 0].set_ylabel("Coordination Records per Episode")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Manager utilization vs team size (only for hierarchical strategies)
        for strategy in ["tree", "hybrid"]:
            strategy_df = df[
                df["Strategy"] == strategy.replace("_", " ").title()
            ]
            if not strategy_df.empty:
                strategy_summary = strategy_df.groupby(
                    "Team Size", as_index=False
                ).agg(
                    mean=("Manager Utilization", "mean"),
                    ci=("Manager Utilization CI", _aggregate_error),
                )
                axes[1, 1].errorbar(
                    strategy_summary["Team Size"].to_numpy(),
                    strategy_summary["mean"].to_numpy(),
                    yerr=strategy_summary["ci"].to_numpy(),
                    marker="d",
                    linewidth=2,
                    capsize=4,
                    label=strategy.replace("_", " ").title(),
                )

        axes[1, 1].set_title(
            "Manager Utilization vs Team Size", fontsize=12, fontweight="bold"
        )
        axes[1, 1].set_xlabel("Team Size (agents)")
        axes[1, 1].set_ylabel("Manager Actions per 100 Steps")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(
            "Hierarchy Strategy Scalability Analysis",
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )
        plt.tight_layout()

        output_path = os.path.join(self.results_dir, output_file)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Scalability analysis saved to {output_path}")
        plt.close()

    def create_hierarchy_overhead_analysis(
        self, output_file: str = "hierarchy_overhead.png"
    ):
        """Analyze hierarchy-specific overhead metrics."""
        if not self.benchmarks:
            print("No benchmark data available")
            return

        # Prepare data
        data_rows = []
        for benchmark in self.benchmarks:
            config = benchmark["configuration"]
            data_rows.append(
                {
                    "Strategy": config["hierarchy_type"]
                    .replace("_", " ")
                    .title(),
                    "Environment": config["environment_type"]
                    .replace("_", " ")
                    .title(),
                    "Manager Utilization": self._metric_mean(
                        benchmark, "manager_utilization"
                    ),
                    "Delegation Success": self._metric_mean(
                        benchmark, "delegation_success_rate"
                    )
                    * 100,
                    "Preemption Rate": self._metric_mean(
                        benchmark, "preemption_rate"
                    ),
                    "Manager Time %": self._metric_mean(
                        benchmark, "manager_time_percent"
                    ),
                    "Worker Time %": self._metric_mean(
                        benchmark, "worker_time_percent"
                    ),
                }
            )

        df = pd.DataFrame(data_rows)

        # Create 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Manager utilization by strategy and environment
        pivot1 = df.pivot_table(
            values="Manager Utilization",
            index="Environment",
            columns="Strategy",
            aggfunc="mean",
        )
        pivot1.plot(kind="bar", ax=axes[0, 0], width=0.8)
        axes[0, 0].set_title(
            "Manager Utilization", fontsize=12, fontweight="bold"
        )
        axes[0, 0].set_ylabel("Actions per 100 Steps")
        axes[0, 0].legend(title="Strategy")
        axes[0, 0].grid(True, alpha=0.3, axis="y")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # Plot 2: Delegation success rate
        pivot2 = df.pivot_table(
            values="Delegation Success",
            index="Environment",
            columns="Strategy",
            aggfunc="mean",
        )
        pivot2.plot(kind="bar", ax=axes[0, 1], width=0.8)
        axes[0, 1].set_title(
            "Delegation Success Rate", fontsize=12, fontweight="bold"
        )
        axes[0, 1].set_ylabel("Success Rate (%)")
        axes[0, 1].set_ylim(0, 105)
        axes[0, 1].legend(title="Strategy")
        axes[0, 1].grid(True, alpha=0.3, axis="y")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # Plot 3: Compute time distribution
        time_df = (
            df[["Strategy", "Manager Time %", "Worker Time %"]]
            .groupby("Strategy")
            .mean()
        )
        time_df.plot(kind="bar", stacked=True, ax=axes[1, 0], width=0.6)
        axes[1, 0].set_title(
            "Compute Time Distribution", fontsize=12, fontweight="bold"
        )
        axes[1, 0].set_ylabel("Time Percentage")
        axes[1, 0].set_ylim(0, 105)
        axes[1, 0].legend(["Manager Time", "Worker Time"])
        axes[1, 0].grid(True, alpha=0.3, axis="y")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Plot 4: Preemption rate (task reassignments)
        pivot4 = df.pivot_table(
            values="Preemption Rate",
            index="Environment",
            columns="Strategy",
            aggfunc="mean",
        )
        pivot4.plot(kind="bar", ax=axes[1, 1], width=0.8)
        axes[1, 1].set_title(
            "Task Preemption Rate", fontsize=12, fontweight="bold"
        )
        axes[1, 1].set_ylabel("Preemptions per Delegation")
        axes[1, 1].legend(title="Strategy")
        axes[1, 1].grid(True, alpha=0.3, axis="y")
        axes[1, 1].tick_params(axis="x", rotation=45)

        plt.suptitle(
            "Hierarchy Overhead Analysis",
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )
        plt.tight_layout()

        output_path = os.path.join(self.results_dir, output_file)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Hierarchy overhead analysis saved to {output_path}")
        plt.close()

    def create_communication_cost_analysis(
        self, output_file: str = "hierarchy_communication.png"
    ):
        """Analyze coordination record costs across strategies."""
        if not self.benchmarks:
            print("No benchmark data available")
            return

        # Prepare data
        data_rows = []
        for benchmark in self.benchmarks:
            config = benchmark["configuration"]
            data_rows.append(
                {
                    "Strategy": config["hierarchy_type"]
                    .replace("_", " ")
                    .title(),
                    "Environment": config["environment_type"]
                    .replace("_", " ")
                    .title(),
                    "Agents": config["num_agents"],
                    "Coordination Records/Episode": self._metric_mean(
                        benchmark, "messages_per_episode"
                    ),
                    "Bytes/Step": self._metric_mean(
                        benchmark, "bytes_per_step"
                    ),
                    "Coordination Latency (ms)": self._metric_mean(
                        benchmark, "coordination_latency_mean"
                    )
                    * 1000,
                }
            )

        df = pd.DataFrame(data_rows)

        # Create 1x3 subplot
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: Coordination records per episode
        pivot1 = df.pivot_table(
            values="Coordination Records/Episode",
            index="Environment",
            columns="Strategy",
            aggfunc="mean",
        )
        pivot1.plot(kind="bar", ax=axes[0], width=0.8)
        axes[0].set_title(
            "Coordination Records per Episode", fontsize=12, fontweight="bold"
        )
        axes[0].set_ylabel("Coordination Record Count")
        axes[0].legend(title="Strategy")
        axes[0].grid(True, alpha=0.3, axis="y")
        axes[0].tick_params(axis="x", rotation=45)

        # Plot 2: Bytes per step
        pivot2 = df.pivot_table(
            values="Bytes/Step",
            index="Environment",
            columns="Strategy",
            aggfunc="mean",
        )
        pivot2.plot(kind="bar", ax=axes[1], width=0.8)
        axes[1].set_title(
            "Estimated Bytes per Step", fontsize=12, fontweight="bold"
        )
        axes[1].set_ylabel("Estimated Bytes per Step")
        axes[1].legend(title="Strategy")
        axes[1].grid(True, alpha=0.3, axis="y")
        axes[1].tick_params(axis="x", rotation=45)

        # Plot 3: Coordination latency
        pivot3 = df.pivot_table(
            values="Coordination Latency (ms)",
            index="Environment",
            columns="Strategy",
            aggfunc="mean",
        )
        pivot3.plot(kind="bar", ax=axes[2], width=0.8)
        axes[2].set_title(
            "Coordination Latency", fontsize=12, fontweight="bold"
        )
        axes[2].set_ylabel("Latency (ms)")
        axes[2].legend(title="Strategy")
        axes[2].grid(True, alpha=0.3, axis="y")
        axes[2].tick_params(axis="x", rotation=45)

        plt.suptitle(
            "Coordination Record Cost Analysis", fontsize=16, fontweight="bold"
        )
        plt.tight_layout()

        output_path = os.path.join(self.results_dir, output_file)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Coordination record cost analysis saved to {output_path}")
        plt.close()

    def create_strategy_radar_chart(
        self, output_file: str = "hierarchy_strategy_radar.png"
    ):
        """Create radar chart comparing strategies across multiple dimensions."""
        if not self.benchmarks:
            print("No benchmark data available")
            return

        # Aggregate metrics by strategy
        strategy_metrics = {}

        for strategy in self.strategies:
            strategy_benchmarks = [
                b
                for b in self.benchmarks
                if b["configuration"]["hierarchy_type"] == strategy
            ]

            if not strategy_benchmarks:
                continue

            # Calculate average metrics
            success_rates = [
                self._metric_mean(b, "success_rate") * 100
                for b in strategy_benchmarks
            ]
            makespans = [
                self._metric_mean(b, "makespan_mean")
                for b in strategy_benchmarks
            ]
            action_eff = [
                self._metric_mean(b, "action_efficiency")
                for b in strategy_benchmarks
            ]
            messages = [
                self._metric_mean(b, "messages_per_episode")
                for b in strategy_benchmarks
            ]
            delegation_success = [
                self._metric_mean(b, "delegation_success_rate") * 100
                for b in strategy_benchmarks
            ]

            # Normalize metrics (0-100 scale, higher is better)
            avg_makespan = np.mean(makespans) if makespans else 50
            avg_actions = np.mean(action_eff) if action_eff else 10
            avg_messages = np.mean(messages) if messages else 30

            # Normalize: convert to 0-100 scale where higher is better
            # For makespan: lower is better, so invert (max 100 steps = 0 score)
            time_eff = (
                max(0, 100 - (avg_makespan / 100 * 100))
                if avg_makespan > 0
                else 50
            )

            # For action efficiency: lower is better (max 1200 actions = 0 score)
            action_score = (
                max(0, 100 - (avg_actions / 1200 * 100))
                if avg_actions > 0
                else 50
            )

            # For coordination records: lower is better (max 60000 records = 0 score)
            comm_score = (
                max(0, 100 - (avg_messages / 60000 * 100))
                if avg_messages > 0
                else 50
            )

            strategy_metrics[strategy] = {
                "Task Success": np.mean(success_rates),
                "Time Efficiency": time_eff,
                "Action Efficiency": action_score,
                "Low Coordination Records": comm_score,
                "Delegation Quality": (
                    np.mean(delegation_success) if delegation_success else 50
                ),
            }

        if not strategy_metrics:
            print("No strategy metrics to plot")
            return

        # Create radar chart
        categories = list(next(iter(strategy_metrics.values())).keys())
        num_categories = len(categories)

        angles = [
            n / float(num_categories) * 2 * np.pi
            for n in range(num_categories)
        ]
        angles += angles[:1]

        fig, ax = plt.subplots(
            figsize=(10, 10), subplot_kw=dict(projection="polar")
        )

        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

        for i, (strategy, metrics) in enumerate(strategy_metrics.items()):
            values = list(metrics.values())
            values += values[:1]

            ax.plot(
                angles,
                values,
                "o-",
                linewidth=2,
                label=strategy.replace("_", " ").title(),
                color=colors[i % len(colors)],
            )
            ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=11)
        ax.set_ylim(0, 100)
        ax.set_title(
            "Hierarchy Strategy Comparison\n(Higher values are better)",
            size=16,
            fontweight="bold",
            pad=20,
        )
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)

        output_path = os.path.join(self.results_dir, output_file)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Strategy radar chart saved to {output_path}")
        plt.close()

    def create_ablation_study_plots(
        self,
        ablation_file: Optional[str] = None,
        output_file: str = "hierarchy_ablation.png",
    ):
        """Create plots for ablation study results."""
        ablation_payloads = []
        if ablation_file is None:
            # Merge all available ablation outputs (one or many strategies).
            pattern = os.path.join(self.results_dir, "ablation_study_*.json")
            files = sorted(glob.glob(pattern), key=os.path.getctime)
            if not files:
                print("No ablation study results found")
                return
            for path in files:
                try:
                    with open(path, "r") as f:
                        ablation_payloads.append(json.load(f))
                except Exception as exc:
                    print(f"Warning: skipping ablation file '{path}': {exc}")
        else:
            try:
                with open(ablation_file, "r") as f:
                    ablation_payloads.append(json.load(f))
            except Exception as e:
                print(f"Error loading ablation study: {e}")
                return

        results = []
        base_config = None
        for payload in ablation_payloads:
            payload_results = payload.get("results", [])
            if isinstance(payload_results, list):
                results.extend(payload_results)
            if base_config is None and isinstance(
                payload.get("base_config"), dict
            ):
                base_config = payload.get("base_config")

        if not results:
            print("No ablation results to plot")
            return

        data_rows = []
        for result in results:
            config = result.get("configuration", result.get("config", {}))
            legacy_cfg = result.get("config", {})
            strategy = (
                config.get("hierarchy_type")
                or legacy_cfg.get("hierarchy_type")
                or "unknown"
            )
            comm_limit = config.get(
                "communication_limit",
                legacy_cfg.get("communication_limit"),
            )
            data_rows.append(
                {
                    "Strategy": str(strategy).replace("_", " ").title(),
                    "Hierarchy Depth": config.get(
                        "hierarchy_depth", legacy_cfg.get("hierarchy_depth", 2)
                    ),
                    "Planning Frequency": config.get(
                        "planning_frequency",
                        legacy_cfg.get("planning_frequency", 1),
                    ),
                    "Coordination Record Limit": comm_limit,
                    "Coordination Record Limit Label": (
                        "Unlimited" if comm_limit is None else str(comm_limit)
                    ),
                    "Success Rate": self._metric_mean(result, "success_rate")
                    * 100,
                    "Success Rate CI": self._metric_ci_half_width(
                        result, "success_rate"
                    )
                    * 100,
                    "Coordination Records/Episode": self._metric_mean(
                        result, "messages_per_episode"
                    ),
                    "Coordination Records/Episode CI": (
                        self._metric_ci_half_width(
                            result, "messages_per_episode"
                        )
                    ),
                    "Manager Util": self._metric_mean(
                        result, "manager_utilization"
                    ),
                    "Manager Util CI": self._metric_ci_half_width(
                        result, "manager_utilization"
                    ),
                }
            )

        df = pd.DataFrame(data_rows)

        if isinstance(base_config, dict):
            baseline_depth = base_config.get("hierarchy_depth")
            baseline_freq = base_config.get("planning_frequency")
            baseline_comm = base_config.get("communication_limit")
        else:
            baseline_depth = (
                df["Hierarchy Depth"].mode().iloc[0] if not df.empty else None
            )
            baseline_freq = (
                df["Planning Frequency"].mode().iloc[0]
                if not df.empty
                else None
            )
            baseline_comm = (
                df["Coordination Record Limit"].mode().iloc[0]
                if not df.empty
                else None
            )

        _unset = object()

        def _filter_ablation(
            data: pd.DataFrame,
            depth: Optional[int] = None,
            freq: Optional[int] = None,
            comm: Any = _unset,
        ) -> pd.DataFrame:
            filtered = data
            if depth is not None:
                filtered = filtered[filtered["Hierarchy Depth"] == depth]
            if freq is not None:
                filtered = filtered[filtered["Planning Frequency"] == freq]
            if comm is not _unset:
                if comm is None:
                    filtered = filtered[
                        filtered["Coordination Record Limit"].isna()
                    ]
                else:
                    filtered = filtered[
                        filtered["Coordination Record Limit"] == comm
                    ]
            return filtered

        def _aggregate_error(series: pd.Series) -> float:
            values = np.asarray(series, dtype=float)
            if values.size == 0:
                return 0.0
            return float(np.sqrt(np.mean(np.square(values))))

        def _plot_grouped_bars(
            ax: Any,
            source_df: pd.DataFrame,
            x_col: str,
            y_col: str,
            err_col: str,
            title: str,
            xlabel: str,
            ylabel: str,
            *,
            integer_x_labels: bool = False,
        ):
            """Render side-by-side bars per strategy for each x-value."""
            if source_df.empty:
                ax.set_title(title, fontsize=12, fontweight="bold")
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_axis_off()
                return

            summary = (
                source_df.groupby([x_col, "Strategy"], as_index=False)
                .agg(
                    mean=(y_col, "mean"),
                    ci=(err_col, _aggregate_error),
                )
                .sort_values([x_col, "Strategy"])
            )

            x_values = sorted(summary[x_col].dropna().unique().tolist())
            x_positions = np.arange(len(x_values), dtype=float)
            num_strategies = max(len(strategies), 1)
            group_width = 0.8
            bar_width = group_width / num_strategies

            for idx, strategy in enumerate(strategies):
                strat_rows = summary[summary["Strategy"] == strategy]
                if strat_rows.empty:
                    continue

                offset = (idx - (num_strategies - 1) / 2.0) * bar_width
                bar_x = []
                bar_h = []
                bar_err = []

                for pos, x_value in zip(x_positions, x_values):
                    row = strat_rows[strat_rows[x_col] == x_value]
                    if row.empty:
                        continue
                    bar_x.append(pos + offset)
                    bar_h.append(float(row["mean"].iloc[0]))
                    bar_err.append(float(row["ci"].iloc[0]))

                if not bar_x:
                    continue

                ax.bar(
                    bar_x,
                    bar_h,
                    width=bar_width * 0.95,
                    yerr=bar_err,
                    capsize=4,
                    alpha=0.9,
                    label=strategy,
                )

            def _format_x_value(value: Any, force_int: bool = False) -> str:
                try:
                    numeric = float(value)
                    if force_int or numeric.is_integer():
                        return str(int(round(numeric)))
                    return str(numeric)
                except (TypeError, ValueError):
                    return str(value)

            if integer_x_labels:
                x_labels = [_format_x_value(v, force_int=True) for v in x_values]
            else:
                x_labels = [_format_x_value(v) for v in x_values]

            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, axis="y", alpha=0.3)
            ax.legend(title="Strategy")

        strategies = sorted(df["Strategy"].unique().tolist())
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Success rate vs hierarchy depth (all strategies)
        depth_df = _filter_ablation(df, freq=baseline_freq, comm=baseline_comm)
        if depth_df.empty:
            depth_df = df
        _plot_grouped_bars(
            axes[0, 0],
            depth_df,
            x_col="Hierarchy Depth",
            y_col="Success Rate",
            err_col="Success Rate CI",
            title="Success Rate vs Hierarchy Depth",
            xlabel="Hierarchy Depth (levels)",
            ylabel="Success Rate (%)",
            integer_x_labels=True,
        )
        axes[0, 0].set_ylim(0, 105)
        axes[0, 0].set_title(
            "Success Rate vs Hierarchy Depth",
            fontsize=12,
            fontweight="bold",
        )

        # Plot 2: Success rate vs planning frequency (all strategies)
        freq_df = _filter_ablation(
            df, depth=baseline_depth, comm=baseline_comm
        )
        if freq_df.empty:
            freq_df = df
        _plot_grouped_bars(
            axes[0, 1],
            freq_df,
            x_col="Planning Frequency",
            y_col="Success Rate",
            err_col="Success Rate CI",
            title="Success Rate vs Planning Frequency",
            xlabel="Planning Frequency (steps)",
            ylabel="Success Rate (%)",
        )
        axes[0, 1].set_ylim(0, 105)
        axes[0, 1].set_title(
            "Success Rate vs Planning Frequency",
            fontsize=12,
            fontweight="bold",
        )

        # Plot 3: Coordination records vs communication limit
        comm_df = _filter_ablation(
            df, depth=baseline_depth, freq=baseline_freq
        )
        if comm_df.empty:
            comm_df = df
        comm_labels = sorted(
            comm_df["Coordination Record Limit Label"].unique().tolist()
        )
        label_to_x = {label: idx for idx, label in enumerate(comm_labels)}
        for strategy in strategies:
            subset = comm_df[comm_df["Strategy"] == strategy]
            if subset.empty:
                continue
            summary = subset.groupby(
                "Coordination Record Limit Label", as_index=False
            ).agg(
                mean=("Coordination Records/Episode", "mean"),
                ci=("Coordination Records/Episode CI", _aggregate_error),
            )
            x_positions = np.array(
                [
                    label_to_x[label]
                    for label in summary["Coordination Record Limit Label"]
                ]
            )
            axes[1, 0].errorbar(
                x_positions,
                summary["mean"].to_numpy(),
                yerr=summary["ci"].to_numpy(),
                marker="^",
                linewidth=2,
                capsize=4,
                label=strategy,
            )
        axes[1, 0].set_xticks(np.arange(len(comm_labels)))
        axes[1, 0].set_xticklabels(comm_labels, rotation=45, ha="right")
        axes[1, 0].set_title(
            "Coordination Records vs Message Limit",
            fontsize=12,
            fontweight="bold",
        )
        axes[1, 0].set_xlabel("Coordination Record Limit")
        axes[1, 0].set_ylabel("Coordination Records per Episode")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend(title="Strategy")

        # Plot 4: Manager utilization vs planning frequency
        util_df = _filter_ablation(
            df, depth=baseline_depth, comm=baseline_comm
        )
        if util_df.empty:
            util_df = df
        for strategy in strategies:
            subset = util_df[util_df["Strategy"] == strategy]
            if subset.empty:
                continue
            summary = subset.groupby("Planning Frequency", as_index=False).agg(
                mean=("Manager Util", "mean"),
                ci=("Manager Util CI", _aggregate_error),
            )
            axes[1, 1].errorbar(
                summary["Planning Frequency"].to_numpy(),
                summary["mean"].to_numpy(),
                yerr=summary["ci"].to_numpy(),
                marker="d",
                linewidth=2,
                capsize=4,
                label=strategy,
            )
        axes[1, 1].set_title(
            "Manager Utilization vs Planning Frequency",
            fontsize=12,
            fontweight="bold",
        )
        axes[1, 1].set_xlabel("Planning Frequency (steps)")
        axes[1, 1].set_ylabel("Manager Actions per 100 Steps")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend(title="Strategy")

        plt.suptitle(
            "Hierarchy Ablation Study Analysis",
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )
        plt.tight_layout()

        output_path = os.path.join(self.results_dir, output_file)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Ablation study plots saved to {output_path}")
        plt.close()

    def generate_hierarchy_report(
        self, output_file: str = "hierarchy_benchmark_report.txt"
    ):
        """Generate comprehensive text report of hierarchy benchmarks."""
        if not self.benchmarks:
            print("No benchmark data available")
            return

        output_path = os.path.join(self.results_dir, output_file)

        with open(output_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("HIERARCHY STRATEGY BENCHMARK REPORT\n")
            f.write("=" * 70 + "\n\n")

            # Overall summary
            f.write("BENCHMARK SUMMARY\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Benchmarks: {len(self.benchmarks)}\n")
            f.write(
                f"Strategies Tested: {', '.join(s.replace('_', ' ').title() for s in self.strategies)}\n"
            )
            f.write(
                f"Environments: {', '.join(e.replace('_', ' ').title() for e in self.environments)}\n"
            )
            f.write(
                f"Agent Counts: {', '.join(map(str, self.agent_counts))}\n\n"
            )

            # Strategy comparison by environment
            for env in self.environments:
                f.write(f"\n{env.replace('_', ' ').upper()}\n")
                f.write("-" * 70 + "\n")

                env_benchmarks = [
                    b
                    for b in self.benchmarks
                    if b["configuration"]["environment_type"] == env
                ]

                # Group by strategy
                for strategy in self.strategies:
                    strategy_benchmarks = [
                        b
                        for b in env_benchmarks
                        if b["configuration"]["hierarchy_type"] == strategy
                    ]

                    if not strategy_benchmarks:
                        continue

                    # Calculate averages
                    avg_success = np.mean(
                        [
                            b["metrics"]["success_rate"] * 100
                            for b in strategy_benchmarks
                        ]
                    )
                    avg_return = np.mean(
                        [
                            b["metrics"]["normalized_return_mean"]
                            for b in strategy_benchmarks
                        ]
                    )
                    avg_makespan = np.mean(
                        [
                            b["metrics"]["makespan_mean"]
                            for b in strategy_benchmarks
                        ]
                    )
                    avg_messages = np.mean(
                        [
                            b["metrics"]["messages_per_episode"]
                            for b in strategy_benchmarks
                        ]
                    )
                    avg_manager_util = np.mean(
                        [
                            b["metrics"]["manager_utilization"]
                            for b in strategy_benchmarks
                        ]
                    )

                    f.write(f"\n  {strategy.replace('_', ' ').title():15s}:\n")
                    f.write(f"    Success Rate:       {avg_success:6.1f}%\n")
                    f.write(f"    Normalized Return:  {avg_return:6.3f}\n")
                    f.write(
                        f"    Avg Makespan:       {avg_makespan:6.1f} steps\n"
                    )
                    f.write(
                        f"    Coordination Records/Episode: {avg_messages:6.1f}\n"
                    )
                    f.write(
                        f"    Manager Util:       {avg_manager_util:6.1f}\n"
                    )

            # Best performing strategy by metric
            f.write("\n\n" + "=" * 70 + "\n")
            f.write("BEST PERFORMING STRATEGIES\n")
            f.write("=" * 70 + "\n\n")

            # Overall best by success rate
            best_success = max(
                self.strategies,
                key=lambda s: np.mean(
                    [
                        b["metrics"]["success_rate"]
                        for b in self.benchmarks
                        if b["configuration"]["hierarchy_type"] == s
                    ]
                ),
            )

            # Best by efficiency (lowest makespan)
            best_efficiency = min(
                self.strategies,
                key=lambda s: np.mean(
                    [
                        b["metrics"]["makespan_mean"]
                        for b in self.benchmarks
                        if b["configuration"]["hierarchy_type"] == s
                    ]
                ),
            )

            # Best by coordination records (fewest records)
            best_communication = min(
                self.strategies,
                key=lambda s: np.mean(
                    [
                        b["metrics"]["messages_per_episode"]
                        for b in self.benchmarks
                        if b["configuration"]["hierarchy_type"] == s
                    ]
                ),
            )

            f.write(
                f"Highest Success Rate:    {best_success.replace('_', ' ').title()}\n"
            )
            f.write(
                f"Most Time Efficient:     {best_efficiency.replace('_', ' ').title()}\n"
            )
            f.write(
                f"Lowest Coordination Records: {best_communication.replace('_', ' ').title()}\n"
            )

            # Recommendations
            f.write("\n\nRECOMMENDATIONS\n")
            f.write("-" * 30 + "\n\n")

            f.write("• Tree Hierarchy:\n")
            f.write(
                "  - Best for: Clear task delegation, centralized control\n"
            )
            f.write(
                "  - Trade-offs: Higher manager overhead, single point of failure\n\n"
            )

            f.write("• Peer-to-Peer:\n")
            f.write(
                "  - Best for: Fault tolerance, distributed decision making\n"
            )
            f.write(
                "  - Trade-offs: Higher coordination record cost, slower consensus\n\n"
            )

            f.write("• Hybrid:\n")
            f.write(
                "  - Best for: Balanced performance, flexible coordination\n"
            )
            f.write(
                "  - Trade-offs: Moderate overhead, complex implementation\n\n"
            )

            f.write("=" * 70 + "\n")

        print(f"✓ Hierarchy report saved to {output_path}")

    def generate_all_visualizations(self):
        """Generate all hierarchy benchmark visualizations."""
        print("\n" + "=" * 60)
        print("Generating hierarchy benchmark visualizations...")
        print("=" * 60)

        self.create_success_rate_comparison()
        self.create_scalability_analysis()
        self.create_hierarchy_overhead_analysis()
        self.create_communication_cost_analysis()
        self.create_strategy_radar_chart()
        self.create_ablation_study_plots()
        self.generate_hierarchy_report()

        print("\n✅ All hierarchy visualizations generated successfully!")
        print(f"   Results saved to: {self.results_dir}/")


def main():
    """Main entry point for hierarchy benchmark analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze and visualize hierarchy strategy benchmarks"
    )

    parser.add_argument(
        "--results-file",
        help="Specific hierarchy results JSON file to analyze",
    )

    parser.add_argument(
        "--results-dir",
        default="results/hierarchy",
        help="Directory containing hierarchy results (default: results/hierarchy)",
    )

    parser.add_argument(
        "--plot",
        choices=[
            "success",
            "scalability",
            "overhead",
            "communication",
            "radar",
            "ablation",
            "all",
        ],
        default="all",
        help="Type of plot to generate (default: all)",
    )

    args = parser.parse_args()

    # Create analyzer
    analyzer = HierarchyBenchmarkAnalyzer(args.results_file, args.results_dir)

    if not analyzer.benchmarks:
        print(
            "No hierarchy benchmark data loaded. Please run benchmarks first."
        )
        print("\nExample:")
        print("  python benchmarks/benchmark_runner.py --hierarchy")
        return

    # Generate requested visualizations
    if args.plot == "all":
        analyzer.generate_all_visualizations()
    elif args.plot == "success":
        analyzer.create_success_rate_comparison()
    elif args.plot == "scalability":
        analyzer.create_scalability_analysis()
    elif args.plot == "overhead":
        analyzer.create_hierarchy_overhead_analysis()
    elif args.plot == "communication":
        analyzer.create_communication_cost_analysis()
    elif args.plot == "radar":
        analyzer.create_strategy_radar_chart()
    elif args.plot == "ablation":
        analyzer.create_ablation_study_plots()


if __name__ == "__main__":
    main()
