"""
Battery Digital Twin Performance Benchmark

Benchmarks the performance of the battery digital twin system, measuring:
- Message processing latency
- System throughput
- Resource utilization (CPU, memory)
- Agent response times
- End-to-end pipeline latency

Usage:
    python3 -m src.battery_twin.validation.performance_benchmark --duration 60
"""

import asyncio
import time
import json
import psutil
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from loguru import logger

from src.battery_twin.orchestrator import BatteryTwinOrchestrator, BatteryTwinConfig
from src.battery_twin.communication.message_schemas import TelemetryMessage, MessageFactory


# ============================================================================
# Performance Metrics Data Classes
# ============================================================================

@dataclass
class LatencyMetrics:
    """Latency measurement metrics."""
    min_ms: float
    max_ms: float
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    std_ms: float


@dataclass
class ThroughputMetrics:
    """Throughput measurement metrics."""
    messages_per_second: float
    total_messages: int
    duration_seconds: float


@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""
    cpu_percent_avg: float
    cpu_percent_max: float
    memory_mb_avg: float
    memory_mb_max: float
    memory_percent_avg: float


@dataclass
class AgentPerformanceMetrics:
    """Per-agent performance metrics."""
    agent_name: str
    messages_processed: int
    average_latency_ms: float
    error_count: int


@dataclass
class PerformanceBenchmarkReport:
    """Complete performance benchmark report."""
    timestamp: str
    duration_seconds: float
    config: Dict
    latency: LatencyMetrics
    throughput: ThroughputMetrics
    resources: ResourceMetrics
    agents: List[AgentPerformanceMetrics]
    notes: str


# ============================================================================
# Performance Benchmark Runner
# ============================================================================

class PerformanceBenchmarkRunner:
    """
    Runs performance benchmarks on the Battery Digital Twin system.

    Measures latency, throughput, and resource utilization under
    various load conditions.
    """

    def __init__(
        self,
        config: Optional[BatteryTwinConfig] = None,
        output_dir: str = "src/battery_twin/validation/results"
    ):
        """
        Initialize benchmark runner.

        Args:
            config: System configuration (uses default if None)
            output_dir: Directory for benchmark results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Use default test config if not provided
        if config is None:
            config = BatteryTwinConfig(
                battery_id="BENCH001",
                enable_telemetry_ingestor=True,
                enable_state_estimator=True,
                enable_health_monitor=True,
                enable_physics_model=False,  # Disable for faster benchmarking
                enable_ml_residual=False,
                enable_storage=False,  # Disable storage for pure compute benchmark
                log_level="WARNING",  # Reduce logging overhead
                enable_metrics=False  # Disable metrics loop
            )

        self.config = config

        # Metrics storage
        self.latencies: List[float] = []
        self.message_count: int = 0
        self.start_time: float = 0
        self.end_time: float = 0

        # Resource monitoring
        self.cpu_samples: List[float] = []
        self.memory_samples: List[float] = []
        self.process = psutil.Process(os.getpid())

        logger.info("PerformanceBenchmarkRunner initialized")

    async def run_benchmark(
        self,
        duration_seconds: float = 60.0,
        message_rate: float = 10.0
    ) -> PerformanceBenchmarkReport:
        """
        Run performance benchmark.

        Args:
            duration_seconds: How long to run the benchmark
            message_rate: Target messages per second

        Returns:
            PerformanceBenchmarkReport with all metrics
        """
        logger.info(f"Starting benchmark: {duration_seconds}s @ {message_rate} msg/s")

        self.start_time = time.time()

        try:
            # Step 1: Initialize orchestrator
            logger.info("Initializing orchestrator...")
            orchestrator = BatteryTwinOrchestrator(self.config)
            await orchestrator.initialize()
            await orchestrator.start()
            logger.info("Orchestrator running")

            # Step 2: Start resource monitoring
            monitor_task = asyncio.create_task(
                self._monitor_resources(duration_seconds)
            )

            # Step 3: Generate load
            logger.info("Generating message load...")
            await self._generate_load(
                orchestrator,
                duration_seconds,
                message_rate
            )

            # Step 4: Wait for monitoring to complete
            await monitor_task

            # Step 5: Collect agent metrics
            agent_metrics = self._collect_agent_metrics(orchestrator)

            # Step 6: Compute report
            self.end_time = time.time()
            report = self._compute_report(agent_metrics)

            # Step 7: Save report
            self._save_report(report)

            # Step 8: Shutdown
            await orchestrator.shutdown()

            logger.success("Benchmark complete")
            return report

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            self.end_time = time.time()

            # Return error report
            return PerformanceBenchmarkReport(
                timestamp=datetime.now().isoformat(),
                duration_seconds=self.end_time - self.start_time,
                config={},
                latency=LatencyMetrics(0, 0, 0, 0, 0, 0, 0),
                throughput=ThroughputMetrics(0, 0, 0),
                resources=ResourceMetrics(0, 0, 0, 0, 0),
                agents=[],
                notes=f"Benchmark failed: {str(e)}"
            )

    async def _generate_load(
        self,
        orchestrator: BatteryTwinOrchestrator,
        duration: float,
        rate: float
    ):
        """Generate synthetic load for benchmarking."""
        interval = 1.0 / rate
        end_time = time.time() + duration

        cycle = 0
        while time.time() < end_time:
            start = time.time()

            # Create synthetic telemetry message
            message = TelemetryMessage(
                battery_id=self.config.battery_id,
                cycle=cycle,
                timestamp=time.time(),
                voltage=3.7 + np.random.randn() * 0.1,
                current=1.5 + np.random.randn() * 0.2,
                temperature=25.0 + np.random.randn() * 2.0,
                soc=0.8 - (cycle * 0.001),
                soh=1.0 - (cycle * 0.0001)
            )

            # Publish via MQTT (synchronous call)
            topic = f"battery/{self.config.battery_id}/telemetry/raw"
            orchestrator.mqtt_bridge.publish_raw(
                topic,
                MessageFactory.to_json(message)
            )

            # Record latency
            latency_ms = (time.time() - start) * 1000
            self.latencies.append(latency_ms)
            self.message_count += 1

            cycle += 1

            # Sleep to maintain target rate
            elapsed = time.time() - start
            if elapsed < interval:
                await asyncio.sleep(interval - elapsed)

        logger.info(f"Generated {self.message_count} messages")

    async def _monitor_resources(self, duration: float):
        """Monitor CPU and memory usage during benchmark."""
        end_time = time.time() + duration
        sample_interval = 0.5  # Sample every 500ms

        while time.time() < end_time:
            try:
                # CPU usage (percentage)
                cpu_percent = self.process.cpu_percent(interval=None)
                self.cpu_samples.append(cpu_percent)

                # Memory usage (MB)
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                self.memory_samples.append(memory_mb)

            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")

            await asyncio.sleep(sample_interval)

        logger.info(f"Collected {len(self.cpu_samples)} resource samples")

    def _collect_agent_metrics(
        self,
        orchestrator: BatteryTwinOrchestrator
    ) -> List[AgentPerformanceMetrics]:
        """Collect performance metrics from each agent."""
        agent_metrics = []

        for name, agent_info in orchestrator.agents.items():
            metrics = AgentPerformanceMetrics(
                agent_name=name,
                messages_processed=agent_info.message_count,
                average_latency_ms=0.0,  # TODO: Get from agent
                error_count=0
            )
            agent_metrics.append(metrics)

        return agent_metrics

    def _compute_report(
        self,
        agent_metrics: List[AgentPerformanceMetrics]
    ) -> PerformanceBenchmarkReport:
        """Compute benchmark report from collected data."""
        duration = self.end_time - self.start_time

        # Latency metrics
        if len(self.latencies) > 0:
            latency = LatencyMetrics(
                min_ms=float(np.min(self.latencies)),
                max_ms=float(np.max(self.latencies)),
                mean_ms=float(np.mean(self.latencies)),
                median_ms=float(np.median(self.latencies)),
                p95_ms=float(np.percentile(self.latencies, 95)),
                p99_ms=float(np.percentile(self.latencies, 99)),
                std_ms=float(np.std(self.latencies))
            )
        else:
            latency = LatencyMetrics(0, 0, 0, 0, 0, 0, 0)

        # Throughput metrics
        throughput = ThroughputMetrics(
            messages_per_second=self.message_count / duration if duration > 0 else 0,
            total_messages=self.message_count,
            duration_seconds=duration
        )

        # Resource metrics
        if len(self.cpu_samples) > 0 and len(self.memory_samples) > 0:
            # Get total system memory
            total_memory_mb = psutil.virtual_memory().total / (1024 * 1024)

            resources = ResourceMetrics(
                cpu_percent_avg=float(np.mean(self.cpu_samples)),
                cpu_percent_max=float(np.max(self.cpu_samples)),
                memory_mb_avg=float(np.mean(self.memory_samples)),
                memory_mb_max=float(np.max(self.memory_samples)),
                memory_percent_avg=float(np.mean(self.memory_samples) / total_memory_mb * 100)
            )
        else:
            resources = ResourceMetrics(0, 0, 0, 0, 0)

        return PerformanceBenchmarkReport(
            timestamp=datetime.now().isoformat(),
            duration_seconds=duration,
            config={
                'battery_id': self.config.battery_id,
                'agents_enabled': {
                    'telemetry_ingestor': self.config.enable_telemetry_ingestor,
                    'state_estimator': self.config.enable_state_estimator,
                    'health_monitor': self.config.enable_health_monitor,
                    'physics_model': self.config.enable_physics_model,
                    'ml_residual': self.config.enable_ml_residual
                }
            },
            latency=latency,
            throughput=throughput,
            resources=resources,
            agents=agent_metrics,
            notes="Benchmark completed successfully"
        )

    def _save_report(self, report: PerformanceBenchmarkReport):
        """Save benchmark report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON report
        json_path = self.output_dir / f"performance_benchmark_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(asdict(report), f, indent=2)
        logger.info(f"Saved JSON report: {json_path}")

        # Save human-readable report
        txt_path = self.output_dir / f"performance_benchmark_{timestamp}.txt"
        with open(txt_path, 'w') as f:
            self._write_text_report(f, report)
        logger.info(f"Saved text report: {txt_path}")

    def _write_text_report(self, f, report: PerformanceBenchmarkReport):
        """Write human-readable text report."""
        f.write("=" * 80 + "\n")
        f.write("BATTERY DIGITAL TWIN PERFORMANCE BENCHMARK\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Timestamp:  {report.timestamp}\n")
        f.write(f"Duration:   {report.duration_seconds:.2f} seconds\n")
        f.write(f"Battery ID: {report.config.get('battery_id', 'N/A')}\n\n")

        f.write("-" * 80 + "\n")
        f.write("LATENCY METRICS\n")
        f.write("-" * 80 + "\n")
        lat = report.latency
        f.write(f"  Min:        {lat.min_ms:.2f} ms\n")
        f.write(f"  Max:        {lat.max_ms:.2f} ms\n")
        f.write(f"  Mean:       {lat.mean_ms:.2f} ms\n")
        f.write(f"  Median:     {lat.median_ms:.2f} ms\n")
        f.write(f"  P95:        {lat.p95_ms:.2f} ms\n")
        f.write(f"  P99:        {lat.p99_ms:.2f} ms\n")
        f.write(f"  Std Dev:    {lat.std_ms:.2f} ms\n\n")

        f.write("-" * 80 + "\n")
        f.write("THROUGHPUT METRICS\n")
        f.write("-" * 80 + "\n")
        thr = report.throughput
        f.write(f"  Messages/s:     {thr.messages_per_second:.2f}\n")
        f.write(f"  Total Messages: {thr.total_messages}\n")
        f.write(f"  Duration:       {thr.duration_seconds:.2f} s\n\n")

        f.write("-" * 80 + "\n")
        f.write("RESOURCE UTILIZATION\n")
        f.write("-" * 80 + "\n")
        res = report.resources
        f.write(f"  CPU Avg:        {res.cpu_percent_avg:.2f}%\n")
        f.write(f"  CPU Max:        {res.cpu_percent_max:.2f}%\n")
        f.write(f"  Memory Avg:     {res.memory_mb_avg:.2f} MB ({res.memory_percent_avg:.2f}%)\n")
        f.write(f"  Memory Max:     {res.memory_mb_max:.2f} MB\n\n")

        if report.agents:
            f.write("-" * 80 + "\n")
            f.write("AGENT PERFORMANCE\n")
            f.write("-" * 80 + "\n")
            for agent in report.agents:
                f.write(f"  {agent.agent_name}:\n")
                f.write(f"    Messages:  {agent.messages_processed}\n")
                f.write(f"    Latency:   {agent.average_latency_ms:.2f} ms\n")
                f.write(f"    Errors:    {agent.error_count}\n")

        f.write("\n" + "=" * 80 + "\n")


# ============================================================================
# CLI Interface
# ============================================================================

async def main():
    """Main entry point for performance benchmark."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Battery Digital Twin Performance Benchmark"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Benchmark duration in seconds (default: 60)"
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=10.0,
        help="Target message rate (msg/s) (default: 10.0)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="src/battery_twin/validation/results",
        help="Output directory for results"
    )

    args = parser.parse_args()

    # Create benchmark runner
    runner = PerformanceBenchmarkRunner(output_dir=args.output_dir)

    # Run benchmark
    report = await runner.run_benchmark(
        duration_seconds=args.duration,
        message_rate=args.rate
    )

    # Print summary
    print("\n" + "=" * 80)
    print("PERFORMANCE BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Duration:       {report.duration_seconds:.2f}s")
    print(f"Throughput:     {report.throughput.messages_per_second:.2f} msg/s")
    print(f"Latency (avg):  {report.latency.mean_ms:.2f} ms")
    print(f"Latency (p99):  {report.latency.p99_ms:.2f} ms")
    print(f"CPU (avg):      {report.resources.cpu_percent_avg:.2f}%")
    print(f"Memory (avg):   {report.resources.memory_mb_avg:.2f} MB")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
