"""
Communication Performance Benchmarking Module
Provides comprehensive metrics and benchmarking capabilities for the REST
communication implementation.

Performance metrics for evaluating communication efficiency, latency,
throughput,
and scalability of the REST-based multi-agent communication system.
"""

import time
import threading
import statistics
import json
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import psutil


@dataclass
class PerformanceMetrics:
    """Container for performance measurement results."""

    # Latency metrics (seconds)
    message_latency_avg: float = 0.0
    message_latency_min: float = 0.0
    message_latency_max: float = 0.0
    message_latency_std: float = 0.0
    message_latency_p95: float = 0.0
    message_latency_p99: float = 0.0

    # Throughput metrics (messages/second)
    throughput_avg: float = 0.0
    throughput_peak: float = 0.0
    messages_per_second: List[float] = field(default_factory=list)

    # Reliability metrics
    success_rate: float = 0.0
    delivery_failures: int = 0
    timeout_failures: int = 0

    # Resource utilization
    cpu_usage_avg: float = 0.0
    memory_usage_avg: float = 0.0  # MB
    network_usage: Dict[str, float] = field(default_factory=dict)

    # Scalability metrics
    agent_count: int = 0
    topology_density: float = 0.0
    total_messages: int = 0
    test_duration: float = 0.0

    # Quality of Service
    message_ordering_violations: int = 0
    duplicate_messages: int = 0
    message_loss_rate: float = 0.0


class LatencyTracker:
    """Tracks message latency with high precision timing."""

    def __init__(self):
        self.pending_messages: Dict[str, float] = {}
        self.completed_latencies: List[float] = []
        self.lock = threading.Lock()

    def start_message_timing(self, message_id: str):
        """Start timing a message."""
        with self.lock:
            self.pending_messages[message_id] = time.perf_counter()

    def end_message_timing(self, message_id: str) -> Optional[float]:
        """End timing a message and return latency."""
        with self.lock:
            if message_id in self.pending_messages:
                start_time = self.pending_messages.pop(message_id)
                latency = time.perf_counter() - start_time
                self.completed_latencies.append(latency)
                return latency
        return None

    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics."""
        with self.lock:
            if not self.completed_latencies:
                return {}

            return {
                "avg": statistics.mean(self.completed_latencies),
                "min": min(self.completed_latencies),
                "max": max(self.completed_latencies),
                "std": (
                    statistics.stdev(self.completed_latencies)
                    if len(self.completed_latencies) > 1
                    else 0.0
                ),
                "p95": self._percentile(self.completed_latencies, 0.95),
                "p99": self._percentile(self.completed_latencies, 0.99),
                "count": len(self.completed_latencies),
            }

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(percentile * (len(sorted_data) - 1))
        return sorted_data[index]


class ThroughputTracker:
    """Tracks message throughput over time."""

    def __init__(self, window_size: float = 1.0):
        self.window_size = window_size
        self.message_timestamps: deque = deque()
        self.lock = threading.Lock()

    def record_message(self):
        """Record a message timestamp."""
        with self.lock:
            current_time = time.perf_counter()
            self.message_timestamps.append(current_time)

            # Remove old timestamps outside window
            cutoff_time = current_time - self.window_size
            while (
                self.message_timestamps
                and self.message_timestamps[0] < cutoff_time
            ):
                self.message_timestamps.popleft()

    def get_current_throughput(self) -> float:
        """Get current throughput (messages per second)."""
        with self.lock:
            return len(self.message_timestamps) / self.window_size

    def get_throughput_history(self, samples: int = 10) -> List[float]:
        """Get historical throughput samples."""
        throughput_samples = []
        current_time = time.perf_counter()

        for i in range(samples):
            sample_end = current_time - (i * self.window_size)
            sample_start = sample_end - self.window_size

            count = sum(
                1
                for ts in self.message_timestamps
                if sample_start <= ts < sample_end
            )
            throughput_samples.append(count / self.window_size)

        return list(reversed(throughput_samples))


class ResourceMonitor:
    """Monitors system resource usage during benchmarks."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.monitoring = False
        self.cpu_samples: List[float] = []
        self.memory_samples: List[float] = []
        self.monitor_thread: Optional[threading.Thread] = None

    def start_monitoring(self):
        """Start resource monitoring."""
        self.monitoring = True
        self.cpu_samples.clear()
        self.memory_samples.clear()

        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True
        )
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)

    def _monitor_loop(self):
        """Main monitoring loop."""
        process = psutil.Process()

        while self.monitoring:
            try:
                cpu_percent = process.cpu_percent()
                memory_mb = process.memory_info().rss / 1024 / 1024

                self.cpu_samples.append(cpu_percent)
                self.memory_samples.append(memory_mb)

                time.sleep(self.interval)
            except Exception:
                pass  # Continue monitoring even if sampling fails

    def get_resource_stats(self) -> Dict[str, float]:
        """Get resource usage statistics."""
        return {
            "cpu_avg": (
                statistics.mean(self.cpu_samples) if self.cpu_samples else 0.0
            ),
            "cpu_peak": max(self.cpu_samples) if self.cpu_samples else 0.0,
            "memory_avg": (
                statistics.mean(self.memory_samples)
                if self.memory_samples
                else 0.0
            ),
            "memory_peak": (
                max(self.memory_samples) if self.memory_samples else 0.0
            ),
        }


class BenchmarkScenario:
    """Represents a specific benchmark scenario."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.setup_func: Optional[Callable] = None
        self.test_func: Optional[Callable] = None
        self.teardown_func: Optional[Callable] = None
        self.expected_duration: float = 10.0
        self.parameters: Dict[str, Any] = {}

    def set_setup(self, func: Callable):
        """Set setup function."""
        self.setup_func = func
        return self

    def set_test(self, func: Callable):
        """Set test function."""
        self.test_func = func
        return self

    def set_teardown(self, func: Callable):
        """Set teardown function."""
        self.teardown_func = func
        return self


class CommunicationBenchmark:
    """Main benchmarking class for communication performance testing."""

    def __init__(self):
        self.latency_tracker = LatencyTracker()
        self.throughput_tracker = ThroughputTracker()
        self.resource_monitor = ResourceMonitor()
        self.scenarios: Dict[str, BenchmarkScenario] = {}

        # Benchmark state
        self.current_results: Optional[PerformanceMetrics] = None
        self.benchmark_history: List[Tuple[str, PerformanceMetrics]] = []

    def add_scenario(self, scenario: BenchmarkScenario):
        """Add a benchmark scenario."""
        self.scenarios[scenario.name] = scenario

    def run_scenario(self, scenario_name: str, **kwargs) -> PerformanceMetrics:
        """Run a specific benchmark scenario."""
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        scenario = self.scenarios[scenario_name]
        scenario.parameters.update(kwargs)

        print(f"Running benchmark scenario: {scenario.name}")
        print(f"Description: {scenario.description}")

        # Setup
        if scenario.setup_func:
            setup_result = scenario.setup_func(scenario.parameters)
            if setup_result:
                scenario.parameters.update(setup_result)

        # Reset tracking
        self.latency_tracker = LatencyTracker()
        self.throughput_tracker = ThroughputTracker()

        # Start monitoring
        self.resource_monitor.start_monitoring()
        start_time = time.perf_counter()

        try:
            # Run test
            if scenario.test_func:
                test_result = scenario.test_func(scenario.parameters, self)
                if test_result:
                    scenario.parameters.update(test_result)

            # Wait for test completion
            time.sleep(0.1)  # Brief pause to capture final metrics

        finally:
            # Stop monitoring
            end_time = time.perf_counter()
            self.resource_monitor.stop_monitoring()

        # Collect results BEFORE teardown to preserve parameters
        duration = end_time - start_time
        results = self._collect_results(scenario, duration)

        # Teardown
        if scenario.teardown_func:
            scenario.teardown_func(scenario.parameters)

        self.current_results = results
        self.benchmark_history.append((scenario_name, results))

        return results

    def _collect_results(
        self, scenario: BenchmarkScenario, duration: float
    ) -> PerformanceMetrics:
        """Collect performance metrics from all trackers."""
        latency_stats = self.latency_tracker.get_latency_stats()
        resource_stats = self.resource_monitor.get_resource_stats()
        throughput_history = self.throughput_tracker.get_throughput_history()

        # Calculate success rate and failures
        total_messages = latency_stats.get("count", 0)
        # These would be updated by the test functions
        delivery_failures = scenario.parameters.get("delivery_failures", 0)
        timeout_failures = scenario.parameters.get("timeout_failures", 0)

        successful_messages = (
            total_messages - delivery_failures - timeout_failures
        )
        success_rate = (
            successful_messages / total_messages if total_messages > 0 else 0.0
        )

        return PerformanceMetrics(
            # Latency metrics
            message_latency_avg=latency_stats.get("avg", 0.0),
            message_latency_min=latency_stats.get("min", 0.0),
            message_latency_max=latency_stats.get("max", 0.0),
            message_latency_std=latency_stats.get("std", 0.0),
            message_latency_p95=latency_stats.get("p95", 0.0),
            message_latency_p99=latency_stats.get("p99", 0.0),
            # Throughput metrics
            throughput_avg=(
                statistics.mean(throughput_history)
                if throughput_history
                else 0.0
            ),
            throughput_peak=(
                max(throughput_history) if throughput_history else 0.0
            ),
            messages_per_second=throughput_history,
            # Reliability metrics
            success_rate=success_rate,
            delivery_failures=delivery_failures,
            timeout_failures=timeout_failures,
            # Resource metrics
            cpu_usage_avg=resource_stats.get("cpu_avg", 0.0),
            memory_usage_avg=resource_stats.get("memory_avg", 0.0),
            # Test metrics
            agent_count=scenario.parameters.get("agent_count", 0),
            topology_density=scenario.parameters.get("topology_density", 0.0),
            total_messages=total_messages,
            test_duration=duration,
        )

    def compare_scenarios(
        self, scenario_names: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Compare performance across multiple scenarios."""
        comparison = {}

        for scenario_name, results in self.benchmark_history:
            if scenario_name in scenario_names:
                comparison[scenario_name] = {
                    "avg_latency_ms": results.message_latency_avg * 1000,
                    "p95_latency_ms": results.message_latency_p95 * 1000,
                    "throughput_msg_per_sec": results.throughput_avg,
                    "success_rate_percent": results.success_rate * 100,
                    "cpu_usage_percent": results.cpu_usage_avg,
                    "memory_usage_mb": results.memory_usage_avg,
                    "messages_total": results.total_messages,
                }

        return comparison

    def export_results(self, filename: str):
        """Export benchmark results to JSON file."""
        export_data = {"benchmark_timestamp": time.time(), "scenarios": []}

        for scenario_name, results in self.benchmark_history:
            scenario_data = {
                "name": scenario_name,
                "metrics": {
                    "latency": {
                        "avg_ms": results.message_latency_avg * 1000,
                        "min_ms": results.message_latency_min * 1000,
                        "max_ms": results.message_latency_max * 1000,
                        "std_ms": results.message_latency_std * 1000,
                        "p95_ms": results.message_latency_p95 * 1000,
                        "p99_ms": results.message_latency_p99 * 1000,
                    },
                    "throughput": {
                        "avg_msg_per_sec": results.throughput_avg,
                        "peak_msg_per_sec": results.throughput_peak,
                        "history": results.messages_per_second,
                    },
                    "reliability": {
                        "success_rate": results.success_rate,
                        "delivery_failures": results.delivery_failures,
                        "timeout_failures": results.timeout_failures,
                    },
                    "resources": {
                        "cpu_usage_avg": results.cpu_usage_avg,
                        "memory_usage_avg_mb": results.memory_usage_avg,
                    },
                    "test_info": {
                        "agent_count": results.agent_count,
                        "total_messages": results.total_messages,
                        "duration_seconds": results.test_duration,
                        "topology_density": results.topology_density,
                    },
                },
            }
            export_data["scenarios"].append(scenario_data)

        with open(filename, "w") as f:
            json.dump(export_data, f, indent=2)

    def print_summary(self, results: Optional[PerformanceMetrics] = None):
        """Print a summary of benchmark results."""
        if results is None:
            results = self.current_results

        if results is None:
            print("No benchmark results available")
            return

        print("\n" + "=" * 60)
        print("COMMUNICATION BENCHMARK RESULTS")
        print("=" * 60)

        print("\nTest Configuration:")
        print(f"  Agents: {results.agent_count}")
        print(f"  Total Messages: {results.total_messages}")
        print(f"  Test Duration: {results.test_duration:.2f}s")
        print(f"  Topology Density: {results.topology_density:.3f}")

        print("\nLatency Metrics:")
        print(f"  Average: {results.message_latency_avg*1000:.2f}ms")
        print(f"  Min: {results.message_latency_min*1000:.2f}ms")
        print(f"  Max: {results.message_latency_max*1000:.2f}ms")
        print(f"  95th percentile: {results.message_latency_p95*1000:.2f}ms")
        print(f"  99th percentile: {results.message_latency_p99*1000:.2f}ms")

        print("\nThroughput Metrics:")
        print(f"  Average: {results.throughput_avg:.1f} msg/s")
        print(f"  Peak: {results.throughput_peak:.1f} msg/s")

        print("\nReliability Metrics:")
        print(f"  Success Rate: {results.success_rate*100:.1f}%")
        print(f"  Delivery Failures: {results.delivery_failures}")
        print(f"  Timeout Failures: {results.timeout_failures}")

        print("\nResource Usage:")
        print(f"  CPU: {results.cpu_usage_avg:.1f}%")
        print(f"  Memory: {results.memory_usage_avg:.1f}MB")

        print("=" * 60)
