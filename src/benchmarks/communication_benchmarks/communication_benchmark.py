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
import random
import string
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import psutil


def generate_payload(size_bytes: int, pattern: str = "random") -> str:
    """Generate a payload of specific size in bytes.

    Args:
        size_bytes: Target size in bytes
        pattern: Type of data to generate ("random", "zeros", "sequential")

    Returns:
        String payload of approximately size_bytes (accounting for encoding)
    """
    if pattern == "zeros":
        # Use repeated zeros (very compressible)
        return "0" * size_bytes
    elif pattern == "sequential":
        # Use sequential numbers (moderately compressible)
        data = ""
        num = 0
        while len(data.encode('utf-8')) < size_bytes:
            data += str(num)
            num += 1
        return data[:size_bytes]
    else:  # random (default)
        # Use random alphanumeric (not compressible)
        # Account for UTF-8 encoding overhead
        return ''.join(random.choices(string.ascii_letters + string.digits, k=size_bytes))


@dataclass
class PerformanceMetrics:
    """Container for performance measurement results."""

    # Latency metrics (seconds)
    message_latency_avg: float = 0.0
    message_latency_min: float = 0.0
    message_latency_max: float = 0.0
    message_latency_std: float = 0.0
    message_latency_p50: float = 0.0
    message_latency_p95: float = 0.0
    message_latency_p99: float = 0.0
    message_latency_p999: float = 0.0
    message_latency_jitter: float = 0.0

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

    # Benchmark configuration
    latency_mode: str = "end_to_end"  # "send_only" or "end_to_end"
    payload_size_bytes: int = 100  # Size of message payload in bytes

    # Latency samples (seconds) for advanced visualizations (down-sampled)
    latency_samples: List[float] = field(default_factory=list)

    # Quality of Service
    message_ordering_violations: int = 0
    duplicate_messages: int = 0
    message_loss_rate: float = 0.0


@dataclass
class BenchmarkTrial:
    """Results from a single benchmark trial."""

    trial_number: int
    metrics: PerformanceMetrics
    timestamp: float = field(default_factory=lambda: time.time())


@dataclass
class MultiTrialResults:
    """Aggregated results from multiple trials with confidence intervals."""

    scenario_name: str
    trial_count: int
    trials: List[BenchmarkTrial] = field(default_factory=list)
    warm_up_operations: int = 0

    # Aggregated latency metrics with 95% confidence intervals
    latency_avg_mean: float = 0.0
    latency_avg_ci_lower: float = 0.0
    latency_avg_ci_upper: float = 0.0

    latency_p50_mean: float = 0.0
    latency_p50_ci_lower: float = 0.0
    latency_p50_ci_upper: float = 0.0

    latency_p95_mean: float = 0.0
    latency_p95_ci_lower: float = 0.0
    latency_p95_ci_upper: float = 0.0

    latency_p99_mean: float = 0.0
    latency_p99_ci_lower: float = 0.0
    latency_p99_ci_upper: float = 0.0

    latency_p999_mean: float = 0.0
    latency_p999_ci_lower: float = 0.0
    latency_p999_ci_upper: float = 0.0

    jitter_mean: float = 0.0
    jitter_ci_lower: float = 0.0
    jitter_ci_upper: float = 0.0

    # Aggregated throughput metrics with confidence intervals
    throughput_mean: float = 0.0
    throughput_ci_lower: float = 0.0
    throughput_ci_upper: float = 0.0

    # Aggregated reliability metrics with confidence intervals
    success_rate_mean: float = 0.0
    success_rate_ci_lower: float = 0.0
    success_rate_ci_upper: float = 0.0

    # Aggregated resource metrics with confidence intervals
    cpu_usage_mean: float = 0.0
    cpu_usage_ci_lower: float = 0.0
    cpu_usage_ci_upper: float = 0.0

    memory_usage_mean: float = 0.0
    memory_usage_ci_lower: float = 0.0
    memory_usage_ci_upper: float = 0.0

    # Test configuration (from first trial)
    agent_count: int = 0
    topology_density: float = 0.0
    total_messages: int = 0
    payload_size_bytes: int = 100


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

            # Calculate jitter (variability in latency)
            sorted_latencies = sorted(self.completed_latencies)
            jitter = (
                statistics.stdev(
                    [sorted_latencies[i+1] - sorted_latencies[i]
                     for i in range(len(sorted_latencies)-1)]
                )
                if len(sorted_latencies) > 2
                else 0.0
            )

            return {
                "avg": statistics.mean(self.completed_latencies),
                "min": min(self.completed_latencies),
                "max": max(self.completed_latencies),
                "std": (
                    statistics.stdev(self.completed_latencies)
                    if len(self.completed_latencies) > 1
                    else 0.0
                ),
                "p50": self._percentile(self.completed_latencies, 0.50),
                "p95": self._percentile(self.completed_latencies, 0.95),
                "p99": self._percentile(self.completed_latencies, 0.99),
                "p99.9": self._percentile(self.completed_latencies, 0.999),
                "jitter": jitter,
                "count": len(self.completed_latencies),
            }

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data using linear interpolation."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        n = len(sorted_data)
        if n == 1:
            return sorted_data[0]

        # Linear interpolation between data points
        index = percentile * (n - 1)
        lower = int(index)
        upper = min(lower + 1, n - 1)
        fraction = index - lower

        return sorted_data[lower] * (1 - fraction) + sorted_data[upper] * fraction


class ThroughputTracker:
    """Tracks message throughput over time."""

    def __init__(self, window_size: float = 1.0):
        self.window_size = window_size
        self.message_timestamps: deque = deque()
        self.total_messages: int = 0
        self.lock = threading.Lock()

    def record_message(self):
        """Record a message timestamp."""
        with self.lock:
            current_time = time.perf_counter()
            self.message_timestamps.append(current_time)
            self.total_messages += 1

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

    def _bootstrap_ci(
        self, values: List[float], confidence: float = 0.95, iterations: int = 2000
    ) -> Tuple[float, float, float]:
        """Calculate mean and confidence interval using bootstrap method.

        Args:
            values: List of values to calculate CI for
            confidence: Confidence level (default 0.95 for 95% CI)
            iterations: Number of bootstrap iterations (default 2000, reduced from 10000 for performance)

        Returns:
            Tuple of (mean, ci_lower, ci_upper)
        """
        if not values:
            return 0.0, 0.0, 0.0

        if len(values) == 1:
            return values[0], values[0], values[0]

        bootstrap_means = []
        n = len(values)

        for _ in range(iterations):
            # Resample with replacement
            sample = [random.choice(values) for _ in range(n)]
            bootstrap_means.append(statistics.mean(sample))

        bootstrap_means.sort()

        # Calculate confidence interval
        alpha = 1 - confidence
        lower_idx = int(alpha / 2 * iterations)
        upper_idx = int((1 - alpha / 2) * iterations)

        mean = statistics.mean(values)
        ci_lower = bootstrap_means[lower_idx]
        ci_upper = bootstrap_means[upper_idx]

        return mean, ci_lower, ci_upper

    def run_scenario(
        self,
        scenario_name: str,
        num_trials: int = 1,
        warm_up_operations: int = 0,
        **kwargs,
    ):
        """Run a specific benchmark scenario with optional multi-trial support.

        Args:
            scenario_name: Name of the scenario to run
            num_trials: Number of trials to run (default 1 for single trial)
            warm_up_operations: Number of warm-up operations to run before measurement (default 0)
            **kwargs: Additional parameters passed to scenario

        Returns:
            PerformanceMetrics if num_trials == 1, MultiTrialResults otherwise
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        scenario = self.scenarios[scenario_name]

        # Multi-trial execution
        if num_trials > 1:
            return self._run_multi_trial_scenario(
                scenario, num_trials, warm_up_operations, **kwargs
            )

        # Single trial execution (backward compatible)
        return self._run_single_trial(scenario, warm_up_operations, **kwargs)

    def _run_single_trial(
        self, scenario: BenchmarkScenario, warm_up_operations: int = 0, **kwargs
    ) -> PerformanceMetrics:
        """Run a single trial of a benchmark scenario.

        Args:
            scenario: The scenario to run
            warm_up_operations: Number of warm-up operations (not implemented yet for single trial)
            **kwargs: Additional parameters

        Returns:
            PerformanceMetrics for the trial
        """
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
        self.benchmark_history.append((scenario.name, results))

        return results

    def _run_multi_trial_scenario(
        self,
        scenario: BenchmarkScenario,
        num_trials: int,
        warm_up_operations: int,
        **kwargs,
    ) -> MultiTrialResults:
        """Run multiple trials of a benchmark scenario and aggregate results.

        Args:
            scenario: The scenario to run
            num_trials: Number of trials to execute
            warm_up_operations: Number of warm-up operations before each trial
            **kwargs: Additional parameters

        Returns:
            MultiTrialResults with aggregated statistics and confidence intervals
        """
        print(f"\nRunning benchmark scenario: {scenario.name}")
        print(f"Description: {scenario.description}")
        print(f"Trials: {num_trials}, Warm-up operations: {warm_up_operations}")

        trials = []

        for trial_num in range(num_trials):
            print(f"\n  Trial {trial_num + 1}/{num_trials}...", end=" ")

            # Update parameters for this trial
            scenario.parameters.clear()
            scenario.parameters.update(kwargs)

            # Setup
            if scenario.setup_func:
                setup_result = scenario.setup_func(scenario.parameters)
                if setup_result:
                    scenario.parameters.update(setup_result)

            # Warm-up phase (if specified)
            if warm_up_operations > 0:
                print(f"warming up...", end=" ")
                # Store original message count
                original_message_count = scenario.parameters.get("message_count", 0)
                original_messages_per_agent = scenario.parameters.get(
                    "messages_per_agent", 0
                )

                # Run warm-up with reduced message count
                scenario.parameters["message_count"] = warm_up_operations
                scenario.parameters["messages_per_agent"] = warm_up_operations

                # Run warm-up (results discarded)
                if scenario.test_func:
                    scenario.test_func(scenario.parameters, self)

                # Restore original message count
                scenario.parameters["message_count"] = original_message_count
                scenario.parameters[
                    "messages_per_agent"
                ] = original_messages_per_agent

            # Reset tracking for actual measurement
            self.latency_tracker = LatencyTracker()
            self.throughput_tracker = ThroughputTracker()

            # Start monitoring
            self.resource_monitor.start_monitoring()
            start_time = time.perf_counter()

            try:
                # Run actual test
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

            # Collect results BEFORE teardown
            duration = end_time - start_time
            metrics = self._collect_results(scenario, duration)

            # Create trial result
            trial = BenchmarkTrial(trial_number=trial_num + 1, metrics=metrics)
            trials.append(trial)

            print(
                f"completed (latency: {metrics.message_latency_avg*1000:.2f}ms, "
                f"throughput: {metrics.throughput_avg:.1f} msg/s)"
            )

            # Teardown after this trial
            if scenario.teardown_func:
                scenario.teardown_func(scenario.parameters)

            # Brief pause between trials
            time.sleep(0.5)

        # Aggregate results across all trials
        multi_trial_results = self._aggregate_trial_results(
            scenario.name, trials, warm_up_operations
        )

        # Store in history
        self.benchmark_history.append((scenario.name, trials[0].metrics))

        return multi_trial_results

    def _aggregate_trial_results(
        self, scenario_name: str, trials: List[BenchmarkTrial], warm_up_operations: int
    ) -> MultiTrialResults:
        """Aggregate results from multiple trials using bootstrap confidence intervals.

        Args:
            scenario_name: Name of the scenario
            trials: List of BenchmarkTrial results
            warm_up_operations: Number of warm-up operations used

        Returns:
            MultiTrialResults with aggregated statistics and 95% confidence intervals
        """
        if not trials:
            raise ValueError("No trials to aggregate")

        # Extract metric values from all trials
        latency_avg_values = [t.metrics.message_latency_avg for t in trials]
        latency_p50_values = [t.metrics.message_latency_p50 for t in trials]
        latency_p95_values = [t.metrics.message_latency_p95 for t in trials]
        latency_p99_values = [t.metrics.message_latency_p99 for t in trials]
        latency_p999_values = [t.metrics.message_latency_p999 for t in trials]
        jitter_values = [t.metrics.message_latency_jitter for t in trials]
        throughput_values = [t.metrics.throughput_avg for t in trials]
        success_rate_values = [t.metrics.success_rate for t in trials]
        cpu_usage_values = [t.metrics.cpu_usage_avg for t in trials]
        memory_usage_values = [t.metrics.memory_usage_avg for t in trials]

        # Calculate means and confidence intervals using bootstrap
        latency_avg_mean, latency_avg_ci_lower, latency_avg_ci_upper = (
            self._bootstrap_ci(latency_avg_values)
        )
        latency_p50_mean, latency_p50_ci_lower, latency_p50_ci_upper = (
            self._bootstrap_ci(latency_p50_values)
        )
        latency_p95_mean, latency_p95_ci_lower, latency_p95_ci_upper = (
            self._bootstrap_ci(latency_p95_values)
        )
        latency_p99_mean, latency_p99_ci_lower, latency_p99_ci_upper = (
            self._bootstrap_ci(latency_p99_values)
        )
        latency_p999_mean, latency_p999_ci_lower, latency_p999_ci_upper = (
            self._bootstrap_ci(latency_p999_values)
        )
        jitter_mean, jitter_ci_lower, jitter_ci_upper = self._bootstrap_ci(
            jitter_values
        )
        throughput_mean, throughput_ci_lower, throughput_ci_upper = (
            self._bootstrap_ci(throughput_values)
        )
        success_rate_mean, success_rate_ci_lower, success_rate_ci_upper = (
            self._bootstrap_ci(success_rate_values)
        )
        cpu_usage_mean, cpu_usage_ci_lower, cpu_usage_ci_upper = self._bootstrap_ci(
            cpu_usage_values
        )
        memory_usage_mean, memory_usage_ci_lower, memory_usage_ci_upper = (
            self._bootstrap_ci(memory_usage_values)
        )

        # Create aggregated result
        first_trial = trials[0].metrics

        return MultiTrialResults(
            scenario_name=scenario_name,
            trial_count=len(trials),
            trials=trials,
            warm_up_operations=warm_up_operations,
            # Latency metrics with CIs
            latency_avg_mean=latency_avg_mean,
            latency_avg_ci_lower=latency_avg_ci_lower,
            latency_avg_ci_upper=latency_avg_ci_upper,
            latency_p50_mean=latency_p50_mean,
            latency_p50_ci_lower=latency_p50_ci_lower,
            latency_p50_ci_upper=latency_p50_ci_upper,
            latency_p95_mean=latency_p95_mean,
            latency_p95_ci_lower=latency_p95_ci_lower,
            latency_p95_ci_upper=latency_p95_ci_upper,
            latency_p99_mean=latency_p99_mean,
            latency_p99_ci_lower=latency_p99_ci_lower,
            latency_p99_ci_upper=latency_p99_ci_upper,
            latency_p999_mean=latency_p999_mean,
            latency_p999_ci_lower=latency_p999_ci_lower,
            latency_p999_ci_upper=latency_p999_ci_upper,
            jitter_mean=jitter_mean,
            jitter_ci_lower=jitter_ci_lower,
            jitter_ci_upper=jitter_ci_upper,
            # Throughput metrics with CIs
            throughput_mean=throughput_mean,
            throughput_ci_lower=throughput_ci_lower,
            throughput_ci_upper=throughput_ci_upper,
            # Reliability metrics with CIs
            success_rate_mean=success_rate_mean,
            success_rate_ci_lower=success_rate_ci_lower,
            success_rate_ci_upper=success_rate_ci_upper,
            # Resource metrics with CIs
            cpu_usage_mean=cpu_usage_mean,
            cpu_usage_ci_lower=cpu_usage_ci_lower,
            cpu_usage_ci_upper=cpu_usage_ci_upper,
            memory_usage_mean=memory_usage_mean,
            memory_usage_ci_lower=memory_usage_ci_lower,
            memory_usage_ci_upper=memory_usage_ci_upper,
            # Test configuration (from first trial)
            agent_count=first_trial.agent_count,
            topology_density=first_trial.topology_density,
            total_messages=first_trial.total_messages,
            payload_size_bytes=first_trial.payload_size_bytes,
        )

    def _collect_results(
        self, scenario: BenchmarkScenario, duration: float
    ) -> PerformanceMetrics:
        """Collect performance metrics from all trackers."""
        latency_stats = self.latency_tracker.get_latency_stats()
        resource_stats = self.resource_monitor.get_resource_stats()
        throughput_history = self.throughput_tracker.get_throughput_history()

        latency_samples = list(self.latency_tracker.completed_latencies)
        sample_limit = int(
            scenario.parameters.get("latency_sample_limit", 2000)
        )
        if sample_limit > 0 and len(latency_samples) > sample_limit:
            step = max(len(latency_samples) // sample_limit, 1)
            latency_samples = latency_samples[::step][:sample_limit]

        total_messages = self.throughput_tracker.total_messages
        delivery_failures = scenario.parameters.get("delivery_failures", 0)
        timeout_failures = scenario.parameters.get("timeout_failures", 0)
        timeout_failures_are_messages = scenario.parameters.get(
            "timeout_failures_are_messages", True
        )

        message_failures = delivery_failures
        if timeout_failures_are_messages:
            message_failures += timeout_failures

        successful_messages = total_messages - message_failures
        success_rate = (
            successful_messages / total_messages if total_messages > 0 else 0.0
        )

        return PerformanceMetrics(
            # Latency metrics
            message_latency_avg=latency_stats.get("avg", 0.0),
            message_latency_min=latency_stats.get("min", 0.0),
            message_latency_max=latency_stats.get("max", 0.0),
            message_latency_std=latency_stats.get("std", 0.0),
            message_latency_p50=latency_stats.get("p50", 0.0),
            message_latency_p95=latency_stats.get("p95", 0.0),
            message_latency_p99=latency_stats.get("p99", 0.0),
            message_latency_p999=latency_stats.get("p99.9", 0.0),
            message_latency_jitter=latency_stats.get("jitter", 0.0),
            # Throughput metrics
            # Calculate actual average throughput from total messages and duration
            throughput_avg=(
                successful_messages / duration if duration > 0 else 0.0
            ),
            # Keep peak as instantaneous maximum from sliding window
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
            # Benchmark configuration
            latency_mode=scenario.parameters.get("latency_mode", "end_to_end"),
            payload_size_bytes=scenario.parameters.get("payload_size_bytes", 100),
            latency_samples=latency_samples,
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
                        "jitter_ms": results.message_latency_jitter * 1000,
                        "p50_ms": results.message_latency_p50 * 1000,
                        "p95_ms": results.message_latency_p95 * 1000,
                        "p99_ms": results.message_latency_p99 * 1000,
                        "p999_ms": results.message_latency_p999 * 1000,
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
                        "payload_size_bytes": results.payload_size_bytes,
                        "latency_mode": results.latency_mode,
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
        print(f"  Payload Size: {results.payload_size_bytes} bytes")
        print(f"  Test Duration: {results.test_duration:.2f}s")
        print(f"  Topology Density: {results.topology_density:.3f}")

        # Only print latency metrics if they were measured
        if results.message_latency_avg > 0 or results.message_latency_max > 0:
            print("\nLatency Metrics:")
            print(f"  Average: {results.message_latency_avg*1000:.2f}ms")
            print(f"  Std Dev: {results.message_latency_std*1000:.2f}ms")
            print(f"  Jitter: {results.message_latency_jitter*1000:.2f}ms")
            print(f"  Min: {results.message_latency_min*1000:.2f}ms")
            print(f"  Max: {results.message_latency_max*1000:.2f}ms")
            print(
                f"  50th percentile (median): {results.message_latency_p50*1000:.2f}ms"
            )
            print(
                f"  95th percentile: {results.message_latency_p95*1000:.2f}ms"
            )
            print(
                f"  99th percentile: {results.message_latency_p99*1000:.2f}ms"
            )
            print(
                f"  99.9th percentile: {results.message_latency_p999*1000:.2f}ms"
            )

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

    def print_multi_trial_summary(self, results: MultiTrialResults):
        """Print a summary of multi-trial benchmark results with confidence intervals.

        Args:
            results: MultiTrialResults to print
        """
        print("\n" + "=" * 70)
        print("MULTI-TRIAL BENCHMARK RESULTS")
        print("=" * 70)

        print("\nTest Configuration:")
        print(f"  Scenario: {results.scenario_name}")
        print(f"  Number of Trials: {results.trial_count}")
        print(f"  Warm-up Operations: {results.warm_up_operations}")
        print(f"  Agents: {results.agent_count}")
        print(f"  Total Messages (per trial): {results.total_messages}")
        print(f"  Payload Size: {results.payload_size_bytes} bytes")
        print(f"  Topology Density: {results.topology_density:.3f}")

        print("\nLatency Metrics (mean ± 95% CI in milliseconds):")
        print(
            f"  Average: {results.latency_avg_mean*1000:.2f} "
            f"[{results.latency_avg_ci_lower*1000:.2f}, {results.latency_avg_ci_upper*1000:.2f}]"
        )
        print(
            f"  Median (p50): {results.latency_p50_mean*1000:.2f} "
            f"[{results.latency_p50_ci_lower*1000:.2f}, {results.latency_p50_ci_upper*1000:.2f}]"
        )
        print(
            f"  p95: {results.latency_p95_mean*1000:.2f} "
            f"[{results.latency_p95_ci_lower*1000:.2f}, {results.latency_p95_ci_upper*1000:.2f}]"
        )
        print(
            f"  p99: {results.latency_p99_mean*1000:.2f} "
            f"[{results.latency_p99_ci_lower*1000:.2f}, {results.latency_p99_ci_upper*1000:.2f}]"
        )
        print(
            f"  p99.9: {results.latency_p999_mean*1000:.2f} "
            f"[{results.latency_p999_ci_lower*1000:.2f}, {results.latency_p999_ci_upper*1000:.2f}]"
        )
        print(
            f"  Jitter: {results.jitter_mean*1000:.2f} "
            f"[{results.jitter_ci_lower*1000:.2f}, {results.jitter_ci_upper*1000:.2f}]"
        )

        print("\nThroughput Metrics (mean ± 95% CI in msg/s):")
        print(
            f"  Average: {results.throughput_mean:.1f} "
            f"[{results.throughput_ci_lower:.1f}, {results.throughput_ci_upper:.1f}]"
        )

        print("\nReliability Metrics (mean ± 95% CI):")
        print(
            f"  Success Rate: {results.success_rate_mean*100:.1f}% "
            f"[{results.success_rate_ci_lower*100:.1f}%, {results.success_rate_ci_upper*100:.1f}%]"
        )

        print("\nResource Usage (mean ± 95% CI):")
        print(
            f"  CPU: {results.cpu_usage_mean:.1f}% "
            f"[{results.cpu_usage_ci_lower:.1f}%, {results.cpu_usage_ci_upper:.1f}%]"
        )
        print(
            f"  Memory: {results.memory_usage_mean:.1f}MB "
            f"[{results.memory_usage_ci_lower:.1f}MB, {results.memory_usage_ci_upper:.1f}MB]"
        )

        print("\nIndividual Trial Results:")
        for trial in results.trials:
            m = trial.metrics
            print(
                f"  Trial {trial.trial_number}: "
                f"latency={m.message_latency_avg*1000:.2f}ms, "
                f"throughput={m.throughput_avg:.1f} msg/s, "
                f"success={m.success_rate*100:.1f}%"
            )

        print("=" * 70)

    def export_multi_trial_results(self, results: MultiTrialResults, filename: str):
        """Export multi-trial benchmark results to JSON file.

        Args:
            results: MultiTrialResults to export
            filename: Output filename
        """
        export_data = {
            "benchmark_timestamp": time.time(),
            "scenario_name": results.scenario_name,
            "trial_count": results.trial_count,
            "warm_up_operations": results.warm_up_operations,
            "aggregated_metrics": {
                "latency_ms": {
                    "avg": {
                        "mean": results.latency_avg_mean * 1000,
                        "ci_lower": results.latency_avg_ci_lower * 1000,
                        "ci_upper": results.latency_avg_ci_upper * 1000,
                    },
                    "p50": {
                        "mean": results.latency_p50_mean * 1000,
                        "ci_lower": results.latency_p50_ci_lower * 1000,
                        "ci_upper": results.latency_p50_ci_upper * 1000,
                    },
                    "p95": {
                        "mean": results.latency_p95_mean * 1000,
                        "ci_lower": results.latency_p95_ci_lower * 1000,
                        "ci_upper": results.latency_p95_ci_upper * 1000,
                    },
                    "p99": {
                        "mean": results.latency_p99_mean * 1000,
                        "ci_lower": results.latency_p99_ci_lower * 1000,
                        "ci_upper": results.latency_p99_ci_upper * 1000,
                    },
                    "p999": {
                        "mean": results.latency_p999_mean * 1000,
                        "ci_lower": results.latency_p999_ci_lower * 1000,
                        "ci_upper": results.latency_p999_ci_upper * 1000,
                    },
                    "jitter": {
                        "mean": results.jitter_mean * 1000,
                        "ci_lower": results.jitter_ci_lower * 1000,
                        "ci_upper": results.jitter_ci_upper * 1000,
                    },
                },
                "throughput_msg_per_sec": {
                    "mean": results.throughput_mean,
                    "ci_lower": results.throughput_ci_lower,
                    "ci_upper": results.throughput_ci_upper,
                },
                "success_rate": {
                    "mean": results.success_rate_mean,
                    "ci_lower": results.success_rate_ci_lower,
                    "ci_upper": results.success_rate_ci_upper,
                },
                "resources": {
                    "cpu_percent": {
                        "mean": results.cpu_usage_mean,
                        "ci_lower": results.cpu_usage_ci_lower,
                        "ci_upper": results.cpu_usage_ci_upper,
                    },
                    "memory_mb": {
                        "mean": results.memory_usage_mean,
                        "ci_lower": results.memory_usage_ci_lower,
                        "ci_upper": results.memory_usage_ci_upper,
                    },
                },
            },
            "test_configuration": {
                "agent_count": results.agent_count,
                "topology_density": results.topology_density,
                "total_messages_per_trial": results.total_messages,
                "payload_size_bytes": results.payload_size_bytes,
            },
            "individual_trials": [
                {
                    "trial_number": trial.trial_number,
                    "timestamp": trial.timestamp,
                    "metrics": {
                        "latency_ms": {
                            "avg": trial.metrics.message_latency_avg * 1000,
                            "p50": trial.metrics.message_latency_p50 * 1000,
                            "p95": trial.metrics.message_latency_p95 * 1000,
                            "p99": trial.metrics.message_latency_p99 * 1000,
                            "p999": trial.metrics.message_latency_p999 * 1000,
                            "jitter": trial.metrics.message_latency_jitter * 1000,
                        },
                        "throughput_msg_per_sec": trial.metrics.throughput_avg,
                        "success_rate": trial.metrics.success_rate,
                        "cpu_percent": trial.metrics.cpu_usage_avg,
                        "memory_mb": trial.metrics.memory_usage_avg,
                    },
                }
                for trial in results.trials
            ],
        }

        with open(filename, "w") as f:
            json.dump(export_data, f, indent=2)

        print(f"\nMulti-trial results exported to {filename}")
