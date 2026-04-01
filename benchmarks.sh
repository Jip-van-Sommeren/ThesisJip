!/usr/bin/bash
set -e
export PYTHONPATH="src"
source venv/bin/activate
python3 -m benchmarks.local.benchmark_runner \
--config-file src/benchmarks/local/benchmark_configs/all_benchmarks_concurrency.yml \
--output-dir src/results/all_benchmarks_extensive_send_only --communication-only --latency-mode send_only
python3 -m benchmarks.local.benchmark_runner \
--config-file src/benchmarks/local/benchmark_configs/all_benchmarks_concurrency.yml \
--output-dir src/results/all_benchmarks_extensive_app_ack --communication-only --latency-mode app_ack
python3 -m benchmarks.local.benchmark_runner \
--config-file src/benchmarks/local/benchmark_configs/all_benchmarks_concurrency.yml \
--output-dir src/results/all_benchmarks_extensive_end_to_end --communication-only --latency-mode end_to_end