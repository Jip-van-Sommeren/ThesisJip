#!/bin/bash
# Generate a distributed benchmark YAML config from Terraform output.
# Usage: ./generate_config.sh [output_file]

set -euo pipefail

OUTPUT_FILE="${1:-../src/benchmarks/distributed/distributed_benchmarks.yml}"
SSH_KEY="${SSH_KEY:-~/.ssh/benchmark-key.pem}"
SSH_USER="${SSH_USER:-ubuntu}"

# Get Terraform outputs
cd "$(dirname "$0")"

AGENT_IPS=$(terraform output -json agent_public_ips 2>/dev/null | python3 -c "
import json, sys
ips = json.load(sys.stdin)
for i, ip in enumerate(ips):
    print(f'    agent_{i}: {{ ip: \"{ip}\", role: agent }}')
")

BROKER_IP=$(terraform output -raw broker_public_ip 2>/dev/null || echo "")

# Build YAML
cat > "$OUTPUT_FILE" << YAMLEOF
# Auto-generated distributed benchmark configuration
# Generated from Terraform outputs on $(date -I)

num_trials: 1
warm_up_operations: 0
agent_counts:
  - 10
  - 15
  - 20

scenarios:
  - point_to_point_latency
  - broadcast_throughput
  - concurrent_messaging
  - scalability_stress

latency_mode: app_ack
output_dir: results/distributed_benchmarks_app_ack

distributed:
  enabled: true
  ssh_user: ${SSH_USER}
  ssh_key: ${SSH_KEY}
  code_path: /home/ubuntu/thesis/src
  agent_placement: round_robin

  time_sync:
    enabled: true
    max_offset_ms: 1.0
    ntp_source: "169.254.169.123"

  hosts:
YAMLEOF

# Add broker if present
if [ -n "$BROKER_IP" ] && [ "$BROKER_IP" != "null" ]; then
cat >> "$OUTPUT_FILE" << YAMLEOF
    broker: { ip: "${BROKER_IP}", role: broker }
YAMLEOF
fi

# Add agent hosts
echo "$AGENT_IPS" >> "$OUTPUT_FILE"

# Add benchmark parameters
cat >> "$OUTPUT_FILE" << 'YAMLEOF'

protocols:
  rest:
    variants:
      http1: {}
      http2: {}
  grpc:
    variants:
      unary: {}
      streaming: {}
  mqtt:
    variants:
      qos0: {}
      qos1: {}
      qos2: {}
  kafka:
    variants:
      acks0:
        parameters:
          kafka_acks: "0"
          compression_type: lz4
      acks1:
        parameters:
          kafka_acks: "1"
          compression_type: lz4
      acksall:
        parameters:
          kafka_acks: all
          compression_type: lz4


YAMLEOF

echo "Config written to: $OUTPUT_FILE"

