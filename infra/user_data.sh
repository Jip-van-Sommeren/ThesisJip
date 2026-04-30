#!/bin/bash
set -euo pipefail

# Bootstrap script for benchmark EC2 instances.
# Installs Python 3.10+, Docker, chrony, and project dependencies.

export DEBIAN_FRONTEND=noninteractive

# --- System updates ---
apt-get update -y
apt-get upgrade -y

# --- Python 3.10+ ---
apt-get install -y python3 python3-pip python3-venv

# --- Docker ---
apt-get install -y docker.io
systemctl enable docker
systemctl start docker
usermod -aG docker ubuntu

# --- Chrony (NTP time sync) ---
apt-get install -y chrony


cat > /etc/chrony/chrony.conf << 'CHRONYEOF'
# Amazon Time Sync Service (sub-ms accuracy)
server 169.254.169.123 prefer iburst minpoll 4 maxpoll 4

# Fallback public NTP pools
pool ntp.ubuntu.com iburst maxsources 4

driftfile /var/lib/chrony/chrony.drift
makestep 1.0 3
rtcsync
logdir /var/log/chrony
CHRONYEOF

systemctl restart chrony

python3 -m venv /home/ubuntu/venv
chown -R ubuntu:ubuntu /home/ubuntu/venv
sudo -u ubuntu /home/ubuntu/venv/bin/pip install --upgrade pip
sudo -u ubuntu /home/ubuntu/venv/bin/pip install \
    flask \
    requests \
    psutil \
    pyyaml \
    confluent-kafka \
    paho-mqtt \
    grpcio \
    grpcio-tools \
    protobuf \
    httpx \
    "httpx[http2]" \
    networkx \
    pydantic

# --- Auto-shutdown after 2 hours (cost safety) ---
cat > /etc/cron.d/auto-shutdown << 'CRONEOF'
SHELL=/bin/bash
0 */2 * * * root /sbin/shutdown -h now "Auto-shutdown: benchmark safety timer"
CRONEOF

echo "Bootstrap complete at $(date)"
