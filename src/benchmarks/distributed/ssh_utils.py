"""
SSH utility functions for distributed benchmark orchestration.

Uses subprocess calls to ssh/scp/rsync (no paramiko dependency).
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SSHConfig:
    """SSH connection parameters."""

    host: str
    user: str
    key_path: str
    port: int = 22
    connect_timeout: int = 10

    @property
    def key_path_expanded(self) -> str:
        return os.path.expanduser(self.key_path)

    @property
    def base_ssh_args(self) -> List[str]:
        return [
            "ssh",
            "-i", self.key_path_expanded,
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", f"ConnectTimeout={self.connect_timeout}",
            "-o", "LogLevel=ERROR",
            "-p", str(self.port),
            f"{self.user}@{self.host}",
        ]

    @property
    def base_scp_args(self) -> List[str]:
        return [
            "scp",
            "-i", self.key_path_expanded,
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", f"ConnectTimeout={self.connect_timeout}",
            "-o", "LogLevel=ERROR",
            "-P", str(self.port),
        ]


def ssh_run(
    config: SSHConfig,
    command: str,
    timeout: int = 120,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run a command on a remote host via SSH."""
    args = config.base_ssh_args + [command]
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=check,
    )


def ssh_run_background(
    config: SSHConfig,
    command: str,
) -> subprocess.CompletedProcess:
    """Run a command on a remote host in the background (nohup)."""
    bg_cmd = f"nohup {command} > /dev/null 2>&1 & echo $!"
    return ssh_run(config, bg_cmd, check=False)


def ssh_check_alive(config: SSHConfig) -> bool:
    """Check if a remote host is reachable via SSH."""
    try:
        result = ssh_run(config, "echo ok", timeout=15, check=False)
        return result.returncode == 0 and "ok" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def ssh_wait_until_ready(
    config: SSHConfig,
    max_retries: int = 20,
    retry_interval: float = 10.0,
) -> bool:
    """Wait until a remote host is reachable via SSH."""
    for i in range(max_retries):
        if ssh_check_alive(config):
            return True
        if i < max_retries - 1:
            print(
                f"  Host {config.host} not ready, "
                f"retrying in {retry_interval}s "
                f"({i + 1}/{max_retries})"
            )
            time.sleep(retry_interval)
    return False


def rsync_deploy(
    config: SSHConfig,
    local_path: str,
    remote_path: str,
    exclude: Optional[List[str]] = None,
    timeout: int = 300,
) -> subprocess.CompletedProcess:
    """Deploy code to a remote host via rsync over SSH."""
    ssh_cmd = (
        f"ssh -i {config.key_path_expanded} "
        f"-o StrictHostKeyChecking=no "
        f"-o UserKnownHostsFile=/dev/null "
        f"-o LogLevel=ERROR "
        f"-p {config.port}"
    )

    args = [
        "rsync",
        "-az",
        "--delete",
        "-e", ssh_cmd,
    ]

    if exclude is None:
        exclude = [
            "__pycache__",
            "*.pyc",
            ".git",
            "venv",
            ".venv",
            "node_modules",
            "results",
            ".mypy_cache",
            ".pytest_cache",
        ]

    for pattern in exclude:
        args.extend(["--exclude", pattern])

    # Ensure local path ends with / for rsync directory sync
    if not local_path.endswith("/"):
        local_path += "/"

    args.append(local_path)
    args.append(f"{config.user}@{config.host}:{remote_path}")

    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )


def scp_fetch(
    config: SSHConfig,
    remote_path: str,
    local_path: str,
    timeout: int = 60,
) -> subprocess.CompletedProcess:
    """Fetch a file from a remote host via SCP."""
    args = config.base_scp_args + [
        f"{config.user}@{config.host}:{remote_path}",
        local_path,
    ]
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )


def deploy_to_all_hosts(
    hosts: list,
    ssh_user: str,
    ssh_key: str,
    local_project_path: str,
    remote_project_path: str,
) -> dict:
    """Deploy project code to all remote hosts.

    Args:
        hosts: List of HostConfig objects.
        ssh_user: SSH username.
        ssh_key: Path to SSH private key.
        local_project_path: Local project root directory.
        remote_project_path: Remote destination path.

    Returns:
        Dict mapping host name to success boolean.
    """
    results = {}
    for host in hosts:
        config = SSHConfig(
            host=host.ip,
            user=ssh_user,
            key_path=ssh_key,
        )
        print(f"Deploying to {host.name} ({host.ip})...")
        try:
            # Ensure remote directory exists
            ssh_run(
                config,
                f"mkdir -p {remote_project_path}",
                check=True,
            )
            rsync_deploy(config, local_project_path, remote_project_path)
            print(f"  Deployed to {host.name}")
            results[host.name] = True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"  Failed to deploy to {host.name}: {e}")
            results[host.name] = False
    return results
