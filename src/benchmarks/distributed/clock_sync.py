"""
Clock synchronization verification for distributed benchmarks.

Checks chrony NTP offsets across all hosts to ensure latency
measurements are accurate. Uses Amazon Time Sync Service by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from benchmarks.distributed.ssh_utils import SSHConfig, ssh_run


@dataclass
class ClockStatus:
    """Clock synchronization status for a single host."""

    host: str
    offset_ms: float
    synced: bool
    source: str
    raw_output: str


def check_chrony_status(ssh_config: SSHConfig) -> ClockStatus:
    """Query chrony tracking on a remote host.

    Returns:
        ClockStatus with offset and sync information.
    """
    try:
        result = ssh_run(
            ssh_config,
            "chronyc tracking 2>/dev/null || ntpq -p 2>/dev/null",
            timeout=15,
            check=False,
        )
        output = result.stdout.strip()

        if "System time" in output:
            return _parse_chrony_output(ssh_config.host, output)
        elif "ntpq" in output or "*" in output:
            return _parse_ntpq_output(ssh_config.host, output)
        else:
            return ClockStatus(
                host=ssh_config.host,
                offset_ms=float("inf"),
                synced=False,
                source="unknown",
                raw_output=output,
            )
    except Exception as e:
        return ClockStatus(
            host=ssh_config.host,
            offset_ms=float("inf"),
            synced=False,
            source="error",
            raw_output=str(e),
        )


def _parse_chrony_output(host: str, output: str) -> ClockStatus:
    """Parse chrony tracking output."""
    offset_ms = float("inf")
    source = "unknown"
    synced = False

    for line in output.splitlines():
        line = line.strip()
        if line.startswith("Reference ID"):
            # e.g. "Reference ID    : A9FEA97B (169.254.169.123)"
            parts = line.split(":", 1)
            if len(parts) > 1:
                source = parts[1].strip()
        elif line.startswith("System time"):
            # e.g. "System time     : 0.000012345 seconds slow of NTP"
            parts = line.split(":", 1)
            if len(parts) > 1:
                try:
                    seconds_str = parts[1].strip().split()[0]
                    offset_ms = abs(float(seconds_str)) * 1000
                    synced = True
                except (ValueError, IndexError):
                    pass
        elif line.startswith("Leap status") and "Normal" in line:
            synced = True

    return ClockStatus(
        host=host,
        offset_ms=offset_ms,
        synced=synced,
        source=source,
        raw_output=output,
    )


def _parse_ntpq_output(host: str, output: str) -> ClockStatus:
    """Parse ntpq -p output (fallback if chrony not available)."""
    offset_ms = float("inf")
    source = "unknown"
    synced = False

    for line in output.splitlines():
        if line.startswith("*"):
            # Active peer line
            parts = line.split()
            if len(parts) >= 9:
                source = parts[0][1:]  # Remove leading *
                try:
                    offset_ms = abs(float(parts[8]))
                    synced = True
                except (ValueError, IndexError):
                    pass
            break

    return ClockStatus(
        host=host,
        offset_ms=offset_ms,
        synced=synced,
        source=source,
        raw_output=output,
    )


def verify_clock_sync(
    hosts: list,
    ssh_user: str,
    ssh_key: str,
    max_offset_ms: float = 1.0,
) -> Dict[str, ClockStatus]:
    """Verify clock synchronization across all hosts.

    Args:
        hosts: List of HostConfig objects.
        ssh_user: SSH username.
        ssh_key: Path to SSH private key.
        max_offset_ms: Maximum acceptable clock offset in ms.

    Returns:
        Dict mapping host name to ClockStatus.
    """
    results: Dict[str, ClockStatus] = {}

    for host in hosts:
        ssh_config = SSHConfig(
            host=host.ip,
            user=ssh_user,
            key_path=ssh_key,
        )
        status = check_chrony_status(ssh_config)
        results[host.name] = status

        if status.synced and status.offset_ms <= max_offset_ms:
            print(
                f"  {host.name} ({host.ip}): "
                f"offset={status.offset_ms:.3f}ms "
                f"[OK]"
            )
        elif status.synced:
            print(
                f"  {host.name} ({host.ip}): "
                f"offset={status.offset_ms:.3f}ms "
                f"[WARNING: exceeds {max_offset_ms}ms]"
            )
        else:
            print(
                f"  {host.name} ({host.ip}): "
                f"NOT SYNCED [WARNING]"
            )

    return results


def get_clock_offsets(
    statuses: Dict[str, ClockStatus],
) -> Dict[str, float]:
    """Extract clock offsets from status results.

    Returns dict mapping host name to offset in ms.
    Used for post-processing latency correction.
    """
    return {
        name: status.offset_ms
        for name, status in statuses.items()
        if status.synced
    }


def all_hosts_synced(
    statuses: Dict[str, ClockStatus],
    max_offset_ms: float = 1.0,
) -> bool:
    """Check if all hosts meet the sync threshold."""
    return all(
        s.synced and s.offset_ms <= max_offset_ms
        for s in statuses.values()
    )
