"""
Battery Data Replay Engine

Replays NASA battery dataset through MQTT for agent testing and development.
Supports variable speed replay and event generation.
"""

import time
import logging
import threading
from typing import Optional, Callable, List
from enum import Enum

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.battery_twin.data.nasa_loader import NASABatteryLoader, CycleData
from src.battery_twin.communication.message_schemas import (
    TelemetryMessage,
    CapacityMessage,
)
from src.battery_twin.communication.mqtt_bridge import MqttBridge, MqttConfig

logger = logging.getLogger(__name__)


class ReplayMode(Enum):
    """Replay mode options."""

    REALTIME = "realtime"  # Replay at original time scale
    FAST = "fast"  # Speed up replay
    BATCH = "batch"  # Process as fast as possible (no delays)


class ReplayEvent(Enum):
    """Events emitted during replay."""

    REPLAY_STARTED = "replay_started"
    CYCLE_STARTED = "cycle_started"
    CYCLE_ENDED = "cycle_ended"
    REPLAY_COMPLETED = "replay_completed"
    REPLAY_STOPPED = "replay_stopped"
    ERROR = "error"


class ReplayEngine:
    """
    Replay engine for battery telemetry data.

    Loads NASA dataset cycles and replays them through MQTT,
    publishing raw telemetry and capacity measurements.

    Features:
    - Variable replay speed (1x, 10x, 100x, batch)
    - MQTT publishing with QoS 1
    - Event callbacks (cycle start/end, etc.)
    - Progress tracking
    - Pause/resume support
    """

    def __init__(
        self,
        loader: NASABatteryLoader,
        mqtt_bridge: Optional[MqttBridge] = None,
        mqtt_config: Optional[MqttConfig] = None,
    ):
        """
        Initialize replay engine.

        Args:
            loader: NASABatteryLoader instance
            mqtt_bridge: Optional MqttBridge instance (will create if None)
            mqtt_config: Optional MqttConfig for creating bridge
        """
        self.loader = loader

        # MQTT bridge
        if mqtt_bridge is not None:
            self.mqtt_bridge = mqtt_bridge
            self.own_bridge = False
        elif mqtt_config is not None:
            self.mqtt_bridge = MqttBridge(
                client_id="replay_engine", mqtt_config=mqtt_config
            )
            self.own_bridge = True
        else:
            # Create default config
            config = MqttConfig(
                broker="localhost",
                port=1883,
                qos=1,
                client_id_prefix="replay_",
            )
            self.mqtt_bridge = MqttBridge(
                client_id="replay_engine", mqtt_config=config
            )
            self.own_bridge = True

        # Replay state
        self.is_running = False
        self.is_paused = False
        self.replay_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()

        # Statistics
        self.stats = {
            "cycles_replayed": 0,
            "samples_published": 0,
            "capacity_messages_published": 0,
            "start_time": None,
            "end_time": None,
            "current_cycle": 0,
        }

        # Event callbacks
        self.event_callbacks: List[Callable[[ReplayEvent, dict], None]] = []

        logger.info("ReplayEngine initialized")

    def add_event_callback(
        self, callback: Callable[[ReplayEvent, dict], None]
    ):
        """
        Add event callback.

        Args:
            callback: Function(event: ReplayEvent, data: dict) -> None
        """
        self.event_callbacks.append(callback)
        logger.debug(f"Added event callback: {callback.__name__}")

    def _emit_event(self, event: ReplayEvent, data: dict = None):
        """Emit event to all callbacks."""
        event_data = data or {}
        event_data["timestamp"] = time.time()

        for callback in self.event_callbacks:
            try:
                callback(event, event_data)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")

    def replay_battery(
        self,
        battery_id: str,
        mode: ReplayMode = ReplayMode.BATCH,
        speed_multiplier: float = 1.0,
        start_cycle: Optional[int] = None,
        end_cycle: Optional[int] = None,
        blocking: bool = True,
    ) -> bool:
        """
        Replay battery discharge cycles.

        Args:
            battery_id: Battery identifier (e.g., "B0005")
            mode: Replay mode (realtime, fast, or batch)
            speed_multiplier: Speed multiplier (e.g., 10.0 for 10x speed)
            start_cycle: Optional starting cycle
            end_cycle: Optional ending cycle
            blocking: If True, block until replay completes. If False, run in background.

        Returns:
            True if replay started successfully
        """
        if self.is_running:
            logger.warning("Replay already running")
            return False

        # Connect MQTT bridge if we own it
        if self.own_bridge and not self.mqtt_bridge.is_connected():
            logger.info("Connecting to MQTT broker...")
            if not self.mqtt_bridge.connect():
                logger.error("Failed to connect to MQTT broker")
                return False

        # Reset state
        self.stop_event.clear()
        self.pause_event.clear()
        self.stats = {
            "cycles_replayed": 0,
            "samples_published": 0,
            "capacity_messages_published": 0,
            "start_time": time.time(),
            "end_time": None,
            "current_cycle": 0,
            "battery_id": battery_id,
        }

        # Start replay
        if blocking:
            self._run_replay(
                battery_id, mode, speed_multiplier, start_cycle, end_cycle
            )
        else:
            self.replay_thread = threading.Thread(
                target=self._run_replay,
                args=(
                    battery_id,
                    mode,
                    speed_multiplier,
                    start_cycle,
                    end_cycle,
                ),
                daemon=True,
            )
            self.replay_thread.start()

        return True

    def _run_replay(
        self,
        battery_id: str,
        mode: ReplayMode,
        speed_multiplier: float,
        start_cycle: Optional[int],
        end_cycle: Optional[int],
    ):
        """Internal replay loop."""
        try:
            self.is_running = True
            logger.info(
                f"Starting replay for {battery_id} in {mode.value} mode"
            )

            self._emit_event(
                ReplayEvent.REPLAY_STARTED,
                {
                    "battery_id": battery_id,
                    "mode": mode.value,
                    "speed_multiplier": speed_multiplier,
                },
            )

            # Load cycles
            cycles = list(
                self.loader.iter_cycles(
                    battery_id, start_cycle=start_cycle, end_cycle=end_cycle
                )
            )

            logger.info(f"Replaying {len(cycles)} cycles")

            # Replay each cycle
            for cycle in cycles:
                if self.stop_event.is_set():
                    logger.info("Replay stopped by user")
                    break

                self._replay_cycle(cycle, mode, speed_multiplier)
                self.stats["cycles_replayed"] += 1

            # Emit completion event
            self.stats["end_time"] = time.time()
            duration = self.stats["end_time"] - self.stats["start_time"]

            self._emit_event(
                ReplayEvent.REPLAY_COMPLETED,
                {
                    "battery_id": battery_id,
                    "cycles_replayed": self.stats["cycles_replayed"],
                    "samples_published": self.stats["samples_published"],
                    "duration_seconds": duration,
                },
            )

            logger.info(
                f"Replay completed: {self.stats['cycles_replayed']} cycles, "
                f"{self.stats['samples_published']} samples in {duration:.1f}s"
            )

        except Exception as e:
            logger.error(f"Replay failed: {e}")
            self._emit_event(ReplayEvent.ERROR, {"error": str(e)})
            raise

        finally:
            self.is_running = False

            # Disconnect if we own the bridge
            if self.own_bridge:
                self.mqtt_bridge.disconnect()

    def _replay_cycle(
        self, cycle: CycleData, mode: ReplayMode, speed_multiplier: float
    ):
        """Replay a single cycle."""
        self.stats["current_cycle"] = cycle.cycle

        logger.debug(
            f"Replaying cycle {cycle.cycle} ({cycle.n_samples} samples)"
        )

        self._emit_event(
            ReplayEvent.CYCLE_STARTED,
            {
                "battery_id": cycle.battery_id,
                "cycle": cycle.cycle,
                "capacity": cycle.capacity,
                "n_samples": cycle.n_samples,
            },
        )

        # Replay samples
        for i in range(cycle.n_samples):
            if self.stop_event.is_set():
                break

            # Handle pause
            while self.pause_event.is_set() and not self.stop_event.is_set():
                time.sleep(0.1)

            # Get sample
            sample = cycle.get_sample(i)

            # Create telemetry message
            telemetry_msg = TelemetryMessage(
                battery_id=sample["battery_id"],
                timestamp=sample["timestamp"],
                cycle=sample["cycle"],
                voltage=sample["voltage"],
                current=sample["current"],
                temperature=sample["temperature"],
                ambient_temperature=sample["ambient_temperature"],
            )

            # Publish to MQTT
            success = self.mqtt_bridge.publish(
                "raw_telemetry", telemetry_msg, battery_id=cycle.battery_id
            )

            if success:
                self.stats["samples_published"] += 1

            # Delay based on mode
            if mode == ReplayMode.REALTIME or mode == ReplayMode.FAST:
                if i < cycle.n_samples - 1:
                    # Calculate delay to next sample
                    next_sample = cycle.get_sample(i + 1)
                    delay = next_sample["timestamp"] - sample["timestamp"]

                    if mode == ReplayMode.FAST:
                        delay /= speed_multiplier

                    if delay > 0:
                        time.sleep(delay)

        # Publish capacity message at end of cycle
        capacity_msg = CapacityMessage(
            battery_id=cycle.battery_id,
            timestamp=time.time(),
            cycle=cycle.cycle,
            capacity=cycle.capacity,
            measurement_type="measured",
        )

        self.mqtt_bridge.publish(
            "capacity", capacity_msg, battery_id=cycle.battery_id
        )
        self.stats["capacity_messages_published"] += 1

        self._emit_event(
            ReplayEvent.CYCLE_ENDED,
            {
                "battery_id": cycle.battery_id,
                "cycle": cycle.cycle,
                "capacity": cycle.capacity,
                "samples_published": cycle.n_samples,
            },
        )

        logger.debug(f"Completed cycle {cycle.cycle}")

    def pause(self):
        """Pause replay."""
        if self.is_running and not self.is_paused:
            self.pause_event.set()
            self.is_paused = True
            logger.info("Replay paused")

    def resume(self):
        """Resume replay."""
        if self.is_running and self.is_paused:
            self.pause_event.clear()
            self.is_paused = False
            logger.info("Replay resumed")

    def stop(self):
        """Stop replay."""
        if self.is_running:
            logger.info("Stopping replay...")
            self.stop_event.set()

            # Wait for thread if running in background
            if self.replay_thread and self.replay_thread.is_alive():
                self.replay_thread.join(timeout=5.0)

            self._emit_event(
                ReplayEvent.REPLAY_STOPPED,
                {
                    "cycles_replayed": self.stats["cycles_replayed"],
                    "samples_published": self.stats["samples_published"],
                },
            )

    def get_stats(self) -> dict:
        """Get replay statistics."""
        stats = self.stats.copy()
        if self.is_running and stats["start_time"]:
            stats["elapsed_time"] = time.time() - stats["start_time"]
        elif stats["end_time"] and stats["start_time"]:
            stats["elapsed_time"] = stats["end_time"] - stats["start_time"]
        else:
            stats["elapsed_time"] = 0.0

        stats["is_running"] = self.is_running
        stats["is_paused"] = self.is_paused

        return stats

    def get_progress(self) -> float:
        """
        Get replay progress as percentage.

        Returns:
            Progress percentage (0-100)
        """
        # This is a simplified version; a more sophisticated implementation
        # would track expected total cycles
        return (
            0.0
            if not self.is_running
            else min(
                100.0,
                (
                    self.stats["cycles_replayed"]
                    / max(1, self.stats.get("total_cycles", 1))
                )
                * 100,
            )
        )

    def __enter__(self):
        """Context manager entry."""
        if self.own_bridge:
            self.mqtt_bridge.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        if self.own_bridge:
            self.mqtt_bridge.disconnect()


__all__ = [
    "ReplayEngine",
    "ReplayMode",
    "ReplayEvent",
]
