"""
Battery Data Pipeline

Coordinates data loading and replay for multiple batteries.
Provides unified interface for data pipeline management.
"""

import logging
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.battery_twin.data.nasa_loader import (
    NASABatteryLoader,
    BatteryDatasetInfo,
)
from src.battery_twin.data.replay_engine import (
    ReplayEngine,
    ReplayMode,
    ReplayEvent,
)
from src.battery_twin.communication.mqtt_bridge import MqttBridge, MqttConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    """Statistics for data pipeline."""

    batteries_loaded: int = 0
    total_cycles: int = 0
    total_samples: int = 0
    cycles_replayed: int = 0
    samples_published: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    errors: List[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """Get pipeline duration in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def throughput(self) -> float:
        """Get samples per second throughput."""
        dur = self.duration
        return self.samples_published / dur if dur > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"Pipeline Statistics:\n"
            f"  Batteries: {self.batteries_loaded}\n"
            f"  Total Cycles: {self.total_cycles}\n"
            f"  Total Samples: {self.total_samples:,}\n"
            f"  Cycles Replayed: {self.cycles_replayed}\n"
            f"  Samples Published: {self.samples_published:,}\n"
            f"  Duration: {self.duration:.2f}s\n"
            f"  Throughput: {self.throughput:.1f} samples/s\n"
            f"  Errors: {len(self.errors)}"
        )


class BatteryDataPipeline:
    """
    Battery data pipeline coordinator.

    Manages:
    - Loading NASA dataset for multiple batteries
    - Coordinating replay across batteries
    - Tracking overall progress and statistics
    - Event handling and logging

    Usage:
        pipeline = BatteryDataPipeline()
        pipeline.load_batteries(["B0005", "B0006"])
        pipeline.replay_all(mode=ReplayMode.BATCH)
        print(pipeline.get_stats())
    """

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        mqtt_config: Optional[MqttConfig] = None,
    ):
        """
        Initialize data pipeline.

        Args:
            dataset_path: Path to NASA discharge.csv file
            mqtt_config: MQTT broker configuration
        """
        self.loader = NASABatteryLoader(dataset_path)

        # MQTT configuration
        if mqtt_config is None:
            mqtt_config = MqttConfig(
                broker="localhost",
                port=1883,
                qos=1,
                client_id_prefix="pipeline_",
            )
        self.mqtt_config = mqtt_config

        # Loaded batteries and their metadata
        self.battery_info: Dict[str, BatteryDatasetInfo] = {}

        # Statistics
        self.stats = PipelineStats()

        # Event handlers
        self.event_handlers: List = []

        logger.info("BatteryDataPipeline initialized")

    def load_batteries(
        self, battery_ids: List[str], validate: bool = True
    ) -> Dict[str, BatteryDatasetInfo]:
        """
        Load battery datasets.

        Args:
            battery_ids: List of battery identifiers
            validate: Whether to validate data

        Returns:
            Dictionary mapping battery_id to BatteryDatasetInfo
        """
        logger.info(f"Loading {len(battery_ids)} batteries...")
        self.stats.start_time = time.time()

        for battery_id in battery_ids:
            try:
                # Load cycles
                cycles = self.loader.load_battery(
                    battery_id, validate=validate
                )

                # Get dataset info
                info = self.loader.get_dataset_info(battery_id)
                self.battery_info[battery_id] = info

                # Update stats
                self.stats.batteries_loaded += 1
                self.stats.total_cycles += info.n_cycles
                self.stats.total_samples += info.n_total_samples

                logger.info(
                    f"Loaded {battery_id}: {info.n_cycles} cycles, "
                    f"{info.n_total_samples:,} samples"
                )

            except Exception as e:
                error_msg = f"Failed to load {battery_id}: {e}"
                logger.error(error_msg)
                self.stats.errors.append(error_msg)
                raise

        logger.info(
            f"Loaded {self.stats.batteries_loaded} batteries successfully"
        )
        return self.battery_info

    def replay_battery(
        self,
        battery_id: str,
        mode: ReplayMode = ReplayMode.BATCH,
        speed_multiplier: float = 1.0,
        start_cycle: Optional[int] = None,
        end_cycle: Optional[int] = None,
    ) -> bool:
        """
        Replay a single battery.

        Args:
            battery_id: Battery identifier
            mode: Replay mode
            speed_multiplier: Speed multiplier for fast replay
            start_cycle: Optional starting cycle
            end_cycle: Optional ending cycle

        Returns:
            True if replay successful
        """
        logger.info(f"Replaying {battery_id} in {mode.value} mode...")

        try:
            # Create replay engine
            engine = ReplayEngine(
                loader=self.loader, mqtt_config=self.mqtt_config
            )

            # Add event handler to track stats
            def on_event(event: ReplayEvent, data: dict):
                if event == ReplayEvent.REPLAY_COMPLETED:
                    self.stats.cycles_replayed += data.get(
                        "cycles_replayed", 0
                    )
                    self.stats.samples_published += data.get(
                        "samples_published", 0
                    )
                elif event == ReplayEvent.ERROR:
                    self.stats.errors.append(
                        f"{battery_id}: {data.get('error', 'Unknown error')}"
                    )

            engine.add_event_callback(on_event)

            # Replay (blocking)
            success = engine.replay_battery(
                battery_id=battery_id,
                mode=mode,
                speed_multiplier=speed_multiplier,
                start_cycle=start_cycle,
                end_cycle=end_cycle,
                blocking=True,
            )

            if success:
                logger.info(f"Completed replay for {battery_id}")
            else:
                logger.error(f"Failed to start replay for {battery_id}")

            return success

        except Exception as e:
            error_msg = f"Replay failed for {battery_id}: {e}"
            logger.error(error_msg)
            self.stats.errors.append(error_msg)
            return False

    def replay_all(
        self,
        mode: ReplayMode = ReplayMode.BATCH,
        speed_multiplier: float = 1.0,
    ) -> bool:
        """
        Replay all loaded batteries sequentially.

        Args:
            mode: Replay mode
            speed_multiplier: Speed multiplier for fast replay

        Returns:
            True if all replays successful
        """
        if not self.battery_info:
            logger.warning("No batteries loaded")
            return False

        logger.info(f"Replaying {len(self.battery_info)} batteries...")
        self.stats.start_time = time.time()

        success = True
        for battery_id in self.battery_info.keys():
            if not self.replay_battery(battery_id, mode, speed_multiplier):
                success = False
                # Continue with other batteries even if one fails

        self.stats.end_time = time.time()
        logger.info("Replay completed for all batteries")

        return success

    def get_battery_info(
        self, battery_id: str
    ) -> Optional[BatteryDatasetInfo]:
        """
        Get metadata for a battery.

        Args:
            battery_id: Battery identifier

        Returns:
            BatteryDatasetInfo if battery is loaded, None otherwise
        """
        return self.battery_info.get(battery_id)

    def get_all_battery_info(self) -> Dict[str, BatteryDatasetInfo]:
        """Get metadata for all loaded batteries."""
        return self.battery_info.copy()

    def get_stats(self) -> PipelineStats:
        """Get pipeline statistics."""
        return self.stats

    def get_available_batteries(self) -> List[str]:
        """Get list of all batteries available in the dataset."""
        return self.loader.get_available_batteries()

    def clear_cache(self):
        """Clear loader cache."""
        self.loader.clear_cache()
        logger.debug("Cleared loader cache")

    def print_summary(self):
        """Print summary of loaded batteries."""
        print("\n" + "=" * 70)
        print("BATTERY DATA PIPELINE SUMMARY")
        print("=" * 70)

        if not self.battery_info:
            print("No batteries loaded")
            return

        for battery_id, info in self.battery_info.items():
            print(f"\n{battery_id}:")
            print(f"  Cycles: {info.n_cycles}")
            print(f"  Samples: {info.n_total_samples:,}")
            print(
                f"  Capacity Range: {info.capacity_range[0]:.3f} - "
                f"{info.capacity_range[1]:.3f} Ah"
            )
            print(
                f"  Degradation: {((1 - info.capacity_range[1]/info.capacity_range[0]) * 100):.1f}%"
            )

        print("\n" + "=" * 70)
        print(str(self.stats))
        print("=" * 70)


def run_simple_replay(
    battery_id: str = "B0005",
    mode: ReplayMode = ReplayMode.BATCH,
    dataset_path: Optional[str] = None,
):
    """
    Simple helper function for quick replay.

    Args:
        battery_id: Battery to replay
        mode: Replay mode
        dataset_path: Optional dataset path
    """
    pipeline = BatteryDataPipeline(dataset_path=dataset_path)
    pipeline.load_batteries([battery_id])
    pipeline.print_summary()

    logger.info(f"Starting replay for {battery_id}...")
    success = pipeline.replay_battery(battery_id, mode=mode)

    if success:
        logger.info("Replay completed successfully")
    else:
        logger.error("Replay failed")

    pipeline.print_summary()


__all__ = [
    "BatteryDataPipeline",
    "PipelineStats",
    "run_simple_replay",
]
