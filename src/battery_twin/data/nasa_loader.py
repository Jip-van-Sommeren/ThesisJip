"""
NASA Battery Dataset Loader

Loads and parses NASA PCoE Battery Aging Dataset for battery digital twin.
Provides clean cycle-based access to discharge data.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Iterator, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CycleData:
    """
    Data for a single discharge cycle.

    Contains telemetry samples (V, I, T) across the cycle and final capacity.
    """
    battery_id: str
    cycle: int
    capacity: float  # Ah
    timestamps: np.ndarray  # seconds
    voltages: np.ndarray  # V
    currents: np.ndarray  # A
    temperatures: np.ndarray  # °C
    ambient_temperature: float  # °C

    @property
    def n_samples(self) -> int:
        """Number of samples in this cycle."""
        return len(self.timestamps)

    @property
    def duration(self) -> float:
        """Duration of cycle in seconds."""
        return float(self.timestamps[-1] - self.timestamps[0]) if len(self.timestamps) > 1 else 0.0

    @property
    def mean_voltage(self) -> float:
        """Mean voltage across cycle."""
        return float(np.mean(self.voltages))

    @property
    def mean_current(self) -> float:
        """Mean current across cycle."""
        return float(np.mean(self.currents))

    @property
    def mean_temperature(self) -> float:
        """Mean temperature across cycle."""
        return float(np.mean(self.temperatures))

    def get_sample(self, index: int) -> Dict:
        """
        Get a single telemetry sample from the cycle.

        Args:
            index: Sample index (0 to n_samples-1)

        Returns:
            Dict with voltage, current, temperature, timestamp
        """
        if index < 0 or index >= self.n_samples:
            raise IndexError(f"Sample index {index} out of range [0, {self.n_samples})")

        return {
            'battery_id': self.battery_id,
            'cycle': self.cycle,
            'timestamp': float(self.timestamps[index]),
            'voltage': float(self.voltages[index]),
            'current': float(self.currents[index]),
            'temperature': float(self.temperatures[index]),
            'ambient_temperature': self.ambient_temperature,
            'capacity': self.capacity
        }

    def __repr__(self) -> str:
        return (
            f"CycleData(battery={self.battery_id}, cycle={self.cycle}, "
            f"capacity={self.capacity:.3f}Ah, samples={self.n_samples}, "
            f"duration={self.duration:.1f}s)"
        )


@dataclass
class BatteryDatasetInfo:
    """Metadata about the loaded dataset."""
    battery_id: str
    n_cycles: int
    n_total_samples: int
    capacity_range: Tuple[float, float]
    temperature_range: Tuple[float, float]
    cycle_range: Tuple[int, int]

    def __str__(self) -> str:
        return (
            f"Battery Dataset: {self.battery_id}\n"
            f"  Cycles: {self.n_cycles} (range: {self.cycle_range[0]}-{self.cycle_range[1]})\n"
            f"  Total Samples: {self.n_total_samples:,}\n"
            f"  Capacity Range: {self.capacity_range[0]:.3f} - {self.capacity_range[1]:.3f} Ah\n"
            f"  Temperature Range: {self.temperature_range[0]:.1f} - {self.temperature_range[1]:.1f} °C"
        )


class NASABatteryLoader:
    """
    Loader for NASA PCoE Battery Aging Dataset.

    Provides efficient cycle-based access to discharge data with
    telemetry (voltage, current, temperature) and capacity measurements.

    Dataset: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
    """

    def __init__(self, dataset_path: Optional[str] = None):
        """
        Initialize NASA dataset loader.

        Args:
            dataset_path: Path to discharge.csv file.
                         Defaults to Digital-Twin-in-python/data/raw/discharge.csv
        """
        if dataset_path is None:
            dataset_path = "Digital-Twin-in-python/data/raw/discharge.csv"

        self.dataset_path = Path(dataset_path)
        self.data: Optional[pd.DataFrame] = None
        self.cycles_cache: Dict[str, List[CycleData]] = {}

        logger.info(f"Initialized NASABatteryLoader with path: {self.dataset_path}")

    def load_battery(
        self,
        battery_id: str = "B0005",
        validate: bool = True
    ) -> List[CycleData]:
        """
        Load all cycles for a specific battery.

        Args:
            battery_id: Battery identifier (e.g., "B0005", "B0006", "B0007", "B0018")
            validate: Whether to validate data ranges

        Returns:
            List of CycleData objects, one per discharge cycle

        Raises:
            FileNotFoundError: If dataset file doesn't exist
            ValueError: If battery_id not found or data invalid
        """
        # Check cache
        if battery_id in self.cycles_cache:
            logger.debug(f"Returning cached data for {battery_id}")
            return self.cycles_cache[battery_id]

        # Load raw data
        if self.data is None:
            self._load_raw_data()

        # Filter to requested battery
        battery_data = self.data[self.data['Battery'] == battery_id].copy()

        if battery_data.empty:
            available = self.data['Battery'].unique().tolist()
            raise ValueError(
                f"Battery {battery_id} not found. Available batteries: {available}"
            )

        logger.info(f"Loading battery {battery_id}: {len(battery_data):,} samples")

        # Group by cycle
        cycles = []
        for cycle_num, cycle_df in battery_data.groupby('id_cycle'):
            # Sort by time
            cycle_df = cycle_df.sort_values('Time').reset_index(drop=True)

            # Extract data
            cycle_data = CycleData(
                battery_id=battery_id,
                cycle=int(cycle_num),
                capacity=float(cycle_df['Capacity'].iloc[0]),
                timestamps=cycle_df['Time'].values,
                voltages=cycle_df['Voltage_measured'].values,
                currents=cycle_df['Current_measured'].values,
                temperatures=cycle_df['Temperature_measured'].values,
                ambient_temperature=float(cycle_df['ambient_temperature'].iloc[0])
            )

            # Validate if requested
            if validate:
                self._validate_cycle(cycle_data)

            cycles.append(cycle_data)

        # Sort by cycle number
        cycles.sort(key=lambda c: c.cycle)

        # Cache results
        self.cycles_cache[battery_id] = cycles

        logger.info(
            f"Loaded {len(cycles)} cycles for {battery_id} "
            f"(capacity range: {cycles[0].capacity:.3f} - {cycles[-1].capacity:.3f} Ah)"
        )

        return cycles

    def load_multiple_batteries(
        self,
        battery_ids: List[str]
    ) -> Dict[str, List[CycleData]]:
        """
        Load multiple batteries at once.

        Args:
            battery_ids: List of battery identifiers

        Returns:
            Dictionary mapping battery_id to list of CycleData
        """
        results = {}
        for battery_id in battery_ids:
            try:
                cycles = self.load_battery(battery_id)
                results[battery_id] = cycles
            except Exception as e:
                logger.error(f"Failed to load {battery_id}: {e}")
                raise

        return results

    def get_cycle(
        self,
        battery_id: str,
        cycle_number: int
    ) -> Optional[CycleData]:
        """
        Get a specific cycle by number.

        Args:
            battery_id: Battery identifier
            cycle_number: Cycle number

        Returns:
            CycleData object if found, None otherwise
        """
        cycles = self.load_battery(battery_id)

        for cycle in cycles:
            if cycle.cycle == cycle_number:
                return cycle

        return None

    def get_dataset_info(self, battery_id: str) -> BatteryDatasetInfo:
        """
        Get metadata about a battery dataset.

        Args:
            battery_id: Battery identifier

        Returns:
            BatteryDatasetInfo object
        """
        cycles = self.load_battery(battery_id)

        n_total_samples = sum(c.n_samples for c in cycles)
        capacities = [c.capacity for c in cycles]
        temperatures = [c.mean_temperature for c in cycles]
        cycle_numbers = [c.cycle for c in cycles]

        return BatteryDatasetInfo(
            battery_id=battery_id,
            n_cycles=len(cycles),
            n_total_samples=n_total_samples,
            capacity_range=(min(capacities), max(capacities)),
            temperature_range=(min(temperatures), max(temperatures)),
            cycle_range=(min(cycle_numbers), max(cycle_numbers))
        )

    def iter_cycles(
        self,
        battery_id: str,
        start_cycle: Optional[int] = None,
        end_cycle: Optional[int] = None
    ) -> Iterator[CycleData]:
        """
        Iterate through cycles.

        Args:
            battery_id: Battery identifier
            start_cycle: Starting cycle number (inclusive)
            end_cycle: Ending cycle number (inclusive)

        Yields:
            CycleData objects
        """
        cycles = self.load_battery(battery_id)

        for cycle in cycles:
            if start_cycle is not None and cycle.cycle < start_cycle:
                continue
            if end_cycle is not None and cycle.cycle > end_cycle:
                break

            yield cycle

    def _load_raw_data(self):
        """Load raw CSV data from disk."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {self.dataset_path}\n"
                f"Please ensure the NASA dataset is downloaded."
            )

        logger.info(f"Loading raw data from {self.dataset_path}")

        try:
            self.data = pd.read_csv(self.dataset_path)
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV file: {e}") from e

        if self.data.empty:
            raise ValueError("Loaded dataset is empty")

        # Validate required columns
        required_cols = [
            'Battery', 'id_cycle', 'Capacity', 'Time',
            'Voltage_measured', 'Current_measured', 'Temperature_measured',
            'ambient_temperature'
        ]

        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        logger.info(
            f"Loaded {len(self.data):,} samples for "
            f"{self.data['Battery'].nunique()} batteries"
        )

    def _validate_cycle(self, cycle: CycleData):
        """
        Validate cycle data ranges.

        Args:
            cycle: CycleData to validate

        Raises:
            ValueError: If data is out of acceptable range
        """
        # Voltage range: 2.0 - 5.0 V (typical Li-ion)
        if np.any(cycle.voltages < 2.0) or np.any(cycle.voltages > 5.0):
            logger.warning(
                f"Cycle {cycle.cycle}: Voltage outside typical range [2.0, 5.0]V"
            )

        # Temperature range: 0 - 60°C
        if np.any(cycle.temperatures < 0) or np.any(cycle.temperatures > 60):
            logger.warning(
                f"Cycle {cycle.cycle}: Temperature outside expected range [0, 60]°C"
            )

        # Capacity should be positive
        if cycle.capacity <= 0:
            raise ValueError(
                f"Cycle {cycle.cycle}: Invalid capacity {cycle.capacity}"
            )

        # Check for NaN values
        if np.any(np.isnan(cycle.voltages)) or np.any(np.isnan(cycle.currents)) \
           or np.any(np.isnan(cycle.temperatures)):
            raise ValueError(f"Cycle {cycle.cycle}: Contains NaN values")

        # Check for reasonable sample count
        if cycle.n_samples < 10:
            logger.warning(
                f"Cycle {cycle.cycle}: Only {cycle.n_samples} samples (very short cycle)"
            )

    def get_available_batteries(self) -> List[str]:
        """
        Get list of all available battery IDs in the dataset.

        Returns:
            List of battery identifiers
        """
        if self.data is None:
            self._load_raw_data()

        return sorted(self.data['Battery'].unique().tolist())

    def clear_cache(self):
        """Clear the cycles cache."""
        self.cycles_cache.clear()
        logger.debug("Cleared cycles cache")


__all__ = [
    'NASABatteryLoader',
    'CycleData',
    'BatteryDatasetInfo',
]
