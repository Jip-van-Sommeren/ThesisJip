"""
Telemetry cleaning utilities that reuse the hybrid digital twin data loader
sanitisation routines for streaming telemetry ingestion.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple

import pandas as pd

from src.battery_twin.hybrid.utils.validators import sanitize_numeric_data


NumericSample = Dict[str, Optional[float]]


class TelemetryCleaner:
    """
    Maintains rolling telemetry windows per battery and uses the hybrid digital
    twin sanitisation utilities to cap outliers and replace invalid numeric
    values.  The cleaned values are returned alongside the set of adjustments
    applied for downstream monitoring.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = max(window_size, 5)
        self._buffers: Dict[str, Deque[NumericSample]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )

    def clean_sample(
        self, battery_id: str, sample: NumericSample
    ) -> Tuple[NumericSample, Dict[str, float]]:
        """
        Add a telemetry sample to the rolling window and return the sanitised
        values along with any adjustments that were applied.
        """
        buffer = self._buffers[battery_id]
        buffer.append(sample.copy())

        df = pd.DataFrame(buffer)
        numeric_columns: List[str] = [
            col
            for col, value in sample.items()
            if value is not None and isinstance(value, (int, float))
        ]

        if numeric_columns:
            sanitised = sanitize_numeric_data(df, numeric_columns)
            last_row = sanitised.iloc[-1]
        else:
            last_row = df.iloc[-1]

        cleaned: NumericSample = sample.copy()
        adjustments: Dict[str, float] = {}

        for col in sample.keys():
            if col not in df.columns:
                continue

            value = last_row[col]
            if pd.isna(value):
                cleaned[col] = None
            else:
                cleaned_value = float(value)
                cleaned[col] = cleaned_value

                original = sample.get(col)
                if (
                    original is not None
                    and abs(cleaned_value - original) > 1e-6
                    and isinstance(original, (int, float))
                ):
                    adjustments[col] = cleaned_value

        return cleaned, adjustments
