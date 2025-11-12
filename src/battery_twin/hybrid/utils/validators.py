"""
Data validation utilities (embedded copy).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from loguru import logger

from .exceptions import InvalidDataError, InvalidParameterError


def validate_input_data(
    data: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    min_rows: int = 1,
    check_numeric: bool = True,
    allow_missing: bool = False,
) -> None:
    if not isinstance(data, pd.DataFrame):
        raise InvalidDataError(f"Expected pandas DataFrame, got {type(data)}")

    if len(data) < min_rows:
        raise InvalidDataError(
            f"Data must have at least {min_rows} rows, got {len(data)}"
        )

    if data.empty:
        raise InvalidDataError("Input data is empty")

    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise InvalidDataError(f"Missing required columns: {missing_columns}")

    if not allow_missing and data.isnull().any().any():
        null_columns = data.columns[data.isnull().any()].tolist()
        raise InvalidDataError(f"Missing values found in columns: {null_columns}")

    if check_numeric:
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if data[col].dtype == object:
                continue
            if not np.isfinite(data[col]).all():
                invalid_count = (~np.isfinite(data[col])).sum()
                raise InvalidDataError(
                    f"Column '{col}' contains {invalid_count} non-finite values (inf, -inf, NaN)"
                )

    logger.debug(
        "Data validation passed: %d rows, %d columns",
        data.shape[0],
        data.shape[1],
    )


def validate_battery_data(data: pd.DataFrame) -> None:
    required_columns = ["Capacity", "Temperature_measured", "Time", "id_cycle"]
    validate_input_data(data, required_columns=required_columns, min_rows=10)

    if (data["Capacity"] <= 0).any():
        raise InvalidDataError("Battery capacity must be positive")

    if (data["Temperature_measured"] < -50).any() or (
        data["Temperature_measured"] > 100
    ).any():
        raise InvalidDataError("Temperature must be between -50°C and 100°C")

    if (data["Time"] <= 0).any():
        raise InvalidDataError("Time values must be positive")

    if (data["id_cycle"] <= 0).any():
        raise InvalidDataError("Cycle numbers must be positive")

    cycles = data["id_cycle"].unique()
    if not np.all(np.diff(cycles) >= 0):
        logger.warning("Cycle numbers are not monotonically increasing")

    logger.debug("Battery data validation passed")


def validate_model_parameters(params: Dict[str, Any]) -> None:
    if "physics_k" in params:
        k = params["physics_k"]
        if not isinstance(k, (int, float)) or k <= 0:
            raise InvalidParameterError(
                "Physics degradation coefficient 'k' must be positive"
            )

    if "ml_model" in params:
        ml_config = params["ml_model"]

        if "hidden_layers" in ml_config:
            layers = ml_config["hidden_layers"]
            if not isinstance(layers, list) or not all(
                isinstance(x, int) and x > 0 for x in layers
            ):
                raise InvalidParameterError(
                    "Hidden layers must be a list of positive integers"
                )

        if "learning_rate" in ml_config:
            lr = ml_config["learning_rate"]
            if not isinstance(lr, (int, float)) or lr <= 0:
                raise InvalidParameterError("Learning rate must be positive")

        if "dropout_rate" in ml_config:
            dropout = ml_config["dropout_rate"]
            if not isinstance(dropout, (int, float)) or not 0 <= dropout <= 1:
                raise InvalidParameterError("Dropout rate must be between 0 and 1")

        if "batch_size" in ml_config:
            batch_size = ml_config["batch_size"]
            if not isinstance(batch_size, int) or batch_size <= 0:
                raise InvalidParameterError("Batch size must be a positive integer")

    logger.debug("Parameter validation passed")


def validate_prediction_inputs(
    data: pd.DataFrame,
    cycles: Optional[np.ndarray] = None,
    temperature: Optional[float] = None,
    charge_time: Optional[float] = None,
) -> None:
    if data is not None:
        validate_input_data(data, min_rows=1)

    if cycles is not None:
        if not isinstance(cycles, np.ndarray):
            raise InvalidDataError("Cycles must be a numpy array")
        if len(cycles) == 0:
            raise InvalidDataError("Cycles array cannot be empty")
        if not np.all(cycles > 0):
            raise InvalidDataError("All cycle numbers must be positive")
        if not np.all(np.diff(cycles) > 0):
            raise InvalidDataError("Cycles must be monotonically increasing")

    if temperature is not None:
        if not isinstance(temperature, (int, float)):
            raise InvalidDataError("Temperature must be a number")
        if temperature < -50 or temperature > 100:
            raise InvalidDataError("Temperature must be between -50°C and 100°C")

    if charge_time is not None:
        if not isinstance(charge_time, (int, float)) or charge_time <= 0:
            raise InvalidDataError("Charge time must be positive")

    logger.debug("Prediction input validation passed")


def validate_array_shapes(arrays: Dict[str, np.ndarray]) -> None:
    if not arrays:
        return

    first_key = list(arrays.keys())[0]
    first_shape = arrays[first_key].shape[0]

    for name, array in arrays.items():
        if array.shape[0] != first_shape:
            raise InvalidDataError(
                f"Array '{name}' has shape {array.shape[0]}, expected {first_shape}"
            )

    logger.debug("Array shape validation passed")


def sanitize_numeric_data(
    data: pd.DataFrame, columns: Optional[List[str]] = None
) -> pd.DataFrame:
    try:
        data_clean = data.copy()

        if columns is None:
            columns = data_clean.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col not in data_clean.columns:
                logger.warning("Column '%s' not found in data", col)
                continue

            data_clean[col] = data_clean[col].replace([np.inf, -np.inf], np.nan)

            Q1 = data_clean[col].quantile(0.25)
            Q3 = data_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR

            outlier_mask = (data_clean[col] < lower_bound) | (
                data_clean[col] > upper_bound
            )
            if outlier_mask.any():
                outlier_count = int(outlier_mask.sum())
                logger.warning(
                    "Found %d outliers in column '%s'", outlier_count, col
                )
                data_clean.loc[data_clean[col] < lower_bound, col] = lower_bound
                data_clean.loc[data_clean[col] > upper_bound, col] = upper_bound

        logger.debug("Data sanitization completed for columns: %s", columns)
        return data_clean

    except Exception as exc:
        raise InvalidDataError(f"Data sanitization failed: {exc}") from exc


def check_data_quality(data: pd.DataFrame) -> Dict[str, Any]:
    quality_metrics: Dict[str, Any] = {
        "n_rows": len(data),
        "n_columns": len(data.columns),
        "missing_values": data.isnull().sum().to_dict(),
        "duplicate_rows": int(data.duplicated().sum()),
        "numeric_columns": data.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_columns": data.select_dtypes(
            include=["object", "category"]
        ).columns.tolist(),
    }

    outliers = {}
    for col in quality_metrics["numeric_columns"]:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
        outliers[col] = int(outlier_mask.sum())

    quality_metrics["outliers"] = outliers

    completeness: Dict[str, float] = {}
    for col in data.columns:
        completeness[col] = float(
            (1 - data[col].isnull().sum() / len(data)) * 100
        )

    quality_metrics["completeness_percent"] = completeness
    quality_metrics["overall_completeness"] = float(
        np.mean(list(completeness.values()))
    )

    logger.debug(
        "Data quality assessment completed: %.1f%% complete",
        quality_metrics["overall_completeness"],
    )
    return quality_metrics

