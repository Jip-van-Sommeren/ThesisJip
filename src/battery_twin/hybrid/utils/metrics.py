"""
Model performance metrics and evaluation utilities (embedded copy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

from .exceptions import InvalidDataError


@dataclass
class MetricsResult:
    """Container for model performance metrics."""

    rmse: float
    mae: float
    r2: float
    mape: float
    median_ae: float
    max_error: float
    mean_error: float
    std_error: float
    n_samples: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
            "mape": self.mape,
            "median_ae": self.median_ae,
            "max_error": self.max_error,
            "mean_error": self.mean_error,
            "std_error": self.std_error,
            "n_samples": self.n_samples,
        }

    def __str__(self) -> str:
        return (
            "MetricsResult(\n"
            f"  RMSE: {self.rmse:.4f}\n"
            f"  MAE: {self.mae:.4f}\n"
            f"  R²: {self.r2:.4f}\n"
            f"  MAPE: {self.mape:.2f}%\n"
            f"  Max Error: {self.max_error:.4f}\n"
            f"  Samples: {self.n_samples}\n"
            ")"
        )


class ModelMetrics:
    """Comprehensive regression metrics calculator."""

    def calculate_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        self._validate_inputs(y_true, y_pred, sample_weight)

        rmse = np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight))
        mae = mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)
        r2 = r2_score(y_true, y_pred, sample_weight=sample_weight)
        median_ae = median_absolute_error(y_true, y_pred)

        errors = y_true - y_pred
        max_error = np.max(np.abs(errors))
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        mape = self._safe_mape(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred, sample_weight=sample_weight)

        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape": mape,
            "median_ae": median_ae,
            "max_error": max_error,
            "mean_error": mean_error,
            "std_error": std_error,
            "mse": mse,
            "n_samples": len(y_true),
        }

    def calculate_metrics_result(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> MetricsResult:
        metrics_dict = self.calculate_all_metrics(y_true, y_pred, sample_weight)
        return MetricsResult(
            rmse=metrics_dict["rmse"],
            mae=metrics_dict["mae"],
            r2=metrics_dict["r2"],
            mape=metrics_dict["mape"],
            median_ae=metrics_dict["median_ae"],
            max_error=metrics_dict["max_error"],
            mean_error=metrics_dict["mean_error"],
            std_error=metrics_dict["std_error"],
            n_samples=metrics_dict["n_samples"],
        )

    def compare_models(
        self,
        y_true: np.ndarray,
        predictions_dict: Dict[str, np.ndarray],
        metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if metrics is None:
            metrics = ["rmse", "mae", "r2", "mape"]

        comparison_results = []

        for model_name, y_pred in predictions_dict.items():
            model_metrics = self.calculate_all_metrics(y_true, y_pred)
            row = {"model": model_name}
            row.update({metric: model_metrics[metric] for metric in metrics})
            comparison_results.append(row)

        df = pd.DataFrame(comparison_results).set_index("model")

        for metric in metrics:
            if metric == "r2":
                df[f"{metric}_rank"] = df[metric].rank(ascending=False)
            else:
                df[f"{metric}_rank"] = df[metric].rank(ascending=True)

        logger.debug("Model comparison completed for %d models", len(predictions_dict))
        return df

    def calculate_time_series_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        time_index: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        base_metrics = self.calculate_all_metrics(y_true, y_pred)
        ts_metrics: Dict[str, float] = {}

        if len(y_true) > 1:
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            ts_metrics["directional_accuracy"] = np.mean(
                true_direction == pred_direction
            )

        ts_metrics["forecast_bias"] = float(np.mean(y_pred - y_true))

        if len(y_true) > 1:
            naive_forecast = np.roll(y_true, 1)[1:]
            actual_next = y_true[1:]
            model_next = y_pred[1:]

            naive_mse = mean_squared_error(actual_next, naive_forecast)
            model_mse = mean_squared_error(actual_next, model_next)

            if naive_mse > 0:
                ts_metrics["theil_u"] = float(
                    np.sqrt(model_mse) / np.sqrt(naive_mse)
                )

        ts_metrics.update(base_metrics)
        return ts_metrics

    def calculate_residual_statistics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        residuals = y_true - y_pred

        stats_dict: Dict[str, float] = {
            "residual_mean": float(np.mean(residuals)),
            "residual_std": float(np.std(residuals)),
            "residual_min": float(np.min(residuals)),
            "residual_max": float(np.max(residuals)),
            "residual_q25": float(np.percentile(residuals, 25)),
            "residual_q50": float(np.percentile(residuals, 50)),
            "residual_q75": float(np.percentile(residuals, 75)),
            "residual_iqr": float(
                np.percentile(residuals, 75) - np.percentile(residuals, 25)
            ),
        }

        if len(residuals) <= 5000:
            stat, p_value = stats.shapiro(residuals)
            stats_dict["normality_test"] = "shapiro"
        else:
            stat, p_value = stats.kstest(residuals, "norm")
            stats_dict["normality_test"] = "ks"

        stats_dict["normality_statistic"] = float(stat)
        stats_dict["normality_p_value"] = float(p_value)
        stats_dict["is_normal"] = bool(p_value > 0.05)

        if len(residuals) > 10:
            stats_dict["durbin_watson"] = float(self._durbin_watson(residuals))

        return stats_dict

    def calculate_confidence_intervals(
        self, y_true: np.ndarray, y_pred: np.ndarray, confidence_level: float = 0.95
    ) -> Dict[str, Tuple[float, float]]:
        n_samples = len(y_true)
        alpha = 1 - confidence_level
        n_bootstrap = 1000
        bootstrap_metrics = []

        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            metrics = self.calculate_all_metrics(y_true_boot, y_pred_boot)
            bootstrap_metrics.append(metrics)

        bootstrap_df = pd.DataFrame(bootstrap_metrics)
        confidence_intervals: Dict[str, Tuple[float, float]] = {}

        for metric in bootstrap_df.columns:
            if metric == "n_samples":
                continue
            lower = float(np.percentile(bootstrap_df[metric], (alpha / 2) * 100))
            upper = float(np.percentile(bootstrap_df[metric], (1 - alpha / 2) * 100))
            confidence_intervals[metric] = (lower, upper)

        return confidence_intervals

    def export_metrics_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        include_residuals: bool = True,
        include_confidence: bool = False,
    ) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "model_name": model_name,
            "basic_metrics": self.calculate_all_metrics(y_true, y_pred),
            "time_series_metrics": self.calculate_time_series_metrics(y_true, y_pred),
        }

        if include_residuals:
            report["residual_statistics"] = self.calculate_residual_statistics(
                y_true, y_pred
            )

        if include_confidence:
            report["confidence_intervals"] = self.calculate_confidence_intervals(
                y_true, y_pred
            )

        logger.info("Generated comprehensive metrics report for %s", model_name)
        return report

    def _validate_inputs(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        if not isinstance(y_true, np.ndarray):
            y_true = np.array(y_true)
        if not isinstance(y_pred, np.ndarray):
            y_pred = np.array(y_pred)

        if len(y_true) != len(y_pred):
            raise InvalidDataError(
                f"y_true and y_pred must have same length: {len(y_true)} != {len(y_pred)}"
            )
        if len(y_true) == 0:
            raise InvalidDataError("Input arrays cannot be empty")

        if sample_weight is not None and len(sample_weight) != len(y_true):
            raise InvalidDataError(
                f"sample_weight length {len(sample_weight)} != {len(y_true)}"
            )

        if not np.all(np.isfinite(y_true)):
            raise InvalidDataError("y_true contains non-finite values")

        if not np.all(np.isfinite(y_pred)):
            raise InvalidDataError("y_pred contains non-finite values")

    def _safe_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator > 1e-8

        if not np.any(mask):
            return float("inf")

        mape_values = np.abs(y_true - y_pred) / denominator
        return float(np.mean(mape_values[mask]) * 100)

    def _durbin_watson(self, residuals: np.ndarray) -> float:
        diff_residuals = np.diff(residuals)
        return float(np.sum(diff_residuals**2) / np.sum(residuals**2))

