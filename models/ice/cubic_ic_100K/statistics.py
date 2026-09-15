"""Retained autocorrelation-aware estimators used for the ice-Ih acceptance audit."""
from __future__ import annotations
import math
from typing import Iterable
import numpy as np
from numpy.typing import NDArray
from scipy import stats

def _newey_west_lag(record_count: int) -> int:
    return max(1, int(math.floor(4.0 * (record_count / 100.0) ** (2.0 / 9.0))))


def _hac_covariance(
    design: NDArray[np.float64], residual: NDArray[np.float64], lag: int
) -> NDArray[np.float64]:
    meat = (design * residual[:, None]).T @ (design * residual[:, None])
    for offset in range(1, lag + 1):
        weight = 1.0 - offset / (lag + 1.0)
        right = design[offset:] * residual[offset:, None]
        left = design[:-offset] * residual[:-offset, None]
        cross = right.T @ left
        meat += weight * (cross + cross.T)
    bread = np.linalg.inv(design.T @ design)
    return bread @ meat @ bread


def hac_linear_trend(values: NDArray[np.float64], interval_ns: float) -> dict[str, float]:
    """Return an OLS slope with a Bartlett-kernel Newey--West uncertainty."""

    if np.ptp(values) == 0.0:
        return {
            "slope_per_ns": 0.0,
            "slope_standard_error_per_ns": 0.0,
            "raw_two_sided_p_value": 1.0,
            "newey_west_lag_records": _newey_west_lag(values.size),
        }
    time_ns = np.arange(values.size, dtype=np.float64) * interval_ns
    centered_time = time_ns - np.mean(time_ns)
    design = np.column_stack((np.ones(values.size), centered_time))
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    residual = values - design @ coefficients
    lag = _newey_west_lag(values.size)
    covariance = _hac_covariance(design, residual, lag)
    standard_error = math.sqrt(max(float(covariance[1, 1]), 0.0))
    if standard_error == 0.0:
        p_value = 0.0 if coefficients[1] != 0.0 else 1.0
    else:
        p_value = 2.0 * stats.norm.sf(abs(float(coefficients[1])) / standard_error)
    return {
        "slope_per_ns": float(coefficients[1]),
        "slope_standard_error_per_ns": standard_error,
        "raw_two_sided_p_value": float(p_value),
        "newey_west_lag_records": lag,
    }


def hac_mean_interval(values: NDArray[np.float64], alpha: float) -> dict[str, float]:
    design = np.ones((values.size, 1), dtype=np.float64)
    mean = float(np.mean(values))
    residual = values - mean
    lag = _newey_west_lag(values.size)
    covariance = _hac_covariance(design, residual, lag)
    standard_error = math.sqrt(max(float(covariance[0, 0]), 0.0))
    critical = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return {
        "mean": mean,
        "standard_deviation": float(np.std(values, ddof=1)),
        "hac_standard_error": standard_error,
        "confidence_level": 1.0 - alpha,
        "confidence_interval": [
            mean - critical * standard_error,
            mean + critical * standard_error,
        ],
        "newey_west_lag_records": lag,
    }


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = np.asarray(list(p_values), dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty(values.size, dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (values.size - rank) * float(values[index]))
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


