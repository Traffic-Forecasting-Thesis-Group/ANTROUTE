from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Sequence
import numpy as np


def route_optimality(c_optimal: np.ndarray, c_predicted: np.ndarray) -> np.ndarray:
    c_optimal = np.asarray(c_optimal, dtype=np.float64)
    c_predicted = np.asarray(c_predicted, dtype=np.float64)
    if np.any(c_predicted <= 0):
        raise ValueError("Route optimality is undefined when a predicted route cost is <= 0.")
    return (c_optimal / c_predicted) * 100.0


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    return float(np.mean(np.abs(actual - predicted)))


def mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    return float(np.mean((actual - predicted) ** 2))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(mse(actual, predicted)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if np.any(actual == 0):
        raise ValueError("MAPE is undefined when an actual value is 0 (division by zero).")
    return float(100.0 * np.mean(np.abs((actual - predicted) / actual)))


def r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    if ss_tot == 0:
        raise ValueError("R^2 is undefined when all actual values are identical (SS_TOT = 0).")
    return float(1.0 - ss_res / ss_tot)


@dataclass
class ScenarioMetrics:
    n_trials: int
    n_eta_trials: int          # ETA observations with a ground truth (one per leg); ETA metrics use only these
    route_optimality_mean: float
    route_optimality_sd: float
    mae: float
    rmse: float
    mse: float
    mape: float
    r_squared: float


def _or_nan(metric, actual: np.ndarray, predicted: np.ndarray) -> float:
    if actual.size == 0:
        return float("nan")
    try:
        return metric(actual, predicted)
    except ValueError:
        return float("nan")


def compute_scenario_metrics(c_optimal, c_predicted, actual_eta, predicted_eta) -> ScenarioMetrics:
    ro = route_optimality(c_optimal, c_predicted)
    if ro.size == 0:
        raise ValueError("cannot compute metrics over zero trials")
    actual_eta = np.asarray(actual_eta, dtype=np.float64)
    predicted_eta = np.asarray(predicted_eta, dtype=np.float64)
    # A trial without a ground-truth ETA (NaN) still counts toward route optimality,
    # but is left out of the ETA error metrics rather than poisoning them.
    has_eta = np.isfinite(actual_eta) & np.isfinite(predicted_eta)
    actual_eta, predicted_eta = actual_eta[has_eta], predicted_eta[has_eta]
    return ScenarioMetrics(
        n_trials=len(ro),
        n_eta_trials=int(has_eta.sum()),
        route_optimality_mean=float(np.mean(ro)),
        route_optimality_sd=float(np.std(ro, ddof=1)) if len(ro) > 1 else 0.0,
        mae=_or_nan(mae, actual_eta, predicted_eta),
        rmse=_or_nan(rmse, actual_eta, predicted_eta),
        mse=_or_nan(mse, actual_eta, predicted_eta),
        mape=_or_nan(mape, actual_eta, predicted_eta),
        r_squared=_or_nan(r_squared, actual_eta, predicted_eta),
    )


def trial_eta_metrics(actual_eta, predicted_eta) -> Dict[str, float]:
    """
    ETA metrics for one trial (Appendix 2's per-trial rows). A single-leg trip has one
    observation, so R^2 is NaN for it; a multi-destination trip is scored over its legs.
    """
    actual = np.atleast_1d(np.asarray(actual_eta, dtype=np.float64))
    predicted = np.atleast_1d(np.asarray(predicted_eta, dtype=np.float64))
    return {
        "mae": _or_nan(mae, actual, predicted),
        "rmse": _or_nan(rmse, actual, predicted),
        "mse": _or_nan(mse, actual, predicted),
        "mape": _or_nan(mape, actual, predicted),
        "r_squared": _or_nan(r_squared, actual, predicted),
    }


def _pooled(trials: Sequence[dict], key: str) -> np.ndarray:
    # A trial's ETA is one value, or one per leg for a multi-destination trip.
    return np.concatenate([np.atleast_1d(np.asarray(t[key], dtype=np.float64)) for t in trials])


def _scenario_metrics(trials: Sequence[dict]) -> ScenarioMetrics:
    return compute_scenario_metrics(
        [t["c_optimal"] for t in trials],
        [t["c_predicted"] for t in trials],
        _pooled(trials, "actual_eta"),
        _pooled(trials, "predicted_eta"),
    )


def compute_metrics_by_scenario(trials: Sequence[dict]) -> Dict[str, ScenarioMetrics]:
    if not trials:
        raise ValueError("cannot compute metrics over zero trials")
    scenario_types = sorted({t["scenario_type"] for t in trials})
    if "overall" in scenario_types:
        raise ValueError("'overall' is reserved for the aggregate row; rename that scenario_type")
    results: Dict[str, ScenarioMetrics] = {"overall": _scenario_metrics(trials)}
    for scenario in scenario_types:
        results[scenario] = _scenario_metrics([t for t in trials if t["scenario_type"] == scenario])
    return results
