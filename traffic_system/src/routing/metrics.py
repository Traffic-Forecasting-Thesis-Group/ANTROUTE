"""
Routing evaluation metrics, computed per trial and summarised per scenario:

    Route Optimality      C_optimal / C_chosen * 100, both timed by the reference (Google Maps)
    congestion exposure   sum of Risk(u, v) over the route's edges (RADR eq. 5)
    event exposure        sum of Event(u, v) over the route's edges (event_layer.event_exposure)
    avoidable event exp.  the same without the edges at the trip's own stops, which no route
                          can avoid
    route length          metres; descriptive only, never tested for significance

ETA is not a metric of this thesis: no system's own predicted travel time is scored here.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Sequence
import numpy as np

# Paired, tested metrics (significance.compare_paired) and the descriptive extra.
TESTED_METRICS = ("route_optimality", "congestion_exposure", "event_exposure", "avoidable_event_exposure")
DESCRIPTIVE_METRICS = ("distance_m",)


def route_optimality(c_optimal: np.ndarray, c_predicted: np.ndarray) -> np.ndarray:
    c_optimal = np.asarray(c_optimal, dtype=np.float64)
    c_predicted = np.asarray(c_predicted, dtype=np.float64)
    if np.any(c_predicted <= 0):
        raise ValueError("Route optimality is undefined when a predicted route cost is <= 0.")
    return (c_optimal / c_predicted) * 100.0


@dataclass
class ScenarioMetrics:
    n_trials: int
    route_optimality_mean: float
    route_optimality_sd: float
    congestion_exposure_mean: float
    event_exposure_mean: float
    avoidable_event_exposure_mean: float
    distance_m_mean: float


def _mean(values: Sequence[float]) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else float("nan")


def _scenario_metrics(trials: Sequence[dict]) -> ScenarioMetrics:
    ro = route_optimality([t["c_optimal"] for t in trials], [t["c_predicted"] for t in trials])
    return ScenarioMetrics(
        n_trials=len(ro),
        route_optimality_mean=float(np.mean(ro)),
        route_optimality_sd=float(np.std(ro, ddof=1)) if len(ro) > 1 else 0.0,
        congestion_exposure_mean=_mean([t["congestion_exposure"] for t in trials]),
        event_exposure_mean=_mean([t["event_exposure"] for t in trials]),
        avoidable_event_exposure_mean=_mean([t["avoidable_event_exposure"] for t in trials]),
        distance_m_mean=_mean([t["distance_m"] for t in trials]),
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
