"""
ANTROUTE-vs-baseline numbers, read from the evaluation outputs rather than typed in by hand.

  GET /routes/comparison-metrics   the thesis evaluation (Section 3.9, Appendix 3 Table 6):
      outputs/routing_eval/metrics.json from scripts/evaluate_routing.py --apple-maps --
      ANTROUTE against the baseline Improved ACO (Cheng 2023) on Route Optimality (Equation 1)
      and ETA MAE, RMSE, MSE, MAPE and R² (Equations 2-6) against Apple Maps typical traffic,
      with Δ% by Equations 7-8 and the Wilcoxon signed-rank p-value. Until that file exists it
      says so; it never substitutes another table.

  GET /routes/forecast-metrics     diagnostics of the Congestion Risk Score itself:
      risk_edges.csv's held-out TEST split against the human Light/Medium/Heavy labels, against
      a constant at the training labels' mean, after a train-split calibration
      (src/models/risk_calibration.py), with the uncalibrated MAE beside it. This scores the
      model, not routes or ETAs, and is not part of the routing comparison.

Both are cached until the file they read changes.
"""

import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from app.config import settings
from app.schemas_route import (
    ComparisonMetricRow,
    ComparisonMetricsResponse,
    RouteOptimalitySummary,
    ScenarioEvaluation,
    TripEvaluationResponse,
)
from src.models import risk_calibration
from src.routing.significance import relative_difference

REPO_ROOT = Path(__file__).resolve().parents[1]
ROUTING_METRICS = REPO_ROOT / "outputs/routing_eval/metrics.json"
# Where evaluate_routing.py writes the same evaluation with each system's ETA calibrated
# (src/routing/eta_calibration.py), relative to ROUTING_METRICS.
CALIBRATED_SUBDIR = "eta_calibrated"


def _metrics_file() -> Path:
    """
    The evaluation the app shows: the ETA-calibrated one when evaluate_routing.py wrote it
    beside the raw metrics.json, else the raw one. Calibration scales each system's predicted
    ETA by a factor fitted on every other trip, so no trip is scored by a factor that saw its
    own Apple Maps time; routes, and so Route Optimality, are the same in both.
    """
    calibrated = ROUTING_METRICS.parent / CALIBRATED_SUBDIR / ROUTING_METRICS.name
    return calibrated if calibrated.exists() else ROUTING_METRICS


def _is_calibrated(report: dict) -> bool:
    return bool((report.get("eta_calibration") or {}).get("applied"))


def eta_display_scale() -> float:
    """
    The factor the app multiplies its ETA engine's times by, so the time on a route card is
    on Apple Maps' scale: ANTROUTE's ETA scale fitted on every test trip. Both systems' cards
    are timed by that one engine (risk_routing._duration_min), so one factor serves both and
    two routes on the same road still show the same time. 1.0 without a calibrated evaluation.
    """
    path = ROUTING_METRICS.parent / CALIBRATED_SUBDIR / ROUTING_METRICS.name
    if not path.exists():
        return 1.0
    return _cached(path, _display_scale)


def _display_scale(path: Path) -> float:
    report = json.loads(path.read_text(encoding="utf-8"))
    scale = _finite(((report.get("eta_calibration") or {}).get("scale_all_trips") or {}).get("antroute"))
    return scale if scale is not None and scale > 0 else 1.0

# (key in metrics.json, label, higher is better)
# (key in metrics.json, label, higher is better, display scale, unit) -- the thesis's
# evaluation metrics (Section 3.9, Appendix 3 Table 6). metrics.json keeps ETA errors in
# seconds; they are shown in minutes (MSE in min^2) as the app's design does.
ROUTING_ROWS = [
    ("route_optimality", "Route Optimality", True, 1.0, "%"),
    ("mae", "MAE", False, 1 / 60, " min"),
    ("rmse", "RMSE", False, 1 / 60, " min"),
    ("mse", "MSE", False, 1 / 3600, " min²"),
    ("mape", "MAPE", False, 1.0, "%"),
    ("r_squared", "R²", True, 1.0, ""),
]


class MetricsUnavailableError(Exception):
    """No evaluation output is on this server yet; the message says what to run."""


_cache: dict = {}


def _cached(path: Path, build):
    """build(path), recomputed only when the file changes."""
    # metrics.json feeds two builders (the test-set table and the per-trip rows), so the
    # builder is part of the key; a new mtime replaces only that builder's entry.
    key = (build.__name__, str(path), path.stat().st_mtime)
    if key not in _cache:
        for stale in [k for k in _cache if k[0] == key[0]]:
            del _cache[stale]
        _cache[key] = build(path)
    return _cache[key]


def _finite(value) -> Optional[float]:
    return float(value) if value is not None and math.isfinite(float(value)) else None


def _improvement_pct(antroute: float, baseline: float, higher_is_better: bool) -> Optional[float]:
    """Relative difference (thesis Equations 7-8): positive when ANTROUTE is better."""
    if baseline == 0:
        return None
    diff = (antroute - baseline) if higher_is_better else (baseline - antroute)
    return round(diff / abs(baseline) * 100, 1)


def _fmt(value: float) -> str:
    return f"{value:,.2f}" if abs(value) < 1000 else f"{value:,.0f}"


def _show(value: float, unit: str) -> str:
    if unit == "%":
        return f"{value:.1f}%"
    if unit == "":
        return f"{value:.2f}"
    return f"{value:,.1f}{unit}" if abs(value) < 1000 else f"{value:,.0f}{unit}"


# The scenarios of Section 3.9, in the order the app lists them; "overall" pools all trips.
SCENARIO_LABELS = [
    ("overall", "All scenarios"),
    ("recommended", "Recommended"),
    ("multi_destination", "Multi-destination"),
    ("incident_exposed", "Incident-exposed"),
]


def _scenario_rows(report: dict, scenario: str) -> Tuple[List[ComparisonMetricRow], int, Optional[RouteOptimalitySummary]]:
    """
    One scenario's table: per metric ANTROUTE, the Improved ACO baseline, Δ% by Equation 7
    (MAE, RMSE, MSE, MAPE) or Equation 8 (Route Optimality, R²), and the Wilcoxon p-value.
    """
    paired_rows = {r["metric"]: r for r in report.get("significance", []) if r.get("scenario_type") == scenario}
    pooled = {s: (report.get("pooled", {}).get(s) or {}).get(scenario) or {} for s in ("antroute", "baseline")}
    rows: List[ComparisonMetricRow] = []
    n_trials = 0
    optimality = None
    for key, label, higher, scale, unit in ROUTING_ROWS:
        r = paired_rows.get(key) or {}
        if key == "route_optimality" or not pooled["antroute"]:
            # Route Optimality is a per-trip ratio, so its mean over trips is the figure.
            ant, base = _finite(r.get("proposed_mean")), _finite(r.get("baseline_mean"))
        else:
            # ETA errors over every evaluated leg pooled (Equations 2-6 over the test set). A
            # mean of per-trip values is not the same number, and per-trip R² over a trip's
            # one to three legs is meaningless -- averaged, it reached -22.
            ant, base = _finite(pooled["antroute"].get(key)), _finite(pooled["baseline"].get(key))
        if ant is None or base is None:
            continue
        n_trials = max(n_trials, int(r.get("n") or 0))
        delta = relative_difference(base, ant, higher_is_better=higher)
        # The Wilcoxon test pairs the two systems trip by trip; R² has no per-trip value to
        # pair, so it carries no p-value.
        paired = key != "r_squared" and r
        rows.append(
            ComparisonMetricRow(
                metric=label,
                antroute=_show(ant * scale, unit),
                baseline=_show(base * scale, unit),
                improvement_pct=round(delta, 1) if math.isfinite(delta) else None,
                higher_is_better=higher,
                p_value=_finite(r.get("wilcoxon_p")) if paired else None,
                significant=bool(r.get("significant")) if paired else None,
            )
        )
        if key == "route_optimality":
            optimality = RouteOptimalitySummary(antroute=round(ant, 1), baseline=round(base, 1), n_trials=int(r.get("n") or 0))
    return rows, n_trials, optimality


def _routing_metrics(path: Path) -> ComparisonMetricsResponse:
    """
    The thesis evaluation (Appendix 3, Table 6) from scripts/evaluate_routing.py --apple-maps:
    every paired test trip ("overall"), and the same table for each scenario.
    """
    report = json.loads(path.read_text(encoding="utf-8"))
    calibrated = _is_calibrated(report)
    rows, n_trials, optimality = _scenario_rows(report, "overall")
    if not rows:
        raise MetricsUnavailableError(f"{path.name} has no paired results yet.")
    scenarios = []
    for key, label in SCENARIO_LABELS:
        scenario_rows, n, _ = _scenario_rows(report, key)
        if scenario_rows:
            scenarios.append(ScenarioEvaluation(scenario=key, label=label, n_trials=n, metrics=scenario_rows))
    return ComparisonMetricsResponse(
        source="routing",
        title="Evaluation Metrics",
        baseline_name="Improved ACO (Cheng 2023)",
        description=(
            f"{n_trials} paired test trips: both models routed on the same origin-destination "
            "pairs and departure windows, scored against Apple Maps typical traffic. Route "
            "Optimality is the mean over trips; the ETA errors pool every leg. Lower is better for "
            "MAE, RMSE, MSE and MAPE; higher is better for Route Optimality and R²."
            + (
                " Each system's predicted ETA is calibrated by one scale factor fitted on every "
                "other trip (leave-one-trip-out), so no trip is scored by a factor fitted on its "
                "own Apple Maps time."
                if calibrated
                else ""
            )
        ),
        metrics=rows,
        route_optimality=optimality,
        scenarios=scenarios,
        eta_calibrated=calibrated,
    )


def _calibrated(df: pd.DataFrame, train: pd.DataFrame, labelled: pd.DataFrame) -> Tuple[np.ndarray, Optional[str]]:
    """
    The test-split predictions on the labels' scale, and how they got there (None = raw).

    The file's own risk_calibrated column wins (scripts/calibrate_risk_edges.py fitted it
    on the train split); otherwise the same train-split affine fit is made here. Too few
    train labels, or a model whose train scores do not vary, leaves the raw scores.
    """
    if "risk_calibrated" in df.columns:
        return labelled["risk_calibrated"].to_numpy(dtype=float), "risk_calibrated column"
    try:
        calibration = risk_calibration.fit(
            train["risk"].to_numpy(dtype=float), train["weak_target"].to_numpy(dtype=float)
        )
    except ValueError:
        return labelled["risk"].to_numpy(dtype=float), None
    return (
        calibration.apply(labelled["risk"].to_numpy(dtype=float)),
        f"clip({calibration.slope:.3f} x risk + {calibration.intercept:.3f}) fitted on the train split",
    )


def _forecast_metrics(path: Path) -> ComparisonMetricsResponse:
    header = pd.read_csv(path, nrows=0).columns
    needed = {"risk", "weak_target", "camera_edge", "split"}
    if not needed <= set(header):
        raise MetricsUnavailableError(f"{path.name} has no labelled test split to score.")
    df = pd.read_csv(path, usecols=list(needed | ({"risk_calibrated"} & set(header))))
    scored = df[df["camera_edge"].astype(str).str.lower().eq("true") & df["weak_target"].notna()]
    labelled, train = scored[scored["split"].eq("test")], scored[scored["split"].eq("train")]
    if labelled.empty:
        raise MetricsUnavailableError(f"{path.name} has no labelled camera edges in its test split.")

    y = labelled["weak_target"].to_numpy(dtype=float)
    raw = labelled["risk"].to_numpy(dtype=float)
    pred, calibration = _calibrated(df, train, labelled)
    # The stronger constant: the training labels' mean, held out from the test split.
    constant = float(train["weak_target"].mean()) if not train.empty else 0.5
    naive = np.full_like(y, constant)

    def scores(p: np.ndarray) -> Tuple[float, float, float]:
        err = p - y
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - float(np.sum(err**2)) / ss_tot if ss_tot > 0 else float("nan")
        return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err**2))), r2

    ant, base = scores(pred), scores(naive)
    rows = []
    for i, (label, higher) in enumerate([("Congestion Risk MAE", False), ("Congestion Risk RMSE", False), ("R²", True)]):
        a, b = ant[i], base[i]
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        rows.append(
            ComparisonMetricRow(
                metric=label,
                antroute=f"{a:.3f}",
                baseline=f"{b:.3f}",
                # R² can be negative, where a percent of the base flips sign; report the
                # difference in points instead.
                improvement_pct=round((a - b) * 100, 1) if label == "R²" else _improvement_pct(a, b, higher),
                higher_is_better=higher,
            )
        )
    if calibration is not None:
        # What the calibration bought, stated rather than absorbed: the raw scores against the
        # same constant, so the table never reads as if the model had always been on scale.
        raw_mae, base_mae = scores(raw)[0], base[0]
        rows.append(
            ComparisonMetricRow(
                metric="Congestion Risk MAE (uncalibrated)",
                antroute=f"{raw_mae:.3f}",
                baseline=f"{base_mae:.3f}",
                improvement_pct=_improvement_pct(raw_mae, base_mae, False),
                higher_is_better=False,
            )
        )
    scale = f"Scores calibrated by {calibration}; the uncalibrated MAE is shown too. " if calibration else ""
    return ComparisonMetricsResponse(
        source="forecast",
        title="Congestion Forecast Accuracy",
        baseline_name=f"Always {constant:.3f} (training mean)",
        description=(
            f"How well ANTRoute's model predicts congestion on {len(labelled):,} held-out test camera "
            f"edges, against a forecaster that always predicts the training labels' mean ({constant:.3f}). "
            f"{scale}This scores the model, not the routes. The routing comparison appears here once "
            "scripts/evaluate_routing.py has been run."
        ),
        metrics=rows,
    )


def _risk_edges_file() -> Path:
    path = Path(settings.risk_edges_path)
    return path if path.is_absolute() else REPO_ROOT / path


def get_comparison_metrics() -> ComparisonMetricsResponse:
    """
    The Comparison page's metrics: the routing evaluation only (Route Optimality and ETA
    accuracy against Apple Maps, Section 3.9). Raises MetricsUnavailableError -- never a
    substitute table -- until scripts/evaluate_routing.py --apple-maps has written it.
    """
    path = _metrics_file()
    if path.exists():
        return _cached(path, _routing_metrics)
    raise MetricsUnavailableError(
        "The routing evaluation hasn't been run yet. Route the test trips with "
        "scripts/evaluate_routing.py --template, fill in the Apple Maps travel times, then score "
        "them with --apple-maps to produce outputs/routing_eval/metrics.json."
    )


def _per_trip(path: Path) -> Tuple[List[dict], bool]:
    report = json.loads(path.read_text(encoding="utf-8"))
    return report.get("per_trip", []), _is_calibrated(report)


def _clock(window: str) -> str:
    when = pd.Timestamp(window)
    return f"{when:%a %b} {when.day}, {when.hour % 12 or 12}:{when.minute:02d} {'AM' if when.hour < 12 else 'PM'}"


def _trip_rows(row: dict) -> Tuple[List[ComparisonMetricRow], Optional[RouteOptimalitySummary]]:
    """One evaluated trip's metrics, formatted like the test-set table (ETA errors in minutes)."""
    rows: List[ComparisonMetricRow] = []
    optimality = None
    for key, label, higher, scale, unit in ROUTING_ROWS:
        ant, base = _finite(row.get(f"{key}_antroute")), _finite(row.get(f"{key}_baseline"))
        if ant is None or base is None:
            continue  # R² is undefined on a single-leg trip
        delta = relative_difference(base, ant, higher_is_better=higher)
        rows.append(
            ComparisonMetricRow(
                metric=label,
                antroute=_show(ant * scale, unit),
                baseline=_show(base * scale, unit),
                improvement_pct=round(delta, 1) if math.isfinite(delta) else None,
                higher_is_better=higher,
            )
        )
        if key == "route_optimality":
            optimality = RouteOptimalitySummary(antroute=round(ant, 1), baseline=round(base, 1), n_trials=1)
    return rows, optimality


def get_trip_evaluation(stops: List[int], window: Optional[str]) -> TripEvaluationResponse:
    """
    The evaluation of the trip the user planned: the test trip with the same stops (graph
    nodes, origin first) and the same departure time of day as `window`, the recorded window
    the app routed on. Only test trips have Apple Maps travel times, so a trip that is not
    one gets no metrics -- never another trip's, and never the test-set mean in their place.
    """
    path = _metrics_file()
    if not path.exists():
        raise MetricsUnavailableError(
            "The routing evaluation hasn't been run yet, so no trip has Apple Maps results."
        )
    per_trip, calibrated = _cached(path, _per_trip)
    trips = [r for r in per_trip if r.get("stop_nodes")]
    same_stops = [r for r in trips if [int(v) for v in str(r["stop_nodes"]).split()] == list(stops)]
    if not same_stops:
        return TripEvaluationResponse(
            status="not_evaluated",
            message=(
                f"This trip isn't one of the {len(trips)} test trips timed in Apple Maps, so it has "
                "no Route Optimality or ETA error to report. Plan a test trip to see its evaluation."
            ),
        )
    minute = (lambda w: (pd.Timestamp(w).hour, pd.Timestamp(w).minute))
    match = next((r for r in same_stops if window and minute(r["window"]) == minute(window)), None)
    if match is None:
        r = same_stops[0]
        return TripEvaluationResponse(
            status="other_time",
            message=(
                f"These stops were evaluated as trip {r['trip_id']}, departing {_clock(r['window'])}. "
                "Set that departure time to see its results; traffic, and so the routes, differ by time."
            ),
            trip_id=r["trip_id"],
            scenario_type=r.get("scenario_type"),
            evaluated_window=r["window"],
        )
    rows, optimality = _trip_rows(match)
    return TripEvaluationResponse(
        status="evaluated",
        message=(
            f"Trip {match['trip_id']} ({str(match.get('scenario_type', '')).replace('_', ' ')}), "
            f"departing {_clock(match['window'])}: both models' routes timed in Apple Maps "
            "typical traffic against Apple's own route."
        ),
        trip_id=match["trip_id"],
        scenario_type=match.get("scenario_type"),
        evaluated_window=match["window"],
        route_optimality=optimality,
        metrics=rows,
        eta_calibrated=calibrated,
    )


def get_forecast_metrics() -> ComparisonMetricsResponse:
    """Congestion-model diagnostics (risk_edges.csv's test split), kept apart from the thesis
    routing evaluation: these score the CRS, not routes or ETAs."""
    risk_edges = _risk_edges_file()
    if risk_edges.exists():
        return _cached(risk_edges, _forecast_metrics)
    raise MetricsUnavailableError("risk_edges.csv isn't on the server, so there is no forecast to score.")
