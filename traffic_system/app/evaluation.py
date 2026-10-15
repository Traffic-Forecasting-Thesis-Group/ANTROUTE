"""
ANTROUTE-vs-baseline numbers, read from the evaluation outputs rather than typed in by hand.

  GET /routes/comparison-metrics   the thesis evaluation (Section 3.9, Appendix 3 Table 6):
      outputs/routing_eval/metrics.json from scripts/evaluate_routing.py --google-maps --
      ANTROUTE against the baseline Improved ACO (Cheng 2023) on Route Optimality (Equation 1,
      both routes timed by Google Maps), congestion exposure and event exposure, with route
      length as a descriptive extra, Δ% by Equations 7-8 and the Wilcoxon signed-rank p-value.
      ETA is not a metric. Until that file exists it says so; it never substitutes another table.

  GET /routes/forecast-metrics     diagnostics of the Congestion Risk Score itself:
      risk_edges.csv's held-out TEST split against the human Light/Medium/Heavy labels, against
      a constant at the training labels' mean, after a train-split calibration
      (src/models/risk_calibration.py), with the uncalibrated MAE beside it. This scores the
      model, not routes, and is not part of the routing comparison.

Both are cached until the file they read changes.
"""

import json
import math
from dataclasses import asdict
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
from src.routing.significance import Q3_FAMILY, compare_paired, holm_adjust, relative_difference

REPO_ROOT = Path(__file__).resolve().parents[1]
ROUTING_METRICS = REPO_ROOT / "outputs/routing_eval/metrics.json"

# (key in metrics.json, label, higher is better, display scale, unit) -- the thesis's routing
# metrics (Section 3.9, Appendix 3 Table 6), each paired per trip and tested. ETA is not one.
ROUTING_ROWS = [
    ("route_optimality", "Route Optimality", True, 1.0, "%"),
    ("congestion_exposure", "Congestion exposure", False, 1.0, ""),
    ("event_exposure", "Event exposure", False, 1.0, ""),
    ("avoidable_event_exposure", "Avoidable event exposure", False, 1.0, ""),
]
# Shown beside them, never tested: route length (metres in metrics.json, shown in km).
DESCRIPTIVE_ROWS = [("distance_m", "Route length (descriptive)", 1 / 1000, " km")]


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
    if unit == " km":
        return f"{value:.2f} km"
    return f"{value:,.1f}{unit}" if abs(value) < 1000 else f"{value:,.0f}{unit}"


# The scenarios of Section 3.9, in the order the app lists them; "overall" pools all trips.
SCENARIO_LABELS = [
    ("overall", "All scenarios"),
    ("single_destination", "Single destination"),
    ("multi_destination", "Multi-destination"),
    ("recommended", "Recommended"),
    ("incident_exposed", "Incident-exposed"),
]

# Research question 1 asks about single- and multi-destination trips. "multi_destination" is a
# scenario type in its own right; "single_destination" is its complement, which newer evaluation
# runs record directly (scripts/evaluate_routing.py). Older metrics.json files predate it, so it
# is derived from per_trip below with the same paired test and Holm families the script uses --
# never re-averaged from the other scenarios' means, which would not be a valid pooling.
DERIVED_SCENARIOS = {
    "single_destination": lambda row: row.get("scenario_type") != "multi_destination",
}


def _derived_significance(report: dict, scenario: str) -> List[dict]:
    """Significance rows for a scenario this evaluation file does not carry, scored from per_trip."""
    predicate = DERIVED_SCENARIOS.get(scenario)
    if predicate is None:
        return []
    rows = [r for r in report.get("per_trip", []) if predicate(r)]
    if not rows:
        return []
    tested = {}
    for key, _label, _higher, _scale, _unit in ROUTING_ROWS:
        proposed = [_finite(r.get(f"{key}_antroute")) for r in rows]
        baseline = [_finite(r.get(f"{key}_baseline")) for r in rows]
        pairs = [(p, b) for p, b in zip(proposed, baseline) if p is not None and b is not None]
        if not pairs:
            continue
        tested[key] = compare_paired(key, [p for p, _ in pairs], [b for _, b in pairs])
    if not tested:
        return []
    # The same three Holm families the evaluation script applies.
    holm_adjust([tested[m] for m in Q3_FAMILY if m in tested])
    holm_adjust([c for m, c in tested.items() if m not in Q3_FAMILY and m != "avoidable_event_exposure"])
    holm_adjust([tested[m] for m in ("avoidable_event_exposure",) if m in tested])
    return [{"scenario_type": scenario, **asdict(c)} for c in tested.values()]


def _derived_means(report: dict, scenario: str) -> dict:
    """Descriptive means (route length) for a derived scenario, straight from its per_trip rows."""
    predicate = DERIVED_SCENARIOS.get(scenario)
    if predicate is None:
        return {}
    rows = [r for r in report.get("per_trip", []) if predicate(r)]
    means: dict = {}
    for key, _label, _scale, _unit in DESCRIPTIVE_ROWS:
        for system in ("antroute", "baseline"):
            values = [v for v in (_finite(r.get(f"{key}_{system}")) for r in rows) if v is not None]
            if values:
                means.setdefault(key, {})[system] = {"mean": float(np.mean(values))}
    return means


def _scenario_rows(report: dict, scenario: str) -> Tuple[List[ComparisonMetricRow], int, Optional[RouteOptimalitySummary]]:
    """
    One scenario's table: per tested metric ANTROUTE, the Improved ACO baseline, Δ% by
    Equation 8 (Route Optimality) or Equation 7 (the exposures), and the Wilcoxon p-value;
    then route length, descriptive only (no Δ%, no test). Every value is the mean over trips.
    """
    significance = [r for r in report.get("significance", []) if r.get("scenario_type") == scenario]
    if not significance:
        significance = _derived_significance(report, scenario)
    paired_rows = {r["metric"]: r for r in significance}
    rows: List[ComparisonMetricRow] = []
    n_trials = 0
    optimality = None
    for key, label, higher, scale, unit in ROUTING_ROWS:
        r = paired_rows.get(key) or {}
        ant, base = _finite(r.get("proposed_mean")), _finite(r.get("baseline_mean"))
        if ant is None or base is None:
            continue
        n_trials = max(n_trials, int(r.get("n") or 0))
        delta = relative_difference(base, ant, higher_is_better=higher)
        rows.append(
            ComparisonMetricRow(
                metric=label,
                antroute=_show(ant * scale, unit),
                baseline=_show(base * scale, unit),
                improvement_pct=round(delta, 1) if math.isfinite(delta) else None,
                higher_is_better=higher,
                p_value=_finite(r.get("wilcoxon_p_holm")) if r.get("wilcoxon_p_holm") is not None else _finite(r.get("wilcoxon_p")),
                significant=bool(r["significant_holm"]) if r.get("significant_holm") is not None else bool(r.get("significant")),
            )
        )
        if key == "route_optimality":
            optimality = RouteOptimalitySummary(antroute=round(ant, 1), baseline=round(base, 1), n_trials=int(r.get("n") or 0))
    if rows:
        means = (report.get("per_trial_mean_sd") or {}).get(scenario) or _derived_means(report, scenario)
        for key, label, scale, unit in DESCRIPTIVE_ROWS:
            m = means.get(key) or {}
            ant, base = _finite((m.get("antroute") or {}).get("mean")), _finite((m.get("baseline") or {}).get("mean"))
            if ant is not None and base is not None:
                rows.append(ComparisonMetricRow(metric=label, antroute=_show(ant * scale, unit),
                                                baseline=_show(base * scale, unit)))
    return rows, n_trials, optimality


def _reference_text(report: dict) -> str:
    """How the Google Maps reference was obtained, from metrics.json -- or that it was not recorded."""
    ref = report.get("reference") or {}
    if not ref.get("access_methods"):
        return "How the reference times were obtained is not recorded in this evaluation."
    dates = ref.get("departure_dates") or {}
    queried = ref.get("queried_on") or {}
    return (
        f"Reference: Google Maps ({'; '.join(ref['access_methods'])}), traffic model "
        f"{'; '.join(ref.get('traffic_models') or ['not recorded'])}, departure slots "
        f"{', '.join(ref.get('departure_slots') or [])} on dates {dates.get('first')} to {dates.get('last')}, "
        f"looked up {queried.get('first')} to {queried.get('last')}."
    )


def _model_text(report: dict) -> str:
    model = report.get("model") or {}
    if not model.get("checkpoint"):
        return "The checkpoint behind these risk scores is not recorded."
    text = "on" if model.get("use_text") else "off"
    return f"Risk scores from {Path(str(model['checkpoint'])).name} (text branch {text})."


def _routing_metrics(path: Path) -> ComparisonMetricsResponse:
    """
    The thesis evaluation (Appendix 3, Table 6) from scripts/evaluate_routing.py --google-maps:
    every paired test trip ("overall"), and the same table for each scenario.
    """
    report = json.loads(path.read_text(encoding="utf-8"))
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
            f"{n_trials} paired test trips in Metro Manila: both models routed on the same "
            "origin-destination pairs and departure windows. Route Optimality times both routes in "
            "Google Maps against Google's own route (higher is better); congestion and event exposure "
            "sum the predicted risk and the reported-incident impact along each route (lower is "
            "better; avoidable event exposure leaves out incidents at a trip's own stops, which no route "
            "can avoid); route length is descriptive. All are means over trips. ETA is not evaluated. "
            f"{_reference_text(report)} {_model_text(report)}"
        ),
        metrics=rows,
        route_optimality=optimality,
        scenarios=scenarios,
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
    The Comparison page's metrics: the routing evaluation only (Route Optimality against Google
    Maps, congestion and event exposure; Section 3.9). Raises MetricsUnavailableError -- never a
    substitute table -- until scripts/evaluate_routing.py --google-maps has written it.
    """
    if ROUTING_METRICS.exists():
        return _cached(ROUTING_METRICS, _routing_metrics)
    raise MetricsUnavailableError(
        "The routing evaluation hasn't been run yet. Route the test trips with "
        "scripts/evaluate_routing.py --template, fill in the Google Maps travel times and how they "
        "were obtained, then score them with --google-maps to produce outputs/routing_eval/metrics.json."
    )


def _per_trip(path: Path) -> List[dict]:
    return json.loads(path.read_text(encoding="utf-8")).get("per_trip", [])


def _clock(window: str) -> str:
    when = pd.Timestamp(window)
    return f"{when:%a %b} {when.day}, {when.hour % 12 or 12}:{when.minute:02d} {'AM' if when.hour < 12 else 'PM'}"


def _trip_rows(row: dict) -> Tuple[List[ComparisonMetricRow], Optional[RouteOptimalitySummary]]:
    """One evaluated trip's metrics, formatted like the test-set table."""
    rows: List[ComparisonMetricRow] = []
    optimality = None
    for key, label, higher, scale, unit in ROUTING_ROWS:
        ant, base = _finite(row.get(f"{key}_antroute")), _finite(row.get(f"{key}_baseline"))
        if ant is None or base is None:
            continue
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
    for key, label, scale, unit in DESCRIPTIVE_ROWS:
        ant, base = _finite(row.get(f"{key}_antroute")), _finite(row.get(f"{key}_baseline"))
        if ant is not None and base is not None:
            rows.append(ComparisonMetricRow(metric=label, antroute=_show(ant * scale, unit),
                                            baseline=_show(base * scale, unit)))
    return rows, optimality


def get_trip_evaluation(stops: List[int], window: Optional[str]) -> TripEvaluationResponse:
    """
    The evaluation of the trip the user planned: the test trip with the same stops (graph
    nodes, origin first) and the same departure time of day as `window`, the recorded window
    the app routed on. Only test trips have Google Maps travel times, so a trip that is not
    one gets no metrics -- never another trip's, and never the test-set mean in their place.
    """
    if not ROUTING_METRICS.exists():
        raise MetricsUnavailableError(
            "The routing evaluation hasn't been run yet, so no trip has Google Maps results."
        )
    per_trip = _cached(ROUTING_METRICS, _per_trip)
    trips = [r for r in per_trip if r.get("stop_nodes")]
    same_stops = [r for r in trips if [int(v) for v in str(r["stop_nodes"]).split()] == list(stops)]
    if not same_stops:
        return TripEvaluationResponse(
            status="not_evaluated",
            message=(
                f"This trip isn't one of the {len(trips)} test trips timed in Google Maps, so it has "
                "no evaluation to report. Plan a test trip to see its evaluation."
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
            f"departing {_clock(match['window'])}: both models' routes timed in Google Maps against "
            "Google's own route, with each route's congestion and event exposure."
        ),
        trip_id=match["trip_id"],
        scenario_type=match.get("scenario_type"),
        evaluated_window=match["window"],
        route_optimality=optimality,
        metrics=rows,
    )


def get_forecast_metrics() -> ComparisonMetricsResponse:
    """Congestion-model diagnostics (risk_edges.csv's test split), kept apart from the thesis
    routing evaluation: these score the CRS, not routes or ETAs."""
    risk_edges = _risk_edges_file()
    if risk_edges.exists():
        return _cached(risk_edges, _forecast_metrics)
    raise MetricsUnavailableError("risk_edges.csv isn't on the server, so there is no forecast to score.")
