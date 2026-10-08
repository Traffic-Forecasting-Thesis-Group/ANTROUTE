"""
ANTROUTE-vs-baseline numbers for GET /routes/comparison-metrics, read from the
evaluation outputs rather than typed in by hand.

Two sources, best first:

  routing     outputs/routing_eval/metrics.json from scripts/evaluate_routing.py
              --apple-maps: the thesis comparison (Appendix 3), ANTROUTE against the
              baseline Improved ACO (Cheng 2023) on Route Optimality and the ETA metrics,
              with the Wilcoxon signed-rank p-value of each.
  forecast    risk_edges.csv's held-out TEST split: how well the congestion model predicts
              the human-labelled Light/Medium/Heavy targets, against a forecaster that
              always predicts Medium (0.5) -- the same calculation as
              scripts/compute_comparison_metrics.py. Shown until the routing evaluation
              has been run.

Neither available raises MetricsUnavailableError, so the app says so instead of showing
made-up numbers. Both are cached until the file they read changes.
"""

import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from app.config import settings
from app.schemas_route import ComparisonMetricRow, ComparisonMetricsResponse

REPO_ROOT = Path(__file__).resolve().parents[1]
ROUTING_METRICS = REPO_ROOT / "outputs/routing_eval/metrics.json"

# (key in metrics.json, label, higher is better)
ROUTING_ROWS = [
    ("route_optimality", "Route Optimality (%)", True),
    ("mae", "ETA MAE (s)", False),
    ("rmse", "ETA RMSE (s)", False),
    ("mse", "ETA MSE (s²)", False),
    ("mape", "ETA MAPE (%)", False),
    ("r_squared", "ETA R²", True),
]


class MetricsUnavailableError(Exception):
    """No evaluation output is on this server yet; the message says what to run."""


_cache: dict = {}


def _cached(path: Path, build):
    """build(path), recomputed only when the file changes."""
    key = (str(path), path.stat().st_mtime)
    if key not in _cache:
        _cache.clear()
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


def _routing_metrics(path: Path) -> ComparisonMetricsResponse:
    report = json.loads(path.read_text(encoding="utf-8"))
    overall = {r["metric"]: r for r in report.get("significance", []) if r.get("scenario_type") == "overall"}
    rows: List[ComparisonMetricRow] = []
    n_trials = 0
    for key, label, higher in ROUTING_ROWS:
        r = overall.get(key)
        if r is None:
            continue
        ant, base = _finite(r.get("proposed_mean")), _finite(r.get("baseline_mean"))
        if ant is None or base is None:
            continue
        n_trials = max(n_trials, int(r.get("n") or 0))
        rows.append(
            ComparisonMetricRow(
                metric=label,
                antroute=_fmt(ant),
                baseline=_fmt(base),
                improvement_pct=_improvement_pct(ant, base, higher),
                higher_is_better=higher,
                p_value=_finite(r.get("wilcoxon_p")),
                significant=bool(r.get("significant")),
            )
        )
    if not rows:
        raise MetricsUnavailableError(f"{path.name} has no paired results yet.")
    return ComparisonMetricsResponse(
        source="routing",
        title="Routing Evaluation",
        baseline_name="Improved ACO (Cheng 2023)",
        description=(
            f"Mean per trip over {n_trials} paired test trips, both models routed on the same "
            "origin-destination pairs and departure windows and scored against Apple Maps."
        ),
        metrics=rows,
    )


def _forecast_metrics(path: Path) -> ComparisonMetricsResponse:
    header = pd.read_csv(path, nrows=0).columns
    needed = {"risk", "weak_target", "camera_edge", "split"}
    if not needed <= set(header):
        raise MetricsUnavailableError(f"{path.name} has no labelled test split to score.")
    df = pd.read_csv(path, usecols=list(needed))
    labelled = df[df["camera_edge"].astype(str).str.lower().eq("true") & df["weak_target"].notna() & df["split"].eq("test")]
    if labelled.empty:
        raise MetricsUnavailableError(f"{path.name} has no labelled camera edges in its test split.")

    y = labelled["weak_target"].to_numpy(dtype=float)
    pred = labelled["risk"].to_numpy(dtype=float)
    naive = np.full_like(y, 0.5)

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
    return ComparisonMetricsResponse(
        source="forecast",
        title="Congestion Forecast Accuracy",
        baseline_name="Always 'Medium'",
        description=(
            f"How well ANTRoute's model predicts congestion on {len(labelled):,} held-out test camera "
            "edges, against a forecaster that always predicts Medium. This scores the model, not the "
            "routes. The routing comparison appears here once scripts/evaluate_routing.py has been run."
        ),
        metrics=rows,
    )


def get_comparison_metrics() -> ComparisonMetricsResponse:
    if ROUTING_METRICS.exists():
        return _cached(ROUTING_METRICS, _routing_metrics)
    risk_edges = Path(settings.risk_edges_path)
    if not risk_edges.is_absolute():
        risk_edges = REPO_ROOT / risk_edges
    if risk_edges.exists():
        return _cached(risk_edges, _forecast_metrics)
    raise MetricsUnavailableError(
        "No evaluation results on the server yet. Run scripts/evaluate_routing.py, or add "
        "risk_edges.csv for the congestion forecast scores."
    )
