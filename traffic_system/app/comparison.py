"""
ANTRoute against the Improved ACO baseline, trip by trip (research question 3).

Everything here is read from the evaluation outputs; nothing is routed or re-tested in a
request. Three files feed it:

    outputs/routing_eval/trials.csv      one row per trip per system: the route it planned
                                         (node path), its Route Optimality against Google
                                         Maps, its exposures and its length
    outputs/routing_eval/metrics.json    the paired Wilcoxon tests with Holm correction, and
                                         how the Google Maps reference was obtained
    outputs/condition_updates/*/summary  recalculation response time for both systems, which
                                         RQ3 pairs with Route Optimality

Travel time is not here. It is not a thesis metric, was never scored against a reference, and
therefore has no paired test -- the app shows it only as an estimate beside a route.
"""

from __future__ import annotations

import csv
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from app import risk_routing as rr
from app.schemas_route import (
    ComparisonProvenance,
    ComparisonTripDetail,
    ComparisonTripRow,
    ComparisonTripSystem,
    ComparisonTripsResponse,
    PathPoint,
    ResponseTimeComparison,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TRIALS_CSV = REPO_ROOT / "outputs" / "routing_eval" / "trials.csv"
METRICS_JSON = REPO_ROOT / "outputs" / "routing_eval" / "metrics.json"
CONDITION_ROOT = REPO_ROOT / "outputs" / "condition_updates"

SYSTEMS = ("antroute", "baseline")


class ComparisonUnavailableError(Exception):
    """The routing evaluation has not been run on this server."""


def _f(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


@lru_cache(maxsize=1)
def _trials() -> List[dict]:
    if not TRIALS_CSV.exists():
        raise ComparisonUnavailableError(
            "The routing evaluation hasn't been run yet, so there are no trials to compare. "
            "Run scripts/evaluate_routing.py --google-maps."
        )
    with TRIALS_CSV.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


@lru_cache(maxsize=1)
def _by_trip() -> Dict[str, Dict[str, dict]]:
    """trip_id -> system -> trial row."""
    out: Dict[str, Dict[str, dict]] = {}
    for row in _trials():
        out.setdefault(row["trip_id"], {})[row["system"]] = row
    return out


@lru_cache(maxsize=1)
def _report() -> dict:
    if not METRICS_JSON.exists():
        raise ComparisonUnavailableError("outputs/routing_eval/metrics.json is not on this server.")
    return json.loads(METRICS_JSON.read_text(encoding="utf-8"))


def _system(row: Optional[dict]) -> Optional[ComparisonTripSystem]:
    if not row:
        return None
    distance = _f(row.get("distance_m"))
    return ComparisonTripSystem(
        route_optimality=_f(row.get("route_optimality")),
        congestion_exposure=_f(row.get("congestion_exposure")),
        event_exposure=_f(row.get("event_exposure")),
        avoidable_event_exposure=_f(row.get("avoidable_event_exposure")),
        distance_km=round(distance / 1000, 2) if distance is not None else None,
    )


def _trip_row(trip_id: str, pair: Dict[str, dict]) -> ComparisonTripRow:
    any_row = pair.get("antroute") or pair.get("baseline") or {}
    return ComparisonTripRow(
        trip_id=trip_id,
        scenario_type=any_row.get("scenario_type") or "",
        window=any_row.get("window") or "",
        stops=any_row.get("stops") or "",
        antroute=_system(pair.get("antroute")),
        baseline=_system(pair.get("baseline")),
    )


def _provenance() -> ComparisonProvenance:
    report = _report()
    reference = report.get("reference") or {}
    departures = reference.get("departure_dates") or {}
    queried = reference.get("queried_on") or {}
    windows = sorted({r.get("window") for r in _trials() if r.get("window")})
    return ComparisonProvenance(
        scope=str(report.get("scope") or ""),
        reference=str(reference.get("reference") or ""),
        access_methods=[str(m) for m in (reference.get("access_methods") or [])],
        reference_departure_first=str(departures.get("first") or "") or None,
        reference_departure_last=str(departures.get("last") or "") or None,
        reference_queried_first=str(queried.get("first") or "") or None,
        reference_queried_last=str(queried.get("last") or "") or None,
        recorded_window_first=windows[0] if windows else None,
        recorded_window_last=windows[-1] if windows else None,
        n_trips=len(_by_trip()),
        baseline_name="Improved ACO (Cheng 2023)",
    )


def response_time_comparison(update: str = "event") -> Optional[ResponseTimeComparison]:
    """
    Recalculation response time for both systems (RQ3b), from the condition-update harness.

    The baseline is the faster of the two in every update kind that was run: it searches on
    distance and flow alone, while ANTRoute also applies the risk tiers and the Event-Aware
    layer before searching. Reported as measured.
    """
    path = CONDITION_ROOT / update / "summary.json"
    if not path.exists():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    by_system = summary.get("by_system") or {}
    ant = (by_system.get("antroute") or {}).get("response_time_s") or {}
    base = (by_system.get("baseline") or {}).get("response_time_s") or {}
    paired = summary.get("response_time_antroute_vs_baseline") or {}
    ant_median, base_median = _f(ant.get("median")), _f(base.get("median"))
    if ant_median is None or base_median is None:
        return None
    p = _f(paired.get("wilcoxon_p_holm"))
    if p is None:
        p = _f(paired.get("wilcoxon_p"))
    significant = paired.get("significant_holm")
    if significant is None:
        significant = paired.get("significant")
    return ResponseTimeComparison(
        update=update,
        n=int(paired.get("n") or ant.get("n") or 0),
        antroute_median_s=ant_median,
        baseline_median_s=base_median,
        antroute_p95_s=_f(ant.get("p95")),
        baseline_p95_s=_f(base.get("p95")),
        faster="antroute" if ant_median < base_median else "baseline",
        relative_difference_pct=_f(paired.get("relative_difference_pct")),
        p_value=p,
        significant=bool(significant) if significant is not None else None,
        definition=str(summary.get("response_time_definition") or ""),
    )


def available_updates() -> List[str]:
    return [k for k in ("traffic", "event", "session") if (CONDITION_ROOT / k / "summary.json").exists()]


def get_comparison_trips(scenario: str = "overall", update: str = "event") -> ComparisonTripsResponse:
    """Every evaluated trip in one scenario, with both systems' results side by side."""
    pairs = _by_trip()
    rows = [
        _trip_row(trip_id, pair)
        for trip_id, pair in pairs.items()
        if scenario == "overall"
        or (scenario == "single_destination" and (pair.get("antroute") or {}).get("scenario_type") != "multi_destination")
        or (pair.get("antroute") or {}).get("scenario_type") == scenario
    ]
    rows.sort(key=lambda r: r.trip_id)
    return ComparisonTripsResponse(
        scenario=scenario,
        trips=rows,
        provenance=_provenance(),
        response_time=response_time_comparison(update),
        available_updates=available_updates(),
    )


def get_comparison_trip_detail(trip_id: str) -> ComparisonTripDetail:
    """One trip with both systems' routes drawn, for the side-by-side maps."""
    pair = _by_trip().get(trip_id)
    if not pair:
        raise ComparisonUnavailableError(f"No evaluated trip '{trip_id}'.")

    paths: Dict[str, List[PathPoint]] = {s: [] for s in SYSTEMS}
    if rr.available():
        net = rr._network()
        known = set(net.node_coords)
        for system in SYSTEMS:
            nodes = [int(v) for v in str((pair.get(system) or {}).get("path", "")).split() if v]
            nodes = [n for n in nodes if n in known]
            if len(nodes) > 1:
                paths[system] = [PathPoint(**p) for p in rr._path_points(net, nodes)]

    return ComparisonTripDetail(
        trip=_trip_row(trip_id, pair),
        antroute_path=paths["antroute"],
        baseline_path=paths["baseline"],
    )
