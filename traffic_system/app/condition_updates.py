"""
Dynamic adaptability (research question 2), read from the condition-update harness.

scripts/evaluate_condition_updates.py plans each test trip under the conditions of one moment
and again under the conditions after an update, then measures:

    recalculation response time   seconds from the updated conditions being loaded to the new
                                  route -- the harness times it with perf_counter, so it is a
                                  measurement, not an estimate
    changes in the route          route_changed, and shared_length_fraction: the share of the
                                  old route's length the recalculated route still uses
    quality before and after      congestion and event exposure of the old route under the NEW
                                  conditions against the recalculated route under the same
                                  conditions -- so both are judged by what the driver meets

Three update kinds were run: "traffic" (the next recorded window), "session" (another recorded
session) and "event" (an MMDA incident injected at a gazetteer intersection). Each has its own
outputs/condition_updates/<kind>/ directory.

Nothing here recomputes a route. The harness already did that offline, with timings taken on
one machine in one run; re-running them inside a web request would measure the server's load,
not the method. Travel time is the one thing computed here, from the stored paths, because the
harness does not record it -- and it carries DURATION_NOTE like every other travel time in the
app: estimated, never scored.
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
    ConditionUpdateCase,
    ConditionUpdateDetail,
    ConditionUpdateSpread,
    ConditionUpdateSummary,
    ConditionUpdatesResponse,
    PathPoint,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "condition_updates"
# The gazetteer the harness injects incidents at, so an event can be put on the map.
INTERSECTIONS_CSV = REPO_ROOT / "configs" / "event_intersections.csv"

UPDATE_KINDS = ("traffic", "event", "session")

UPDATE_LABELS = {
    "traffic": "Traffic update (next recorded window)",
    "session": "Session update (another recorded session)",
    "event": "Road event (injected incident)",
}


class ConditionUpdatesUnavailableError(Exception):
    """The harness has not been run for this update kind; the message says what to run."""


def _f(value) -> Optional[float]:
    """A finite float, or None -- the CSV writes '' and 'nan' for a missing measurement."""
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _nodes(text: str) -> List[int]:
    return [int(v) for v in str(text).split()] if text else []


def _spread(block: Optional[dict]) -> Optional[ConditionUpdateSpread]:
    if not block:
        return None
    return ConditionUpdateSpread(
        n=int(block.get("n") or 0),
        median=_f(block.get("median")),
        p95=_f(block.get("p95")),
        mean=_f(block.get("mean")),
        max=_f(block.get("max")),
    )


@lru_cache(maxsize=1)
def _intersection_coords() -> Dict[str, tuple]:
    """intersection name -> (lat, lon), from the same gazetteer the harness injects at."""
    if not INTERSECTIONS_CSV.exists():
        return {}
    out: Dict[str, tuple] = {}
    with INTERSECTIONS_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            lat, lon = _f(row.get("lat")), _f(row.get("lon"))
            name = (row.get("intersection") or "").strip()
            if name and lat is not None and lon is not None:
                out.setdefault(name, (lat, lon))
    return out


@lru_cache(maxsize=len(UPDATE_KINDS))
def _load(kind: str) -> tuple:
    """(summary dict, per-trip rows) for one update kind, read once."""
    directory = OUTPUT_ROOT / kind
    summary_path, trips_path = directory / "summary.json", directory / "per_trip.csv"
    if not summary_path.exists() or not trips_path.exists():
        raise ConditionUpdatesUnavailableError(
            f"The condition-update harness hasn't been run for '{kind}'. Run "
            f"scripts/evaluate_condition_updates.py --update {kind} "
            f"--out-dir outputs/condition_updates/{kind}."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with trips_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return summary, rows


def _case(row: dict) -> ConditionUpdateCase:
    """One trip's update, without geometry."""
    old_congestion = _f(row.get("old_congestion_exposure"))
    new_congestion = _f(row.get("recalculated_congestion_exposure"))
    old_event = _f(row.get("old_event_exposure"))
    new_event = _f(row.get("recalculated_event_exposure"))
    return ConditionUpdateCase(
        trip_id=row.get("trip_id") or "",
        system=row.get("system") or "",
        window_before=row.get("window_before") or "",
        window_after=row.get("window_after") or "",
        injected_intersection=row.get("injected_intersection") or None,
        incidents_in_force_after=int(_f(row.get("incidents_in_force_after")) or 0),
        recalculation_seconds=_f(row.get("recalculation_seconds")),
        route_changed=str(row.get("route_changed")).lower() == "true",
        shared_length_fraction=_f(row.get("shared_length_fraction")),
        old_congestion_exposure=old_congestion,
        recalculated_congestion_exposure=new_congestion,
        congestion_exposure_improvement=_f(row.get("congestion_exposure_improvement")),
        old_event_exposure=old_event,
        recalculated_event_exposure=new_event,
        event_exposure_improvement=_f(row.get("event_exposure_improvement")),
        old_distance_km=round(_f(row.get("old_distance_m")) / 1000, 2)
        if _f(row.get("old_distance_m")) is not None
        else None,
        recalculated_distance_km=round(_f(row.get("recalculated_distance_m")) / 1000, 2)
        if _f(row.get("recalculated_distance_m")) is not None
        else None,
    )


def _summary(kind: str, summary: dict, system: str) -> ConditionUpdateSummary:
    block = (summary.get("by_system") or {}).get(system) or {}
    changes = block.get("route_changes") or {}
    return ConditionUpdateSummary(
        update=kind,
        update_label=UPDATE_LABELS.get(kind, kind),
        system=system,
        n_trips=int(block.get("n_trips") or 0),
        response_time_s=_spread(block.get("response_time_s")),
        n_route_changed=int(changes.get("n_changed") or 0),
        share_route_changed=_f(changes.get("share_changed")),
        shared_length_fraction=_spread(changes.get("shared_length_fraction_all")),
        response_time_definition=summary.get("response_time_definition") or "",
        shared_length_definition=summary.get("shared_length_fraction_definition") or "",
        quality_definition=str(summary.get("quality_definition") or ""),
        sampling=str(summary.get("sampling") or ""),
        machine=str(summary.get("machine") or ""),
    )


def available_kinds() -> List[str]:
    """The update kinds this server has results for."""
    return [k for k in UPDATE_KINDS if (OUTPUT_ROOT / k / "summary.json").exists()]


def get_condition_updates(kind: str, system: str = "antroute") -> ConditionUpdatesResponse:
    """The summary and every case for one update kind, without route geometry."""
    if kind not in UPDATE_KINDS:
        raise ConditionUpdatesUnavailableError(
            f"Unknown update kind '{kind}'. Choose one of: {', '.join(UPDATE_KINDS)}."
        )
    summary, rows = _load(kind)
    cases = [_case(r) for r in rows if (r.get("system") or "") == system]
    return ConditionUpdatesResponse(
        available_updates=available_kinds(),
        summary=_summary(kind, summary, system),
        cases=cases,
    )


def get_condition_update_detail(kind: str, trip_id: str, system: str = "antroute") -> ConditionUpdateDetail:
    """
    One case with both routes drawn. Travel time is computed here from the stored paths with
    the app's own estimator -- the harness does not record it, and it is not a scored metric.
    """
    summary, rows = _load(kind)
    row = next(
        (r for r in rows if (r.get("trip_id") or "") == trip_id and (r.get("system") or "") == system),
        None,
    )
    if row is None:
        raise ConditionUpdatesUnavailableError(
            f"No {system} case '{trip_id}' in the '{kind}' condition-update results."
        )

    old_nodes, new_nodes = _nodes(row.get("old_route", "")), _nodes(row.get("recalculated_route", ""))
    old_path: List[PathPoint] = []
    new_path: List[PathPoint] = []
    old_minutes = new_minutes = None

    # Geometry and travel time need the routing graph; without it the case still reports its
    # measured numbers rather than failing outright.
    if rr.available() and old_nodes:
        net = rr._network()
        known = set(net.node_coords)
        old_nodes = [n for n in old_nodes if n in known]
        new_nodes = [n for n in new_nodes if n in known]
        old_path = [PathPoint(**p) for p in rr._path_points(net, old_nodes)]
        new_path = [PathPoint(**p) for p in rr._path_points(net, new_nodes)]
        try:
            wg = rr._weighted_graph(row.get("window_after") or None)
            if len(old_nodes) > 1:
                old_minutes = rr._duration_min(net, wg, wg.evaluate_path(old_nodes))
            if len(new_nodes) > 1:
                new_minutes = rr._duration_min(net, wg, wg.evaluate_path(new_nodes))
        except Exception:
            # A travel time is a nicety here; never fail the panel over one.
            old_minutes = new_minutes = None

    return ConditionUpdateDetail(
        case=_case(row),
        old_path=old_path,
        recalculated_path=new_path,
        old_duration_min=old_minutes,
        recalculated_duration_min=new_minutes,
        incidents=_incidents_after(row),
    )


def _incidents_after(row: dict) -> List[Dict[str, object]]:
    """
    The event this update introduced, for the Event Details table. The harness records the
    intersection it was injected at and how many incidents were in force afterwards; the
    recorded incident feed itself is per window, not per case.
    """
    intersection = row.get("injected_intersection")
    if not intersection:
        return []
    coords = _intersection_coords().get(str(intersection).strip())
    return [
        {
            "kind": "Accident / incident",
            "location": str(intersection),
            "lat": coords[0] if coords else None,
            "lon": coords[1] if coords else None,
            "reported_at": row.get("window_after") or row.get("window_before") or None,
            "in_force_after": int(_f(row.get("incidents_in_force_after")) or 0),
        }
    ]
