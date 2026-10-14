"""
Condition-update harness (src/routing/condition_update.py) on the whole Metro Manila road
graph, routed by the app's own code: plan a trip under conditions A, apply an update,
recalculate, compare.

    python scripts/evaluate_condition_updates.py --update traffic --n-trips 30 --out-dir outputs/condition_updates/traffic
    python scripts/evaluate_condition_updates.py --update event   --n-trips 30 --out-dir outputs/condition_updates/event

Updates (--update):

    traffic         a different recorded window: t1 = the first window of the same session
                    starting >= --update-minutes after t0 (predicted risk and the recorded
                    incidents in force at t1)
    session         a different recorded session: t1 = the same clock time in the nearest
                    other recorded day's session (AM or PM, as t0; the next day, else the previous)
    event           a road event: an MMDA incident injected at a chosen intersection
                    (--intersection, else one drawn per trip from the event gazetteer), with
                    --lanes lanes occupied, at --event-age-minutes after it was reported; it is
                    active for its decay time (the Event-Aware layer's 60 minutes). Risk stays
                    at t0's, so the incident is the only change. The trip's origin and
                    destination are drawn on opposite sides of the incident, and kept only when
                    the old route runs into it.
    recorded-event  a real incident reported between t0 and t1 (was --require-events)

For each trip and each system (ANTROUTE, and the baseline, Improved ACO (Cheng 2023)):

    old route            planned under conditions A
    recalculated route   planned again under the updated conditions B

and three measures:

    Route Recalculation Response Time   seconds per recalculation: from conditions B being
        loaded (the window's risk and the baseline's observations are cached beforehand) to
        the new route -- applying the incidents plus the search. Median and 95th percentile
        per system, on the whole-city graph, both systems on the same machine in one run.
    Changes in Recommended Routes       share of trips whose route changed, and the
        shared-length fraction: the share of the old route's length the new route still uses.
    Route Quality Before and After      the old route under conditions B vs the recalculated
        route, on congestion exposure and event exposure (and avoidable event exposure);
        improvement = old - recalculated, positive = recalculating helped. Route length is
        descriptive. Both systems' routes are measured with the same yardstick: ANTROUTE's
        risk tiers and the Event-Aware layer at B.

The baseline plans on Cheng's cost from camera observations; it has no incident input, so
under an injected event its recalculated route is its old route -- which is the point of the
comparison, not a defect of the harness.

--warm-start also recalculates ANTROUTE starting from the old colony's pheromone
(aco_routing.warm_pheromone) and reports convergence: the iteration at which the colony
first reached its final best cost, cold vs warm, and the time and cost of each.

Outputs: per_trip.csv (one row per trip and system) and summary.json (with the model,
event-layer, update and machine settings). Refuses to run with EVENT_MU = 0: that is the
ablation, not the method.
"""

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app import risk_routing as rr  # noqa: E402
from src.data.alignment import WINDOW_STEPS  # noqa: E402
from src.routing.aco_routing import (  # noqa: E402
    AntColonyConfig,
    ant_colony_shortest_path,
    converged_at,
    warm_pheromone,
)
from src.routing.baseline_router import BaselineNoRouteError, plan_baseline  # noqa: E402
from src.routing.condition_update import METRICS, compare_update  # noqa: E402
from src.routing.event_layer import (  # noqa: E402
    EVENT_TTL_MINUTES,
    TrafficEvent,
    active_at,
    apply_events,
    describe_event_layer,
    event_exposure,
    event_impact,
)
from src.routing.route_trials import pick_waypoints  # noqa: E402
from src.routing.significance import compare_paired  # noqa: E402

ANTROUTE, BASELINE = "antroute", "baseline"
UPDATES = ("traffic", "session", "event", "recorded-event")

# Straight-line leg length, as scripts/citywide_trips.py: shorter is a few blocks where
# nothing can change, longer dominates the sample.
MIN_LEG_KM, MAX_LEG_KM = 2.0, 12.0


def haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    (lat1, lon1), (lat2, lon2) = a, b
    p1, p2 = math.radians(lat1), math.radians(lat2)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def _split_of(net, w: str) -> str:
    rows = net.risk_by_window[w]
    return str(rows["split"].iloc[0]) if "split" in rows.columns else "all"


def _in_split(net, split: str) -> List[str]:
    return [w for w in sorted(net.risk_by_window) if split == "all" or _split_of(net, w) == split]


def update_pairs(net, split: str, minutes: int) -> List[Tuple[str, str]]:
    """(t0, t1) per window t0 in `split`: t1 the first window of the same day and session
    starting >= `minutes` later. Windows past the end of their session have no update."""
    windows = sorted(net.risk_by_window)
    starts = {w: datetime.fromisoformat(w) for w in windows}
    pairs = []
    for w in _in_split(net, split):
        later = [v for v in windows
                 if starts[v].date() == starts[w].date() and (starts[v].hour < 12) == (starts[w].hour < 12)
                 and starts[v] - starts[w] >= timedelta(minutes=minutes)]
        if later:
            pairs.append((w, min(later, key=lambda v: starts[v])))
    return pairs


def session_pairs(net, split: str) -> List[Tuple[str, str]]:
    """(t0, t1) per window t0 in `split`: t1 the window at the closest clock time in the nearest
    other recorded day's same session (AM or PM; the next day when there is one, else the
    previous) -- the same moment of the day, different traffic."""
    windows = sorted(net.risk_by_window)
    starts = {w: datetime.fromisoformat(w) for w in windows}
    clock = lambda t: t.hour * 60 + t.minute  # noqa: E731
    pairs = []
    for w in _in_split(net, split):
        t0 = starts[w]
        other = [v for v in windows if starts[v].date() != t0.date() and (starts[v].hour < 12) == (t0.hour < 12)]
        if not other:
            continue
        days = {starts[v].date() for v in other}
        later = [d for d in days if d > t0.date()]
        day = min(later) if later else max(days)
        same_day = [v for v in other if starts[v].date() == day]
        pairs.append((w, min(same_day, key=lambda v: abs(clock(starts[v]) - clock(t0)))))
    return pairs


def draw_trip(rng, node_ids: np.ndarray, coords) -> Tuple[int, int]:
    while True:
        o = int(rng.choice(node_ids))
        for _ in range(200):
            d = int(rng.choice(node_ids))
            if d != o and MIN_LEG_KM <= haversine_km(coords[o], coords[d]) <= MAX_LEG_KM:
                return o, d


def incident_nodes(net, window: str) -> List[int]:
    """Node ids at the intersections carrying an incident penalty at this window."""
    impact = rr._window_impact(window)
    if impact is None:
        return []
    src, dst = net.graph.edge_index[0].numpy(), net.graph.edge_index[1].numpy()
    hit = impact > 0
    return sorted({int(net.graph.node_ids[i]) for i in np.concatenate([src[hit], dst[hit]])})


def draw_trip_across(rng, net, centre: int) -> Optional[Tuple[int, int]]:
    """
    An origin and destination on opposite sides of `centre` (an incident), each 1.5-5 km from
    it, so the trip's natural route runs past the incident. Random city-wide trips almost never
    do, and an exposure that is 0 on both routes says nothing about road-event conditions.
    """
    lat0, lon0 = net.node_coords[centre]
    dlat = np.radians(net.coord_lat - lat0)
    dlon = np.radians(net.coord_lon - lon0) * math.cos(math.radians(lat0))
    km = 6371.0 * np.hypot(dlat, dlon)
    bearing = np.arctan2(dlon, dlat)
    ring = np.flatnonzero((km >= 1.5) & (km <= 5.0))
    if ring.size < 2:
        return None
    for _ in range(200):
        i, j = rng.choice(ring, size=2, replace=False)
        gap = abs((bearing[i] - bearing[j] + np.pi) % (2 * np.pi) - np.pi)
        if gap >= np.radians(120):
            return int(net.coord_ids[i]), int(net.coord_ids[j])
    return None


def injected_conditions(net, window: str, label: str, lanes: int, age_minutes: float):
    """
    Conditions B for a road event: the window's own incidents plus one MMDA incident injected
    at intersection `label`, reported `age_minutes` before the window's moment and decaying
    linearly over the Event-Aware layer's EVENT_TTL_MINUTES. Returns (impact [E], injected
    impact [E] alone).
    """
    when = rr._as_of(window)
    event = TrafficEvent(at=when - timedelta(minutes=age_minutes), kind="crash",
                         location_text="injected", landmark="injected", intersection=label,
                         lanes_occupied=lanes, direction=None)
    injected = active_at([event], when)
    if not injected:
        raise SystemExit(f"--event-age-minutes {age_minutes} is past the incident's {EVENT_TTL_MINUTES}-minute decay")
    alone = event_impact(net.graph, injected, net.event_intersections)
    return event_impact(net.graph, rr.live_events(net, window) + injected, net.event_intersections), alone


class Conditions:
    """One moment's conditions: the risk graph ANTROUTE is measured on, the incident impact,
    and what the baseline plans with."""

    def __init__(self, window: str, impact: Optional[np.ndarray]):
        self.window = window
        self.risk_graph = rr._weighted_graph(window)       # cached per window: loading B, not timed
        self.impact = impact
        self.baseline_inputs = rr._baseline_inputs(window)  # likewise

    def antroute_graph(self):
        """ANTROUTE's cost at these conditions -- the timed part of loading B is this."""
        if self.impact is None or rr.settings.event_mu <= 0:
            return self.risk_graph
        return apply_events(self.risk_graph, self.impact, mu=rr.settings.event_mu)


def plan_antroute(cond: Conditions, origin: int, destination: int, config: AntColonyConfig, pheromone=None):
    """(result or None, seconds): applying the incidents plus the colony search."""
    start = time.perf_counter()
    try:
        result = ant_colony_shortest_path(cond.antroute_graph(), origin, destination, config, pheromone=pheromone)
    except (ValueError, KeyError):
        result = None
    return result, time.perf_counter() - start


def plan_iaco(cond: Conditions, origin: int, destination: int):
    """(baseline route or None, seconds), with the app's baseline settings."""
    start = time.perf_counter()
    try:
        route = plan_baseline(cond.baseline_inputs, [origin, destination], rr.BASELINE_CONFIG,
                              fallback=rr.settings.baseline_fallback, corridor_margin=rr.BASELINE_CORRIDOR_MARGIN)
    except (BaselineNoRouteError, ValueError, KeyError):
        route = None
    return route, time.perf_counter() - start


def summarise(rows: List[dict]) -> Dict[str, dict]:
    out = {}
    for metric in METRICS:
        old = [r[f"old_{metric}"] for r in rows]
        new = [r[f"recalculated_{metric}"] for r in rows]
        out[metric] = asdict(compare_paired(metric, new, old))   # proposed = recalculated
    lengths = np.array([r["distance_m_change"] for r in rows], dtype=float)
    out["distance_m_change"] = {"mean": float(lengths.mean()) if lengths.size else float("nan")}
    timed = [r for r in rows if r.get("old_route_optimality") not in (None, "")]
    if timed:
        out["route_optimality"] = asdict(compare_paired(
            "route_optimality",
            [r["recalculated_route_optimality"] for r in timed], [r["old_route_optimality"] for r in timed],
        ))
    return out


def spread(values) -> dict:
    x = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if not x.size:
        return {"n": 0}
    return {"n": int(x.size), "median": float(np.median(x)), "p95": float(np.percentile(x, 95)),
            "mean": float(x.mean()), "max": float(x.max())}


def system_summary(rows: List[dict]) -> dict:
    changed = [r for r in rows if r["route_changed"]]
    out = {
        "n_trips": len(rows),
        "response_time_s": spread(r["recalculation_seconds"] for r in rows),
        "route_changes": {
            "n_changed": len(changed),
            "share_changed": len(changed) / len(rows) if rows else float("nan"),
            "shared_length_fraction_all": spread(r["shared_length_fraction"] for r in rows),
            "shared_length_fraction_changed": spread(r["shared_length_fraction"] for r in changed),
        },
        "route_quality_all_trips": summarise(rows),
        "route_quality_changed_trips": summarise(changed) if changed else None,
    }
    warm = [r for r in rows if r.get("warm_converged_at") not in (None, "")]
    if warm:
        out["warm_start"] = {
            "n": len(warm),
            "converged_at_cold": spread(r["cold_converged_at"] for r in warm),
            "converged_at_warm": spread(r["warm_converged_at"] for r in warm),
            "seconds_cold": spread(r["recalculation_seconds"] for r in warm),
            "seconds_warm": spread(r["warm_seconds"] for r in warm),
            "same_route_as_cold": sum(bool(r["warm_same_route"]) for r in warm) / len(warm),
            "cost_ratio_warm_to_cold": spread(r["warm_cost"] / r["cold_cost"] for r in warm),
        }
    return out


def paired_response_time(rows: List[dict]) -> Optional[dict]:
    """ANTROUTE vs the baseline on recalculation seconds, paired by trip (lower is better)."""
    by = {(r["trip_id"], r["system"]): r for r in rows}
    trips = sorted({t for t, s in by if (t, ANTROUTE) in by and (t, BASELINE) in by})
    if not trips:
        return None
    return asdict(compare_paired("recalculation_seconds",
                                 [by[(t, ANTROUTE)]["recalculation_seconds"] for t in trips],
                                 [by[(t, BASELINE)]["recalculation_seconds"] for t in trips]))


def google_times(t1: str, old: List[int], new: List[int], key: str) -> Optional[dict]:
    """
    Google Maps times at t1's departure slot: Google's own route, the old route and the
    recalculated one (each forced through six of its own points as via waypoints), and Route
    Optimality for both. None when Google returns no route for any of them.
    """
    import google_routes_lookup as g

    net = rr._network()
    wg = rr._weighted_graph(t1)
    slot = datetime.fromisoformat(t1).strftime("%a %H:%M")
    today = datetime.now(g.LOCAL_TZ).date()
    depart = g.departure(slot, today)
    latlon = lambda n: "{:.6f},{:.6f}".format(*net.node_coords[n])  # noqa: E731

    def seconds(stops: List[int]) -> Optional[float]:
        row = {"start": latlon(old[0]), "end": latlon(old[-1]), "stops": " | ".join(latlon(v) for v in stops)}
        routes = g.compute_route(g.request_body(row, depart), key).get("routes") or []
        return g.seconds(routes[0]["duration"]) if routes and "duration" in routes[0] else None

    own = seconds([])
    old_s = seconds(pick_waypoints(wg, old, 6))
    new_s = old_s if new == old else seconds(pick_waypoints(wg, new, 6))
    if None in (own, old_s, new_s):
        return None
    return {
        "google_own_s": own, "google_old_s": old_s, "google_recalculated_s": new_s,
        "old_route_optimality": own / old_s * 100, "recalculated_route_optimality": own / new_s * 100,
        "google_departure_slot": slot, "google_departure_date": depart.date().isoformat(),
        "google_queried_on": today.isoformat(),
    }


def _summaries(rows: List[dict]) -> dict:
    systems = sorted({r["system"] for r in rows})
    return {s: system_summary([r for r in rows if r["system"] == s]) for s in systems}


def rescore(out_dir: Path) -> None:
    """Recompute every exposure of a finished run from its saved routes, keeping its times."""
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    table = pd.read_csv(out_dir / "per_trip.csv", keep_default_na=False)
    if "system" not in table.columns:              # a run from before the baseline was added
        table["system"] = ANTROUTE
        table["recalculation_seconds"] = float("nan")
    net = rr._network()
    update = summary.get("update", "traffic")
    rows = []
    for r in table.to_dict("records"):
        old = [int(v) for v in str(r["old_route"]).split()]
        new = [int(v) for v in str(r["recalculated_route"]).split()]
        t1 = str(r["window_after"])
        impact = rr._window_impact(t1)
        if update == "event":
            impact, _ = injected_conditions(net, t1, str(r["injected_intersection"]),
                                            int(summary["lanes"]), float(summary["event_age_minutes"]))
        r.update(compare_update(rr._weighted_graph(t1), impact, old, new))
        for k in ("old_route_optimality", "recalculated_route_optimality"):
            if k in r and r[k] == "":
                r[k] = None
        rows.append(r)
    summary["by_system"] = _summaries(rows)
    summary["response_time_antroute_vs_baseline"] = paired_response_time(rows)
    pd.DataFrame(rows).to_csv(out_dir / "per_trip.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"rescored {len(rows)} rows in {out_dir}")
    report(summary)


def report(summary: dict) -> None:
    for system, s in summary["by_system"].items():
        t, c = s["response_time_s"], s["route_changes"]
        print(f"\n{system}: {s['n_trips']} trips")
        if t.get("n"):
            print(f"  response time       median {t['median']:.2f} s   p95 {t['p95']:.2f} s")
        print(f"  routes changed      {c['n_changed']} of {s['n_trips']} ({c['share_changed']:.0%}); "
              f"shared length median {c['shared_length_fraction_all'].get('median', float('nan')):.2f}")
        for metric in METRICS:
            q = s["route_quality_all_trips"][metric]
            print(f"  {metric:26s} old {q['baseline_mean']:.3f}  recalculated {q['proposed_mean']:.3f}  "
                  f"Wilcoxon p {q['wilcoxon_p']}")
        if s.get("warm_start"):
            w = s["warm_start"]
            print(f"  warm start          converged at iteration {w['converged_at_cold']['median']:.0f} cold vs "
                  f"{w['converged_at_warm']['median']:.0f} warm (median); same route {w['same_route_as_cold']:.0%}")


def model_record() -> dict:
    path = rr.risk_edges_path().with_name("risk_summary.json")
    if not path.exists():
        return {"checkpoint": None, "use_text": None, "note": f"{path} not found"}
    info = json.loads(path.read_text(encoding="utf-8"))
    return {k: info.get(k) for k in ("checkpoint", "use_text", "context", "context_drop", "flood_hazard")}


def machine_record() -> dict:
    import os
    return {"platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "python": platform.python_version()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--update", choices=UPDATES, default="traffic")
    p.add_argument("--require-events", action="store_true", help="same as --update recorded-event")
    p.add_argument("--n-trips", type=int, default=30)
    p.add_argument("--split", default="test", help='risk-file split to draw t0 from, or "all"')
    p.add_argument("--update-minutes", type=int, default=WINDOW_STEPS,
                   help="--update traffic: how long after t0 the conditions are updated (default: one window)")
    p.add_argument("--intersection", default=None,
                   help="--update event: gazetteer intersection label to inject the incident at "
                        "(default: one drawn per trip)")
    p.add_argument("--lanes", type=int, default=2, help="--update event: lanes the incident occupies")
    p.add_argument("--event-age-minutes", type=float, default=0.0,
                   help="--update event: minutes since the incident was reported when the route is recalculated")
    p.add_argument("--systems", default=f"{ANTROUTE},{BASELINE}")
    p.add_argument("--warm-start", action="store_true",
                   help="also recalculate ANTROUTE from the old colony's pheromone, and report convergence")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--google", action="store_true",
                   help="also time ANTROUTE's old and recalculated routes, and Google's own, in Google Maps at "
                        "t1's departure slot (Routes API, GOOGLE_ROUTES_API_KEY)")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs/condition_updates")
    p.add_argument("--rescore", action="store_true",
                   help="recompute the metrics of an earlier run in --out-dir from its saved routes; "
                        "nothing is re-routed and no Google request is sent")
    a = p.parse_args()
    if a.require_events:
        a.update = "recorded-event"
    if a.rescore:
        rescore(a.out_dir)
        return
    systems = [s.strip() for s in a.systems.split(",") if s.strip()]
    if not set(systems) <= {ANTROUTE, BASELINE} or not systems:
        raise SystemExit(f"--systems takes {ANTROUTE} and/or {BASELINE}")
    key = None
    if a.google:
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import google_routes_lookup

        key = google_routes_lookup.api_key()

    if not rr.settings.event_mu > 0:
        raise SystemExit("EVENT_MU is 0: the Event-Aware layer is off, which is the ablation, not the method.")
    net = rr._network()
    if not net.risk_by_window:
        raise SystemExit(f"{rr.risk_edges_path()} not found: there are no conditions to update")

    if a.update == "traffic":
        pairs = update_pairs(net, a.split, a.update_minutes)
    elif a.update == "session":
        pairs = session_pairs(net, a.split)
    elif a.update == "event":
        pairs = [(w, w) for w in _in_split(net, a.split)]
    else:
        # A real incident that was NOT in force at t0: one already in force then was avoided by
        # the old route as well, so it is not an update.
        fresh = {}
        for t0, t1 in update_pairs(net, a.split, a.update_minutes):
            new = sorted(set(incident_nodes(net, t1)) - set(incident_nodes(net, t0)))
            if new:
                fresh[(t0, t1)] = new
        pairs = sorted(fresh)
    if not pairs:
        raise SystemExit(f"no (t0, t1) update pairs for --update {a.update} in split {a.split}")
    labels = sorted(net.event_intersections)
    if a.intersection is not None and a.intersection not in net.event_intersections:
        raise SystemExit(f"--intersection {a.intersection!r} is not in the event gazetteer")
    print(f"{len(pairs)} update pairs available (--update {a.update})")

    rng = np.random.default_rng(a.seed)
    node_ids = np.array(sorted(net.node_coords))
    config = AntColonyConfig(n_ants=8, n_iterations=15, seed=a.seed)   # the app's colony
    rows: List[dict] = []
    n_trips, attempts = 0, 0
    budget = a.n_trips * (60 if a.update in ("event", "recorded-event") else 5)
    while n_trips < a.n_trips and attempts < budget:
        attempts += 1
        t0, t1 = pairs[int(rng.integers(len(pairs)))]
        label = None
        if a.update == "event":
            label = a.intersection or labels[int(rng.integers(len(labels)))]
            centre = int(net.graph.node_ids[net.event_intersections[label]])
            trip = draw_trip_across(rng, net, centre)
        elif a.update == "recorded-event":
            trip = draw_trip_across(rng, net, int(rng.choice(fresh[(t0, t1)])))
        else:
            trip = draw_trip(rng, node_ids, net.node_coords)
        if trip is None:
            continue
        origin, destination = trip

        before = Conditions(t0, rr._window_impact(t0))
        if a.update == "event":
            impact_after, injected = injected_conditions(net, t1, label, a.lanes, a.event_age_minutes)
        else:
            impact_after, injected = rr._window_impact(t1), None
        after = Conditions(t1, impact_after)

        old_ant, _ = plan_antroute(before, origin, destination, config)
        if old_ant is None:
            continue
        old_nodes = old_ant.best.nodes
        # A road-event trip shows something only when the incident lands on the old route.
        if a.update == "event" and event_exposure(after.risk_graph, injected, old_nodes) <= 0:
            continue
        if a.update == "recorded-event" and event_exposure(after.risk_graph, impact_after, old_nodes) <= 0:
            continue

        trip_rows = []
        common = {"origin": origin, "destination": destination, "window_before": t0, "window_after": t1,
                  "update": a.update, "injected_intersection": label,
                  "incidents_in_force_after": len(rr.live_events(net, t1)) + (1 if label else 0)}
        if ANTROUTE in systems:
            new_ant, seconds = plan_antroute(after, origin, destination, config)
            if new_ant is None:
                continue
            row = {"system": ANTROUTE, **common, "recalculation_seconds": seconds,
                   **compare_update(after.risk_graph, impact_after, old_nodes, new_ant.best.nodes),
                   "cold_converged_at": converged_at(new_ant.best_cost_by_iteration),
                   "cold_cost": new_ant.best.dynamic_cost,
                   "old_route": " ".join(map(str, old_nodes)),
                   "recalculated_route": " ".join(map(str, new_ant.best.nodes))}
            if a.warm_start:
                warm, warm_s = plan_antroute(after, origin, destination, config,
                                             pheromone=warm_pheromone(old_ant.pheromone, config.initial_pheromone))
                if warm is not None:
                    row.update({"warm_seconds": warm_s, "warm_converged_at": converged_at(warm.best_cost_by_iteration),
                                "warm_cost": warm.best.dynamic_cost,
                                "warm_same_route": warm.best.nodes == new_ant.best.nodes})
            if key:
                try:
                    times = google_times(t1, old_nodes, new_ant.best.nodes, key)
                except RuntimeError as err:
                    if str(err).startswith(("HTTP 400", "HTTP 401", "HTTP 403")):
                        raise SystemExit(f"Google Routes API: {err}") from None
                    times = None
                row.update(times or {"google_note": "Google returned no route for one of the three lookups"})
            trip_rows.append(row)
        if BASELINE in systems:
            old_base, _ = plan_iaco(before, origin, destination)
            new_base, seconds = plan_iaco(after, origin, destination)
            if old_base is not None and new_base is not None:
                trip_rows.append({
                    "system": BASELINE, **common, "recalculation_seconds": seconds,
                    "baseline_algorithm": new_base.algorithm,
                    **compare_update(after.risk_graph, impact_after, old_base.nodes, new_base.nodes),
                    "old_route": " ".join(map(str, old_base.nodes)),
                    "recalculated_route": " ".join(map(str, new_base.nodes)),
                })
        if not trip_rows:
            continue
        n_trips += 1
        trip_id = f"U{n_trips:02d}"
        for r in trip_rows:
            r["trip_id"] = trip_id
            rows.append(r)
            print(f"{trip_id} {r['system']:8s} {t0} -> {t1}{' @' + label if label else ''}  "
                  f"{r['recalculation_seconds']:.2f}s  changed={r['route_changed']}  "
                  f"shared {r['shared_length_fraction']:.2f}  "
                  f"congestion {r['old_congestion_exposure']:.2f} -> {r['recalculated_congestion_exposure']:.2f}  "
                  f"event {r['old_event_exposure']:.2f} -> {r['recalculated_event_exposure']:.2f}")
    if not rows:
        raise SystemExit("no trip could be routed")

    sampling = {
        "traffic": f"uniform road points {MIN_LEG_KM}-{MAX_LEG_KM} km apart; t1 = next window >= "
                   f"{a.update_minutes} min later, same session",
        "session": f"uniform road points {MIN_LEG_KM}-{MAX_LEG_KM} km apart; t1 = same clock time, nearest other "
                   "recorded day's same session",
        "event": "an MMDA incident injected at a gazetteer intersection; origin and destination 1.5-5 km on "
                 "opposite sides of it, kept only when the old route runs into it; risk unchanged",
        "recorded-event": "a real incident reported between t0 and t1; origin and destination 1.5-5 km on "
                          "opposite sides of it, kept only when the old route runs into it",
    }[a.update]
    summary = {
        "harness": "plan under conditions A, apply an update, recalculate, compare",
        "update": a.update,
        "sampling": sampling,
        "scope": "Metro Manila, whole-city graph",
        "split": a.split,
        "update_minutes": a.update_minutes if a.update in ("traffic", "recorded-event") else None,
        "intersection": a.intersection,
        "lanes": a.lanes,
        "event_age_minutes": a.event_age_minutes,
        "event_active_minutes": EVENT_TTL_MINUTES,
        "systems": systems,
        "antroute_colony": asdict(config),
        "baseline_colony": asdict(rr.BASELINE_CONFIG),
        "baseline_corridor_margin": rr.BASELINE_CORRIDOR_MARGIN,
        "response_time_definition": (
            "seconds from conditions B being loaded (window risk and baseline observations cached) to the new "
            "route: applying the incidents plus the search; both systems on the same machine in one run"
        ),
        "shared_length_fraction_definition": "share of the old route's length the recalculated route still uses",
        "quality_definition": (
            "congestion and event exposure of the old route under conditions B vs the recalculated route, both "
            "measured with ANTROUTE's risk tiers and Event-Aware layer at B; improvement = old - recalculated; "
            "route length descriptive"
        ),
        "warm_start": a.warm_start,
        "machine": machine_record(),
        "lambda": rr.DEFAULT_LAMBDA,
        "event_layer": describe_event_layer(rr.settings.event_mu),
        "model": model_record(),
        "by_system": _summaries(rows),
        "response_time_antroute_vs_baseline": paired_response_time(rows),
        "google_reference": {
            "access_method": "Google Routes API v2 computeRoutes (DRIVE, TRAFFIC_AWARE_OPTIMAL, BEST_GUESS; "
                             "six route points as via waypoints)",
            "departure": "t1's weekday and time, on the next such date",
        } if a.google else None,
    }
    a.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out_dir / "per_trip.csv", index=False)
    (a.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    report(summary)
    print(f"\nwrote per_trip.csv and summary.json to {a.out_dir}")


if __name__ == "__main__":
    main()
