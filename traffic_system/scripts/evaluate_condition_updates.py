"""
Route Quality Before and After Condition Updates (src/routing/condition_update.py), on the
whole Metro Manila road graph, routed by the app's own code.

    python scripts/evaluate_condition_updates.py --n-trips 30 --out-dir outputs/condition_updates

Per trip: an origin and destination anywhere in the city, a recorded window t0 from the risk
file's --split, and an update to t1 = the first window of the same session starting at least
--update-minutes later. ANTROUTE routes the trip under the conditions at t0 (the old route)
and again under the conditions at t1 (the recalculated route) -- each time with the predicted
risk of every road (observed / propagated / assumed tiers) and the Event-Aware layer, exactly
as the app does. Both routes are then measured under the conditions at t1:

    congestion exposure   sum of Risk_t1 over the route
    event exposure        sum of Event_t1 over the route
    route length          descriptive only

improvement = old - recalculated (positive: recalculating helped). Over trips, the two
exposures are compared pairwise (Wilcoxon signed-rank, Shapiro-Wilk, t-test if normal; Δ% by
Equation 7), overall and on the trips whose route actually changed.

--require-events keeps only updates that bring an incident reported between t0 and t1 (one
already in force at t0 was avoided by the old route too), draws each trip's origin and
destination on opposite sides of it, and keeps the trip only when the old route runs into it --
the only trips where the road-event half of the question can show anything. Run it as its own
sample and report it as such, never pooled with the uniform one.

Outputs: per_trip.csv and summary.json (with the model, event-layer and update settings).
Refuses to run with EVENT_MU = 0: that is the ablation, not the method.
"""

import argparse
import json
import math
import sys
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
from src.routing.aco_routing import AntColonyConfig, ant_colony_shortest_path  # noqa: E402
from src.routing.condition_update import METRICS, compare_update  # noqa: E402
from src.routing.event_layer import describe_event_layer, event_exposure  # noqa: E402
from src.routing.significance import compare_paired  # noqa: E402
from src.routing.route_trials import pick_waypoints  # noqa: E402

# Straight-line leg length, as scripts/citywide_trips.py: shorter is a few blocks where
# nothing can change, longer dominates the sample.
MIN_LEG_KM, MAX_LEG_KM = 2.0, 12.0


def haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    (lat1, lon1), (lat2, lon2) = a, b
    p1, p2 = math.radians(lat1), math.radians(lat2)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def update_pairs(net, split: str, minutes: int) -> List[Tuple[str, str]]:
    """(t0, t1) per window t0 in `split`: t1 the first window of the same day and session
    starting >= `minutes` later. Windows past the end of their session have no update."""
    def split_of(w):
        rows = net.risk_by_window[w]
        return str(rows["split"].iloc[0]) if "split" in rows.columns else "all"

    windows = sorted(net.risk_by_window)
    starts = {w: datetime.fromisoformat(w) for w in windows}
    pairs = []
    for w in windows:
        if split != "all" and split_of(w) != split:
            continue
        later = [v for v in windows
                 if starts[v].date() == starts[w].date() and (starts[v].hour < 12) == (starts[w].hour < 12)
                 and starts[v] - starts[w] >= timedelta(minutes=minutes)]
        if later:
            pairs.append((w, min(later, key=lambda v: starts[v])))
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


def route(window: str, origin: int, destination: int, config: AntColonyConfig) -> Optional[List[int]]:
    """ANTROUTE's best route under the conditions at `window`: risk tiers + Event-Aware layer."""
    wg, _, _ = rr.event_graph(window)
    try:
        return ant_colony_shortest_path(wg, origin, destination, config).best.nodes
    except (ValueError, KeyError):
        return None


def summarise(rows: List[dict]) -> Dict[str, dict]:
    out = {}
    for metric in METRICS:
        old = [r[f"old_{metric}"] for r in rows]
        new = [r[f"recalculated_{metric}"] for r in rows]
        out[metric] = asdict(compare_paired(metric, new, old))   # proposed = recalculated
    lengths = np.array([r["distance_m_change"] for r in rows], dtype=float)
    out["distance_m_change"] = {"mean": float(lengths.mean()) if lengths.size else float("nan")}
    timed = [r for r in rows if r.get("old_route_optimality") is not None]
    if timed:
        out["route_optimality"] = asdict(compare_paired(
            "route_optimality",
            [r["recalculated_route_optimality"] for r in timed], [r["old_route_optimality"] for r in timed],
        ))
    return out


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


def rescore(out_dir: Path) -> None:
    """Recompute every exposure of a finished run from its saved routes, keeping its Google times."""
    table = pd.read_csv(out_dir / "per_trip.csv", keep_default_na=False)
    rows = []
    for r in table.to_dict("records"):
        old = [int(v) for v in str(r["old_route"]).split()]
        new = [int(v) for v in str(r["recalculated_route"]).split()]
        t1 = str(r["window_after"])
        r.update(compare_update(rr._weighted_graph(t1), rr._window_impact(t1), old, new))
        for k in ("old_route_optimality", "recalculated_route_optimality"):
            if k in r and r[k] == "":
                r[k] = None
        rows.append(r)
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    changed = [r for r in rows if r["route_changed"]]
    summary["all_trips"] = summarise(rows)
    summary["route_changed_trips"] = summarise(changed) if changed else None
    pd.DataFrame(rows).to_csv(out_dir / "per_trip.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"rescored {len(rows)} trips in {out_dir}")
    for metric in METRICS:
        s = summary["all_trips"][metric]
        print(f"  {metric:26s} old {s['baseline_mean']:.3f}  recalculated {s['proposed_mean']:.3f}  "
              f"Wilcoxon p {s['wilcoxon_p']}")


def model_record() -> dict:
    path = rr.risk_edges_path().with_name("risk_summary.json")
    if not path.exists():
        return {"checkpoint": None, "use_text": None, "note": f"{path} not found"}
    info = json.loads(path.read_text(encoding="utf-8"))
    return {k: info.get(k) for k in ("checkpoint", "use_text", "context", "context_drop", "flood_hazard")}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-trips", type=int, default=30)
    p.add_argument("--split", default="test", help='risk-file split to draw t0 from, or "all"')
    p.add_argument("--update-minutes", type=int, default=WINDOW_STEPS,
                   help="how long after t0 the conditions are updated (default: one window length)")
    p.add_argument("--require-events", action="store_true",
                   help="only updates with an incident in force at t1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--google", action="store_true",
                   help="also time the old and recalculated routes, and Google's own, in Google Maps at "
                        "t1's departure slot (Routes API, GOOGLE_ROUTES_API_KEY), for Route Optimality "
                        "before and after the update")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs/condition_updates")
    p.add_argument("--rescore", action="store_true",
                   help="recompute the metrics of an earlier run in --out-dir from its saved routes "
                        "(per_trip.csv); nothing is re-routed and no Google request is sent")
    a = p.parse_args()
    if a.rescore:
        rescore(a.out_dir)
        return
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
    pairs = update_pairs(net, a.split, a.update_minutes)
    # With --require-events an update must bring an incident that was NOT in force at t0: one
    # already in force then was avoided by the old route as well, so it is not an update.
    fresh: Dict[Tuple[str, str], List[int]] = {}
    if a.require_events:
        for t0, t1 in pairs:
            new = sorted(set(incident_nodes(net, t1)) - set(incident_nodes(net, t0)))
            if new:
                fresh[(t0, t1)] = new
        pairs = sorted(fresh)
    if not pairs:
        raise SystemExit("no (t0, t1) update pairs in this split"
                         + (" with an incident reported between t0 and t1" if a.require_events else ""))
    print(f"{len(pairs)} update pairs available (t1 = t0 + >= {a.update_minutes} min, same session)")

    rng = np.random.default_rng(a.seed)
    node_ids = np.array(sorted(net.node_coords))
    config = AntColonyConfig(n_ants=8, n_iterations=15, seed=a.seed)   # the app's colony
    rows: List[dict] = []
    attempts, budget = 0, a.n_trips * (60 if a.require_events else 5)
    while len(rows) < a.n_trips and attempts < budget:
        attempts += 1
        t0, t1 = pairs[int(rng.integers(len(pairs)))]
        if a.require_events:
            trip = draw_trip_across(rng, net, int(rng.choice(fresh[(t0, t1)])))
            if trip is None:
                continue
            origin, destination = trip
        else:
            origin, destination = draw_trip(rng, node_ids, net.node_coords)
        old = route(t0, origin, destination, config)
        if old is None:
            continue
        wg_after, impact_after = rr._weighted_graph(t1), rr._window_impact(t1)
        # Only an old route that the new incident actually lands on can show what the update did.
        if a.require_events and event_exposure(wg_after, impact_after, old) <= 0:
            continue
        new = route(t1, origin, destination, config)
        if new is None:
            continue
        row = {
            "trip_id": f"U{len(rows) + 1:02d}",
            "origin": origin,
            "destination": destination,
            "window_before": t0,
            "window_after": t1,
            "incidents_in_force_after": len(rr.live_events(net, t1)),
            **compare_update(wg_after, impact_after, old, new),
            "old_route": " ".join(map(str, old)),
            "recalculated_route": " ".join(map(str, new)),
        }
        if key:
            try:
                times = google_times(t1, old, new, key)
            except RuntimeError as err:
                if str(err).startswith(("HTTP 400", "HTTP 401", "HTTP 403")):
                    raise SystemExit(f"Google Routes API: {err}") from None
                times = None
            row.update(times or {"google_note": "Google returned no route for one of the three lookups"})
        rows.append(row)
        print(f"{row['trip_id']} {t0} -> {t1}  changed={row['route_changed']}  "
              f"congestion {row['old_congestion_exposure']:.2f} -> {row['recalculated_congestion_exposure']:.2f}  "
              f"event {row['old_event_exposure']:.2f} -> {row['recalculated_event_exposure']:.2f}")
    if not rows:
        raise SystemExit("no trip could be routed")

    changed = [r for r in rows if r["route_changed"]]
    summary = {
        "metric": "Route Quality Before and After Condition Updates",
        "definition": (
            "congestion exposure and event exposure of the old route (planned under the conditions at t0) "
            "under the new conditions at t1, vs the route recalculated under t1; route length descriptive. "
            "improvement = old - recalculated"
        ),
        "scope": "Metro Manila, whole-city graph",
        "split": a.split,
        "update_minutes": a.update_minutes,
        "require_events": a.require_events,
        "sampling": ("an incident reported between t0 and t1; origin and destination 1.5-5 km on opposite "
                     "sides of it, kept only when the old route runs into it"
                     if a.require_events else f"uniform road points {MIN_LEG_KM}-{MAX_LEG_KM} km apart"),
        "n_trips": len(rows),
        "n_route_changed": len(changed),
        "lambda": rr.DEFAULT_LAMBDA,
        "event_layer": describe_event_layer(rr.settings.event_mu),
        "model": model_record(),
        "all_trips": summarise(rows),
        "route_changed_trips": summarise(changed) if len(changed) else None,
        "google_reference": {
            "access_method": "Google Routes API v2 computeRoutes (DRIVE, TRAFFIC_AWARE_OPTIMAL, BEST_GUESS; "
                             "six route points as via waypoints)",
            "departure": "t1's weekday and time, on the next such date",
            "n_timed": sum(1 for r in rows if r.get("old_route_optimality") is not None),
        } if a.google else None,
    }
    a.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out_dir / "per_trip.csv", index=False)
    (a.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"\n{len(changed)} of {len(rows)} routes changed after the update")
    for metric in METRICS:
        s = summary["all_trips"][metric]
        print(f"  {metric:20s} old {s['baseline_mean']:.3f}  recalculated {s['proposed_mean']:.3f}  "
              f"change {s['relative_difference_pct']:.1f}%  Wilcoxon p {s['wilcoxon_p']:.4f}")
    ro = summary["all_trips"].get("route_optimality")
    if ro:
        print(f"  route_optimality     old {ro['baseline_mean']:.1f}%  recalculated {ro['proposed_mean']:.1f}%  "
              f"(Google Maps, n={ro['n']})  Wilcoxon p {ro['wilcoxon_p']:.4f}")
    print(f"wrote per_trip.csv and summary.json to {a.out_dir}")


if __name__ == "__main__":
    main()
