"""
More Apple Maps test trips, this time anywhere in Metro Manila rather than only between the
seven EDSA camera intersections.

    python scripts/citywide_trips.py --out outputs/appendix/template_citywide.csv

The first batch (scripts/evaluate_routing.py --template) draws its trips from the camera
intersections and routes them on the camera subgraph. Trips the app plans start and end
anywhere, on the whole-city graph, so this batch is drawn from road points across the city
and routed by the app's own code (app.risk_routing.plan_real_routes): the routes the team
times in Apple Maps are exactly the routes the app shows for that trip and departure.

Per trip, both systems' recommended route and each system's own per-leg ETA are written in
evaluate_routing.py's template format, plus three columns it uses for these trips:

    graph                           "full": distance and risk are measured on the city graph
    antroute_predicted_eta_seconds  ANTROUTE's ETA engine, per leg (what the app shows)
    baseline_predicted_eta_seconds  IACO's own estimate, the observed t_ij it planned with

Departures reuse the first batch's slots (Monday, the same times), so whoever does the
lookups sets "Leave at" to times they already know. Trip ids continue the first batch's
numbering (R11..., M11...) so the two sheets can be scored together:

    python scripts/apple_maps_worklist.py --template outputs/appendix/template_citywide.csv \
        --out outputs/appendix/lookups_citywide.csv
    # fill eta_seconds, then --apply, then score both sheets with evaluate_routing.py --apple-maps
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app import risk_routing as rr  # noqa: E402
from src.routing.baseline_router import BaselineNoRouteError  # noqa: E402
from src.routing.eta_engine import path_eta_seconds  # noqa: E402
from src.routing.route_trials import MULTI_DESTINATION, RECOMMENDED, pick_waypoints  # noqa: E402

PREFIX = {RECOMMENDED: "R", MULTI_DESTINATION: "M"}
# A leg shorter than this is a few blocks, where both systems take the same street; longer
# than this and one trip dominates a slot's lookups. Straight-line kilometres.
MIN_LEG_KM, MAX_LEG_KM = 2.0, 12.0


def haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    (lat1, lon1), (lat2, lon2) = a, b
    p1, p2 = math.radians(lat1), math.radians(lat2)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def named_nodes(net) -> Tuple[np.ndarray, Dict[int, str]]:
    """Road points with a named road leaving them, and that name -- a place a person could
    plausibly start from, and a label to tell the lookups apart."""
    src = net.graph.edge_index[0].numpy()
    names: Dict[int, str] = {}
    for e, name in enumerate(net.edge_names if net.edge_names is not None else []):
        if name:
            names.setdefault(int(net.graph.node_ids[src[e]]), str(name))
    return np.array(sorted(names)), names


def split_legs(nodes: Sequence[int], stops: Sequence[int]) -> List[List[int]]:
    """A route through `stops` in order, cut into one node list per leg."""
    cuts, start = [0], 0
    for stop in stops[1:]:
        start = list(nodes).index(stop, start + 1) if stop in nodes[start + 1:] else -1
        if start < 0:
            raise ValueError(f"route does not pass through stop {stop}")
        cuts.append(start)
    return [list(nodes[a:b + 1]) for a, b in zip(cuts, cuts[1:])]


def baseline_eta(window: str, leg: Sequence[int]) -> float:
    """IACO's own estimate for a leg: the observed travel times it planned with."""
    inputs = rr._baseline_inputs(window)
    g = inputs.graph
    return float(inputs.travel_time[g.edges_of([g.index_of(v) for v in leg])].sum())


def draw_stops(rng, candidates, coords, n_stops: int) -> List[int]:
    """n_stops road points, each leg between MIN_LEG_KM and MAX_LEG_KM long."""
    while True:
        stops = [int(rng.choice(candidates))]
        for _ in range(n_stops - 1):
            here = coords[stops[-1]]
            for _ in range(200):
                nxt = int(rng.choice(candidates))
                if nxt not in stops and MIN_LEG_KM <= haversine_km(here, coords[nxt]) <= MAX_LEG_KM:
                    stops.append(nxt)
                    break
            else:
                break
        if len(stops) == n_stops:
            return stops


def route_trip(window: str, stops: List[int]) -> Optional[Dict[str, List[List[int]]]]:
    """Both systems' recommended route per leg, exactly as the app plans it; None if either has none."""
    points = [rr._network().node_coords[s] for s in stops]
    legs = {}
    for system in ("antroute", "baseline"):
        try:
            option = rr.plan_real_routes(points[0], points[1:], system, window)[0]
        except (rr.NoRouteError, rr.RouteOutsideNetworkError, BaselineNoRouteError) as err:
            print(f"  skipped ({system}: {err})")
            return None
        legs[system] = split_legs(option["nodes"], stops)
    return legs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--first-batch", type=Path, default=REPO_ROOT / "outputs/appendix/template.csv",
                   help="the first template: its departure slots are reused and its trip ids continued")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/appendix/template_citywide.csv")
    p.add_argument("--n-trials", type=int, default=10, help="trips per scenario")
    p.add_argument("--n-stops", type=int, default=3, help="stops per multi-destination trip")
    p.add_argument("--n-waypoints", type=int, default=6, help="Apple Maps stops per leg")
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()

    first = pd.read_csv(a.first_batch, dtype=str, keep_default_na=False)
    net = rr._network()
    windows = sorted(w for w in first["window"].unique() if w in net.risk_by_window)
    if not windows:
        raise SystemExit(f"none of {a.first_batch}'s windows are in risk_edges.csv")
    leave_at = dict(zip(first["window"], first["apple_leave_at"]))
    taken = {
        prefix: max((int(t[1:]) for t in first["trip_id"].unique() if t.startswith(prefix) and t[1:].isdigit()),
                    default=0)
        for prefix in PREFIX.values()
    }

    candidates, road_name = named_nodes(net)
    coords = net.node_coords
    label = lambda n: f"{road_name.get(n, 'Road')} ({coords[n][0]:.5f},{coords[n][1]:.5f})"  # noqa: E731
    latlon = lambda n: f"{coords[n][0]:.6f},{coords[n][1]:.6f}"  # noqa: E731
    rng = np.random.default_rng(a.seed)

    rows = []
    for scenario, n_stops in ((RECOMMENDED, 2), (MULTI_DESTINATION, a.n_stops)):
        number = taken[PREFIX[scenario]]
        made = 0
        while made < a.n_trials:
            window = str(rng.choice(windows))
            stops = draw_stops(rng, candidates, coords, n_stops)
            trip_id = f"{PREFIX[scenario]}{number + made + 1:02d}"
            print(f"{trip_id} {leave_at[window]}  {' -> '.join(label(s) for s in stops)}")
            legs = route_trip(window, stops)
            if legs is None:
                continue
            wg = rr._weighted_graph(window)
            for leg_no, (o, d) in enumerate(zip(stops, stops[1:]), start=1):
                ant, base = legs["antroute"][leg_no - 1], legs["baseline"][leg_no - 1]
                row = {
                    "trip_id": trip_id,
                    "scenario_type": scenario,
                    "leg": leg_no,
                    "window": window,
                    "apple_leave_at": leave_at[window],
                    "origin": label(o),
                    "destination": label(d),
                    "origin_latlon": latlon(o),
                    "destination_latlon": latlon(d),
                    "same_route": ant == base,
                    "apple_eta_seconds": "",
                    "baseline_algorithm": "iaco",
                    "events_live": 0,
                    "antroute_event_edges": 0,
                    "baseline_event_edges": 0,
                }
                for system, nodes in (("antroute", ant), ("baseline", base)):
                    row[f"{system}_waypoints"] = " | ".join(
                        latlon(v) for v in pick_waypoints(wg, nodes, a.n_waypoints)
                    )
                    row[f"{system}_apple_eta_seconds"] = ""
                for system, nodes in (("antroute", ant), ("baseline", base)):
                    row[f"{system}_route"] = " ".join(str(v) for v in nodes)
                row["graph"] = "full"
                row["antroute_predicted_eta_seconds"] = round(
                    path_eta_seconds(wg, net.free_flow_seconds, ant), 1
                )
                row["baseline_predicted_eta_seconds"] = round(baseline_eta(window, base), 1)
                rows.append(row)
            made += 1

    a.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    trips = len({r["trip_id"] for r in rows})
    same = sum(r["same_route"] for r in rows)
    print(f"\nwrote {len(rows)} legs for {trips} trips to {a.out} ({same} legs where both systems took the same path)")
    print(f"next: python scripts/apple_maps_worklist.py --template {a.out} "
          f"--out {a.out.with_name('lookups_citywide.csv')}")


if __name__ == "__main__":
    main()
