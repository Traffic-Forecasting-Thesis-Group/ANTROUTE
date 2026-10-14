"""
Google Maps as a congestion reference for the risk scores, on road segments across Metro Manila.

    python scripts/google_segment_reference.py --per-tier 100 --out-dir outputs/google_eval/segments

The routing evaluation (evaluate_routing.py) uses Google to time whole routes. This checks the
congestion risk itself, in every risk tier -- including the assumed tier, where nothing else
does. It does not store Google travel times for the whole network (that would be pre-fetching a
dataset, which the Maps Platform terms forbid, and per-edge times of 20-100 m roads are noise
anyway); it samples segments.

1. Sample.   --per-tier segments per risk tier (observed / propagated / assumed, by the
             length-weighted majority of their edges), each 1-2 km along the road graph and
             nearly straight (straight-line >= 75% of road length), starting on a road with
             free-flow speed >= 30 km/h -- through-roads, where congestion shows; on a
             residential lane Google's traffic and no-traffic times are the same.
2. Query.    Each segment at an AM-peak and a PM-peak recorded window (--split test), as a
             Google departure slot (weekday + time, next such date): one Routes API request,
             TRAFFIC_AWARE_OPTIMAL / BEST_GUESS, with two interior points as via waypoints so
             Google stays on the segment. It returns `duration` (with predicted traffic) and
             `staticDuration` (without). Google's congestion index = duration / staticDuration.
             A segment Google routes more than 25% longer or shorter than ours is flagged
             off_segment and left out of the comparison.
3. Compare.  ANTROUTE's predicted risk on the segment (mean over its edges, that window) against
             Google's congestion index: Spearman rank correlation overall, per tier and per
             slot, and the mean index per congestion level (clear / moderate / heavy by the
             app's own cutoffs).

segments.csv is saved after every request, so an interrupted run resumes; summary.json holds
the comparison and how the reference was obtained. Google predicts typical traffic for a future
date, not the recorded May days -- the same stated limitation as the routing evaluation.
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.sparse.csgraph import dijkstra  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from app import risk_routing as rr  # noqa: E402
from google_routes_lookup import LOCAL_TZ, api_key, compute_route, departure, latlng  # noqa: E402
from src.routing.dynamic_weight import TIERS  # noqa: E402
from src.routing.route_trials import pick_waypoints  # noqa: E402

FIELD_MASK = "routes.duration,routes.staticDuration,routes.distanceMeters"
MIN_M, MAX_M = 1000.0, 2000.0
MIN_SPEED_KMH = 30.0
MAX_LENGTH_MISMATCH = 0.25
ACCESS_METHOD = ("Google Routes API v2 computeRoutes (DRIVE, TRAFFIC_AWARE_OPTIMAL, BEST_GUESS; "
                 "duration vs staticDuration; two interior via points)")


def pick_windows(net, split: str) -> List[str]:
    """One AM-peak and one PM-peak recorded window from `split`, each the middle of its peak."""
    def split_of(w):
        rows = net.risk_by_window[w]
        return str(rows["split"].iloc[0]) if "split" in rows.columns else "all"

    windows = [w for w in sorted(net.risk_by_window) if split == "all" or split_of(w) == split]
    picked = []
    for am in (True, False):
        session = [w for w in windows if (datetime.fromisoformat(w).hour < 12) == am]
        if session:
            picked.append(session[len(session) // 2])
    if not picked:
        raise SystemExit(f"no recorded windows in split {split!r}")
    return picked


def straight_km(net, a: int, b: int) -> float:
    (la1, lo1), (la2, lo2) = net.node_coords[a], net.node_coords[b]
    p1, p2 = math.radians(la1), math.radians(la2)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def sample_segments(net, wg, tiers: np.ndarray, speed_kmh: np.ndarray, per_tier: int, rng) -> pd.DataFrame:
    n = wg.n_nodes
    roads = sp.csr_matrix((wg.distance, (wg.src, wg.dst)), shape=(n, n))
    rows, seen = [], set()
    for tier in TIERS:
        starts = np.flatnonzero((tiers == tier) & (speed_kmh >= MIN_SPEED_KMH))
        made, tries = 0, 0
        while made < per_tier and tries < per_tier * 40 and starts.size:
            tries += 1
            e = int(rng.choice(starts))
            s = int(wg.src[e])
            dist, pred = dijkstra(roads, directed=True, indices=s, limit=MAX_M, return_predecessors=True)
            ends = np.flatnonzero((dist >= MIN_M) & (dist <= MAX_M))
            if not ends.size:
                continue
            t = int(rng.choice(ends))
            if straight_km(net, int(wg.node_ids[s]), int(wg.node_ids[t])) * 1000 < 0.75 * dist[t]:
                continue
            path = [t]
            while path[-1] != s:
                path.append(int(pred[path[-1]]))
            path.reverse()
            edges = np.array([wg.edge_id(u, v) for u, v in zip(path, path[1:])])
            length = wg.distance[edges]
            share = {k: float(length[tiers[edges] == k].sum() / length.sum()) for k in TIERS}
            if max(share, key=share.get) != tier or (s, t) in seen:
                continue
            seen.add((s, t))
            nodes = [int(wg.node_ids[i]) for i in path]
            rows.append({
                "segment_id": f"{tier[0].upper()}{made + 1:03d}",
                "tier": tier,
                "tier_share": round(share[tier], 3),
                "length_m": round(float(length.sum()), 1),
                "nodes": " ".join(map(str, nodes)),
                "start": "{:.6f},{:.6f}".format(*net.node_coords[nodes[0]]),
                "end": "{:.6f},{:.6f}".format(*net.node_coords[nodes[-1]]),
                "stops": " | ".join("{:.6f},{:.6f}".format(*net.node_coords[v]) for v in pick_waypoints(wg, nodes, 2)),
                "road": (net.edge_names[edges[np.argmax(length)]] if net.edge_names is not None else None) or "",
            })
            made += 1
        print(f"  {tier:10s} {made} segments ({tries} tries)")
    return pd.DataFrame(rows)


def query(table: pd.DataFrame, path: Path, key: str, pause: float) -> pd.DataFrame:
    today = datetime.now(LOCAL_TZ).date()
    todo = [i for i in table.index if not str(table.at[i, "google_duration_s"]).strip()]
    print(f"{len(todo)} of {len(table)} segment-slot queries to do")
    for n, i in enumerate(todo, start=1):
        row = table.loc[i]
        depart = departure(row["departure_slot"], today)
        body = {
            "origin": latlng(row["start"]),
            "destination": latlng(row["end"]),
            "intermediates": [{**latlng(p), "via": True} for p in str(row["stops"]).split("|") if p.strip()],
            "travelMode": "DRIVE",
            "routingPreference": "TRAFFIC_AWARE_OPTIMAL",
            "trafficModel": "BEST_GUESS",
            "departureTime": depart.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        try:
            result = compute_route(body, key, field_mask=FIELD_MASK)
        except RuntimeError as err:
            if str(err).startswith(("HTTP 400", "HTTP 401", "HTTP 403")):
                raise SystemExit(f"stopped at {row['segment_id']}: {err}") from None
            print(f"  {row['segment_id']} {row['departure_slot']}: FAILED -- {err}")
            continue
        route = (result.get("routes") or [{}])[0]
        if "duration" not in route or "staticDuration" not in route:
            table.at[i, "google_duration_s"] = "none"
            print(f"  {row['segment_id']} {row['departure_slot']}: no route")
        else:
            table.at[i, "google_duration_s"] = float(str(route["duration"]).rstrip("s"))
            table.at[i, "google_static_s"] = float(str(route["staticDuration"]).rstrip("s"))
            table.at[i, "google_distance_m"] = float(route.get("distanceMeters", "nan"))
        table.at[i, "google_departure_date"] = depart.date().isoformat()
        table.at[i, "google_queried_on"] = today.isoformat()
        table.to_csv(path, index=False)
        if n % 25 == 0:
            print(f"  {n} done")
        time.sleep(pause)
    return table


def compare(table: pd.DataFrame, net) -> dict:
    t = table[pd.to_numeric(table["google_duration_s"], errors="coerce").notna()].copy()
    for c in ("google_duration_s", "google_static_s", "google_distance_m", "length_m", "predicted_risk"):
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t["congestion_index"] = t["google_duration_s"] / t["google_static_s"]
    t["off_segment"] = (t["google_distance_m"] / t["length_m"] - 1).abs() > MAX_LENGTH_MISMATCH
    kept = t[~t["off_segment"] & t["congestion_index"].notna()]

    def corr(frame):
        if len(frame) < 5 or frame["predicted_risk"].nunique() < 2 or frame["congestion_index"].nunique() < 2:
            return {"n": int(len(frame)), "spearman_rho": None, "p": None}
        rho, p = spearmanr(frame["predicted_risk"], frame["congestion_index"])
        return {"n": int(len(frame)), "spearman_rho": round(float(rho), 4), "p": float(p)}

    def level(r):
        return "clear" if r <= net.risk_low else "moderate" if r <= net.risk_high else "heavy"

    kept = kept.assign(level=kept["predicted_risk"].map(level))
    return {
        "n_queries": int(len(table)),
        "n_answered": int(len(t)),
        "n_off_segment": int(t["off_segment"].sum()),
        "overall": corr(kept),
        "by_tier": {tier: corr(kept[kept["tier"] == tier]) for tier in TIERS},
        "by_slot": {slot: corr(g) for slot, g in kept.groupby("departure_slot")},
        "by_tier_and_slot": {f"{tier} | {slot}": corr(g) for (tier, slot), g in kept.groupby(["tier", "departure_slot"])},
        "mean_congestion_index_by_predicted_level": {
            lvl: {"n": int(len(g)), "mean": round(float(g["congestion_index"].mean()), 4)}
            for lvl, g in kept.groupby("level")
        },
        "_kept": kept,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-tier", type=int, default=100)
    p.add_argument("--split", default="test")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pause", type=float, default=0.2)
    p.add_argument("--sample-only", action="store_true", help="sample the segments and stop; nothing is sent")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs/google_eval/segments")
    a = p.parse_args()

    net = rr._network()
    if not net.risk_by_window:
        raise SystemExit("risk_edges.csv is not loaded")
    windows = pick_windows(net, a.split)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    path = a.out_dir / "segments.csv"

    if path.exists():
        table = pd.read_csv(path, dtype=str, keep_default_na=False)
        print(f"resuming {path}: {table['segment_id'].nunique()} segments x {table['departure_slot'].nunique()} slots")
    else:
        wg = rr._weighted_graph(windows[0])
        tiers = rr._edge_tiers(windows[0])
        context = rr._context_model()
        if context is None:
            raise SystemExit("the assumed-tier estimate is unavailable; it is needed for the road speeds")
        speed = context[0].speed * 100.0
        print(f"sampling {a.per_tier} segments per tier")
        segments = sample_segments(net, wg, tiers, speed, a.per_tier, np.random.default_rng(a.seed))
        rows = []
        for w in windows:
            wgw = rr._weighted_graph(w)
            for seg in segments.to_dict("records"):
                m = wgw.evaluate_path([int(v) for v in seg["nodes"].split()])
                rows.append({**seg, "window": w, "departure_slot": datetime.fromisoformat(w).strftime("%a %H:%M"),
                             "predicted_risk": round(m.mean_risk, 4)})
        table = pd.DataFrame(rows)
        for c in ("google_duration_s", "google_static_s", "google_distance_m", "google_departure_date", "google_queried_on"):
            table[c] = ""
        table.to_csv(path, index=False)
        print(f"wrote {len(table)} segment-slot rows to {path}")
    if a.sample_only:
        return

    table = query(table, path, api_key(), a.pause)
    result = compare(table, net)
    kept = result.pop("_kept")
    kept.to_csv(a.out_dir / "segments_compared.csv", index=False)
    answered = table[table["google_queried_on"].astype(str).str.strip() != ""]
    summary = {
        "reference": {
            "reference": "Google Maps",
            "access_methods": [ACCESS_METHOD],
            "traffic_models": ["BEST_GUESS"],
            "departure_dates": sorted(answered["google_departure_date"].astype(str).unique().tolist()),
            "queried_on": sorted(answered["google_queried_on"].astype(str).unique().tolist()),
            "departure_slots": sorted(table["departure_slot"].unique().tolist()),
            "recorded_windows": windows,
        },
        "definition": "congestion_index = Google duration with predicted traffic / duration without traffic; "
                      "compared with ANTROUTE's mean predicted risk on the same segment and window",
        "sampling": f"{a.per_tier} segments per risk tier, {MIN_M:.0f}-{MAX_M:.0f} m, near-straight, "
                    f"starting on roads >= {MIN_SPEED_KMH:.0f} km/h free-flow; seed {a.seed}",
        "risk_level_cutoffs": {"clear_max": net.risk_low, "moderate_max": net.risk_high},
        **result,
    }
    (a.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("n_answered", "n_off_segment", "overall", "by_tier",
                                              "mean_congestion_index_by_predicted_level")}, indent=1))


if __name__ == "__main__":
    main()
