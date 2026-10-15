"""
Route Optimality for the routing ablation (scripts/evaluate_ablation.py): time every variant's
route in Google Maps the same way the evaluation timed ANTROUTE's and the baseline's.

    python scripts/ablation_google.py --dry-run      # how many requests it would send
    python scripts/ablation_google.py                # send them (GOOGLE_ROUTES_API_KEY)

Per trip and leg, a variant's route is sent as six evenly spaced via points (pick_waypoints) at
the evaluation's own departure date and slot for that trip (template.csv), with the evaluation's
access method and traffic model. Route Optimality = Google's own time for the trip (its own
route, from the evaluation) / the variant's time, x 100, summed over legs. The baseline keeps the
times the evaluation already recorded. Identical legs are looked up once; answers are appended
to outputs/ablation/google_lookups.csv as they arrive, so an interrupted run resumes.
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import google_routes_lookup as g  # noqa: E402
from app import risk_routing as rr  # noqa: E402
from src.routing.route_trials import pick_waypoints  # noqa: E402
from src.routing.significance import compare_paired, holm_adjust  # noqa: E402

BASELINE = "baseline (Improved ACO)"
ANTROUTE = "ANTROUTE (lambda 2, mu 1)"
FIELDS = ["key", "departure", "start", "stops", "end", "duration_seconds", "queried_on"]


def legs_of(path: Sequence[int], stops: Sequence[int]) -> List[List[int]]:
    """Split a multi-stop route at its stops, in order."""
    legs, start = [], 0
    for stop in stops[1:]:
        end = path.index(stop, start + 1) if start + 1 < len(path) else -1
        if end < 0:
            raise ValueError(f"stop {stop} not on the route after position {start}")
        legs.append(list(path[start:end + 1]))
        start = end
    return legs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ablation-dir", type=Path, default=REPO_ROOT / "outputs/ablation")
    p.add_argument("--template", type=Path, default=REPO_ROOT / "outputs/appendix/template.csv")
    p.add_argument("--template-citywide", type=Path, default=REPO_ROOT / "outputs/appendix/template_citywide.csv")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    table = pd.read_csv(a.ablation_dir / "per_trip.csv")
    template = pd.concat([pd.read_csv(t, dtype=str) for t in (a.template, a.template_citywide) if t.exists()])
    template = template.sort_values(["trip_id", "leg"], key=lambda c: c.astype(int) if c.name == "leg" else c)
    plan: Dict[str, dict] = {}
    for trip_id, rows in template.groupby("trip_id"):
        plan[trip_id] = {
            "depart": datetime.fromisoformat(f"{rows['google_departure_date'].iloc[0]}T{rows['google_departure_slot'].iloc[0].split()[1]}"),
            "own": [float(x) for x in rows["google_duration_seconds"]],
            "baseline": [float(x) if str(x) not in ("", "nan") else np.nan for x in rows["baseline_google_duration_seconds"]],
            "stops": None,
        }
    per_trip = pd.read_csv(REPO_ROOT / "outputs/routing_eval/per_trip.csv")
    for r in per_trip.itertuples():
        if r.trip_id in plan:
            plan[r.trip_id]["stops"] = [int(v) for v in str(r.stop_nodes).split()]

    net = rr._network()
    latlon = lambda n: "{:.6f},{:.6f}".format(*net.node_coords[int(n)])  # noqa: E731
    requests: Dict[str, dict] = {}
    route_legs: Dict[tuple, List[str]] = {}
    for r in table.itertuples():
        if r.variant == BASELINE or r.trip_id not in plan or plan[r.trip_id]["stops"] is None:
            continue
        info = plan[r.trip_id]
        path = [int(v) for v in str(r.path).split()]
        wg = rr._weighted_graph(r.window)
        keys = []
        for leg in legs_of(path, info["stops"]):
            row = {"start": latlon(leg[0]), "end": latlon(leg[-1]),
                   "stops": " | ".join(latlon(v) for v in pick_waypoints(wg, leg, 6))}
            key = json.dumps([info["depart"].isoformat(), row["start"], row["stops"], row["end"]])
            requests[key] = {**row, "departure": info["depart"]}
            keys.append(key)
        route_legs[(r.trip_id, r.variant)] = keys

    out = a.ablation_dir / "google_lookups.csv"
    done = {}
    if out.exists():
        done = {row["key"]: float(row["duration_seconds"]) for row in csv.DictReader(out.open(encoding="utf-8"))}
    todo = [k for k in requests if k not in done]
    print(f"{len(route_legs)} variant routes, {len(requests)} distinct legs, {len(done)} already answered, "
          f"{len(todo)} requests to send")
    if a.dry_run:
        return

    if todo:
        key = g.api_key()
        new_file = not out.exists()
        with out.open("a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new_file:
                w.writeheader()
            for i, k in enumerate(todo, 1):
                req = requests[k]
                depart = req["departure"].replace(tzinfo=g.LOCAL_TZ)
                try:
                    routes = g.compute_route(g.request_body(req, depart), key).get("routes") or []
                except RuntimeError as err:
                    if str(err).startswith(("HTTP 400", "HTTP 401", "HTTP 403")):
                        raise SystemExit(f"Google Routes API: {err}") from None
                    routes = []
                if not routes or "duration" not in routes[0]:
                    continue
                done[k] = g.seconds(routes[0]["duration"])
                w.writerow({"key": k, "departure": req["departure"].isoformat(), "start": req["start"],
                            "stops": req["stops"], "end": req["end"], "duration_seconds": done[k],
                            "queried_on": datetime.now(g.LOCAL_TZ).date().isoformat()})
                f.flush()
                if i % 25 == 0:
                    print(f"  {i}/{len(todo)}")

    # Route Optimality per trip and variant
    rows = []
    for (trip_id, variant), keys in route_legs.items():
        if any(k not in done for k in keys):
            continue
        own = sum(plan[trip_id]["own"])
        rows.append({"trip_id": trip_id, "variant": variant, "google_seconds": sum(done[k] for k in keys),
                     "route_optimality": own / sum(done[k] for k in keys) * 100})
    for trip_id, info in plan.items():
        if not any(np.isnan(info["baseline"])) and info["stops"] is not None:
            rows.append({"trip_id": trip_id, "variant": BASELINE, "google_seconds": sum(info["baseline"]),
                         "route_optimality": sum(info["own"]) / sum(info["baseline"]) * 100})
    ro = pd.DataFrame(rows)
    ro.to_csv(a.ablation_dir / "route_optimality.csv", index=False)
    by = ro.set_index(["trip_id", "variant"])["route_optimality"]
    tests = {}
    for variant in sorted(v for v in ro["variant"].unique() if v != ANTROUTE):
        trips = sorted(set(ro.loc[ro["variant"] == variant, "trip_id"]) & set(ro.loc[ro["variant"] == ANTROUTE, "trip_id"]))
        tests[variant] = compare_paired("route_optimality", [by[(t, ANTROUTE)] for t in trips], [by[(t, variant)] for t in trips])
    holm_adjust(list(tests.values()))
    summary_path = a.ablation_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["route_optimality"] = {
        "reference": "Google Routes API v2 computeRoutes (DRIVE, TRAFFIC_AWARE_OPTIMAL, BEST_GUESS; six via points per "
                     "leg), the evaluation's departure date and slot per trip; baseline times from the evaluation",
        "means": ro.groupby("variant")["route_optimality"].mean().to_dict(),
        "vs_antroute": {v: c.__dict__ for v, c in tests.items()},
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print("\nRoute Optimality (mean %), Holm-adjusted Wilcoxon p vs ANTROUTE:")
    print(f"  {ANTROUTE:34s} {ro.loc[ro['variant'] == ANTROUTE, 'route_optimality'].mean():.1f}")
    for v, c in tests.items():
        print(f"  {v:34s} {c.baseline_mean:.1f}   n={c.n}  p={c.wilcoxon_p_holm}")


if __name__ == "__main__":
    main()
