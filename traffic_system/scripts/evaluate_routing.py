"""
Evaluate ANTROUTE against the baseline with Apple Maps as ground truth, as the thesis specifies
(Sections 3.8-3.9, Appendices 1-3).

    # 1. route the test trips with both systems and write the Apple Maps lookup sheet
    python scripts/evaluate_routing.py --risk-edges data/processed/risk_scores/risk_edges.csv \
        --template apple_maps.csv

    # 2. fill in the *_apple_eta_seconds columns (see below), then score
    python scripts/evaluate_routing.py --risk-edges data/processed/risk_scores/risk_edges.csv \
        --apple-maps apple_maps.csv

Trips (--n-trials per scenario, default 10 as in Appendices 1-3) are drawn from the risk file's
held-out --split (default "test", per Section 3.3) and from CCTV intersections that are
connected on the road graph:

    recommended        best ACO route, origin -> destination
    alternative        next-best ACO route for the same origin/destination/window
    multi_destination  --n-stops CCTV intersections visited in order, routed leg by leg

The sheet has one row per leg (single-destination trips have one). Per row, in Apple Maps
(Driving, "Leave at" = apple_leave_at; Apple Maps only predicts typical traffic for a weekday
and time, not a past date):

    apple_eta_seconds            origin -> destination, Apple's own fastest route
    antroute_apple_eta_seconds   origin -> antroute_waypoints (as stops, in order) -> destination
    baseline_apple_eta_seconds   origin -> baseline_waypoints (as stops, in order) -> destination

When same_route is True one of the last two is enough. An alternative trip's
apple_eta_seconds may be left blank: it is the same lookup as the recommended trip's.

Scoring (stored routes are re-scored as-is, ACO is not re-run):

    Route Optimality      = sum(apple_eta_seconds) / sum(<system>_apple_eta_seconds) * 100
    MAE/RMSE/MSE/MAPE/R^2 = <system>'s own per-leg ETA vs <system>_apple_eta_seconds
    significance          Shapiro-Wilk, Wilcoxon signed-rank, paired t-test (if normal),
                          d = baseline - proposed, Δ% (Equation 15)

Outputs (in --out-dir):
    trials.csv                           one row per (system, trip)
    appendix1_route_optimality.csv       per-trial C_optimal, C_predicted, Route Optimality
    appendix2_eta.csv                    per-trial MAE, RMSE, MSE, MAPE, R^2
    appendix3_routing_results.csv        per-trial path, distance, risk exposure, ETA (min)
    appendix3_route_optimality_pairs.csv per-trial paired Route Optimality, d_i, Δ%
    appendix3_significance.csv           per scenario and metric: Shapiro-Wilk, Wilcoxon, t-test
    metrics.json                         all of the above plus pooled metrics per scenario
"""

import argparse
import json
import math
import sys
from dataclasses import asdict
from datetime import datetime
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.data.graph_data import DEFAULT_K, build_subgraph  # noqa: E402
from src.routing.aco_routing import AntColonyConfig, ant_colony_shortest_path, remaining_cost_to  # noqa: E402
from src.routing.dynamic_weight import (  # noqa: E402
    DEFAULT_LAMBDA,
    DEFAULT_MISSING_RISK,
    load_risk_edges,
    window_key,
    weighted_graph_from_risk_file,
)
from src.routing.eta_engine import DEFAULT_GAMMA, load_free_flow_seconds  # noqa: E402
from src.routing.metrics import compute_metrics_by_scenario, route_optimality, trial_eta_metrics  # noqa: E402
from src.routing.route_trials import (  # noqa: E402
    ALTERNATIVE,
    ANTROUTE,
    BASELINE,
    MULTI_DESTINATION,
    RECOMMENDED,
    SCENARIOS,
    SYSTEMS,
    pick_waypoints,
    score_trip,
)
from src.routing.significance import compare_paired, relative_difference  # noqa: E402

ETA_METRICS = ("mae", "rmse", "mse", "mape", "r_squared")
TRIP_PREFIX = {RECOMMENDED: "R", ALTERNATIVE: "A", MULTI_DESTINATION: "M"}


# ---------------------------------------------------------------- template

def camera_labels(graph) -> Dict[int, str]:
    return {int(graph.node_ids[i]): label for label, i in graph.camera_nodes.items()}


def load_coordinates(spatial_dir: Path) -> Dict[int, Tuple[float, float]]:
    frame = pd.read_csv(spatial_dir / "full_network_static_features.csv", usecols=["node_id", "lat", "lon"])
    return {int(n): (float(lat), float(lon)) for n, lat, lon in frame.itertuples(index=False)}


def latlon(coords: Dict[int, Tuple[float, float]], node_id: int) -> str:
    lat, lon = coords[int(node_id)]
    return f"{lat:.6f},{lon:.6f}"


def leave_at(window: str) -> str:
    try:
        return datetime.fromisoformat(window).strftime("%a %H:%M")
    except ValueError:
        return ""


def evaluation_windows(a) -> List[str]:
    frame = load_risk_edges(a.risk_edges)
    keys = window_key(frame)
    if a.split == "all":
        return sorted(keys.unique())
    if "split" not in frame.columns:
        raise SystemExit(f"{a.risk_edges} has no split column; pass --split all to use every window")
    windows = sorted(keys[frame["split"] == a.split].unique())
    if not windows:
        raise SystemExit(f"{a.risk_edges} has no windows in split {a.split!r}")
    return windows


def reachability(wg, cameras: Sequence[int]) -> Dict[Tuple[int, int], bool]:
    """Which ordered CCTV pairs are connected on the road graph (independent of the window)."""
    out = {}
    for d in cameras:
        remaining = remaining_cost_to(wg, wg.index_of(d))
        for o in cameras:
            out[(o, d)] = o != d and bool(np.isfinite(remaining[wg.index_of(o)]))
    return out


def sample_trips(cameras, connected, windows, a, rng) -> Dict[str, List[Tuple[str, Tuple[int, ...]]]]:
    pairs = [(o, d) for o in cameras for d in cameras if connected[(o, d)]]
    sequences = [
        s for s in permutations(cameras, a.n_stops) if all(connected[leg] for leg in zip(s, s[1:]))
    ]
    if not pairs:
        raise SystemExit("no two CCTV intersections are connected on this graph")
    if not sequences:
        raise SystemExit(f"no {a.n_stops}-stop sequence of CCTV intersections is connected; lower --n-stops")

    def draw(options):
        combos = [(w, s) for w in windows for s in options]
        picks = rng.choice(len(combos), size=min(a.n_trials, len(combos)), replace=False)
        return [combos[i] for i in sorted(picks)]

    single = draw(pairs)
    return {RECOMMENDED: single, ALTERNATIVE: single, MULTI_DESTINATION: draw(sequences)}


def write_template(a, graph, config: AntColonyConfig) -> None:
    windows = evaluation_windows(a)
    coords = load_coordinates(a.spatial_dir)
    labels = camera_labels(graph)
    cameras = sorted(labels)
    rng = np.random.default_rng(a.seed)

    graphs: Dict[str, Dict[str, object]] = {}

    def graphs_for(window):
        if window not in graphs:
            wg = weighted_graph_from_risk_file(
                graph, a.risk_edges, window=window, lam=a.lam, missing_risk=a.missing_risk
            )
            graphs[window] = {ANTROUTE: wg, BASELINE: wg.with_lambda(a.baseline_lambda)}
        return graphs[window]

    connected = reachability(graphs_for(windows[0])[ANTROUTE], cameras)
    trips = sample_trips(cameras, connected, windows, a, rng)

    rows = []
    routed: Dict[Tuple[str, Tuple[int, ...]], Dict[str, object]] = {}   # one ACO run per system, trip
    for scenario in SCENARIOS:
        for number, (window, stops) in enumerate(trips[scenario], start=1):
            g = graphs_for(window)
            key = (window, tuple(stops))
            if key not in routed:
                routed[key] = {
                    s: [ant_colony_shortest_path(g[s], o, d, config) for o, d in zip(stops, stops[1:])]
                    for s in SYSTEMS
                }
            # per system, per leg: the leg's route for this scenario, or None if ACO had no alternative
            legs = {
                s: [
                    (r.alternatives[0].nodes if r.alternatives else None) if scenario == ALTERNATIVE else r.best.nodes
                    for r in routed[key][s]
                ]
                for s in SYSTEMS
            }
            trip_id = f"{TRIP_PREFIX[scenario]}{number:02d}"
            for leg_no, (o, d) in enumerate(zip(stops, stops[1:]), start=1):
                routes = {s: legs[s][leg_no - 1] for s in SYSTEMS}
                row = {
                    "trip_id": trip_id,
                    "scenario_type": scenario,
                    "leg": leg_no,
                    "window": window,
                    "apple_leave_at": leave_at(window),
                    "origin": labels[o],
                    "destination": labels[d],
                    "origin_latlon": latlon(coords, o),
                    "destination_latlon": latlon(coords, d),
                    "same_route": routes[ANTROUTE] is not None and routes[ANTROUTE] == routes[BASELINE],
                    "apple_eta_seconds": "",
                }
                for s in SYSTEMS:
                    stops_on_route = pick_waypoints(g[ANTROUTE], routes[s], a.n_waypoints) if routes[s] else []
                    row[f"{s}_waypoints"] = (
                        " | ".join(latlon(coords, v) for v in stops_on_route) if routes[s] else "NO ROUTE"
                    )
                    row[f"{s}_apple_eta_seconds"] = ""
                for s in SYSTEMS:
                    row[f"{s}_route"] = " ".join(str(v) for v in routes[s]) if routes[s] else ""
                rows.append(row)
            print(f"{trip_id} {window} {' -> '.join(labels[v] for v in stops)}")

    missing = sum(1 for r in rows if not (r["antroute_route"] and r["baseline_route"]))
    a.template.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.template, index=False)
    print(f"\nwrote {len(rows)} Apple Maps lookups for {sum(len(t) for t in trips.values())} trips to {a.template}")
    if missing:
        print(
            f"NOTE: {missing} alternative leg(s) have NO ROUTE: ACO found no second route for that system. "
            "Those trips cannot be paired; raise --n-iterations / --n-ants or accept fewer alternative trials."
        )
    print("Fill in the *_apple_eta_seconds columns from Apple Maps, then re-run with --apple-maps.")


# ---------------------------------------------------------------- scoring

def seconds(value) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)) or str(value).strip() == "":
        return float("nan")
    return float(value)


def nodes(value: str) -> Optional[List[int]]:
    return [int(v) for v in value.split()] if value and value.strip() else None


def json_safe(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def mean_sd(values: Sequence[float]) -> Dict[str, float]:
    x = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    return {
        "mean": float(x.mean()) if x.size else float("nan"),
        "sd": float(x.std(ddof=1)) if x.size > 1 else float("nan"),
    }


def score_sheet(a, graph, free_flow_seconds) -> Tuple[List[dict], List[str]]:
    sheet = pd.read_csv(a.apple_maps, dtype=str, keep_default_na=False)
    needed = {"trip_id", "scenario_type", "leg", "window", "apple_eta_seconds"} | {
        f"{s}_{c}" for s in SYSTEMS for c in ("route", "apple_eta_seconds")
    }
    if not needed <= set(sheet.columns):
        raise SystemExit(f"{a.apple_maps} is missing {sorted(needed - set(sheet.columns))}; make it with --template")

    # Apple's own route is the same lookup for every trip with this window and leg endpoints.
    optimal_by_leg = {
        (r["window"], r["origin"], r["destination"]): r["apple_eta_seconds"]
        for _, r in sheet.iterrows()
        if r["apple_eta_seconds"].strip()
    }
    gammas = {ANTROUTE: a.gamma, BASELINE: a.baseline_gamma}
    graphs: Dict[str, object] = {}
    trials: List[dict] = []
    notes: List[str] = []

    for trip_id, rows in sheet.groupby("trip_id", sort=True):
        rows = rows.sort_values("leg", key=lambda c: c.astype(int))
        window, scenario = rows["window"].iloc[0], rows["scenario_type"].iloc[0]
        if window not in graphs:
            try:
                graphs[window] = weighted_graph_from_risk_file(
                    graph, a.risk_edges, window=window, missing_risk=a.missing_risk
                )
            except ValueError as err:
                raise SystemExit(f"{a.apple_maps}: {err}") from None
        wg = graphs[window]

        optimal = [
            seconds(r["apple_eta_seconds"].strip() or optimal_by_leg.get((r["window"], r["origin"], r["destination"])))
            for _, r in rows.iterrows()
        ]
        measured = {s: [seconds(r[f"{s}_apple_eta_seconds"]) for _, r in rows.iterrows()] for s in SYSTEMS}
        same = [str(r.get("same_route", "")).strip().lower() == "true" for _, r in rows.iterrows()]
        for i, is_same in enumerate(same):
            if is_same:
                shared = next((measured[s][i] for s in SYSTEMS if np.isfinite(measured[s][i])), float("nan"))
                for s in SYSTEMS:
                    measured[s][i] = shared

        routes = {s: [nodes(r[f"{s}_route"]) for _, r in rows.iterrows()] for s in SYSTEMS}
        if any(leg is None for s in SYSTEMS for leg in routes[s]):
            notes.append(f"{trip_id}: a system has no route for some leg; trip left out of the paired comparison")
            continue
        if not all(np.isfinite(v) for v in optimal + measured[ANTROUTE] + measured[BASELINE]):
            notes.append(f"{trip_id}: Apple Maps times not all filled in; skipped")
            continue
        for s in SYSTEMS:
            trial = score_trip(
                wg, routes[s], free_flow_seconds, gammas[s], s, scenario, optimal, measured[s], trip_id
            )
            trial["route_optimality"] = float(route_optimality([trial["c_optimal"]], [trial["c_predicted"]])[0])
            trial.update(trial_eta_metrics(trial["actual_eta"], trial["predicted_eta"]))
            trials.append(trial)
    return trials, notes


def describe(stops: Sequence[int], labels: Dict[int, str]) -> str:
    return " -> ".join(labels.get(v, str(v)) for v in stops)


def write_reports(a, graph, trials: List[dict]) -> dict:
    labels = camera_labels(graph)
    by = {(t["trip_id"], t["system"]): t for t in trials}
    trip_ids = sorted({t["trip_id"] for t in trials})
    scenario_of = {t["trip_id"]: t["scenario_type"] for t in trials}
    groups = {"overall": trip_ids}
    groups.update({sc: [i for i in trip_ids if scenario_of[i] == sc] for sc in SCENARIOS})
    groups = {k: v for k, v in groups.items() if v}

    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)

    flat = []
    for t in trials:
        row = dict(t)
        row["stops"] = describe(t["stops"], labels)
        row["path"] = " ".join(str(v) for v in t["path"])
        row["actual_eta"] = " | ".join(f"{v:g}" for v in t["actual_eta"])
        row["predicted_eta"] = " | ".join(f"{v:.1f}" for v in t["predicted_eta"])
        flat.append(row)
    pd.DataFrame(flat).to_csv(out / "trials.csv", index=False)

    def per_trial(columns, name):
        rows = [
            {"scenario_type": scenario_of[i], "trial": i, "system": s, **{c: by[(i, s)][c] for c in columns}}
            for i in trip_ids for s in SYSTEMS
        ]
        pd.DataFrame(rows).to_csv(out / name, index=False)

    per_trial(("c_optimal", "c_predicted", "route_optimality"), "appendix1_route_optimality.csv")
    per_trial(ETA_METRICS, "appendix2_eta.csv")
    pd.DataFrame(
        [
            {
                "scenario_type": scenario_of[i],
                "trial": i,
                "model": s,
                "path": describe(by[(i, s)]["stops"], labels),
                "hops": by[(i, s)]["hops"],
                "total_distance_m": by[(i, s)]["distance_m"],
                "route_risk_exposure": by[(i, s)]["risk_exposure"],
                "eta_min": by[(i, s)]["eta_minutes"],
            }
            for i in trip_ids for s in (BASELINE, ANTROUTE)
        ]
    ).to_csv(out / "appendix3_routing_results.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_type": scenario_of[i],
                "trial": i,
                "proposed": by[(i, ANTROUTE)]["route_optimality"],
                "baseline": by[(i, BASELINE)]["route_optimality"],
                "d_i": by[(i, BASELINE)]["route_optimality"] - by[(i, ANTROUTE)]["route_optimality"],
                "relative_difference_pct": relative_difference(
                    by[(i, BASELINE)]["route_optimality"], by[(i, ANTROUTE)]["route_optimality"]
                ),
            }
            for i in trip_ids
        ]
    ).to_csv(out / "appendix3_route_optimality_pairs.csv", index=False)

    significance, summary = [], {}
    for group, ids in groups.items():
        summary[group] = {}
        for metric in ("route_optimality",) + ETA_METRICS:
            proposed = [by[(i, ANTROUTE)][metric] for i in ids]
            baseline = [by[(i, BASELINE)][metric] for i in ids]
            c = compare_paired(metric, proposed, baseline)
            significance.append({"scenario_type": group, **asdict(c)})
            summary[group][metric] = {ANTROUTE: mean_sd(proposed), BASELINE: mean_sd(baseline)}
    pd.DataFrame(significance).to_csv(out / "appendix3_significance.csv", index=False)

    pooled = {s: compute_metrics_by_scenario([t for t in trials if t["system"] == s]) for s in SYSTEMS}
    report = {
        "risk_edges": str(a.risk_edges),
        "apple_maps": str(a.apple_maps),
        ANTROUTE: {"gamma": a.gamma},
        BASELINE: {"gamma": a.baseline_gamma},
        "per_trial_mean_sd": summary,
        "pooled": {s: {k: asdict(m) for k, m in r.items()} for s, r in pooled.items()},
        "significance": significance,
    }
    (out / "metrics.json").write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")
    return report


def score(a, graph) -> None:
    if not (a.spatial_dir / "metro_manila_travel_time.npz").exists():
        raise SystemExit("metro_manila_travel_time.npz not found; the systems' own ETAs need it")
    free_flow_seconds = load_free_flow_seconds(a.spatial_dir, graph)
    trials, notes = score_sheet(a, graph, free_flow_seconds)
    for note in notes:
        print(f"NOTE: {note}")
    if not trials:
        raise SystemExit(f"{a.apple_maps} has no fully filled-in trips")
    over = sum(t["c_optimal"] > t["c_predicted"] for t in trials)
    if over:
        print(
            f"NOTE: {over} trial(s) score above 100%: Apple timed the forced route faster than its own. "
            "Usually traffic shifted between lookups -- re-check those rows."
        )

    report = write_reports(a, graph, trials)
    sig = {(r["scenario_type"], r["metric"]): r for r in report["significance"]}
    print(f"\n{'scenario':<18}{'n':>3}  {'optimality % ant / base':>24}  {'Wilcoxon p':>10}  {'MAE s ant / base':>18}")
    for group, metrics in report["per_trial_mean_sd"].items():
        ro, ma = metrics["route_optimality"], metrics["mae"]
        p = sig[(group, "route_optimality")]["wilcoxon_p"]
        print(
            f"{group:<18}{sig[(group, 'route_optimality')]['n']:>3}  "
            f"{ro[ANTROUTE]['mean']:>10.2f} / {ro[BASELINE]['mean']:<11.2f}  "
            f"{p:>10.4f}  {ma[ANTROUTE]['mean']:>8.1f} / {ma[BASELINE]['mean']:<8.1f}"
        )
    print(f"\nwrote appendix tables, trials.csv and metrics.json to {a.out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--risk-edges", required=True, type=Path)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--template", type=Path, help="route the test trips and write the Apple Maps sheet")
    mode.add_argument("--apple-maps", type=Path, help="the filled-in sheet from --template, to score")
    p.add_argument("--spatial-dir", type=Path, default=REPO_ROOT / "data/processed/spatial")
    p.add_argument("-k", type=int, default=DEFAULT_K)
    p.add_argument("--missing-risk", type=float, default=DEFAULT_MISSING_RISK)
    p.add_argument("--lambda", dest="lam", type=float, default=DEFAULT_LAMBDA, help="ANTROUTE risk penalty")
    p.add_argument("--gamma", type=float, default=DEFAULT_GAMMA, help="ANTROUTE ETA risk sensitivity")
    p.add_argument("--baseline-lambda", type=float, default=0.0)
    p.add_argument("--baseline-gamma", type=float, default=0.0)
    p.add_argument("--split", default="test", help='--template: risk-file split to draw windows from, or "all"')
    p.add_argument("--n-trials", type=int, default=10, help="--template: trials per scenario")
    p.add_argument("--n-stops", type=int, default=3, help="--template: stops per multi-destination trip")
    p.add_argument("--n-waypoints", type=int, default=6, help="--template: Apple Maps stops per leg")
    p.add_argument("--n-ants", type=int, default=20)
    p.add_argument("--n-iterations", type=int, default=60)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--evaporation", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs/routing_eval")
    a = p.parse_args()
    if a.n_stops < 3:
        p.error("--n-stops must be at least 3 (origin, one intermediate stop, destination)")

    graph = build_subgraph(a.spatial_dir, a.k)
    if a.template:
        config = AntColonyConfig(
            n_ants=a.n_ants,
            n_iterations=a.n_iterations,
            alpha=a.alpha,
            beta=a.beta,
            evaporation=a.evaporation,
            n_alternatives=1,
            seed=a.seed,
        )
        write_template(a, graph, config)
    else:
        score(a, graph)


if __name__ == "__main__":
    main()
