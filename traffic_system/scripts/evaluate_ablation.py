"""
Routing ablation: does the risk term do anything beyond distance? On the evaluation's own trips
(outputs/routing_eval/per_trip.csv, 145 paired trips, their stops and recorded windows), routed
on the whole Metro Manila graph with the app's colony (8 ants x 15 iterations):

    cost variants  W = dist * (1 + lambda * Risk) * (1 + mu * Event)
        distance only        lambda 0, mu 0
        lambda 0 + events    lambda 0, mu 1     (the "it is just distance" comparison: same
                                                 incident layer, no risk)
        risk only            lambda 2, mu 0
        ANTROUTE             lambda 2, mu 1     (the method)
        baseline             Improved ACO (Cheng 2023), its evaluated route (trials.csv)
    fill-rule variants (ANTROUTE's cost; how unscored roads get their risk)
        fill 0               every road without a prediction gets 0
        fill median          ... the window's median
        fill nearest         ... the nearest scored road's risk, at any distance
        ANTROUTE             nearest within 1.5 km, else the assumed-tier estimate

Every route is measured with ONE yardstick -- ANTROUTE's risk tiers and Event-Aware layer for
that window (the same measure as the evaluation):

    distance_m             metres
    congestion_exposure    sum of edge risk (RADR eq. 5); grows with the number of edges
    risk_per_km            length-weighted mean risk: congestion per kilometre, independent of
                           route length -- the fair test of "it is just distance"
    high_risk_share        share of the route's length on edges in the window's top risk tercile
    event_exposure         sum of edge incident impact
    shared_with_antroute   share of ANTROUTE's route length this route also uses

Each variant is compared with ANTROUTE pairwise over trips (Wilcoxon signed-rank, Holm across
the variants per metric). Route Optimality needs Google Maps times for every variant's route;
it is not computed here.

    python scripts/evaluate_ablation.py --out-dir outputs/ablation
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app import risk_routing as rr  # noqa: E402
from src.routing.aco_routing import AntColonyConfig, multi_stop_route  # noqa: E402
from src.routing.condition_update import shared_length_fraction  # noqa: E402
from src.routing.dynamic_weight import (  # noqa: E402
    DEFAULT_MISSING_RISK,
    build_weighted_graph,
    fill_unscored_risk,
    risk_vector,
    scored_mask,
)
from src.routing.event_layer import apply_events, event_exposure  # noqa: E402
from src.routing.significance import compare_paired, holm_adjust  # noqa: E402

ANTROUTE = "ANTROUTE (lambda 2, mu 1)"
COST_VARIANTS = {               # name -> (lambda, mu, fill rule)
    "distance only (lambda 0, mu 0)": (0.0, 0.0, "method"),
    "lambda 0 + events (mu 1)": (0.0, 1.0, "method"),
    "risk only (lambda 2, mu 0)": (2.0, 0.0, "method"),
    ANTROUTE: (2.0, 1.0, "method"),
}
FILL_VARIANTS = {
    "fill 0": (2.0, 1.0, "zero"),
    "fill median": (2.0, 1.0, "median"),
    "fill nearest (any distance)": (2.0, 1.0, "nearest"),
}
BASELINE = "baseline (Improved ACO)"
METRICS = ("distance_m", "congestion_exposure", "risk_per_km", "high_risk_share", "event_exposure")


def fill_rule_risk(rule: str, graph, frame, method_risk: np.ndarray) -> np.ndarray:
    """Risk of every edge for one window under a fill rule; 'method' is ANTROUTE's own."""
    if rule == "method":
        return method_risk
    risk, _ = risk_vector(graph, frame, DEFAULT_MISSING_RISK)          # unscored edges = 0
    if rule == "zero":
        return risk
    mask = scored_mask(graph, frame)
    if rule == "median":
        filled, _ = fill_unscored_risk(graph, risk, mask, rr._nearest_scored(mask.tobytes()), max_m=-1.0)
        return filled
    if rule == "nearest":
        filled, _ = fill_unscored_risk(graph, risk, mask, rr._nearest_scored(mask.tobytes()), max_m=float("inf"))
        return filled
    raise ValueError(f"unknown fill rule {rule!r}")


def measure(yardstick, impact: Optional[np.ndarray], high_cut: float, path: Sequence[int]) -> Dict[str, float]:
    """A route's metrics under the method's risk and incidents for its window."""
    m = yardstick.evaluate_path(path, impact)
    idx = [yardstick.index_of(int(v)) for v in path]
    edges = np.array([yardstick.edge_id(u, v) for u, v in zip(idx, idx[1:])], dtype=np.int64)
    length = yardstick.distance[edges]
    risk = yardstick.risk[edges]
    total = float(length.sum())
    return {
        "distance_m": m.distance_m,
        "congestion_exposure": m.risk_exposure,
        "risk_per_km": float((risk * length).sum() / total) if total > 0 else float("nan"),
        "high_risk_share": float(length[risk >= high_cut].sum() / total) if total > 0 else float("nan"),
        "event_exposure": m.event_exposure,
    }


class Graphs:
    """Weighted graphs per (window, variant), built once."""

    def __init__(self):
        self.net = rr._network()
        self._cache: Dict[tuple, object] = {}

    def method(self, window: str):
        return rr._weighted_graph(window)

    def variant(self, window: str, lam: float, mu: float, rule: str):
        key = (window, lam, mu, rule)
        if key not in self._cache:
            method = self.method(window)
            risk = fill_rule_risk(rule, self.net.graph, self.net.risk_by_window[window], method.risk)
            wg = build_weighted_graph(self.net.graph, risk, lam=lam, window=window)
            impact = rr._window_impact(window)
            if mu > 0 and impact is not None:
                wg = apply_events(wg, impact, mu=mu)
            self._cache[key] = wg
        return self._cache[key]


def summarise(rows: pd.DataFrame) -> dict:
    """Mean per variant and metric, and each variant vs ANTROUTE, Holm-adjusted across variants."""
    variants = [v for v in rows["variant"].unique() if v != ANTROUTE]
    by = rows.set_index(["trip_id", "variant"])
    means = rows.groupby("variant")[list(METRICS) + ["shared_with_antroute"]].mean().to_dict(orient="index")
    tests: Dict[str, Dict[str, dict]] = {}
    for metric in METRICS:
        family = {}
        for variant in variants:
            trips = sorted(set(rows.loc[rows["variant"] == variant, "trip_id"])
                           & set(rows.loc[rows["variant"] == ANTROUTE, "trip_id"]))
            ant = [by.loc[(t, ANTROUTE), metric] for t in trips]
            other = [by.loc[(t, variant), metric] for t in trips]
            family[variant] = compare_paired(metric, ant, other)   # proposed = ANTROUTE
        holm_adjust(list(family.values()))
        tests[metric] = {v: asdict(c) for v, c in family.items()}
    changed = rows[rows["variant"] != ANTROUTE].groupby("variant")["shared_with_antroute"].apply(
        lambda s: float((s < 0.999).mean())).to_dict()
    return {"means": means, "share_of_trips_with_a_different_route": changed, "vs_antroute": tests}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dir", type=Path, default=REPO_ROOT / "outputs/routing_eval")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs/ablation")
    p.add_argument("--limit", type=int, default=None, help="only the first N trips (a quick check)")
    a = p.parse_args()

    per_trip = pd.read_csv(a.eval_dir / "per_trip.csv")
    trials = pd.read_csv(a.eval_dir / "trials.csv")
    baseline_path = {r.trip_id: [int(v) for v in str(r.path).split()]
                     for r in trials[trials["system"] == "baseline"].itertuples()}
    if a.limit:
        per_trip = per_trip.head(a.limit)
    graphs = Graphs()
    config = AntColonyConfig(n_ants=8, n_iterations=15, seed=0)   # the app's colony
    variants = {**COST_VARIANTS, **FILL_VARIANTS}
    rows: List[dict] = []
    notes: List[str] = []
    for n, trip in enumerate(per_trip.itertuples(), 1):
        window = trip.window
        stops = [int(v) for v in str(trip.stop_nodes).split()]
        yardstick = graphs.method(window)
        impact = rr._window_impact(window)
        scored = yardstick.risk
        high_cut = float(np.quantile(scored, 2 / 3))
        routes: Dict[str, List[int]] = {}
        for name, (lam, mu, rule) in variants.items():
            try:
                routes[name] = multi_stop_route(graphs.variant(window, lam, mu, rule), stops, config).best.nodes
            except (ValueError, KeyError) as err:
                notes.append(f"{trip.trip_id} {name}: no route ({err})")
        if trip.trip_id in baseline_path:
            routes[BASELINE] = baseline_path[trip.trip_id]
        if ANTROUTE not in routes:
            continue
        for name, path in routes.items():
            try:
                metrics = measure(yardstick, impact, high_cut, path)
                shared = shared_length_fraction(yardstick, routes[ANTROUTE], path)
            except (KeyError, ValueError) as err:
                notes.append(f"{trip.trip_id} {name}: not on the city graph ({err})")
                continue
            rows.append({"trip_id": trip.trip_id, "scenario_type": trip.scenario_type, "graph": trip.graph,
                         "window": window, "variant": name, **metrics, "shared_with_antroute": shared,
                         "path": " ".join(map(str, path))})
        if n % 10 == 0:
            print(f"{n}/{len(per_trip)} trips")

    table = pd.DataFrame(rows)
    summary = {
        "trips": int(table["trip_id"].nunique()),
        "graph": "whole Metro Manila graph for every variant (the app's routing)",
        "yardstick": "ANTROUTE's risk tiers and Event-Aware layer for each trip's window",
        "variants": {k: {"lambda": v[0], "mu": v[1], "fill_rule": v[2]} for k, v in variants.items()},
        **summarise(table),
        "notes": notes[:50],
    }
    a.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.out_dir / "per_trip.csv", index=False)
    (a.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    means = pd.DataFrame(summary["means"]).T
    means["different_route"] = pd.Series(summary["share_of_trips_with_a_different_route"])
    pd.set_option("display.width", 220)
    print(f"\n{summary['trips']} trips; means per variant (one yardstick):\n")
    print(means.round(3).to_string())
    print("\nvs ANTROUTE, Holm-adjusted Wilcoxon p (risk_per_km / congestion_exposure / distance_m):")
    for v in summary["vs_antroute"]["risk_per_km"]:
        ps = [summary["vs_antroute"][m][v]["wilcoxon_p_holm"] for m in ("risk_per_km", "congestion_exposure", "distance_m")]
        print(f"  {v:34s} " + "  ".join("n/a" if x is None else f"{x:.3g}" for x in ps))
    print(f"\nwrote per_trip.csv and summary.json to {a.out_dir}" + (f"; {len(notes)} note(s)" if notes else ""))


if __name__ == "__main__":
    main()
