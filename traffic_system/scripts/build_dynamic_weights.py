"""
Run the Dynamic Weight Engine over a risk_edges.csv and write the risk-penalised graph
the ACO routing engine consumes.

    python scripts/build_dynamic_weights.py --risk-edges outputs/risk_scores/risk_edges.csv

    # one window, a stronger penalty, and a static-vs-dynamic route comparison
    python scripts/build_dynamic_weights.py --risk-edges outputs/risk_scores/risk_edges.csv \
        --window "2026-05-25T17:30:00" --lambda 3 --compare "EDSA-Quezon Ave" "Roxas Blvd-Kalaw"

Outputs (in --out-dir):
    dynamic_weights.csv   one row per road edge: distance_m, risk, dynamic_weight
    dynamic_weights.json  lambda, risk coverage, and the summary printed below

--compare is a verification harness, not the routing engine: it runs Dijkstra at
lambda = 0 and at the chosen lambda over the same graph to show that the risk penalty
actually moves the chosen route. The thesis routes with ACO.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.graph_data import DEFAULT_K, build_subgraph  # noqa: E402
from src.routing.dynamic_weight import (  # noqa: E402
    DEFAULT_LAMBDA,
    DEFAULT_MISSING_RISK,
    WeightedGraph,
    weighted_graph_from_risk_file,
)


def summarise(wg: WeightedGraph) -> dict:
    inflation = wg.weight / wg.distance          # 1 + lambda * risk
    return {
        "window": wg.window,
        "lambda": wg.lam,
        "nodes": wg.n_nodes,
        "edges": wg.n_edges,
        "scored_edges": wg.scored_edges,
        "risk_coverage": round(wg.coverage, 4),
        "risk_mean": round(float(wg.risk.mean()), 4),
        "risk_max": round(float(wg.risk.max()), 4),
        "distance_total_km": round(float(wg.distance.sum()) / 1000, 3),
        "weight_total_km": round(float(wg.weight.sum()) / 1000, 3),
        "cost_inflation_mean": round(float(inflation.mean()), 4),
        "cost_inflation_max": round(float(inflation.max()), 4),
    }


def compare_routes(wg: WeightedGraph, origin: int, destination: int) -> dict:
    """Shortest path under distance only vs under the dynamic weight."""
    import networkx as nx

    def route(graph_at_lambda: WeightedGraph) -> dict:
        g = graph_at_lambda.to_networkx()
        try:
            path = nx.shortest_path(g, origin, destination, weight="weight")
        except nx.NetworkXNoPath:
            return {"error": f"no route from {origin} to {destination}"}
        m = graph_at_lambda.evaluate_path(path)
        return {
            "hops": m.n_edges,
            "distance_m": round(m.distance_m, 2),
            "risk_exposure": round(m.risk_exposure, 4),
            "mean_risk": round(m.mean_risk, 4),
            "path": m.nodes,
        }

    static = route(wg.with_lambda(0.0))
    dynamic = route(wg)
    out = {"origin": origin, "destination": destination, "static": static, "dynamic": dynamic}

    if "error" not in static and "error" not in dynamic:
        out["distance_change_pct"] = round(
            100 * (dynamic["distance_m"] - static["distance_m"]) / static["distance_m"], 2
        )
        if static["risk_exposure"] > 0:
            out["risk_change_pct"] = round(
                100 * (dynamic["risk_exposure"] - static["risk_exposure"]) / static["risk_exposure"], 2
            )
        out["same_route"] = static["path"] == dynamic["path"]
    return out


def resolve_endpoint(wg: WeightedGraph, graph, name: str) -> int:
    """Accept either a CCTV intersection label or a raw road-network node id."""
    if name in graph.camera_nodes:
        return int(wg.node_ids[graph.camera_nodes[name]])
    try:
        node_id = int(name)
    except ValueError:
        raise SystemExit(
            f"unknown endpoint {name!r}; use a node id or one of: "
            f"{sorted(graph.camera_nodes)}"
        ) from None
    wg.index_of(node_id)          # raises if the node is outside the subgraph
    return node_id


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--risk-edges", required=True, type=Path, help="risk_edges.csv from the decoder")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs/dynamic_weights")
    p.add_argument("--spatial-dir", type=Path, default=REPO_ROOT / "data/processed/spatial")
    p.add_argument("-k", type=int, default=DEFAULT_K, help="subgraph radius; must match the checkpoint")
    p.add_argument("--lambda", dest="lam", type=float, default=DEFAULT_LAMBDA)
    p.add_argument("--window", default=None, help="window to weight (default: the latest)")
    p.add_argument("--missing-risk", type=float, default=DEFAULT_MISSING_RISK)
    p.add_argument("--compare", nargs=2, metavar=("ORIGIN", "DESTINATION"), default=None)
    a = p.parse_args()

    graph = build_subgraph(a.spatial_dir, a.k)
    wg = weighted_graph_from_risk_file(
        graph, a.risk_edges, window=a.window, lam=a.lam, missing_risk=a.missing_risk
    )

    summary = summarise(wg)
    if a.compare:
        origin = resolve_endpoint(wg, graph, a.compare[0])
        destination = resolve_endpoint(wg, graph, a.compare[1])
        summary["route_comparison"] = compare_routes(wg, origin, destination)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    wg.to_dataframe().to_csv(a.out_dir / "dynamic_weights.csv", index=False)
    (a.out_dir / "dynamic_weights.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    if wg.coverage < 1.0:
        print(
            f"\nWARNING: {wg.n_edges - wg.scored_edges} of {wg.n_edges} edges had no "
            f"predicted risk and fell back to {a.missing_risk}."
        )
    print(f"\nwrote {a.out_dir / 'dynamic_weights.csv'}")


if __name__ == "__main__":
    main()
