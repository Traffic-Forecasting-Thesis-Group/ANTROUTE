import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from src.data.graph_data import GraphData
from src.models.congestion_risk_score import edge_index_from_adjacency
from src.models.radr_stgnn import normalize_adjacency
from src.routing.aco_routing import AntColonyConfig, ant_colony_shortest_path, multi_stop_route
from src.routing.dynamic_weight import build_weighted_graph, risk_vector

NODE_IDS = np.array([100, 101, 102, 103])
EDGES = {(0, 3): 500.0, (0, 1): 200.0, (1, 2): 200.0, (2, 3): 200.0}


def make_graph() -> GraphData:
    rows, cols = zip(*EDGES)
    adjacency = sp.csr_matrix(
        (list(EDGES.values()), (list(rows), list(cols))), shape=(4, 4), dtype=np.float64
    )
    return GraphData(
        node_ids=NODE_IDS,
        adjacency=adjacency,
        a_hat=normalize_adjacency(adjacency),
        edge_index=edge_index_from_adjacency(adjacency),
        camera_nodes={"origin": 0, "destination": 3},
    )


def risk_rows(direct: float, detour: float) -> pd.DataFrame:
    risks = {(100, 103): direct, (100, 101): detour, (101, 102): detour, (102, 103): detour}
    return pd.DataFrame(
        {
            "window_start": "2026-05-25T17:30:00",
            "source_node_id": [s for s, _ in risks],
            "target_node_id": [t for _, t in risks],
            "risk": list(risks.values()),
        }
    )


def weighted_risky_direct():
    graph = make_graph()
    risk, _ = risk_vector(graph, risk_rows(direct=0.9, detour=0.1))
    return build_weighted_graph(graph, risk, lam=2.0)


def test_aco_finds_the_same_optimum_dijkstra_would():
    wg = weighted_risky_direct()
    config = AntColonyConfig(n_ants=10, n_iterations=30, seed=0)
    result = ant_colony_shortest_path(wg, 100, 103, config)
    assert result.best.nodes == [100, 101, 102, 103]
    assert result.best.dynamic_cost == pytest.approx(3 * 200.0 * 1.2)


def test_the_only_other_route_appears_as_an_alternative():
    wg = weighted_risky_direct()
    config = AntColonyConfig(n_ants=10, n_iterations=30, seed=0)
    result = ant_colony_shortest_path(wg, 100, 103, config)
    alt_node_sets = [tuple(m.nodes) for m in result.alternatives]
    assert (100, 103) in alt_node_sets
    direct_metrics = next((m for m in result.alternatives if m.nodes == [100, 103]))
    assert direct_metrics.dynamic_cost > result.best.dynamic_cost


def test_lambda_zero_makes_aco_prefer_the_physically_shorter_route():
    wg = weighted_risky_direct().with_lambda(0.0)
    config = AntColonyConfig(n_ants=10, n_iterations=30, seed=0)
    result = ant_colony_shortest_path(wg, 100, 103, config)
    assert result.best.nodes == [100, 103]


def test_origin_equal_to_destination_is_rejected():
    wg = weighted_risky_direct()
    with pytest.raises(ValueError):
        ant_colony_shortest_path(wg, 100, 100, AntColonyConfig(n_iterations=5))


def test_an_unreachable_destination_raises_rather_than_returning_silently():
    rows, cols = ([0, 1], [1, 2])
    adjacency = sp.csr_matrix(([100.0, 100.0], (rows, cols)), shape=(4, 4), dtype=np.float64)
    graph = GraphData(
        node_ids=NODE_IDS,
        adjacency=adjacency,
        a_hat=normalize_adjacency(adjacency),
        edge_index=edge_index_from_adjacency(adjacency),
        camera_nodes={},
    )
    wg = build_weighted_graph(graph, np.zeros(2))
    with pytest.raises(ValueError, match="not reachable"):
        ant_colony_shortest_path(wg, 100, 103, AntColonyConfig(n_ants=5, n_iterations=5, seed=0))


def diamond_graph_with_two_equal_routes() -> GraphData:
    edges = {(0, 1): 100.0, (1, 3): 100.0, (0, 2): 100.0, (2, 3): 100.0}
    rows, cols = zip(*edges)
    adjacency = sp.csr_matrix(
        (list(edges.values()), (list(rows), list(cols))), shape=(4, 4), dtype=np.float64
    )
    return GraphData(
        node_ids=np.array([200, 201, 202, 203]),
        adjacency=adjacency,
        a_hat=normalize_adjacency(adjacency),
        edge_index=edge_index_from_adjacency(adjacency),
        camera_nodes={},
    )


def test_two_equally_good_routes_both_surface_as_best_and_alternative():
    graph = diamond_graph_with_two_equal_routes()
    wg = build_weighted_graph(graph, np.zeros(4), lam=2.0)
    config = AntColonyConfig(n_ants=20, n_iterations=40, seed=1)
    result = ant_colony_shortest_path(wg, 200, 203, config)
    found = {tuple(result.best.nodes)} | {tuple(a.nodes) for a in result.alternatives}
    assert (200, 201, 203) in found and (200, 202, 203) in found
    assert result.best.dynamic_cost == pytest.approx(200.0)


def test_multi_stop_concatenates_legs_without_duplicating_the_junction_node():
    wg = weighted_risky_direct().with_lambda(0.0)
    config = AntColonyConfig(n_ants=10, n_iterations=20, seed=0)
    result = multi_stop_route(wg, [100, 101, 103], config)
    assert result.best.nodes == [100, 101, 102, 103]
    assert result.best.n_edges == 3


def test_multi_stop_needs_at_least_two_stops():
    wg = weighted_risky_direct()
    with pytest.raises(ValueError):
        multi_stop_route(wg, [100], AntColonyConfig(n_iterations=5))