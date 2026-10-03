from typing import List
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
    # A close cost gap (not weighted_risky_direct's wide one): the goal-directed heuristic
    # that makes ACO able to find long real-world routes (see aco_routing.remaining_cost_to)
    # also makes it strongly favor the better option at every junction, so a route that is
    # genuinely ~2x worse is (correctly) found with ~1 in 500,000 ant-tries, not something a
    # 300-try test budget should expect to see. A route that is only marginally worse should
    # still surface as a real alternative, which is what this checks.
    graph = make_graph()
    risk, _ = risk_vector(graph, risk_rows(direct=0.22001, detour=0.1))
    wg = build_weighted_graph(graph, risk, lam=2.0)
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


def long_chain_with_dead_end_traps(n_hops: int) -> GraphData:
    """
    A single n_hops-long route (node i -> i+1, cost 10 each) from node 0 to node n_hops, plus
    one dead-end trap edge off of every interior node (i -> a node with no further outgoing
    edges at all). This is the real failure pattern found on the actual Metro Manila subgraph:
    a 63-hop route where most nodes had 2-3 outgoing choices, one correct and the others
    leading nowhere near the destination -- without filtering out the unreachable options,
    an ant can wander onto one and simply dead-end (`_candidate_edges` returns empty), and a
    long enough route makes that happen to effectively every ant, every time.
    """
    node_ids = np.arange(2 * n_hops + 1, dtype=np.int64)  # 0..n_hops main chain, n_hops+1..2n_hops traps
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    for i in range(n_hops):
        rows.append(i)
        cols.append(i + 1)
        data.append(10.0)
        if i > 0:  # also give node i a dead-end trap (node i has no outgoing edges of its own)
            trap = n_hops + i
            rows.append(i)
            cols.append(trap)
            data.append(10.0)
    n = 2 * n_hops + 1
    adjacency = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)
    return GraphData(
        node_ids=node_ids,
        adjacency=adjacency,
        a_hat=normalize_adjacency(adjacency),
        edge_index=edge_index_from_adjacency(adjacency),
        camera_nodes={},
    )


def test_a_long_route_with_dead_end_branches_is_still_found():
    """
    Regression test for the real bug: ACO used to pick among a node's outgoing edges using
    only that edge's own local cost, with no idea whether the edge it was choosing could even
    reach the destination. On a 63-hop real route this meant it essentially never completed a
    single tour (confirmed against the actual Metro Manila subgraph: 0/500 single-ant tours
    succeeded). A 40-hop chain riddled with dead-end branches reproduces the same shape of
    failure at test scale.
    """
    graph = long_chain_with_dead_end_traps(n_hops=40)
    risk = np.zeros(graph.edge_index.shape[1])
    wg = build_weighted_graph(graph, risk, lam=0.0)
    config = AntColonyConfig(n_ants=10, n_iterations=20, seed=0)
    result = ant_colony_shortest_path(wg, 0, 40, config)
    assert result.best.nodes == list(range(41))
    assert result.best.dynamic_cost == pytest.approx(400.0)