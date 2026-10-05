import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from src.data.graph_data import GraphData
from src.models.congestion_risk_score import edge_index_from_adjacency
from src.models.radr_stgnn import normalize_adjacency
from src.routing.dynamic_weight import build_weighted_graph, risk_vector
from src.routing.eta_engine import congested_eta, load_free_flow_seconds, path_eta_seconds

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
        camera_nodes={},
    )


def test_gamma_zero_reproduces_free_flow_time():
    free_flow = np.array([36.0, 54.0])
    assert congested_eta(free_flow, np.array([0.9, 0.1]), gamma=0.0) == pytest.approx(free_flow)


def test_higher_risk_and_higher_gamma_both_increase_eta():
    base = congested_eta(np.array([60.0]), np.array([0.5]), gamma=1.0)
    assert congested_eta(np.array([60.0]), np.array([0.9]), gamma=1.0) > base
    assert congested_eta(np.array([60.0]), np.array([0.5]), gamma=3.0) > base


@pytest.mark.parametrize(
    "free_flow, risk, gamma",
    [
        (np.array([60.0]), np.array([1.5]), 1.0),
        (np.array([60.0]), np.array([-0.1]), 1.0),
        (np.array([0.0]), np.array([0.5]), 1.0),
        (np.array([60.0]), np.array([0.5]), -1.0),
    ],
)
def test_rejects_inputs_that_would_corrupt_the_estimate(free_flow, risk, gamma):
    with pytest.raises(ValueError):
        congested_eta(free_flow, risk, gamma)


def test_load_free_flow_seconds_aligns_with_the_subgraph_by_node_id(tmp_path):
    full_node_order = np.array([999, 100, 101, 102, 103, 888])
    n = len(full_node_order)
    full_adj = sp.csr_matrix((6, 6), dtype=np.float64).tolil()
    full_tt = sp.csr_matrix((6, 6), dtype=np.float64).tolil()
    position = {int(nid): i for i, nid in enumerate(full_node_order)}
    for (u, v), length in EDGES.items():
        iu, iv = (position[int(NODE_IDS[u])], position[int(NODE_IDS[v])])
        full_adj[iu, iv] = length
        full_tt[iu, iv] = length / 10.0
    np.save(tmp_path / "metro_manila_node_order.npy", full_node_order)
    sp.save_npz(tmp_path / "metro_manila_travel_time.npz", full_tt.tocsr())
    graph = make_graph()
    free_flow = load_free_flow_seconds(tmp_path, graph)
    src, dst = (graph.edge_index[0].numpy(), graph.edge_index[1].numpy())
    expected = np.array(
        [
            EDGES[list(NODE_IDS).index(NODE_IDS[s]), list(NODE_IDS).index(NODE_IDS[d])] / 10.0
            for s, d in zip(src, dst)
        ]
    )
    assert free_flow == pytest.approx(expected)


def test_load_free_flow_seconds_rejects_a_subgraph_node_missing_from_the_full_file(tmp_path):
    full_node_order = np.array([100, 101, 102])
    np.save(tmp_path / "metro_manila_node_order.npy", full_node_order)
    sp.save_npz(tmp_path / "metro_manila_travel_time.npz", sp.csr_matrix((3, 3)))
    with pytest.raises(ValueError, match="not in"):
        load_free_flow_seconds(tmp_path, make_graph())


def test_load_free_flow_seconds_catches_a_misaligned_travel_time_matrix(tmp_path):
    full_node_order = NODE_IDS
    full_tt = sp.lil_matrix((4, 4), dtype=np.float64)
    full_tt[0, 1] = 999.0
    np.save(tmp_path / "metro_manila_node_order.npy", full_node_order)
    sp.save_npz(tmp_path / "metro_manila_travel_time.npz", full_tt.tocsr())
    with pytest.raises(ValueError, match="disagree on edge order"):
        load_free_flow_seconds(tmp_path, make_graph())


def test_path_eta_sums_congested_time_over_the_route():
    graph = make_graph()
    risk_rows = pd.DataFrame(
        {
            "window_start": "2026-05-25T17:30:00",
            "source_node_id": [100, 100, 101, 102],
            "target_node_id": [103, 101, 102, 103],
            "risk": [0.9, 0.1, 0.1, 0.1],
        }
    )
    risk, _ = risk_vector(graph, risk_rows)
    wg = build_weighted_graph(graph, risk, lam=2.0)
    free_flow = np.array(
        [
            {(0, 3): 60.0, (0, 1): 20.0, (1, 2): 20.0, (2, 3): 20.0}[int(wg.src[e]), int(wg.dst[e])]
            for e in range(wg.n_edges)
        ]
    )
    eta = path_eta_seconds(wg, free_flow, [100, 101, 102, 103], gamma=1.0)
    assert eta == pytest.approx(3 * 20.0 * 1.1)


def test_path_eta_needs_at_least_two_nodes():
    graph = make_graph()
    risk = np.zeros(graph.edge_index.shape[1])
    wg = build_weighted_graph(graph, risk)
    with pytest.raises(ValueError):
        path_eta_seconds(wg, np.ones(wg.n_edges), [100], gamma=1.0)