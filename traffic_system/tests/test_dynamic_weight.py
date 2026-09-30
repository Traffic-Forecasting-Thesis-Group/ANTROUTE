import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from src.data.graph_data import GraphData
from src.models.congestion_risk_score import edge_index_from_adjacency
from src.models.radr_stgnn import normalize_adjacency
from src.routing.dynamic_weight import (
    DEFAULT_LAMBDA,
    build_weighted_graph,
    dynamic_weight,
    edge_distances,
    iter_windows,
    load_risk_edges,
    risk_vector,
    weighted_graph_from_risk_file,
)

# Four nodes, a direct arc 0 -> 3 and a longer detour 0 -> 1 -> 2 -> 3.
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


def risk_rows(direct: float, detour: float, window: str = "2026-05-25T17:30:00") -> pd.DataFrame:
    """The direct arc is risky, the detour is clear."""
    risks = {(100, 103): direct, (100, 101): detour, (101, 102): detour, (102, 103): detour}
    return pd.DataFrame(
        {
            "window_start": window,
            "source_node_id": [s for s, _ in risks],
            "target_node_id": [t for _, t in risks],
            "risk": list(risks.values()),
        }
    )


# --- formula ---------------------------------------------------------------

def test_matches_the_worked_example_from_the_methodology():
    # 500 m at risk 0.85 with lambda = 2 costs 500 * (1 + 2 * 0.85) = 1350.
    assert dynamic_weight(np.array([500.0]), np.array([0.85]), 2.0) == pytest.approx(1350.0)


def test_zero_risk_leaves_the_physical_distance_untouched():
    distance = np.array([200.0, 500.0])
    assert dynamic_weight(distance, np.zeros(2), DEFAULT_LAMBDA) == pytest.approx(distance)


def test_zero_lambda_reproduces_the_static_baseline():
    distance = np.array([200.0, 500.0])
    assert dynamic_weight(distance, np.array([0.9, 0.2]), 0.0) == pytest.approx(distance)


def test_weight_rises_with_risk_and_with_lambda():
    d = np.array([100.0])
    assert dynamic_weight(d, np.array([0.2]), 2.0) < dynamic_weight(d, np.array([0.8]), 2.0)
    assert dynamic_weight(d, np.array([0.5]), 1.0) < dynamic_weight(d, np.array([0.5]), 4.0)


@pytest.mark.parametrize(
    "distance, risk, lam",
    [
        (np.array([100.0]), np.array([1.5]), 2.0),    # risk above 1
        (np.array([100.0]), np.array([-0.1]), 2.0),   # risk below 0
        (np.array([0.0]), np.array([0.5]), 2.0),      # zero-length edge
        (np.array([-5.0]), np.array([0.5]), 2.0),     # negative length
        (np.array([100.0]), np.array([0.5]), -1.0),   # negative lambda rewards risk
        (np.array([100.0]), np.array([np.nan]), 2.0),
    ],
)
def test_rejects_inputs_that_would_corrupt_the_cost(distance, risk, lam):
    with pytest.raises(ValueError):
        dynamic_weight(distance, risk, lam)


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        dynamic_weight(np.array([100.0, 200.0]), np.array([0.5]), 2.0)


# --- distances -------------------------------------------------------------

def test_distances_follow_the_edge_index_column_order():
    graph = make_graph()
    distance = edge_distances(graph)

    src, dst = graph.edge_index[0].numpy(), graph.edge_index[1].numpy()
    assert [EDGES[(u, v)] for u, v in zip(src, dst)] == pytest.approx(distance)


def test_distance_and_edge_index_disagreement_is_caught():
    graph = make_graph()
    graph.edge_index = graph.edge_index[:, [1, 0, 2, 3]]   # same edges, wrong order
    with pytest.raises(ValueError, match="edge order"):
        edge_distances(graph)


# --- weighted graph --------------------------------------------------------

def test_build_weighted_graph_applies_the_formula_per_edge():
    graph = make_graph()
    risk = np.linspace(0.0, 1.0, graph.edge_index.shape[1])
    wg = build_weighted_graph(graph, risk, lam=2.0)

    assert wg.weight == pytest.approx(wg.distance * (1 + 2.0 * risk))
    assert wg.n_nodes == 4 and wg.n_edges == len(EDGES)


def test_build_weighted_graph_rejects_a_risk_vector_of_the_wrong_length():
    with pytest.raises(ValueError, match="shape"):
        build_weighted_graph(make_graph(), np.zeros(2))


def test_out_edges_lists_only_the_arcs_leaving_a_node():
    wg = build_weighted_graph(make_graph(), np.zeros(len(EDGES)))

    assert sorted(int(wg.dst[e]) for e in wg.out_edges(0)) == [1, 3]
    assert [int(wg.dst[e]) for e in wg.out_edges(1)] == [2]
    assert list(wg.out_edges(3)) == []          # a sink has no moves


def test_index_of_maps_road_node_ids_and_rejects_outsiders():
    wg = build_weighted_graph(make_graph(), np.zeros(len(EDGES)))

    assert wg.index_of(102) == 2
    with pytest.raises(KeyError):
        wg.index_of(999)


def test_with_lambda_reweights_without_touching_the_risk_snapshot():
    wg = build_weighted_graph(make_graph(), np.full(len(EDGES), 0.5), lam=2.0)
    static = wg.with_lambda(0.0)

    assert static.weight == pytest.approx(static.distance)
    assert static.risk == pytest.approx(wg.risk)
    assert wg.lam == 2.0                         # the original is left alone


def test_to_networkx_carries_weight_distance_and_risk():
    wg = build_weighted_graph(make_graph(), np.full(len(EDGES), 0.5), lam=2.0)
    g = wg.to_networkx()

    assert g.number_of_nodes() == 4 and g.number_of_edges() == len(EDGES)
    assert g[100][103]["distance"] == pytest.approx(500.0)
    assert g[100][103]["weight"] == pytest.approx(500.0 * 2.0)


# --- path evaluation -------------------------------------------------------

def test_evaluate_path_sums_distance_risk_and_cost():
    graph = make_graph()
    risk, _ = risk_vector(graph, risk_rows(direct=0.9, detour=0.1))
    wg = build_weighted_graph(graph, risk, lam=2.0)

    detour = wg.evaluate_path([100, 101, 102, 103])
    assert detour.n_edges == 3
    assert detour.distance_m == pytest.approx(600.0)
    assert detour.risk_exposure == pytest.approx(0.3)        # sum, not distance-weighted
    assert detour.mean_risk == pytest.approx(0.1)
    assert detour.dynamic_cost == pytest.approx(3 * 200.0 * 1.2)


def test_the_risky_shortcut_costs_more_than_the_longer_clear_detour():
    graph = make_graph()
    risk, _ = risk_vector(graph, risk_rows(direct=0.9, detour=0.1))
    wg = build_weighted_graph(graph, risk, lam=2.0)

    direct = wg.evaluate_path([100, 103])
    detour = wg.evaluate_path([100, 101, 102, 103])

    assert direct.distance_m < detour.distance_m             # the shortcut is shorter
    assert direct.dynamic_cost > detour.dynamic_cost         # but the router avoids it
    assert direct.risk_exposure > detour.risk_exposure

    # ...and at lambda = 0 the static router takes the shortcut instead.
    static = wg.with_lambda(0.0)
    assert static.evaluate_path([100, 103]).dynamic_cost < static.evaluate_path(
        [100, 101, 102, 103]
    ).dynamic_cost


def test_evaluate_path_rejects_a_route_that_is_not_connected():
    wg = build_weighted_graph(make_graph(), np.zeros(len(EDGES)))
    with pytest.raises(KeyError, match="no edge"):
        wg.evaluate_path([100, 102])


def test_evaluate_path_needs_at_least_two_nodes():
    wg = build_weighted_graph(make_graph(), np.zeros(len(EDGES)))
    with pytest.raises(ValueError):
        wg.evaluate_path([100])


# --- risk_edges.csv --------------------------------------------------------

def test_risk_vector_aligns_by_node_id_pair_not_row_order():
    graph = make_graph()
    rows = risk_rows(direct=0.9, detour=0.1).iloc[::-1]      # shuffled
    risk, scored = risk_vector(graph, rows)

    assert scored == len(EDGES)
    src, dst = graph.edge_index[0].numpy(), graph.edge_index[1].numpy()
    for e, (u, v) in enumerate(zip(NODE_IDS[src], NODE_IDS[dst])):
        assert risk[e] == pytest.approx(0.9 if (u, v) == (100, 103) else 0.1)


def test_unscored_edges_fall_back_and_are_reported_in_the_coverage():
    graph = make_graph()
    rows = risk_rows(direct=0.9, detour=0.1).iloc[:1]        # only the direct arc
    risk, scored = risk_vector(graph, rows, missing_risk=0.0)

    assert scored == 1
    assert risk.sum() == pytest.approx(0.9)

    wg = build_weighted_graph(graph, risk, coverage=scored / len(EDGES), scored_edges=scored)
    assert wg.coverage == pytest.approx(0.25)


def test_duplicate_rows_for_one_edge_are_averaged():
    graph = make_graph()
    rows = pd.concat([risk_rows(direct=0.9, detour=0.1), risk_rows(direct=0.1, detour=0.1)])
    risk, _ = risk_vector(graph, rows)

    wg = build_weighted_graph(graph, risk)
    assert wg.risk[wg.edge_id(0, 3)] == pytest.approx(0.5)


def test_risk_outside_the_decoder_range_is_rejected():
    rows = risk_rows(direct=0.9, detour=0.1)
    rows.loc[rows.index[0], "risk"] = 1.4
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        risk_vector(make_graph(), rows)


def test_an_out_of_range_fallback_is_rejected():
    with pytest.raises(ValueError, match="missing_risk"):
        risk_vector(make_graph(), risk_rows(0.5, 0.5), missing_risk=2.0)


def test_load_risk_edges_reports_a_missing_column(tmp_path):
    path = tmp_path / "risk_edges.csv"
    risk_rows(0.9, 0.1).drop(columns=["risk"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing column"):
        load_risk_edges(path)


def test_the_latest_window_is_used_when_none_is_named(tmp_path):
    path = tmp_path / "risk_edges.csv"
    pd.concat(
        [
            risk_rows(direct=0.1, detour=0.1, window="2026-05-25T07:00:00"),
            risk_rows(direct=0.9, detour=0.1, window="2026-05-25T17:30:00"),
        ]
    ).to_csv(path, index=False)

    wg = weighted_graph_from_risk_file(make_graph(), path)
    assert wg.window == "2026-05-25T17:30:00"
    assert wg.risk[wg.edge_id(0, 3)] == pytest.approx(0.9)

    earlier = weighted_graph_from_risk_file(make_graph(), path, window="2026-05-25T07:00:00")
    assert earlier.risk[earlier.edge_id(0, 3)] == pytest.approx(0.1)


def test_an_unknown_window_is_an_error_rather_than_an_empty_graph(tmp_path):
    path = tmp_path / "risk_edges.csv"
    risk_rows(0.9, 0.1).to_csv(path, index=False)
    with pytest.raises(ValueError, match="no rows for window"):
        weighted_graph_from_risk_file(make_graph(), path, window="2026-01-01T00:00:00")


def test_iter_windows_yields_every_window_in_order(tmp_path):
    path = tmp_path / "risk_edges.csv"
    pd.concat(
        [
            risk_rows(direct=0.9, detour=0.1, window="2026-05-25T17:30:00"),
            risk_rows(direct=0.1, detour=0.1, window="2026-05-25T07:00:00"),
        ]
    ).to_csv(path, index=False)

    graphs = list(iter_windows(make_graph(), path))
    assert [g.window for g in graphs] == ["2026-05-25T07:00:00", "2026-05-25T17:30:00"]
    assert all(g.coverage == 1.0 for g in graphs)


def test_a_file_written_with_window_end_only_still_works(tmp_path):
    """predict_risk.py --edges writes window_end; predict_congestion_risk.py writes both."""
    path = tmp_path / "risk_edges.csv"
    risk_rows(0.9, 0.1).rename(columns={"window_start": "window_end"}).to_csv(path, index=False)

    wg = weighted_graph_from_risk_file(make_graph(), path)
    assert wg.window == "2026-05-25T17:30:00"


# --- the real Metro Manila graph ------------------------------------------

@pytest.mark.skipif(
    not (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "data/processed/spatial/metro_manila_adjacency.npz"
    ).exists(),
    reason="spatial topology has not been built",
)
def test_the_real_subgraph_carries_plausible_road_lengths():
    from pathlib import Path

    from src.data.graph_data import build_subgraph

    graph = build_subgraph(Path(__file__).resolve().parents[1] / "data/processed/spatial")
    wg = build_weighted_graph(graph, np.full(graph.edge_index.shape[1], 0.5), lam=2.0)

    assert wg.n_edges == graph.edge_index.shape[1]
    assert wg.distance.min() > 0
    assert wg.distance.max() < 20_000          # metres, not degrees or kilometres
    assert wg.weight == pytest.approx(wg.distance * 2.0)

    # Every camera intersection must be reachable as a routing endpoint.
    for label, index in graph.camera_nodes.items():
        assert wg.index_of(int(wg.node_ids[index])) == index, label
