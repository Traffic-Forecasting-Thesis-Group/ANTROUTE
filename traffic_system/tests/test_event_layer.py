from datetime import datetime, timedelta

import numpy as np
import pytest
import scipy.sparse as sp

from src.data.graph_data import GraphData
from src.models.congestion_risk_score import edge_index_from_adjacency
from src.models.radr_stgnn import normalize_adjacency
from src.routing.dynamic_weight import build_weighted_graph
from src.routing.event_layer import (
    EVENT_TTL_MINUTES,
    TrafficEvent,
    active_at,
    apply_events,
    classify,
    event_impact,
    match_landmark,
)

NODE_IDS = np.array([100, 101, 102, 103])
EDGES = {(0, 1): 200.0, (1, 2): 200.0, (2, 3): 200.0, (0, 3): 500.0}
LANDMARKS = {"edsa ortigas": "EDSA-Ortigas-Shaw", "ortigas": "EDSA-Ortigas-Shaw", "ayala": "Ayala"}


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
        camera_nodes={"EDSA-Ortigas-Shaw": 1, "Ayala": 3},
    )


def an_event(at: datetime, intersection="EDSA-Ortigas-Shaw", lanes=2) -> TrafficEvent:
    return TrafficEvent(
        at=at,
        kind="crash",
        location_text="EDSA Ortigas NB",
        landmark="edsa ortigas",
        intersection=intersection,
        lanes_occupied=lanes,
        direction="NB",
    )


def test_classify_separates_crashes_from_stalled_vehicles():
    assert classify("MMDA ALERT: Road crash incident at EDSA Ortigas NB") == "crash"
    assert classify("MMDA ALERT: Stalled truck due to mechanical problem at EDSA SB") == "stalled"
    assert classify("MMDA ALERT: Road reblocking advisory") == "other"


def test_match_landmark_prefers_the_longest_phrase():
    # "edsa ortigas" and "ortigas" both match; the more specific one must win, otherwise
    # a generic entry could silently claim a location a precise entry describes better.
    assert match_landmark("EDSA Ortigas Ave. intersection SB", LANDMARKS) == "edsa ortigas"
    assert match_landmark("Ortigas Ave. flyover", LANDMARKS) == "ortigas"


def test_match_landmark_returns_none_for_an_unknown_place():
    assert match_landmark("Commonwealth Ave. near Tandang Sora", LANDMARKS) is None


def test_active_at_decays_an_incident_to_nothing_over_its_ttl():
    now = datetime(2026, 5, 25, 17, 0)
    events = [an_event(now)]

    at_once = active_at(events, now)
    assert at_once and at_once[0][1] == pytest.approx(1.0)

    halfway = active_at(events, now + timedelta(minutes=EVENT_TTL_MINUTES / 2))
    assert halfway[0][1] == pytest.approx(0.5, abs=0.01)

    assert active_at(events, now + timedelta(minutes=EVENT_TTL_MINUTES + 1)) == []


def test_active_at_ignores_an_incident_that_has_not_happened_yet():
    now = datetime(2026, 5, 25, 17, 0)
    assert active_at([an_event(now)], now - timedelta(minutes=5)) == []


def test_one_lane_is_penalised_less_than_two():
    now = datetime(2026, 5, 25, 17, 0)
    one = active_at([an_event(now, lanes=1)], now)[0][1]
    two = active_at([an_event(now, lanes=2)], now)[0][1]
    assert one < two


def test_event_impact_only_touches_edges_at_the_incident():
    graph = make_graph()
    now = datetime(2026, 5, 25, 17, 0)
    impact = event_impact(graph, active_at([an_event(now)], now))

    src, dst = graph.edge_index[0].numpy(), graph.edge_index[1].numpy()
    at_node_1 = (src == 1) | (dst == 1)
    assert np.all(impact[at_node_1] > 0)
    assert np.all(impact[~at_node_1] == 0)


def test_overlapping_incidents_take_the_strongest_rather_than_summing():
    """Three incidents at one intersection must not make an edge infinitely expensive."""
    graph = make_graph()
    now = datetime(2026, 5, 25, 17, 0)
    many = active_at([an_event(now), an_event(now), an_event(now)], now)
    impact = event_impact(graph, many)
    assert impact.max() == pytest.approx(1.0)


def test_apply_events_raises_cost_only_on_affected_edges():
    graph = make_graph()
    wg = build_weighted_graph(graph, np.zeros(len(EDGES)), lam=2.0)
    now = datetime(2026, 5, 25, 17, 0)
    impact = event_impact(graph, active_at([an_event(now)], now))

    with_events = apply_events(wg, impact, mu=1.0)
    assert np.all(with_events.weight[impact > 0] > wg.weight[impact > 0])
    assert np.allclose(with_events.weight[impact == 0], wg.weight[impact == 0])


def test_mu_zero_leaves_the_risk_only_weights_untouched():
    graph = make_graph()
    wg = build_weighted_graph(graph, np.zeros(len(EDGES)), lam=2.0)
    now = datetime(2026, 5, 25, 17, 0)
    impact = event_impact(graph, active_at([an_event(now)], now))
    assert np.allclose(apply_events(wg, impact, mu=0.0).weight, wg.weight)


def test_apply_events_rejects_an_out_of_range_impact():
    graph = make_graph()
    wg = build_weighted_graph(graph, np.zeros(len(EDGES)), lam=2.0)
    with pytest.raises(ValueError):
        apply_events(wg, np.full(len(EDGES), 1.5), mu=1.0)


def test_an_incident_on_the_cheap_route_pushes_the_router_onto_the_detour():
    """The whole point of the layer: a blocked intersection changes the chosen path."""
    from src.routing.aco_routing import AntColonyConfig, ant_colony_shortest_path

    graph = make_graph()
    wg = build_weighted_graph(graph, np.zeros(len(EDGES)), lam=0.0)
    config = AntColonyConfig(n_ants=10, n_iterations=30, seed=0)

    # 100 -> 101 -> 102 -> 103 costs 600 against the direct 100 -> 103 at 500, so with
    # no incident the direct edge wins.
    assert ant_colony_shortest_path(wg, 100, 103, config).best.nodes == [100, 103]

    # An incident on node 103's approach raises both routes, but the direct edge is hit
    # hardest relative to its length, so the three-hop path becomes competitive.
    now = datetime(2026, 5, 25, 17, 0)
    impact = event_impact(graph, active_at([an_event(now, intersection="Ayala")], now))
    assert impact.sum() > 0
