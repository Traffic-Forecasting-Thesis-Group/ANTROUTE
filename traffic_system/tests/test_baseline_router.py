import numpy as np
import pandas as pd
import pytest

from src.routing.baseline_iaco import IacoConfig, IacoGraph, ahp_weights, iaco_search
from src.routing.baseline_router import (
    IACO,
    MIXED,
    SHORTEST_DISTANCE,
    BaselineNoRouteError,
    build_inputs,
    camera_flow,
    camera_observations,
    load_vehicle_counts,
    plan_baseline,
)

# Always take the best-looking road, so a run is deterministic.
GREEDY = IacoConfig(n_ants=3, n_iterations=2, q0=1.0, seed=0)
# The same colony as the paper wrote it: an ant at a dead end dies.
LITERAL = IacoConfig(n_ants=3, n_iterations=2, q0=1.0, seed=0, dead_end_recovery=False)


def graph(node_ids, edges, coords, distance=None):
    """IacoGraph from (u, v) node-id pairs and {node_id: (x, y)} grid coordinates."""
    position = {n: i for i, n in enumerate(node_ids)}
    return IacoGraph(
        node_ids=np.array(node_ids),
        src=np.array([position[u] for u, _ in edges]),
        dst=np.array([position[v] for _, v in edges]),
        distance=np.full(len(edges), 100.0) if distance is None else np.asarray(distance, dtype=float),
        # tiny grid near the equator, so haversine distance tracks grid distance
        lat=np.array([coords[n][1] * 1e-3 for n in node_ids]),
        lon=np.array([coords[n][0] * 1e-3 for n in node_ids]),
    )


def dead_end_trap():
    """
    0 -> 1 is a cul-de-sac pointing straight at the destination (its only exit is back to 0);
    0 -> 2 -> 3 is the real way round. The paper's greedy rule always prefers 1, and under its
    no-revisit rule an ant at 1 has nowhere to go -- so IACO, unaided, can never arrive.
    """
    edges = [(0, 1), (1, 0), (0, 2), (2, 3)]
    coords = {0: (0, 0), 1: (0, 1.5), 2: (1, 1), 3: (0, 2)}
    return graph([0, 1, 2, 3], edges, coords)


def free_inputs(g, seconds=10.0):
    return build_inputs(g, np.full(g.n_edges, seconds), camera_congestion={})


def test_the_trap_really_defeats_unaided_iaco():
    # Guards the fixture: if this ever finds a route, the fallback tests below prove nothing.
    g = dead_end_trap()
    s = free_inputs(g).s
    assert iaco_search(g, 0, 3, s, np.ones(g.n_edges), LITERAL, np.random.default_rng(0)) is None


def test_dead_end_recovery_lets_iaco_escape_the_trap():
    route = plan_baseline(free_inputs(dead_end_trap()), [0, 3], GREEDY)
    assert route.nodes == [0, 2, 3]
    assert route.algorithm == IACO and route.fallback_reason is None


def test_iaco_route_is_used_when_an_ant_arrives():
    g = graph([0, 1, 2], [(0, 1), (1, 2)], {0: (0, 0), 1: (1, 0), 2: (2, 0)})
    route = plan_baseline(free_inputs(g, seconds=12.0), [0, 2], GREEDY)
    assert route.nodes == [0, 1, 2]
    assert route.algorithm == IACO and route.fallback_reason is None
    assert route.legs_by_iaco == route.legs == 1
    assert route.travel_time_s == pytest.approx(24.0)


def test_corridor_search_reports_edges_of_the_whole_graph():
    # Far-off roads listed first, so a corridor's own edge numbering differs from g's.
    edges = [(3, 4), (4, 3), (0, 1), (1, 2)]
    coords = {0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (0, 40), 4: (1, 40)}
    g = graph([0, 1, 2, 3, 4], edges, coords)
    inputs = build_inputs(g, np.array([5.0, 5.0, 12.0, 13.0]), camera_congestion={})
    route = plan_baseline(inputs, [0, 2], GREEDY, corridor_margin=0.2)
    assert route.nodes == [0, 1, 2]
    assert route.edges == [2, 3]
    assert route.travel_time_s == pytest.approx(25.0)


@pytest.mark.parametrize("detour_y", [5, 8])  # reached by widening; only by the whole graph
def test_corridor_widens_until_it_joins_the_stops(detour_y):
    g = graph([0, 1, 2], [(0, 1), (1, 2)], {0: (0, 0), 1: (1, detour_y), 2: (2, 0)})
    route = plan_baseline(free_inputs(g), [0, 2], GREEDY, corridor_margin=0.2)
    assert route.nodes == [0, 1, 2]
    assert route.algorithm == IACO


def test_failed_iaco_falls_back_to_shortest_distance_and_says_so():
    route = plan_baseline(free_inputs(dead_end_trap()), [0, 3], LITERAL, fallback=True)
    assert route.nodes == [0, 2, 3]
    assert route.algorithm == SHORTEST_DISTANCE
    assert route.legs_by_iaco == 0
    assert "found no route" in route.fallback_reason and "shortest-distance" in route.fallback_reason


def test_without_the_fallback_a_failed_search_raises():
    with pytest.raises(BaselineNoRouteError, match="found no route"):
        plan_baseline(free_inputs(dead_end_trap()), [0, 3], LITERAL)


def test_multi_stop_routes_leg_by_leg_and_reports_a_mixed_result():
    # The trap as the first leg, then a plain 3 -> 4 leg IACO can do.
    edges = [(0, 1), (1, 0), (0, 2), (2, 3), (3, 4)]
    coords = {0: (0, 0), 1: (0, 1.5), 2: (1, 1), 3: (0, 2), 4: (0, 3)}
    g = graph([0, 1, 2, 3, 4], edges, coords)
    route = plan_baseline(free_inputs(g), [0, 3, 4], LITERAL, fallback=True)
    assert route.nodes == [0, 2, 3, 4]
    assert route.algorithm == MIXED
    assert (route.legs_by_iaco, route.legs) == (1, 2)
    assert "1 of 2 legs" in route.fallback_reason


def test_repeated_stops_are_skipped_and_an_empty_trip_is_rejected():
    g = graph([0, 1], [(0, 1)], {0: (0, 0), 1: (1, 0)})
    assert plan_baseline(free_inputs(g), [0, 0, 1], GREEDY).nodes == [0, 1]
    with pytest.raises(ValueError, match="same node"):
        plan_baseline(free_inputs(g), [1, 1], GREEDY)


def test_observed_congestion_slows_travel_time_and_raises_cost():
    g = graph([0, 1, 2], [(0, 1), (1, 2)], {0: (0, 0), 1: (1, 0), 2: (2, 0)})
    calm = build_inputs(g, np.array([10.0, 10.0]), camera_congestion={})
    jam = build_inputs(g, np.array([10.0, 10.0]), camera_congestion={0: 1.0}, gamma=1.0)
    assert calm.travel_time.tolist() == [10.0, 10.0]          # nothing observed: free flow
    assert jam.travel_time.tolist() == [20.0, 20.0]           # level 1.0 at gamma 1 doubles it
    assert jam.cameras_observed == 1 and calm.cameras_observed == 0


def test_without_vehicle_counts_the_flow_term_is_zero():
    g = graph([0, 1, 2], [(0, 1), (1, 2)], {0: (0, 0), 1: (1, 0), 2: (2, 0)}, distance=[50.0, 100.0])
    inputs = build_inputs(g, np.array([10.0, 20.0]), camera_congestion={})
    w = ahp_weights().weights
    assert not inputs.flow_observed
    assert inputs.s.tolist() == pytest.approx([w[0] * 0.5 + w[1] * 0.5, w[0] + w[1]])


def test_camera_observations_average_per_camera_and_skip_edges_it_cannot_own():
    node_ids = np.array([100, 101, 102, 103])
    cameras = [0, 3]                       # nodes 100 and 103
    rows = pd.DataFrame({
        "source_node_id": [100, 101, 101, 100, 102],
        "target_node_id": [101, 100, 102, 103, 103],
        "weak_target":    [0.5, 1.0, 0.0, 1.0, np.nan],
    })
    # 100-101 twice -> camera 0 averages 0.5 and 1.0; 101-102 touches no camera; 100-103
    # touches two; 102-103 has no observation.
    assert camera_observations(rows, node_ids, cameras) == {0: 0.75}


def test_vehicle_counts_become_the_flow_term(tmp_path):
    labels = tmp_path / "auto_labels.csv"
    pd.DataFrame({
        "camera_id": ["camA", "camA", "camA", "camB"],
        "timestamp": ["2026-05-25T17:00:00", "2026-05-25T17:10:00", "2026-05-25T18:00:00", "2026-05-25T17:05:00"],
        "n_vehicles": [10, 20, 99, 4],
    }).to_csv(labels, index=False)
    camera_csv = tmp_path / "camera_nodes.csv"
    pd.DataFrame({"camera_id": ["camA", "camB"], "intersection": ["A", "B"]}).to_csv(camera_csv, index=False)

    counts = load_vehicle_counts([labels], {"A": 0, "B": 2}, camera_csv)
    flow = camera_flow(counts, pd.Timestamp("2026-05-25T17:00:00"), pd.Timestamp("2026-05-25T17:30:00"))
    assert flow == {0: 15.0, 2: 4.0}           # the 18:00 frame is outside the window

    g = graph([0, 1, 2], [(0, 1), (1, 2)], {0: (0, 0), 1: (1, 0), 2: (2, 0)})
    inputs = build_inputs(g, np.array([10.0, 10.0]), camera_congestion={}, camera_flow=flow)
    w = ahp_weights().weights
    assert inputs.flow_observed
    # edge 0-1 takes camera 0's 15, edge 1-2 camera 2's 4; flow is divided by its max (15)
    assert inputs.s.tolist() == pytest.approx([w[0] + w[1] + w[2], w[0] + w[1] + w[2] * 4 / 15])
