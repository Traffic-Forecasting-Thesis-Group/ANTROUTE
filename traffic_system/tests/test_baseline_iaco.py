import numpy as np
import pytest

from src.routing.baseline_iaco import (
    IacoConfig,
    IacoGraph,
    TrafficSnapshots,
    ahp_weights,
    comprehensive_cost,
    drive,
    extremum_normalise,
    fill_from_nearest_camera,
    iaco_dynamic_trip,
    iaco_search,
    rescale_pheromone,
    spatial_shortest_path,
    _transfer,
)


def test_ahp_reproduces_the_papers_weights_and_consistency():
    ahp = ahp_weights()
    assert ahp.weights == pytest.approx([0.637, 0.258, 0.105], abs=1e-3)
    assert ahp.lambda_max == pytest.approx(3.0385, abs=2e-3)
    assert ahp.cr == pytest.approx(0.0333, abs=2e-3)
    assert ahp.consistent


def test_ahp_rejects_a_non_reciprocal_matrix():
    with pytest.raises(ValueError):
        ahp_weights(np.array([[1.0, 3.0], [3.0, 1.0]]))


def test_extremum_normalise_maps_the_maximum_to_one():
    assert extremum_normalise(np.array([2.0, 4.0, 8.0])).tolist() == [0.25, 0.5, 1.0]


def test_comprehensive_cost_is_the_weighted_sum_of_normalised_criteria():
    s = comprehensive_cost(np.array([1.0, 2.0]), np.array([10.0, 10.0]), np.array([0.0, 4.0]))
    w = ahp_weights().weights
    assert s.tolist() == pytest.approx([w[0] * 0.5 + w[1], w[0] + w[1] + w[2]])


def test_fill_takes_the_nearest_cameras_value():
    # path 0 - 1 - 2 - 3 - 4, cameras at 0 (value 0.0) and 4 (value 1.0)
    src = np.array([0, 1, 2, 3])
    dst = np.array([1, 2, 3, 4])
    filled = fill_from_nearest_camera(src, dst, 5, {0: 0.0, 4: 1.0})
    # edge 0-1 touches camera 0, edge 3-4 touches camera 4; 1-2 is nearer 0, 2-3 nearer 4
    assert filled.tolist() == [0.0, 0.0, 1.0, 1.0]


def test_fill_averages_cameras_at_equal_distance_and_ignores_missing_ones():
    src = np.array([0, 1])
    dst = np.array([1, 2])
    filled = fill_from_nearest_camera(src, dst, 3, {0: 0.0, 2: 1.0, 1: float("nan")})
    assert filled.tolist() == [0.0, 1.0]
    middle = fill_from_nearest_camera(np.array([0, 1, 2, 3]), np.array([1, 2, 3, 4]), 5, {0: 0.0, 4: 1.0})
    assert middle[1] == 0.0 and middle[2] == 1.0


def test_eq9_is_applied_literally_by_default():
    tau = np.array([1.0, 1.0])
    rescale_pheromone(tau, np.array([10.0, 10.0]), np.array([20.0, 5.0]))
    assert tau.tolist() == [2.0, 0.5]
    tau = np.array([1.0, 1.0])
    rescale_pheromone(tau, np.array([10.0, 10.0]), np.array([20.0, 5.0]), invert=True)
    assert tau.tolist() == [0.5, 2.0]


def test_q0_of_one_always_exploits():
    rng = np.random.default_rng(0)
    config = IacoConfig(q0=1.0)
    tau = np.ones(3)
    picks = {_transfer(np.array([0, 1, 2]), tau, np.array([0.1, 0.9, 0.5]), config, rng) for _ in range(50)}
    assert picks == {1}


# Two equal-length routes: 0 -> 1 -> 3 and 0 -> 2 -> 3. Only traffic differs.
def fork_graph() -> IacoGraph:
    return IacoGraph(
        node_ids=np.array([10, 11, 12, 13]),
        src=np.array([0, 1, 0, 2]),
        dst=np.array([1, 3, 2, 3]),
        distance=np.array([100.0, 100.0, 100.0, 100.0]),
        lat=np.array([14.0, 14.001, 13.999, 14.0]),
        lon=np.array([121.0, 121.001, 121.001, 121.002]),
    )


def test_search_prefers_the_less_congested_branch():
    g = fork_graph()
    s = comprehensive_cost(g.distance, np.array([60.0, 60.0, 20.0, 20.0]), np.array([30.0, 30.0, 5.0, 5.0]))
    result = iaco_search(g, 0, 3, s, np.ones(g.n_edges), IacoConfig(seed=0), np.random.default_rng(0))
    assert [int(g.node_ids[i]) for i in result.path] == [10, 12, 13]
    assert result.cost == pytest.approx(s[2] + s[3])
    assert 1 <= result.converged_at <= 60
    assert len(result.curve) == 60


def test_spatial_shortest_path_ignores_traffic():
    g = fork_graph()
    g.distance[2] = 150.0
    assert spatial_shortest_path(g, 10, 13) == [10, 11, 13]


# 0 -> 1 is shared, then the vehicle chooses 1 -> 2 -> 4 or 1 -> 3 -> 4.
def two_stage_graph() -> IacoGraph:
    return IacoGraph(
        node_ids=np.array([0, 1, 2, 3, 4]),
        src=np.array([0, 1, 2, 1, 3]),
        dst=np.array([1, 2, 4, 3, 4]),
        distance=np.full(5, 100.0),
        lat=np.array([14.0, 14.001, 14.002, 14.0, 14.001]),
        lon=np.array([121.0, 121.001, 121.002, 121.002, 121.003]),
    )


def jam_after_departure() -> TrafficSnapshots:
    # At t = 10 s (vehicle still on 0 -> 1) the 2-branch jams and the 3-branch clears.
    before = np.array([20.0, 10.0, 10.0, 40.0, 40.0])
    after = np.array([20.0, 200.0, 200.0, 10.0, 10.0])
    return TrafficSnapshots(
        times=np.array([0.0, 10.0]),
        travel_time=np.stack([before, after]),
        flow=np.stack([np.array([5.0, 1.0, 1.0, 9.0, 9.0]), np.array([5.0, 9.0, 9.0, 1.0, 1.0])]),
        congestion=np.stack([np.array([0.5, 0.0, 0.0, 1.0, 1.0]), np.array([0.5, 1.0, 1.0, 0.0, 0.0])]),
    )


def test_drive_prices_each_edge_with_the_snapshot_in_force():
    g = two_stage_graph()
    seconds, exposure = drive(g, [0, 1, 2], departure=0.0, traffic=jam_after_departure())
    # 0 -> 1 uses snapshot 0 (20 s); the next two edges use snapshot 1 (200 s each)
    assert seconds == 420.0
    assert exposure == 0.5 + 1.0 + 1.0


def test_literal_replan_keeps_the_reinforced_route():
    # With literal eq. 9 the old route's pheromone is too strong, so it keeps the jammed route.
    g = two_stage_graph()
    trip = iaco_dynamic_trip(g, 0, 4, 0.0, jam_after_departure(), IacoConfig(seed=0))
    assert trip.replans == 1
    assert trip.path == [0, 1, 2, 4]
    assert trip.travel_time_s == 420.0


def test_replan_with_fresh_pheromone_avoids_the_jam():
    g = two_stage_graph()
    trip = iaco_dynamic_trip(g, 0, 4, 0.0, jam_after_departure(), IacoConfig(seed=0, reset_on_replan=True))
    assert trip.replans == 1
    assert trip.path == [0, 1, 3, 4]
    assert trip.travel_time_s == 40.0


def test_dynamic_trip_without_an_update_keeps_its_plan():
    g = two_stage_graph()
    traffic = jam_after_departure()
    static = TrafficSnapshots(traffic.times[:1], traffic.travel_time[:1], traffic.flow[:1], traffic.congestion[:1])
    trip = iaco_dynamic_trip(g, 0, 4, 0.0, static, IacoConfig(seed=0))
    assert trip.replans == 0
    assert trip.path == [0, 1, 2, 4]
    assert trip.travel_time_s == 40.0

# ---------------------------------------------------------------------------
# The paper's network (sec. 5.1): Figure 3 with Tables 5-7 (already normalised).
# ---------------------------------------------------------------------------
PAPER_EDGES = {  # (from, to): (length, travel time, traffic flow)
    ("O", "1"): (0.549, 0.659, 0.533), ("1", "4"): (0.704, 0.634, 0.615), ("3", "10"): (1.000, 1.000, 0.770),
    ("5", "6"): (0.282, 0.254, 0.256), ("7", "8"): (0.268, 0.241, 0.304), ("8", "13"): (0.394, 0.245, 0.488),
    ("11", "12"): (0.282, 0.338, 0.450), ("O", "3"): (0.718, 0.462, 0.347), ("2", "6"): (0.746, 0.987, 1.000),
    ("4", "5"): (0.282, 0.298, 0.254), ("5", "7"): (0.423, 0.634, 0.464), ("7", "12"): (0.437, 0.357, 0.687),
    ("9", "D"): (0.423, 0.585, 0.411), ("12", "13"): (0.310, 0.254, 0.251), ("1", "2"): (0.676, 0.704, 0.507),
    ("3", "4"): (0.563, 0.423, 0.450), ("4", "11"): (0.972, 0.921, 0.758), ("6", "8"): (0.451, 0.272, 0.303),
    ("8", "9"): (0.282, 0.241, 0.258), ("10", "11"): (0.563, 0.441, 0.488), ("13", "D"): (0.282, 0.220, 0.256),
}
PAPER_GRID = {"O": (0, 0), "1": (1, 0), "2": (3, 0), "3": (0, 1), "4": (1, 1), "5": (2, 1), "6": (3, 1),
              "7": (2, 2), "8": (3, 2), "9": (4, 2), "10": (0, 3), "11": (1, 3), "12": (2, 3), "13": (3, 3),
              "D": (4, 3)}
PAPER_NAMES = list(PAPER_GRID)


def paper_network():
    idx = {n: i for i, n in enumerate(PAPER_NAMES)}
    edges = list(PAPER_EDGES)
    d, t, n = (np.array([PAPER_EDGES[e][k] for e in edges]) for k in range(3))
    g = IacoGraph(
        node_ids=np.arange(len(PAPER_NAMES)),
        src=np.array([idx[u] for u, _ in edges]),
        dst=np.array([idx[v] for _, v in edges]),
        distance=d,
        # small grid near the equator, so haversine distance matches grid distance
        lat=np.array([-PAPER_GRID[k][1] * 1e-3 for k in PAPER_NAMES]),
        lon=np.array([PAPER_GRID[k][0] * 1e-3 for k in PAPER_NAMES]),
    )
    return g, idx, comprehensive_cost(d, t, n)


def paper_path_totals(route: str):
    g, idx, s = paper_network()
    edges = g.edges_of([idx[x] for x in route.split("-")])
    return g.distance[edges].sum(), s[edges].sum()


def test_reproduces_table_8_for_the_basic_aco_path():
    length, combined = paper_path_totals("O-1-4-5-7-8-13-D")
    assert length == pytest.approx(2.902, abs=5e-4)
    assert combined == pytest.approx(2.911, abs=5e-4)


def test_reproduces_table_8_for_the_improved_aco_path():
    length, combined = paper_path_totals("O-3-4-5-7-12-13-D")
    assert length == pytest.approx(3.015, abs=5e-4)
    # Paper prints 2.885; its own tables give 2.888.
    assert combined == pytest.approx(2.885, abs=5e-3)


def test_the_papers_reported_path_is_not_the_optimum_of_its_own_cost():
    # O-3-4-5-6-8-13-D (2.701) beats the paper's reported path (2.888), which ranks 5th of 15.
    assert paper_path_totals("O-3-4-5-6-8-13-D")[1] == pytest.approx(2.701, abs=5e-4)
    g, idx, s = paper_network()
    result = iaco_search(g, idx["O"], idx["D"], s, np.ones(g.n_edges),
                         IacoConfig(n_ants=23, n_iterations=100, q0=0.2, seed=0), np.random.default_rng(0))
    assert result.cost < paper_path_totals("O-3-4-5-7-12-13-D")[1]


def test_stable_from_reads_the_flat_tail_of_the_iteration_best_curve():
    from src.routing.baseline_iaco import stable_from
    assert stable_from([3.0, 2.9, 2.95, 2.88, 2.88, 2.88]) == 4
    assert stable_from([2.5]) == 1
    assert stable_from([]) is None
