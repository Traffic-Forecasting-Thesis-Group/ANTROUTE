"""Shared ACO kernel: stopping rule, deadlines, outcomes, reproducibility and the exact reference."""
import pytest

from src.routing.aco_kernel import (NoFeasibleRoute, SearchConfig, SearchTimeLimit, aco_search, converged,
                                    exact_shortest_path)
from src.routing.condition_update import Conditions
from src.routing.dynamic_weight import WeightedRoad, prepare_graph
from src.routing.historical_traffic import bpr_time
from src.routing.network import Road


def graph(network, **conditions):
    return prepare_graph(network, Conditions(time_of_day=8, weather="clear", **conditions))


def positions(network):
    return {node.id: (node.longitude, node.latitude) for node in network.nodes}


def test_cumulative_patience_window():
    # Item 15: [100, 99.5, 99, 98.5], p = 3, abs 0, rel 0.01: 100 - 98.5 = 1.5 > 0.01 * 100, so continue.
    assert not converged([100, 99.5, 99, 98.5], 3, 0, 0.01)
    # Consecutive small improvements accumulate: each 0.5 step alone is below 1, the window total is not.
    assert converged([100, 99.8, 99.6, 99.4], 3, 0, 0.01)  # 0.6 <= 1
    assert not converged([100, 99.5, 99], 3, 0, 0.01)  # The window needs B[k - p].
    assert not converged([None, None, None, 5], 3, 0, 0)  # No route at the window start: not comparable.
    assert converged([5, 5, 5, 5], 3, 0, 0)  # Zero tolerances: no improvement over p iterations.


def test_bpr_time_with_independent_demand():
    # Item 15: T0 = 10 min, q = 1000 veh/h; K = 2000 -> 10.09375 min; K = 1000 -> 11.5 min (capacity cut raises time).
    assert bpr_time(10, 1000, 2000) == pytest.approx(10.09375)
    assert bpr_time(10, 1000, 1000) == pytest.approx(11.5)
    with pytest.raises(ValueError):
        bpr_time(10, 1000, 0)


def test_deadline_with_and_without_a_route(network, monkeypatch):
    import src.routing.aco_kernel as kernel
    clock = iter(range(0, 10_000))
    # Each perf_counter() call advances one second; the 2.5 s budget runs out during the search.
    monkeypatch.setattr(kernel, "perf_counter", lambda: float(next(clock)))
    adjacency = graph(network)
    with pytest.raises(SearchTimeLimit):
        aco_search(adjacency, "a", "d", SearchConfig(wall_time_ms=500), positions(network), 1e9, started=0.0)
    clock = iter(range(0, 10_000))
    # 10 s: enough clock ticks for the first ants to finish a route, too few for 20 iterations of patience.
    result = aco_search(adjacency, "a", "d", SearchConfig(wall_time_ms=10_000, n_ants=2), positions(network), 1e9, started=0.0)
    assert result.stop_reason == "time_limit" and result.roads  # Best route so far, labelled.
    assert result.converged_at is None


def test_disconnection_is_a_separate_outcome(network):
    with pytest.raises(NoFeasibleRoute):
        aco_search(graph(network, closed_road_ids={"ab", "ad"}), "a", "d", SearchConfig(), positions(network), 1e9)


def test_records_and_reproducibility(network):
    adjacency = graph(network)
    first = aco_search(adjacency, "a", "d", SearchConfig(seed=3), positions(network), 1e9)
    again = aco_search(adjacency, "a", "d", SearchConfig(seed=3), positions(network), 1e9)
    assert first.best_cost_by_iteration == again.best_cost_by_iteration and first.pheromone == again.pheromone
    assert first.stop_reason == "converged" and first.iterations_completed == first.converged_at
    assert first.first_best_found_iteration >= 1 and first.search_time_ms >= 0
    assert all(value >= SearchConfig().pheromone_floor for value in first.pheromone.values())


def test_aco_matches_the_exact_reference_and_keeps_parallel_edges(network):
    # A parallel a -> b road with its own ID and a lower cost: both the exact reference and ACO must use it by ID.
    adjacency = graph(network)
    original = next(edge for edge in adjacency["a"] if edge.road.id == "ab")
    parallel = WeightedRoad(Road.model_construct(**{**original.road.__dict__, "id": "ab-2"}), original.cost / 2)
    adjacency["a"].append(parallel)
    exact = exact_shortest_path(adjacency, "a", "d")
    assert [edge.road.id for edge in exact] == ["ab-2", "bd"]
    found = aco_search(adjacency, "a", "d", SearchConfig(), positions(network), 1e9)
    assert [edge.road.id for edge in found.roads] == ["ab-2", "bd"]
    assert sum(edge.cost for edge in found.roads) == pytest.approx(sum(edge.cost for edge in exact))
