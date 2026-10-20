from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app.main import app
from src.routing.baseline_cheng import rescale_pheromone
from src.routing.dynamic_trip import TripEvent, changed_roads, simulate_trip
from src.routing.condition_update import MANILA
from src.routing.dynamic_trip import ReplanConfig
from src.routing.network import RoadNetwork

DEPARTURE = datetime(2025, 6, 2, 8, 0, tzinfo=MANILA)


@pytest.fixture
def loop_network(network_data):
    """The shared fixture plus b -> a, so closing b -> d forces a detour back through a."""
    back = {**network_data["roads"][0], "id": "ba", "source": "b", "target": "a",
            "geometry": list(reversed(network_data["roads"][0]["geometry"]))}
    network_data["roads"].append(back)
    return RoadNetwork.model_validate(network_data)


def test_replan_trigger_counts_large_changes_closures_and_reopenings():
    old = {"x": 1.0, "y": 1.0, "z": 1.0, "closed": 1.0}
    new = {"x": 1.09, "y": 1.2, "z": 0.5, "reopened": 1.0}
    assert changed_roads(old, new, 0.10) == 4  # y, z, closed, reopened; x is below the threshold
    assert changed_roads(old, dict(old), 0.10) == 0


def test_cheng_eq9_is_applied_as_printed():
    pheromone = {"slower": 1.0, "faster": 1.0, "same": 1.0, "closed": 1.0}
    old = {"slower": 10.0, "faster": 10.0, "same": 10.0, "closed": 10.0}
    new = {"slower": 20.0, "faster": 5.0, "same": 10.0, "reopened": 7.0}
    changed = rescale_pheromone(pheromone, old, new)
    assert pheromone == {"slower": 2.0, "faster": 0.5, "same": 1.0, "reopened": 1.0}
    assert set(changed) == {"slower", "faster", "closed", "reopened"}


def test_closure_mid_trip_reroutes_from_the_current_node(loop_network):
    # The first road takes 10 s, so the closure (at 6 s) is seen on arrival at b.
    events = [TripEvent(at_minute=0.1, close_road_ids={"bd"})]
    result = simulate_trip(loop_network, "a", "d", DEPARTURE, "clear", frozenset(), events, "antroute")
    assert [road.id for road in result.roads] == ["ab", "ba", "ad"]
    assert [search.at_node for search in result.searches] == ["a", "b"]
    assert result.searches[1].trigger == "closures changed"
    assert [search.kind for search in result.searches] == ["initial", "reroute"]
    evaluation = result.evaluation()
    assert evaluation["reroutes"] == 1
    assert evaluation["initial_routing_time_ms"] == result.searches[0].routing_time_ms
    assert evaluation["dynamic_rerouting_time_ms"] == result.searches[1].routing_time_ms
    assert evaluation["computational_time_ms"] == pytest.approx(
        evaluation["initial_routing_time_ms"] + evaluation["dynamic_rerouting_time_ms"] + result.monitoring_time_ms)
    convergence = evaluation["convergence"]
    assert convergence["iterations_to_convergence"] == result.searches[0].iterations_to_convergence
    assert convergence["converged"]
    assert convergence["iterations_to_convergence"] == result.searches[0].best_found_at + 20 == result.searches[0].iterations


def test_cheng_reroutes_with_eq9_after_closure_and_reopening(loop_network):
    events = [TripEvent(at_minute=0.1, close_road_ids={"bd"}, reopen_road_ids={"ad"})]
    reference = ReplanConfig(policy="cheng_reference_printed")
    result = simulate_trip(loop_network, "a", "d", DEPARTURE, "clear", frozenset({"ad"}), events, "cheng", replan=reference)
    assert [road.id for road in result.roads] == ["ab", "ba", "ad"]
    assert result.searches[1].affected_roads == 2  # bd closed, ad reopened
    assert result.searches[1].stop_reason == "converged"  # Same convergence rule as ANTROUTE.
    assert result.searches[1].decision == "adopted_incumbent_invalid"  # The closure invalidated the remainder.
    with pytest.raises(ValueError, match="Cheng baseline only"):
        simulate_trip(loop_network, "a", "d", DEPARTURE, "clear", frozenset(), [], "antroute", replan=reference)


def test_naive_departure_is_rejected(network):
    with pytest.raises(ValueError, match="timezone-aware"):
        simulate_trip(network, "a", "d", datetime(2025, 6, 2, 8, 0), "clear", frozenset(), [], "antroute")


@pytest.mark.parametrize("cost_factor, decision", [
    (2.0, "kept_incumbent"),  # A candidate scored worse than the incumbent remainder b -> d.
    (1.0, "kept_incumbent"),  # A tie never replaces the incumbent.
    (None, "kept_incumbent_search_failed"),  # A failed search keeps the valid incumbent.
])
def test_valid_incumbent_survives_worse_tied_or_failed_candidates(loop_network, monkeypatch, cost_factor, decision):
    from dataclasses import replace
    import src.routing.dynamic_trip as trip
    from src.routing.aco_kernel import SearchExhausted, SearchResult
    real = trip.ant_colony_route

    def candidate(adjacency, origin, destination, *args, **kwargs):
        if origin == "a":  # The departure plan is a real search: a -> b -> d.
            return real(adjacency, origin, destination, *args, **kwargs)
        if cost_factor is None:
            raise SearchExhausted("no route found")
        route = [replace(edge, cost=edge.cost * cost_factor) for edge in adjacency["b"] if edge.road.id == "bd"]
        return SearchResult(route, [route[0].cost], "converged", 1, 21, 21, 1.0)
    monkeypatch.setattr(trip, "ant_colony_route", candidate)
    rain = [TripEvent(at_minute=0.1, weather="rain")]  # Changes every native cost, so a re-plan runs at b.
    result = simulate_trip(loop_network, "a", "d", DEPARTURE, "clear", frozenset(), rain, "antroute")
    assert [road.id for road in result.roads] == ["ab", "bd"]
    assert result.searches[1].decision == decision
    assert result.searches[1].incumbent_cost is not None


def test_weather_change_triggers_replanning(network):
    events = [TripEvent(at_minute=0.1, weather="rain")]
    result = simulate_trip(network, "a", "d", DEPARTURE, "clear", frozenset(), events, "antroute")
    assert result.searches[1].trigger == "weather clear -> rain"
    assert result.searches[1].affected_roads == 3
    # Each road is scored when entered: ab in clear weather, bd in rain.
    assert result.path_costs[1] > result.path_costs[0]


def test_hour_change_without_cost_change_does_not_replan(network):
    # Fixture profiles are flat, so crossing 09:00 changes no cost or travel time.
    for algorithm in ("antroute", "cheng"):
        result = simulate_trip(network, "a", "d", datetime(2025, 6, 2, 8, 59, 55, tzinfo=MANILA), "clear", frozenset({"ad"}), [], algorithm)
        assert len(result.searches) == 1
        assert result.unsearched_changes[0]["change"] == "hour 08:00 -> 09:00"
        evaluation = result.evaluation()
        assert evaluation["reroutes"] == 0 and evaluation["dynamic_rerouting_time_ms"] == 0
        assert evaluation["mean_rerouting_time_ms"] is None
        assert result.monitoring_time_ms > 0  # The hour check itself still costs time.


def test_multiple_stops_are_visited_in_order(loop_network):
    # a -> b, then b -> a -> d: the second leg is planned from b when the first ends.
    result = simulate_trip(loop_network, "a", ["b", "d"], DEPARTURE, "clear", frozenset(), [], "antroute")
    assert [road.id for road in result.roads] == ["ab", "bd"]
    assert [search.kind for search in result.searches] == ["initial", "initial"]
    assert [search.at_node for search in result.searches] == ["a", "b"]
    assert result.stop_arrivals_s == [10, 20]
    evaluation = result.evaluation()
    assert evaluation["reroutes"] == 0
    assert evaluation["initial_routing_time_ms"] == pytest.approx(sum(s.routing_time_ms for s in result.searches))
    back = simulate_trip(loop_network, "a", ["b", "a"], DEPARTURE, "clear", frozenset(), [], "cheng")
    assert [road.id for road in back.roads] == ["ab", "ba"]
    with pytest.raises(ValueError, match="differ"):
        simulate_trip(loop_network, "a", ["b", "b"], DEPARTURE, "clear", frozenset(), [], "antroute")


def test_trip_api(loop_network, monkeypatch):
    import app.main as service
    from app.study_policy import require_historical_release
    monkeypatch.setitem(app.dependency_overrides, require_historical_release, lambda: None)
    monkeypatch.setattr(service, "get_network", lambda: loop_network)
    payload = {"origin": "a", "destination": "d", "conditions": {"time_of_day": 8, "weather": "clear"},
               "events": [{"at_minute": 0.1, "close_road_ids": ["bd"]}]}
    with TestClient(app) as client:
        body = client.post("/routes/trip", json=payload).json()  # algorithm defaults to antroute
        assert body["algorithm"]["name"] == "antroute"
        assert body["route"]["road_ids"] == ["ab", "ba", "ad"]
        assert {"distance_travel_m", "path_cost", "search_time_ms", "response_time_ms", "recalculation_time_ms", "detection_delay_s",
                "computational_time_ms", "reroutes", "convergence"} <= set(body["evaluation"])
        assert body["evaluation"]["detection_delay_s"] == [pytest.approx(4)]  # Closure at 6 s, seen on arrival at b at 10 s.
        assert body["decision"]["road_ids"] == ["ab", "bd"]  # The departure decision, before the closure.
        assert body["evaluation"]["path_cost"] == pytest.approx(sum(s["path_cost"] for s in body["route"]["segments"]))
        stops = {**payload, "destination": None, "destinations": ["b", "d"], "events": []}
        body = client.post("/routes/trip", json=stops).json()
        assert body["trip"]["stops"] == ["b", "d"] and len(body["trip"]["stop_arrival_minutes"]) == 2
        assert client.post("/routes/trip", json={**stops, "destinations": []}).status_code == 422
        assert client.post("/routes/trip", json={**stops, "destinations": ["b"] * 3 + ["d"] * 3}).status_code == 422
        for removed in ("cspa", "reset", "other"):
            assert client.post("/routes/trip", json={**payload, "algorithm": removed}).status_code == 422
        assert client.post("/routes/trip", json={**payload, "events": [{"at_minute": 0, "close_road_ids": ["nope"]}]}).status_code == 422
