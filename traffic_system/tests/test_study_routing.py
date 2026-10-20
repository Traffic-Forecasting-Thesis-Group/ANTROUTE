import copy
import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app, get_network
from src.routing.aco_routing import (AntColonyConfig, NoFeasibleRoute, SearchExhausted, ant_colony_route, distance_reference,
                                     node_positions)
from src.routing.condition_update import Conditions
from src.routing.dynamic_weight import prepare_graph
from src.routing.metrics import evaluate
from src.routing.network import FEATURES, RoadNetwork
from conftest import fixture_cost_config


def conditions(**changes):
    return Conditions(time_of_day=changes.pop("time_of_day", 8), weather=changes.pop("weather", "clear"), **changes)


def solve(network, snapshot=None, **settings):
    return ant_colony_route(prepare_graph(network, snapshot or conditions()), "a", "d", AntColonyConfig(**settings),
                            node_positions(network), distance_reference(network, "antroute"))


def test_aco_route_and_evaluation_items(network):
    config = AntColonyConfig()
    result = solve(network)
    metrics = evaluate(result, config, 2.5)
    assert [edge.road.id for edge in result.roads] == ["ab", "bd"]
    assert set(metrics) == {"distance_travel_m", "path_cost", "search_time_ms", "computational_time_ms", "convergence_rule"}
    assert metrics["distance_travel_m"] == 200
    # Two 100 m edges: 0.1 x (0.1 km + 10/60 min + (0.1 km / 4000 veh/h) / 1e-3).
    assert metrics["path_cost"] == pytest.approx(2 * 0.1 * (0.1 + 10 / 60 + 0.1 / 4000 / 1e-3))
    assert metrics["search_time_ms"] == 2.5
    rule = metrics["convergence_rule"]
    assert rule["stop_reason"] == "converged"
    assert rule["iterations_to_convergence"] == rule["first_best_found_iteration"] + 20 == rule["iterations"]


def test_contributions_add_up_to_path_cost_and_name_sources(network):
    from src.routing.contributions import route_contributions
    from src.routing.dynamic_weight import segment_values
    network.provenance["estimation_registry"] = {"distance": {"classification": "calculated", "basis": "OSM length", "source": "osm"}}
    network.roads[0].feature_sources = {"distance": "osm_polyline"}
    result = solve(network)
    roads = [edge.road for edge in result.roads]
    nodes = {node.id: node for node in network.nodes}
    segments = [segment_values(network, road, conditions(), nodes) for road in roads]
    records = route_contributions(network, roads, segments)
    assert [record["feature"] for record in records] == list(FEATURES)
    assert sum(record["contribution"] for record in records) == pytest.approx(sum(edge.cost for edge in result.roads))
    assert sum(record["share"] for record in records) == pytest.approx(1)
    distance = records[0]
    assert distance["route_value"] == pytest.approx(0.2) and distance["unit"] == "km"
    assert distance["classification"] == "calculated" and distance["data_sources"] == {"osm_polyline": 1, "not_recorded": 1}


def test_explanation_separates_costlier_roads_from_search_misses(network_data):
    from src.routing.aco_routing import ColonyResult
    from src.routing.dynamic_weight import segment_values
    from src.routing.explanation import explain
    # The direct road a->d is shorter (150 m vs 200 m) but lies in a class 3 hazard area along its length.
    direct = next(road for road in network_data["roads"] if road["id"] == "ad")
    direct.update(distance_m=150, flood_km_by_class={"3": 0.15})
    network_data["cost_config"] = fixture_cost_config(flood_exposure=.5, distance=.05, congestion=.05, travel_time=.05, intersections=.05,
                                                     signals=.05, road_type=.05, time_profile=.05, weather=.05, capacity_burden=.1)
    network = RoadNetwork.model_validate(network_data)
    graph = prepare_graph(network, conditions())
    nodes = {node.id: node for node in network.nodes}
    edges = {edge.road.id: edge for roads in graph.values() for edge in roads}

    def run(chosen, visits):
        colony = ColonyResult([edges[road] for road in chosen], [1.0], "converged", 1, 1, 1, 0.0, visits=visits, ant_routes=10)
        return explain(network, graph, colony.roads, colony, "a", "d", lambda road: segment_values(network, road, conditions(), nodes))

    avoided = run(["ab", "bd"], {"ab": 10, "bd": 10})
    stretch = avoided["stretches"][0]
    assert avoided["references"]["shortest_distance"]["distance_m"] == 150
    assert avoided["references"]["lowest_cost"]["same_as_chosen"]
    assert stretch["verdict"] == "costlier_alternative" and stretch["distance_saved_m"] == 50
    assert next(iter(stretch["feature_differences"])) == "flood_exposure" and stretch["cost_difference"] > 0
    assert tuple(stretch["alternative"]["geometry"][0]) == (nodes["a"].longitude, nodes["a"].latitude)

    direct["flood_km_by_class"] = {}  # Now a->d is shorter and cheaper: taking a-b-d is a search miss.
    network = RoadNetwork.model_validate(network_data)
    graph = prepare_graph(network, conditions())
    edges = {edge.road.id: edge for roads in graph.values() for edge in roads}
    missed = run(["ab", "bd"], {"ab": 10, "bd": 10})["stretches"][0]
    assert missed["verdict"] == "missed_by_search" and missed["ant_routes_using_alternative"] == 0
    assert run(["ad"], {"ad": 10})["stretches"] == []


def test_weather_and_flood_exposure_stay_separate(network_data):
    from src.routing.dynamic_weight import segment_values
    network_data["roads"][0]["flood_km_by_class"] = {"3": 0.1}  # ab lies in a class 3 hazard area.
    network = RoadNetwork.model_validate(network_data)
    nodes = {node.id: node for node in network.nodes}
    rain = {road.id: segment_values(network, road, conditions(weather="rain"), nodes).criteria for road in network.roads}
    clear = {road.id: segment_values(network, road, conditions(), nodes).criteria for road in network.roads}
    # Weather is severity x length on every road, unchanged by mapped flood hazard.
    assert rain["ab"]["weather"] == pytest.approx(.8 * .1) and rain["bd"]["weather"] == pytest.approx(.8 * .1)
    # Flood exposure is the mapped hazard, whatever the weather.
    assert rain["ab"]["flood_exposure"] == clear["ab"]["flood_exposure"] == pytest.approx(.1)
    assert clear["ab"]["weather"] == 0


def test_closure_reroutes_and_does_not_mutate_network(network):
    rerouted = solve(network, conditions(closed_road_ids={"ab"}))
    assert [edge.road.id for edge in rerouted.roads] == ["ad"]
    assert [edge.road.id for edge in solve(network).roads] == ["ab", "bd"]


def test_closure_can_disconnect_network(network):
    with pytest.raises(NoFeasibleRoute):
        solve(network, conditions(closed_road_ids={"ab", "ad"}))


def test_direction_and_isolated_node(network):
    graph = prepare_graph(network, conditions())
    with pytest.raises(NoFeasibleRoute):
        ant_colony_route(graph, "d", "a", AntColonyConfig(), node_positions(network), 1e9)
    with pytest.raises(NoFeasibleRoute):
        ant_colony_route(graph, "a", "c", AntColonyConfig(), node_positions(network), 1e9)


def test_weather_changes_cost_and_leaves_original_snapshot_unchanged(network):
    clear = solve(network)
    rainy = solve(network, conditions(weather="rain"))
    assert sum(edge.cost for edge in rainy.roads) > sum(edge.cost for edge in clear.roads)
    assert solve(network).best_cost_by_iteration == clear.best_cost_by_iteration


@pytest.mark.parametrize("feature", FEATURES)
def test_each_of_the_ten_criteria_affects_cost(network_data, feature):
    original = RoadNetwork.model_validate(network_data)
    updated = copy.deepcopy(network_data)
    road = updated["roads"][0]
    if feature == "distance":
        road["distance_m"] = 500
    elif feature == "travel_time":
        road["travel_time_s"][8] = 500
    elif feature == "congestion":
        road["traffic_congestion"][8] = 0.8
    elif feature == "intersections":
        updated["nodes"][1]["is_intersection"] = True
    elif feature == "signals":
        updated["nodes"][1]["traffic_signal"] = True
    elif feature == "road_type":
        updated["cost_config"]["road_class_severity"]["residential"] = 0.8
    elif feature == "time_profile":
        road["time_profile"]["weekend"][8] = 0.8  # The default study date, 2025-06-01, is a Sunday.
    elif feature == "weather":
        road["weather"]["clear"]["severity"] = 0.8
    elif feature == "flood_exposure":
        road["flood_km_by_class"] = {"2": 0.05}
    elif feature == "capacity_burden":
        road["road_capacity_vph"] = 500
    changed = RoadNetwork.model_validate(updated)
    before = prepare_graph(original, conditions())["a"][0].cost
    after = prepare_graph(changed, conditions())["a"][0].cost
    assert after > before


def test_time_profile_follows_local_hour_and_day_category(network_data):
    profile = network_data["roads"][0]["time_profile"]
    profile["weekday"][8], profile["weekend"][8] = 1, 0.2
    network = RoadNetwork.model_validate(network_data)
    nodes = {node.id: node for node in network.nodes}
    from datetime import date
    from src.routing.dynamic_weight import segment_values
    weekday = Conditions(departure_date=date(2025, 6, 2), time_of_day=8, weather="clear")  # Monday
    road = network.roads[0]
    assert segment_values(network, road, weekday, nodes).criteria["time_profile"] == pytest.approx(0.1)
    assert segment_values(network, road, conditions(), nodes).criteria["time_profile"] == pytest.approx(0.02)
    assert segment_values(network, road, conditions(time_of_day=3), nodes).criteria["time_profile"] == 0


def test_convergence_is_online_and_resets(network):
    result = solve(network, patience=2, max_iterations=20)
    assert result.converged
    assert len(result.best_cost_by_iteration) == 3
    assert result.converged_at == 3
    limited = solve(network, patience=10, max_iterations=2)
    assert not limited.converged and limited.converged_at is None
    assert len(limited.best_cost_by_iteration) == 2
    assert evaluate(limited, AntColonyConfig(max_iterations=2), 1)["convergence_rule"]["stop_reason"] == "iteration_limit"


def test_search_limit_is_not_disconnection(network):
    # Closing the one-edge route leaves a feasible two-edge path, beyond this step limit.
    with pytest.raises(SearchExhausted):
        solve(network, conditions(closed_road_ids={"ad"}), max_steps=1, max_iterations=1)


@pytest.mark.parametrize("mutation", ["city", "node", "geometry", "endpoint", "missing_feature", "nan", "profile", "extra", "weights", "weather"])
def test_invalid_scope_and_features_are_rejected(network_data, mutation):
    if mutation == "city":
        network_data["nodes"][0]["city"] = "Quezon City"
    elif mutation == "node":
        network_data["nodes"][0]["longitude"] = -1
    elif mutation == "geometry":
        network_data["roads"][0]["geometry"].insert(1, [-1, 0.5])
    elif mutation == "endpoint":
        network_data["roads"][0]["geometry"][0] = [0.3, 0.5]
    elif mutation == "missing_feature":
        del network_data["roads"][0]["road_capacity_vph"]
    elif mutation == "nan":
        network_data["roads"][0]["distance_m"] = float("nan")
    elif mutation == "profile":
        network_data["roads"][0]["traffic_congestion"] = [0] * 23
    elif mutation == "extra":
        network_data["roads"][0]["event_exposure"] = 1
    elif mutation == "weights":
        network_data["cost_config"]["weights"] = {"distance": 1}
    elif mutation == "weather":
        del network_data["roads"][0]["weather"]["rain"]
    with pytest.raises(ValidationError):
        RoadNetwork.model_validate(network_data)


def test_unknown_conditions_rejected(network):
    with pytest.raises(ValueError, match="Unknown closed"):
        prepare_graph(network, conditions(closed_road_ids={"not-a-road"}))
    with pytest.raises(ValueError, match="Weather"):
        prepare_graph(network, conditions(weather="unknown"))
    with pytest.raises(ValidationError):
        conditions(time_of_day=24)


def test_api(network, monkeypatch):
    import app.main as service
    from app.study_policy import require_historical_release
    # Exercise the engine with explicitly synthetic fixtures only. This is not a
    # production setting and does not certify a historical dataset.
    monkeypatch.setitem(app.dependency_overrides, require_historical_release, lambda: None)
    monkeypatch.setattr(service, "get_network", lambda: network)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/network").json()["cities"] == ["San Juan", "Manila", "Mandaluyong", "Makati", "Pasig"]
        payload = {"origin": "a", "destination": "d", "conditions": conditions().model_dump(mode="json")}
        response = client.post("/routes/plan", json=payload)
        assert response.status_code == 200
        result = response.json()
        assert result["route"]["road_ids"] == ["ab", "bd"]
        assert result["evaluation"]["computational_time_ms"] >= 0
        payload["conditions"]["closed_road_ids"] = ["ab", "ad"]
        assert client.post("/routes/plan", json=payload).status_code == 404
        payload["conditions"]["closed_road_ids"] = ["unknown"]
        assert client.post("/routes/plan", json=payload).status_code == 422
        payload["conditions"]["closed_road_ids"] = []
        payload["origin"] = "outside"
        assert client.post("/routes/plan", json=payload).status_code == 422
        assert client.post("/auth/login").status_code == 404
        assert client.get("/health/model").status_code == 404


def test_missing_or_invalid_network_is_unavailable(tmp_path, monkeypatch):
    path = tmp_path / "study_network.json"
    monkeypatch.setenv("STUDY_NETWORK_PATH", str(path))
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/network").status_code == 503
        path.write_text(json.dumps({"roads": []}), encoding="utf-8")
        assert client.get("/network").status_code == 503


def test_network_reload_on_changed_file(network, tmp_path, monkeypatch):
    path = tmp_path / "network.json"
    path.write_text(network.model_dump_json(), encoding="utf-8")
    monkeypatch.setenv("STUDY_NETWORK_PATH", str(path))
    assert get_network().roads[0].distance_m == 100
    data = network.model_dump()
    data["roads"][0]["distance_m"] = 8000
    path.write_text(json.dumps(data), encoding="utf-8")
    assert get_network().roads[0].distance_m == 8000
