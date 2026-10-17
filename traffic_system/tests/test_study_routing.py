import copy
import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app, get_network
from src.routing.aco_routing import AntColonyConfig, NoFeasibleRoute, SearchExhausted, ant_colony_route
from src.routing.condition_update import Conditions
from src.routing.dynamic_weight import prepare_graph
from src.routing.metrics import evaluate
from src.routing.network import FEATURES, RoadNetwork


def conditions(**changes):
    return Conditions(time_of_day=changes.pop("time_of_day", 8), weather=changes.pop("weather", "clear"), **changes)


def solve(network, snapshot=None, **settings):
    return ant_colony_route(prepare_graph(network, snapshot or conditions()), "a", "d", AntColonyConfig(**settings))


def test_aco_route_and_exact_four_evaluation_items(network):
    config = AntColonyConfig()
    result = solve(network)
    metrics = evaluate(result, config, 2.5)
    assert [edge.road.id for edge in result.roads] == ["ab", "bd"]
    assert set(metrics) == {"distance_travel_m", "path_cost", "computational_time_ms", "convergence_rule"}
    assert metrics["distance_travel_m"] == 200
    assert metrics["path_cost"] == pytest.approx(2 * (0.1 * 0.1 + 0.1 * 10 / 600))
    assert metrics["computational_time_ms"] == 2.5


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
        ant_colony_route(graph, "d", "a", AntColonyConfig())
    with pytest.raises(NoFeasibleRoute):
        ant_colony_route(graph, "a", "c", AntColonyConfig())


def test_weather_changes_cost_and_leaves_original_snapshot_unchanged(network):
    clear = solve(network)
    rainy = solve(network, conditions(weather="rain"))
    assert sum(edge.cost for edge in rainy.roads) > sum(edge.cost for edge in clear.roads)
    assert solve(network).best_cost_by_iteration == clear.best_cost_by_iteration


@pytest.mark.parametrize("feature", FEATURES)
def test_each_of_the_ten_features_affects_cost(network_data, feature):
    original = RoadNetwork.model_validate(network_data)
    updated = copy.deepcopy(network_data)
    road = updated["roads"][0]
    if feature == "distance":
        road["distance_m"] = 500
    elif feature == "travel_time":
        road["travel_time_s"][8] = 500
    elif feature == "number_of_intersections":
        updated["nodes"][1]["is_intersection"] = True
    elif feature == "traffic_signal":
        updated["nodes"][1]["traffic_signal"] = True
    elif feature == "road_type":
        updated["cost_config"]["road_type_costs"]["residential"] = 0.8
    elif feature == "weather":
        road["weather"]["clear"]["cost"] = 0.8
    elif feature == "flood_risk":
        road["flood_risk"] = 0.8
    elif feature == "road_capacity":
        road["road_capacity_vph"] = 500
    else:
        road[feature][8] = 0.8
    changed = RoadNetwork.model_validate(updated)
    before = prepare_graph(original, conditions())["a"][0].cost
    after = prepare_graph(changed, conditions())["a"][0].cost
    assert after > before


def test_time_of_day_updates_local_hour_features(network_data):
    network_data["roads"][0]["traffic_congestion"][8] = 1
    network_data["roads"][0]["time_of_day"][8] = 1
    network = RoadNetwork.model_validate(network_data)
    assert prepare_graph(network, conditions(time_of_day=8))["a"][0].cost > prepare_graph(network, conditions(time_of_day=3))["a"][0].cost


def test_convergence_is_online_and_resets(network):
    result = solve(network, patience=2, max_iterations=20)
    assert result.converged
    assert len(result.best_cost_by_iteration) == 3
    limited = solve(network, patience=10, max_iterations=2)
    assert not limited.converged
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
