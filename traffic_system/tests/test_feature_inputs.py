import copy
import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from shapely.geometry import LineString, box, mapping

from app import places, weather_service
from app.main import app
from scripts.preprocess_road_features import directional_lanes, road_geometry
from src.routing.condition_update import Conditions
from src.routing.dynamic_weight import prepare_graph
from src.routing.network import CITIES, RoadNetwork
from src.routing.weather import fetch_current_weather, observed_effect, weather_condition


def weather_snapshot(age=0):
    return {"source": "Weatherstack", "cities": {city: {
        "retrieved_at": (datetime.now(timezone.utc) - timedelta(seconds=age)).isoformat(),
        "location": {"localtime": "2026-10-17 20:40"},
        "current": {"precip": 0.5, "weather_descriptions": ["Light rain"]},
    } for city in CITIES}}


@pytest.mark.parametrize("data,expected,source", [
    ({"lanes": "4", "oneway": "False"}, 2, "osm_lanes"),
    ({"lanes": "3", "oneway": "True"}, 3, "osm_lanes"),
    ({"lanes:forward": "2", "lanes": "3"}, 2, "osm_directional_lanes"),
    ({"lanes:backward": "1", "reversed": "True"}, 1, "osm_directional_lanes"),
    ({}, 1, "road_type_default_lanes"),
])
def test_directional_capacity_estimates_are_labeled(data, expected, source):
    assert directional_lanes(data, 1) == (expected, source)


def test_documented_osmnx_geometry_convention():
    import networkx as nx
    graph = nx.MultiDiGraph(created_with="OSMnx 2.1.1")
    graph.add_node("a", x=121, y=14.5)
    graph.add_node("b", x=121.001, y=14.5)
    line, source = road_geometry(graph, "a", "b", {})
    assert source == "osm_endpoint_segment_osmnx_convention"
    assert list(line.coords) == [(121, 14.5), (121.001, 14.5)]
    del graph.graph["created_with"]
    with pytest.raises(ValueError, match="Only OSMnx"):
        road_geometry(graph, "a", "b", {})


def test_observed_weather_conversion_is_separate_from_measurement():
    scenarios = {"clear": {"severity": 0, "travel_time_multiplier": 1},
                 "rain": {"severity": 0.4, "travel_time_multiplier": 1.15},
                 "heavy_rain": {"severity": 0.8, "travel_time_multiplier": 1.35}}
    rain = {"current": {"precip": 1, "weather_descriptions": ["Rain"]}}
    assert weather_condition(rain) == "rain"
    assert observed_effect(rain, scenarios).travel_time_multiplier == 1.15
    # Thresholds on 1-hour accumulations: >= 7.5 mm/h heavy rain, > 0 rain, 0 clear.
    assert weather_condition({"current": {"precip": 7.5}}) == "heavy_rain"
    assert weather_condition({"current": {"precip": 7.4}}) == "rain"
    assert weather_condition({"current": {"precip": 0, "weather_descriptions": ["Clear"]}}) == "clear"


def test_missing_weather_observation_is_not_assumed_clear():
    from src.routing.weather import MissingWeatherObservation
    # Missing precipitation: the description decides.
    assert weather_condition({"current": {"precip": None, "weather_descriptions": ["Patchy light drizzle"]}}) == "rain"
    assert weather_condition({"current": {"precip": None, "weather_descriptions": ["Sunny"]}}) == "clear"
    with pytest.raises(MissingWeatherObservation):
        weather_condition({"current": {"precip": None, "weather_descriptions": []}})


def test_weather_provider_errors_never_echo_key(monkeypatch):
    import requests
    secret = "test-secret-never-echo"
    monkeypatch.setenv("WEATHERSTACK_API_KEY", secret)
    def failure(*args, **kwargs):
        raise requests.ConnectionError(f"https://provider.example/?access_key={secret}")
    monkeypatch.setattr(requests, "get", failure)
    with pytest.raises(RuntimeError) as error:
        fetch_current_weather({"Manila": mapping(box(120, 14, 121, 15))})
    assert secret not in str(error.value)


def test_current_weather_age_and_refresh_cooldown(network, tmp_path, monkeypatch):
    path = tmp_path / "weather.json"
    monkeypatch.setenv("WEATHER_SNAPSHOT_PATH", str(path))
    path.write_text(json.dumps(weather_snapshot()), encoding="utf-8")
    snapshot = weather_service.read_snapshot(network)
    assert not weather_service.public_snapshot(snapshot)["stale"]
    def no_call(*args):
        raise AssertionError("Fresh weather should not incur another API request")
    monkeypatch.setattr(weather_service, "fetch_current_weather", no_call)
    assert weather_service.refresh_weather(network)["source"] == "Weatherstack"
    path.write_text(json.dumps(weather_snapshot(7200)), encoding="utf-8")
    with pytest.raises(HTTPException, match="older than one hour"):
        weather_service.current_effects(network)


def test_current_weather_override_does_not_mutate_road_snapshot(network_data):
    for road in network_data["roads"]:
        road["weather"]["current"] = {"severity": 0, "travel_time_multiplier": 1}
    network = RoadNetwork.model_validate(network_data)
    before = copy.deepcopy(network.model_dump())
    from src.routing.network import WeatherEffect
    from src.routing.criteria import CriterionInputError
    calm = {city: WeatherEffect(severity=0, travel_time_multiplier=1) for city in CITIES}
    override = {city: WeatherEffect(severity=1, travel_time_multiplier=3) for city in CITIES}
    snapshot = Conditions(time_of_day=8, weather="current")
    graph = prepare_graph(network, snapshot, override)
    assert graph["a"][0].cost > prepare_graph(network, snapshot, calm)["a"][0].cost
    with pytest.raises(CriterionInputError, match="observed weather"):  # Never silently the road's default.
        prepare_graph(network, snapshot)
    assert network.model_dump() == before


def test_local_places_and_scope_aware_snapping(network_data, monkeypatch):
    # A bidirectional test component gives search/snap the same connected coverage policy.
    for road in list(network_data["roads"]):
        reverse = copy.deepcopy(road)
        reverse.update(id=road["id"] + "-reverse", source=road["target"], target=road["source"], geometry=list(reversed(road["geometry"])))
        network_data["roads"].append(reverse)
    network_data["nodes"][0]["label"] = "Test Street · San Juan"
    network = RoadNetwork.model_validate(network_data)
    import app.main as service
    monkeypatch.setattr(service, "get_network", lambda: network)
    with TestClient(app) as client:
        assert client.get("/places/search", params={"query": "Test Street"}).json()["results"][0]["id"] == "a"
        assert client.get("/places/nearest", params={"latitude": 0.5, "longitude": 0.2}).json()["id"] == "a"
        assert client.get("/places/nearest", params={"latitude": 0.5, "longitude": -1}).status_code == 422
        road = client.get("/roads/nearest", params={"latitude": 0.5, "longitude": 0.4}).json()
        assert road["id"] in {"ab", "ab-reverse", "ad", "ad-reverse"}
        assert road["opposite_roads"]
        assert client.get("/network?include_nodes=false").json()["nodes"] == []


def test_named_places_snap_to_the_nearest_connected_node(network_data, tmp_path, monkeypatch):
    from app import places
    for road in list(network_data["roads"]):
        network_data["roads"].append({**road, "id": road["id"] + "-reverse", "source": road["target"], "target": road["source"],
                                      "geometry": road["geometry"][::-1]})
    network_data["nodes"][1]["label"] = "Mall Road · San Juan"
    gazetteer = tmp_path / "places.csv"
    gazetteer.write_text("name,category,city,latitude,longitude,osm_element,osm_id\n"
                         "Test Mall,mall,Manila,0.5,1.15,node,1\nOld Mall Chapel,place_of_worship,San Juan,0.5,0.25,node,2\n",
                         encoding="utf-8")
    monkeypatch.setattr(places, "GAZETTEER", gazetteer)
    monkeypatch.setattr(places, "CACHED", None)
    results = places.search_nodes(RoadNetwork.model_validate(network_data), "mall")
    # Word-start matches first (places before roads), then the rest; each place resolves to a road node.
    assert [(item["label"], item["id"]) for item in results] == [
        ("Test Mall · Mall · Manila", "d"), ("Old Mall Chapel · Place_Of_Worship · San Juan", "a"),
        ("Mall Road · San Juan", "b")]


def test_builder_uses_mapped_flood_and_labels_estimates(tmp_path):
    import networkx as nx
    from scripts.build_study_network import build
    boundaries = {city: mapping(box(121 + i * .01, 14.5, 121 + (i + 1) * .01, 14.51)) for i, city in enumerate(CITIES)}
    boundary_path = tmp_path / "boundaries.json"
    boundary_path.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"properties": {"city": city}, "geometry": geometry} for city, geometry in boundaries.items()
    ]}), encoding="utf-8")
    graph = nx.MultiDiGraph(created_with="OSMnx 2.1.1")
    graph.add_node("a", x=121.001, y=14.505, street_count=3, highway="traffic_signals")
    graph.add_node("b", x=121.002, y=14.505, street_count=2)
    graph.add_edge("a", "b", key=0, highway="residential", name="Test Street", lanes="2", oneway="False", maxspeed="20")
    graph_path = tmp_path / "roads.graphml"
    nx.write_graphml(graph, graph_path)
    flood_path = tmp_path / "flood.json"
    flood_path.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"properties": {"Var": 3}, "geometry": mapping(box(121, 14.50, 121.01, 14.51))}
    ]}), encoding="utf-8")
    from scripts import preprocess_road_features as static
    from src.routing.historical_traffic import FIELDS, bpr
    assumptions = {"saturation_flow_vph_per_lane": 1500, "assumed_effective_green_ratio": .5, "peak_hours": [8],
        "road_types": {"residential": {"speed_kph": 20, "lanes": 1, "traffic_factor": 0.5}},
        "weather_scenarios": {"clear": {"severity": 0, "travel_time_multiplier": 1}},
        "road_class_severity": {"method": "speed_median", "status": "assumed_preference"},
        "flood_class_severity": {"values": {"1": 1 / 3, "2": 2 / 3, "3": 1}, "status": "assumed_scoring_scenario"},
        "bpr_alpha": .15, "bpr_beta": 4, "weekend_demand_factor": .75, "hourly_demand_capacity_ratio": [1] * 24,
        "traffic_variation": {"seed": 1, "strength": 0, "day_coefficient": .2, "area_coefficient": .25,
                              "road_coefficient": .35, "location_hour_coefficient": .25},
        "sources": dict.fromkeys(["map", "flood", "weather", "here", "assumptions", "live_assumptions"], "test")}
    assumptions_path = tmp_path / "assumptions.json"
    assumptions_path.write_text(json.dumps(assumptions), encoding="utf-8")
    nodes, segments, floods, excluded, _ = static.preprocess(boundary_path, graph_path, flood_path, assumptions)
    features = tmp_path / "features"
    features.mkdir()
    static.write_csv(features / "road_nodes.csv", static.NODE_FIELDS, nodes)
    static.write_csv(features / "road_segments.csv", static.SEGMENT_FIELDS, segments)
    static.write_csv(features / "segment_flood.csv", static.FLOOD_FIELDS, floods)
    assert segments[0]["traffic_location"] == segments[0]["road_id"]
    assert "lanes_source" in segments[0] and not any("volume" in field for field in static.SEGMENT_FIELDS + FIELDS)
    from datetime import date, timedelta
    development = [date(2025, 6, 1) + timedelta(days=offset) for offset in range(14)]
    network = build(features, boundary_path, assumptions_path, development)
    road = network.roads[0]
    # The whole road lies in the class 3 polygon: its flood portion is its full length, by class.
    assert road.flood_km_by_class == {"3": pytest.approx(road.distance_m / 1000, rel=1e-6)} and road.flood_coverage == "mapped"
    assert road.road_capacity_vph == 1500  # two-way lanes=2 is one lane per direction; no signal at b
    assert segments[0]["capacity_method"] == "directional_lanes_x_assumed_base_saturation_flow_x_gC"
    assert "unvalidated" in segments[0]["capacity_uncertainty"]
    factor, congestion = bpr(.5, assumptions)
    assert road.traffic_congestion == pytest.approx([congestion] * 24, abs=1e-6)
    assert road.travel_time_s[0] == pytest.approx(road.free_flow_time_s * factor, rel=1e-5)
    assert road.distance_m > 100
    assert network.nodes[0].traffic_signal
    assert "estimated" in road.feature_sources["capacity_burden"]
    assert network.provenance["mode"] == "historical_period_with_estimates" and network.provenance["contract_version"] == 2
    assert network.cost_config.calibration_id == "provisional-uncalibrated"  # Never published before calibration.
    assert network.cost_config.road_class_severity == {"residential": 0.0}  # The only class is the fastest class.


def test_flood_portions_resolve_overlaps_to_the_highest_class():
    from shapely.geometry import LineString
    from shapely.strtree import STRtree
    from scripts.preprocess_road_features import flood_portions, geodesic_km
    line = LineString([(121.000, 14.5), (121.010, 14.5)])
    class2 = box(120.999, 14.49, 121.005, 14.51)  # First half.
    class3 = box(121.003, 14.49, 121.008, 14.51)  # Overlaps class 2 on 121.003-121.005.
    polygons = [class2, class3]
    portions, coverage = flood_portions(line, STRtree(polygons), [2, 3], box(120, 14, 122, 15))
    total = geodesic_km(line)
    assert portions[3] == pytest.approx(total * 0.5, rel=1e-3)  # 121.003-121.008
    assert portions[2] == pytest.approx(total * 0.3, rel=1e-3)  # 121.000-121.003 only; the overlap counts once, as class 3
    assert 1 not in portions and coverage == "mapped"
    _, outside = flood_portions(line, STRtree(polygons), [2, 3], box(121.001, 14.4, 121.002, 14.6))
    assert outside == "unknown"  # Beyond the hazard dataset's extent: unknown, not safe.
