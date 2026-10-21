"""Small synthetic geometry exists only in tests, never in the served dataset."""
import pytest
from shapely.geometry import box, mapping

from src.routing.criteria import CRITERIA
from src.routing.network import CITIES, RoadNetwork


@pytest.fixture(autouse=True)
def historical_serving_mode(monkeypatch):
    """API tests exercise the historical release unless a test selects live mode itself."""
    monkeypatch.setenv("ANTROUTE_MODE", "historical")


@pytest.fixture
def network_data():
    nodes = [
        {"id": "a", "city": "San Juan", "latitude": 0.5, "longitude": 0.2, "is_intersection": False, "traffic_signal": False},
        {"id": "b", "city": "San Juan", "latitude": 0.5, "longitude": 0.6, "is_intersection": False, "traffic_signal": False},
        {"id": "c", "city": "San Juan", "latitude": 0.8, "longitude": 0.8, "is_intersection": False, "traffic_signal": False},
        {"id": "d", "city": "Manila", "latitude": 0.5, "longitude": 1.2, "is_intersection": False, "traffic_signal": False},
    ]
    lookup = {node["id"]: node for node in nodes}
    roads = []
    for source, target, distance in (("a", "b", 100), ("b", "d", 100), ("a", "d", 1000)):
        roads.append({
            "id": source + target, "source": source, "target": target,
            "distance_m": distance, "road_type": "residential", "road_capacity_vph": 4000,
            "flood_km_by_class": {}, "traffic_congestion": [0] * 24,
            "travel_time_s": [10] * 24, "time_profile": {"weekday": [0] * 24, "weekend": [0] * 24},
            "weather": {
                "clear": {"severity": 0, "travel_time_multiplier": 1},
                "rain": {"severity": 0.8, "travel_time_multiplier": 2},
            },
            "geometry": [[lookup[node]["longitude"], lookup[node]["latitude"]] for node in (source, target)],
        })
    return {
        "boundaries": {city: mapping(box(i, 0, i + 1, 1)) for i, city in enumerate(CITIES)},
        "nodes": nodes, "roads": roads,
        "cost_config": fixture_cost_config(),
        # Frozen references a calibrated release would carry (fixture values). The synthetic nodes are degrees apart
        # while roads are 100 m long, so a very large guidance reference keeps straight-line guidance negligible here.
        "provenance": {"distance_reference_m": {"antroute": 1e9, "cheng": 1e9},
                       "cheng_references": {"distance_m": 1000, "travel_time_s": 600, "flow_vph": 4000}},
    }


def fixture_cost_config(**weights):
    """Equal weights and unit development maxima (capacity burden on its own scale); fixtures only."""
    maxima = {**dict.fromkeys(CRITERIA, 1.0), "capacity_burden": 1e-3}
    return {"weights": {**dict.fromkeys(CRITERIA, 0.1), **weights}, "development_max": maxima,
            "road_class_severity": {"residential": 0}, "flood_class_severity": {"1": 1 / 3, "2": 2 / 3, "3": 1}}


@pytest.fixture
def network(network_data):
    return RoadNetwork.model_validate(network_data)
