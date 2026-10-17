"""Small synthetic geometry exists only in tests, never in the served dataset."""
import pytest
from shapely.geometry import box, mapping

from src.routing.network import CITIES, RoadNetwork


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
            "flood_risk": 0, "traffic_congestion": [0] * 24,
            "travel_time_s": [10] * 24, "time_of_day": [0] * 24,
            "weather": {
                "clear": {"cost": 0, "travel_time_multiplier": 1},
                "rain": {"cost": 0.8, "travel_time_multiplier": 2, "flood_risk": 0.7},
            },
            "geometry": [[lookup[node]["longitude"], lookup[node]["latitude"]] for node in (source, target)],
        })
    return {
        "boundaries": {city: mapping(box(i, 0, i + 1, 1)) for i, city in enumerate(CITIES)},
        "nodes": nodes, "roads": roads,
        "cost_config": {"road_type_costs": {"residential": 0}},
    }


@pytest.fixture
def network(network_data):
    return RoadNetwork.model_validate(network_data)
