"""Normalized weighted costs for exactly the ten study features."""
from dataclasses import dataclass

from .condition_update import Conditions, validate_conditions
from .network import Road, RoadNetwork, WeatherEffect
from .estimation import traffic_estimate


@dataclass(frozen=True)
class WeightedRoad:
    road: Road
    cost: float


def prepare_graph(network: RoadNetwork, conditions: Conditions, weather_by_city: dict[str, WeatherEffect] | None = None) -> dict[str, list[WeightedRoad]]:
    validate_conditions(network, conditions)
    nodes = {node.id: node for node in network.nodes}
    adjacency: dict[str, list[WeightedRoad]] = {node.id: [] for node in network.nodes}
    config = network.cost_config
    hour = conditions.time_of_day
    for road in network.roads:
        if road.id in conditions.closed_road_ids:
            continue
        weather = road.weather[conditions.weather]
        if weather_by_city is not None and conditions.weather in {"current", "historical"}:
            weather = weather_by_city[nodes[road.source].city]
        travel_time = road.travel_time_s[hour]
        congestion = road.traffic_congestion[hour]
        if road.free_flow_time_s is not None and road.volume_capacity_ratio is not None:
            travel_time, congestion = traffic_estimate(road.free_flow_time_s, road.volume_capacity_ratio[hour],
                                                      network.provenance["assumptions"], conditions.departure_date)
        target = nodes[road.target]
        features = {
            "distance": min(road.distance_m / config.distance_scale_m, 1.0),
            "traffic_congestion": congestion,
            "travel_time": min(travel_time * weather.travel_time_multiplier / config.travel_time_scale_s, 1.0),
            "number_of_intersections": float(target.is_intersection),
            "traffic_signal": float(target.traffic_signal),
            "road_type": config.road_type_costs[road.road_type],
            "time_of_day": road.time_of_day[hour],
            "weather": weather.cost,
            "flood_risk": road.flood_risk if weather.flood_risk is None else weather.flood_risk,
            "road_capacity": 1.0 - min(road.road_capacity_vph / config.capacity_scale_vph, 1.0),
        }
        cost = sum(config.weights[name] * value for name, value in features.items())
        adjacency[road.source].append(WeightedRoad(road, cost))
    return adjacency
