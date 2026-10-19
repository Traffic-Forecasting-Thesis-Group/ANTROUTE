"""Normalized weighted costs for exactly the ten study features."""
from dataclasses import dataclass

from .condition_update import Conditions, validate_conditions
from .network import Road, RoadNetwork, WeatherEffect


@dataclass(frozen=True)
class WeightedRoad:
    road: Road
    cost: float


def prepare_graph(network: RoadNetwork, conditions: Conditions, weather_by_city: dict[str, WeatherEffect] | None = None,
                  traffic: dict[str, tuple[float, float]] | None = None) -> dict[str, list[WeightedRoad]]:
    validate_conditions(network, conditions)
    nodes = {node.id: node for node in network.nodes}
    adjacency: dict[str, list[WeightedRoad]] = {node.id: [] for node in network.nodes}
    for road in network.roads:
        if road.id in conditions.closed_road_ids:
            continue
        _, _, cost = segment_values(network, road, conditions, nodes, weather_by_city, traffic)
        adjacency[road.source].append(WeightedRoad(road, cost))
    return adjacency


def segment_values(network, road, conditions, nodes, weather_by_city=None, traffic=None):
    """One implementation for routing, dataset exports and snapshot inspection.

    traffic: {location_id: (traffic_congestion, travel_time_factor)} for the requested
    date and hour, from the hourly traffic table. Without it, the road's 24-hour profile is used.
    """
    config, hour = network.cost_config, conditions.time_of_day
    weather = road.weather[conditions.weather]
    if weather_by_city is not None and conditions.weather in {"current", "historical"}:
        weather = weather_by_city.get(road.source, weather_by_city[nodes[road.source].city])
    travel_time, congestion = road.travel_time_s[hour], road.traffic_congestion[hour]
    if traffic is not None and road.traffic_location is not None and road.free_flow_time_s is not None:
        congestion, factor = traffic[road.traffic_location]
        travel_time = road.free_flow_time_s * factor
    target = nodes[road.target]
    flood = road.flood_risk if weather.flood_risk is None else weather.flood_risk
    share = config.weather_flood_exposure_share
    raw = {"distance": road.distance_m, "traffic_congestion": congestion,
           "travel_time": travel_time * weather.travel_time_multiplier,
           "number_of_intersections": int(target.is_intersection), "traffic_signal": int(target.traffic_signal),
           "road_type": road.road_type, "time_of_day": hour,
           # Rain costs more on flood-prone roads, so the weather feature differs between roads.
           "weather": weather.cost * ((1 - share) + share * flood),
           "flood_risk": flood,
           "road_capacity": road.road_capacity_vph}
    features = {**raw,
                "distance": min(raw["distance"] / config.distance_scale_m, 1),
                "travel_time": min(raw["travel_time"] / config.travel_time_scale_s, 1),
                "road_type": config.road_type_costs[road.road_type],
                "time_of_day": road.time_of_day[hour],
                "road_capacity": 1 - min(road.road_capacity_vph / config.capacity_scale_vph, 1)}
    return raw, features, sum(config.weights[name] * value for name, value in features.items())
