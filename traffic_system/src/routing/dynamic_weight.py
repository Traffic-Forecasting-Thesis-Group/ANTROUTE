"""Score road segments under one condition snapshot with the contract in criteria.py."""
from dataclasses import dataclass
from typing import NamedTuple

from .condition_update import Conditions, day_category, validate_conditions
from .criteria import CriterionInputError, SegmentInputs, criteria, scale
from .network import Road, RoadNetwork, WeatherEffect

OBSERVED_WEATHER = {"current", "historical"}


@dataclass(frozen=True)
class WeightedRoad:
    road: Road
    cost: float


class SegmentScore(NamedTuple):
    physical: dict  # Measurements in their own units, for simulation and display.
    criteria: dict[str, float]  # The ten additive criteria.
    scaled: dict[str, float | None]  # criterion / development maximum
    cost: float  # sum of weight * scaled
    flags: tuple[str, ...]  # Missing-data/provenance flags and reference exceedances.


def prepare_graph(network: RoadNetwork, conditions: Conditions, weather_by_city: dict[str, WeatherEffect] | None = None,
                  traffic: dict[str, tuple[float, float]] | None = None) -> dict[str, list[WeightedRoad]]:
    validate_conditions(network, conditions)
    nodes = {node.id: node for node in network.nodes}
    adjacency: dict[str, list[WeightedRoad]] = {node.id: [] for node in network.nodes}
    for road in network.roads:
        if road.id in conditions.closed_road_ids:
            continue
        adjacency[road.source].append(WeightedRoad(road, segment_values(network, road, conditions, nodes, weather_by_city, traffic).cost))
    return adjacency


def segment_inputs(network, road, conditions, nodes, weather_by_city=None, traffic=None) -> tuple[SegmentInputs, dict]:
    """Physical inputs of one road for one snapshot, and the physical values reported with them.

    traffic: {location_id: (congestion, travel_time_factor)} for the requested date and hour from the hourly
    table; without it the road's 24-hour fallback profile is used and flagged.
    weather_by_city: observed weather effects by node id or city; required for observed weather conditions.
    """
    config, hour, flags = network.cost_config, conditions.time_of_day, []
    if conditions.weather in OBSERVED_WEATHER:
        if weather_by_city is None:
            raise CriterionInputError(f"Weather '{conditions.weather}' needs an observed weather snapshot")
        weather = weather_by_city.get(road.source) or weather_by_city.get(nodes[road.source].city)
        if weather is None:
            raise CriterionInputError(f"No observed weather for road {road.id}")
    else:
        weather = road.weather[conditions.weather]
    if traffic is not None and road.traffic_location is not None and road.free_flow_time_s is not None:
        if road.traffic_location not in traffic:
            raise CriterionInputError(f"No hourly traffic record for road {road.id}")
        congestion, factor = traffic[road.traffic_location]
        base_time = road.free_flow_time_s * factor
    else:
        congestion, base_time = road.traffic_congestion[hour], road.travel_time_s[hour]
        flags.append("traffic_fallback_profile")
    if road.flood_coverage == "unknown":
        flags.append("flood_coverage_unknown")
    category = day_category(conditions.departure_date)
    target = nodes[road.target]
    portions = tuple((km, config.flood_class_severity[hazard]) for hazard, km in sorted(road.flood_km_by_class.items()))
    inputs = SegmentInputs(
        length_m=road.distance_m, traversal_s=base_time * weather.travel_time_multiplier, congestion=congestion,
        class_severity=config.road_class_severity[road.road_type], profile_severity=road.time_profile[category][hour],
        weather_severity=weather.severity, flood_portions=portions, capacity_vph=road.road_capacity_vph,
        intersection=int(target.is_intersection), signal=int(target.traffic_signal), flags=tuple(flags))
    km = road.distance_m / 1000
    physical = {
        "distance_m": road.distance_m, "travel_time_s": inputs.traversal_s, "congestion": congestion,
        "road_type": road.road_type, "class_severity": inputs.class_severity, "time_profile": inputs.profile_severity,
        "day_category": category, "hour": hour, "weather_condition": conditions.weather, "weather_severity": weather.severity,
        "flood_km_by_class": dict(road.flood_km_by_class), "flood_coverage": road.flood_coverage,
        "flood_severity": sum(k * s for k, s in portions) / km, "capacity_vph": road.road_capacity_vph,
        "intersection": inputs.intersection, "signal": inputs.signal,
    }
    return inputs, physical


def segment_values(network, road, conditions, nodes, weather_by_city=None, traffic=None) -> SegmentScore:
    """One implementation for calibration, routing, dataset exports and explanations."""
    inputs, physical = segment_inputs(network, road, conditions, nodes, weather_by_city, traffic)
    values = criteria(inputs)
    config = network.cost_config
    result = scale(values, config.weights, config.development_max)
    flags = inputs.flags + tuple(f"exceeds_development_max:{name}" for name in result.exceedances)
    return SegmentScore(physical, values, result.values, result.cost, flags)
