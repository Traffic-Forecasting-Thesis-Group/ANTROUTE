"""How much each of the ten features contributed to a route's path cost, and where its values came from.

Path cost = sum over roads and features of CRITIC weight x normalized feature value, so the
per-feature contributions below add up exactly to the route's path cost.
"""
from collections import Counter

from .network import FEATURES

# How the route's raw values are summarized for display.
UNITS = {"distance": "m", "traffic_congestion": "index 0-1", "travel_time": "s", "number_of_intersections": "intersections",
         "traffic_signal": "signals", "road_type": "roads", "time_of_day": "hour", "weather": "cost 0-1",
         "flood_risk": "hazard 0-1", "road_capacity": "vehicles/hour"}
TOTALS = {"distance", "travel_time", "number_of_intersections", "traffic_signal"}
NODE_FEATURES = {"number_of_intersections", "traffic_signal"}


def _route_value(feature, values):
    if feature == "road_type":
        return dict(Counter(values).most_common())
    if feature == "time_of_day":
        return sorted(set(values))
    if feature in TOTALS:
        return sum(values)
    return {"mean": sum(values) / len(values), "max": max(values)}


def route_contributions(network, roads, segments):
    """segments: one (raw, normalized) feature pair per road, as scored when the road was entered.

    Returns one record per feature: CRITIC weight, contribution to path cost and its share,
    the route's raw value, the feature's classification/basis/reference from the estimation
    registry, and how many of the route's roads (or road-end nodes) took each data source.
    """
    if not roads:
        return []
    weights = network.cost_config.weights
    registry = network.provenance.get("estimation_registry", {})
    nodes = {node.id: node for node in network.nodes}
    total = sum(weights[name] * normalized[name] for _, normalized in segments for name in FEATURES)
    records = []
    for name in FEATURES:
        contribution = sum(weights[name] * normalized[name] for _, normalized in segments)
        sources = Counter((nodes[road.target].feature_sources if name in NODE_FEATURES else road.feature_sources).get(name, "not_recorded")
                          for road in roads)
        entry = registry.get(name, {})
        records.append({
            "feature": name, "weight": weights[name], "contribution": contribution,
            "share": contribution / total if total > 0 else 0.0,
            "normalized_mean": sum(normalized[name] for _, normalized in segments) / len(segments),
            "route_value": _route_value(name, [raw[name] for raw, _ in segments]), "unit": UNITS[name],
            "classification": entry.get("classification", ""), "basis": entry.get("basis", ""),
            "reference": entry.get("source", ""), "data_sources": dict(sources.most_common()),
        })
    return records
