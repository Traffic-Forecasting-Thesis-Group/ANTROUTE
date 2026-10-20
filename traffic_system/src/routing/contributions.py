"""How much each of the ten criteria contributed to a route's path cost, and where its values came from.

Path cost = sum over roads and criteria of weight x criterion / development maximum, so the
per-criterion contributions below add up exactly to the route's path cost.
"""
from collections import Counter

from .criteria import CRITERIA, LENGTH_WEIGHTED, UNITS

NODE_CRITERIA = {"intersections", "signals"}


def route_contributions(network, roads, segments):
    """segments: one SegmentScore per road, as scored when the road was entered.

    Returns one record per criterion: weight, contribution to path cost and its share, the route's additive
    criterion total (and, for severity x length criteria, the length-weighted mean severity), the
    classification/basis/reference from the estimation registry, and how many of the route's roads (or
    road-end nodes) took each data source.
    """
    if not roads:
        return []
    weights = network.cost_config.weights
    registry = network.provenance.get("estimation_registry", {})
    nodes = {node.id: node for node in network.nodes}
    kilometres = sum(segment.criteria["distance"] for segment in segments)
    total = sum(sum(weights[name] * (segment.scaled[name] or 0) for name in CRITERIA) for segment in segments)
    records = []
    for name in CRITERIA:
        contribution = sum(weights[name] * (segment.scaled[name] or 0) for segment in segments)
        value = sum(segment.criteria[name] for segment in segments)
        sources = Counter((nodes[road.target].feature_sources if name in NODE_CRITERIA else road.feature_sources).get(name, "not_recorded")
                          for road in roads)
        entry = registry.get(name, {})
        records.append({
            "feature": name, "weight": weights[name], "contribution": contribution,
            "share": contribution / total if total > 0 else 0.0,
            "route_value": value, "unit": UNITS[name],
            "mean_severity": value / kilometres if name in LENGTH_WEIGHTED and kilometres else None,
            "development_max": network.cost_config.development_max[name],
            "classification": entry.get("classification", ""), "basis": entry.get("basis", ""),
            "reference": entry.get("source", ""), "data_sources": dict(sources.most_common()),
        })
    return records
