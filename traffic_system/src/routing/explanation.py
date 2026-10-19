"""Why ANTROUTE's route left out a shorter or cheaper road: stretch-by-stretch comparison with reference routes.

The reference routes (shortest distance and lowest ten-feature cost) come from Dijkstra on the same
snapshot graph. They are used only to explain a finished ACO search, never to plan the route.

For each stretch where a reference route leaves ANTROUTE's route and rejoins it later:
  - if the reference stretch costs more under the ten features, ANTROUTE avoided it for a reason:
    the features that make it costlier are listed;
  - if it costs less, the colony did not find it: how many ant routes used its roads is listed,
    because a lower-cost road the ants never reinforced is a search miss, not a decision.
"""
import heapq
import math

from .network import FEATURES

EPSILON = 1e-9


def _dijkstra(adjacency, origin, destination, weight):
    distance, previous, heap = {origin: 0.0}, {}, [(0.0, origin)]
    while heap:
        value, node = heapq.heappop(heap)
        if node == destination:
            break
        if value > distance[node]:
            continue
        for edge in adjacency[node]:
            candidate = value + weight(edge)
            if candidate < distance.get(edge.road.target, math.inf):
                distance[edge.road.target], previous[edge.road.target] = candidate, edge
                heapq.heappush(heap, (candidate, edge.road.target))
    if destination not in previous:
        return None
    route, node = [], destination
    while node != origin:
        route.append(previous[node])
        node = previous[node].road.source
    return route[::-1]


def _stretches(chosen, reference):
    """(start index in chosen, end index in chosen, reference edges) for each place the routes differ."""
    nodes = [chosen[0].road.source] + [edge.road.target for edge in chosen]
    position = {node: index for index, node in enumerate(nodes)}
    found, index = [], 0
    while index < len(reference):
        edge = reference[index]
        start = position.get(edge.road.source)
        if start is not None and start < len(chosen) and chosen[start].road.id == edge.road.id:
            index += 1
            continue
        # The reference leaves the chosen route here; follow it until it rejoins at a later chosen node.
        detour, end = [], None
        for later in reference[index:]:
            detour.append(later)
            rejoin = position.get(later.road.target)
            if rejoin is not None and start is not None and rejoin > start:
                end = rejoin
                break
        if start is None or end is None:
            break
        found.append((start, end, detour))
        index += len(detour)
    return found


def _facts(edges, score):
    """What is on these roads: the ten features' raw values, summarized as a driver would read them."""
    raws = [score(edge)[0] for edge in edges]
    normalized = [score(edge)[1] for edge in edges]
    length = sum(raw["distance"] for raw in raws)

    def by_length(name, values=raws):
        return sum(value[name] * raw["distance"] for value, raw in zip(values, raws)) / length if length else 0.0
    return {"distance_m": length, "cost": sum(score(edge)[2] for edge in edges),
            "roads": len(edges), "intersections": sum(raw["number_of_intersections"] for raw in raws),
            "signals": sum(raw["traffic_signal"] for raw in raws), "travel_time_s": sum(raw["travel_time"] for raw in raws),
            "max_flood_risk": max(raw["flood_risk"] for raw in raws), "flood_risk": by_length("flood_risk"),
            "congestion": by_length("traffic_congestion"), "capacity_vph": by_length("road_capacity"),
            "weather_cost": by_length("weather"), "time_of_day_load": by_length("time_of_day", normalized),
            "road_types": list(dict.fromkeys(raw["road_type"] for raw in raws)),
            "road_names": list(dict.fromkeys(e.road.name for e in edges if e.road.name))}


def _geometry(edges):
    coordinates = []
    for edge in edges:
        coordinates.extend(edge.road.geometry if not coordinates else edge.road.geometry[1:])
    return coordinates


def explain(network, adjacency, chosen, colony, origin, destination, segment_score):
    """segment_score(road) -> (raw, normalized, cost) under this snapshot, as used for routing."""
    weights = network.cost_config.weights
    cache = {}

    def score(edge):
        if edge.road.id not in cache:
            cache[edge.road.id] = segment_score(edge.road)
        return cache[edge.road.id]
    references = {"shortest_distance": _dijkstra(adjacency, origin, destination, lambda edge: edge.road.distance_m),
                  "lowest_cost": _dijkstra(adjacency, origin, destination, lambda edge: edge.cost)}
    total = _facts(chosen, score)
    result = {"chosen": total, "ant_routes": colony.ant_routes, "references": {}, "stretches": []}
    seen = set()
    for kind, reference in references.items():
        if reference is None:
            continue
        result["references"][kind] = {**_facts(reference, score), "same_as_chosen": [e.road.id for e in reference] == [e.road.id for e in chosen]}
        for start, end, detour in _stretches(chosen, reference):
            key = tuple(edge.road.id for edge in detour)
            if key in seen:  # Both references can take the same stretch.
                continue
            seen.add(key)
            taken = chosen[start:end]
            mine, other = _facts(taken, score), _facts(detour, score)
            difference = {name: sum(weights[name] * score(e)[1][name] for e in detour)
                          - sum(weights[name] * score(e)[1][name] for e in taken) for name in FEATURES}
            visits = [colony.visits.get(edge.road.id, 0) for edge in detour]
            costlier = other["cost"] > mine["cost"] + EPSILON
            geometry = _geometry(detour)
            result["stretches"].append({
                "reference": kind, "verdict": "costlier_alternative" if costlier else "missed_by_search",
                "chosen": {**mine, "geometry": _geometry(taken)}, "alternative": {**other, "geometry": geometry},
                "marker": geometry[len(geometry) // 2],
                "distance_saved_m": mine["distance_m"] - other["distance_m"], "cost_difference": other["cost"] - mine["cost"],
                "feature_differences": dict(sorted(difference.items(), key=lambda item: -abs(item[1]))),
                "ant_routes_using_alternative": min(visits), "ant_routes_using_first_road": visits[0],
            })
    return result
