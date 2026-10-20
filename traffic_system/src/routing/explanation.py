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
    """What is on these roads, summarized as a driver would read them. Severities are length-weighted means."""
    scores = [score(edge) for edge in edges]
    kilometres = sum(item.criteria["distance"] for item in scores)

    def per_km(name):
        return sum(item.criteria[name] for item in scores) / kilometres if kilometres else 0.0
    return {"distance_m": sum(item.physical["distance_m"] for item in scores), "cost": sum(item.cost for item in scores),
            "roads": len(edges), "intersections": sum(item.physical["intersection"] for item in scores),
            "signals": sum(item.physical["signal"] for item in scores),
            "travel_time_s": sum(item.physical["travel_time_s"] for item in scores),
            "max_flood_severity": max(item.physical["flood_severity"] for item in scores), "flood_severity": per_km("flood_exposure"),
            "congestion": per_km("congestion"), "weather_severity": per_km("weather"), "time_profile_severity": per_km("time_profile"),
            # Length-weighted harmonic mean: total length / total capacity burden.
            "capacity_vph": kilometres / sum(item.criteria["capacity_burden"] for item in scores) if kilometres else 0.0,
            "road_types": list(dict.fromkeys(item.physical["road_type"] for item in scores)),
            "road_names": list(dict.fromkeys(e.road.name for e in edges if e.road.name))}


def _geometry(edges):
    coordinates = []
    for edge in edges:
        coordinates.extend(edge.road.geometry if not coordinates else edge.road.geometry[1:])
    return coordinates


def explain(network, adjacency, chosen, colony, origin, destination, segment_score):
    """segment_score(road) -> SegmentScore under this snapshot, as used for routing."""
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
            difference = {name: sum(weights[name] * (score(e).scaled[name] or 0) for e in detour)
                          - sum(weights[name] * (score(e).scaled[name] or 0) for e in taken) for name in FEATURES}
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
