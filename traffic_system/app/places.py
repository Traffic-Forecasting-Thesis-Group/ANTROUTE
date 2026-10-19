"""Local place and road-name search and in-scope snapping; no external Places queries."""
import csv
import math
import threading
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
from fastapi import HTTPException
from shapely.geometry import LineString, Point, shape
from shapely.ops import transform, unary_union
from shapely.prepared import prep
from shapely.strtree import STRtree
from pyproj import Transformer

from src.routing.network import RoadNetwork

LOCK = threading.Lock()
CACHED = None
GAZETTEER = Path(__file__).resolve().parents[1] / "data/raw/study/places_gazetteer.csv"


@dataclass
class NetworkIndex:
    nodes: dict
    roads: dict
    connected: set
    area: object
    search: list
    road_tree: STRtree
    road_order: list
    project: object


def index_for(network: RoadNetwork) -> NetworkIndex:
    global CACHED
    with LOCK:
        if CACHED is not None and CACHED[0] is network:
            return CACHED[1]
        nodes = {node.id: node for node in network.nodes}
        roads = {road.id: road for road in network.roads}
        graph = nx.DiGraph((road.source, road.target) for road in network.roads)
        connected = max(nx.strongly_connected_components(graph), key=len)
        project = Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True).transform
        order = list(network.roads)
        roads_search = [(node_info(node), f"{node.label} {node.city} {node.id}".casefold())
                        for node in network.nodes if node.id in connected]
        index = NetworkIndex(
            nodes, roads, connected,
            prep(unary_union([shape(geometry) for geometry in network.boundaries.values()])),
            place_entries([nodes[node_id] for node_id in sorted(connected)]) + roads_search,
            STRtree([transform(project, LineString(road.geometry)) for road in order]), order, project,
        )
        CACHED = (network, index)
        return index


def place_entries(candidates):
    """Named OSM places (barangays, malls, schools, ...) as aliases of their nearest connected road node."""
    path = GAZETTEER
    if not path.exists() or not candidates:
        return []
    with path.open(encoding="utf-8") as stream:
        places = list(csv.DictReader(stream))
    latitude = np.array([node.latitude for node in candidates])
    longitude = np.array([node.longitude for node in candidates])
    entries = []
    for place in places:
        lat, lon = float(place["latitude"]), float(place["longitude"])
        distance = np.hypot((longitude - lon) * math.cos(math.radians(lat)), latitude - lat)
        node = candidates[int(distance.argmin())]
        label = f"{place['name']} · {place['category'].title()} · {place['city']}"
        entries.append(({**node_info(node), "label": label, "kind": "place"},
                         f"{place['name']} {place['category']} {place['city']}".casefold()))
    return entries


def node_info(node):
    return {"id": node.id, "label": node.label or f"Road node {node.id} · {node.city}",
            "city": node.city, "latitude": node.latitude, "longitude": node.longitude}


def search_nodes(network, query, limit=12):
    """Matches at a word start before other matches; within each, places before roads."""
    query = query.strip()
    tokens = query.casefold().split()
    if not tokens:
        return []
    matches, seen = [], set()
    for order, (info, text) in enumerate(index_for(network).search):
        if all(token in text for token in tokens):
            # Keep suggestions distinct while allowing precise ID lookup.
            if info["label"] in seen and query != info["id"]:
                continue
            seen.add(info["label"])
            starts = text.startswith(tokens[0]) or f" {tokens[0]}" in text
            matches.append((not starts, info.get("kind") != "place", order, info))
    return [{key: value for key, value in info.items() if key != "kind"} for *_, info in sorted(matches, key=lambda m: m[:3])[:limit]]


def checked_point(index, latitude, longitude):
    point = Point(longitude, latitude)
    if not index.area.covers(point):
        raise HTTPException(422, "Selected point is outside the five-city study area")
    return point


def nearest_node(network, latitude, longitude):
    index = index_for(network)
    checked_point(index, latitude, longitude)
    longitude_scale = math.cos(math.radians(latitude))
    node = min((index.nodes[node_id] for node_id in index.connected), key=lambda item:
               ((item.longitude - longitude) * longitude_scale) ** 2 + (item.latitude - latitude) ** 2)
    distance = math.hypot((node.longitude - longitude) * longitude_scale, node.latitude - latitude) * 111195
    if distance > 300:
        raise HTTPException(422, "No connected study road node within 300 meters of this point")
    return {**node_info(node), "snap_distance_m": round(distance, 1)}


def nearest_road(network, latitude, longitude):
    index = index_for(network)
    point = checked_point(index, latitude, longitude)
    projected = transform(index.project, point)
    position = int(index.road_tree.nearest(projected))
    road = index.road_order[position]
    distance = index.road_tree.geometries[position].distance(projected)
    if distance > 100:
        raise HTTPException(422, "No study road within 100 meters of this point")
    opposite = [candidate for candidate in network.roads if candidate.source == road.target
                and candidate.target == road.source and candidate.name == road.name]
    return {"id": road.id, "name": road.name or road.id, "geometry": road.geometry,
            "opposite_roads": [{"id": candidate.id, "name": candidate.name or candidate.id,
                                "geometry": candidate.geometry} for candidate in opposite],
            "snap_distance_m": round(distance, 1)}
