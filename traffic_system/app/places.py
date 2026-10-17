"""Local road-name search and in-scope snapping; no external Places queries."""
import math
import threading
from dataclasses import dataclass

import networkx as nx
from fastapi import HTTPException
from shapely.geometry import LineString, Point, shape
from shapely.ops import transform, unary_union
from shapely.prepared import prep
from shapely.strtree import STRtree
from pyproj import Transformer

from src.routing.network import RoadNetwork

LOCK = threading.Lock()
CACHED = None


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
        index = NetworkIndex(
            nodes, roads, connected,
            prep(unary_union([shape(geometry) for geometry in network.boundaries.values()])),
            [(node, f"{node.label} {node.city} {node.id}".casefold()) for node in network.nodes if node.id in connected],
            STRtree([transform(project, LineString(road.geometry)) for road in order]), order, project,
        )
        CACHED = (network, index)
        return index


def node_info(node):
    return {"id": node.id, "label": node.label or f"Road node {node.id} · {node.city}",
            "city": node.city, "latitude": node.latitude, "longitude": node.longitude}


def search_nodes(network, query, limit=12):
    tokens = query.strip().casefold().split()
    if not tokens:
        return []
    results, seen = [], set()
    for node, text in index_for(network).search:
        if all(token in text for token in tokens):
            # Keep suggestions distinct while allowing precise ID lookup.
            label = node.label or node.id
            if label in seen and query != node.id:
                continue
            seen.add(label)
            results.append(node_info(node))
            if len(results) >= limit:
                break
    return results


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
