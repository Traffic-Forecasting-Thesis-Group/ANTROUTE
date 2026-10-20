"""Preprocess the static input features from OSM (OSMnx) and Project NOAH into clean tables.

Raw sources (data/raw):
  study/five_city_boundaries.geojson   city polygons (OSM via Nominatim)
  spatial/metro_manila_road_graph.graphml   OSMnx drive graph
  spatial/flood_hazard.geojson         NOAH five-year flood hazard polygons (Var 1-3)

Outputs (data/processed/features):
  road_nodes.csv      intersections and traffic signals per node
  road_segments.csv   distance, road type, lanes, speed, capacity per directed segment
  segment_flood.csv   NOAH hazard class and flood risk per segment
and data/quality/feature_preprocessing.json (counts, exclusions, fallbacks, checks).

Run: python -m scripts.preprocess_road_features
"""
import ast
import hashlib
import csv
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
from pyproj import Geod
from shapely import make_valid, wkt
from shapely.geometry import LineString, Point, box, shape
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.strtree import STRtree

from src.routing.network import CITIES

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data/processed/features"
GEOD = Geod(ellps="WGS84")
NODE_FIELDS = ["node_id", "city", "latitude", "longitude", "street_count", "is_intersection", "traffic_signal", "signal_source"]
SEGMENT_FIELDS = ["road_id", "source_node", "target_node", "city", "road_name", "road_type", "distance_m", "geometry_source",
                  "lanes", "lanes_source", "speed_kph", "speed_source", "free_flow_time_s", "signal_at_target",
                  "road_capacity_vph", "capacity_source", "capacity_method", "capacity_uncertainty", "traffic_location", "geometry_wkt"]
FLOOD_FIELDS = ["road_id", "flood_km_class1", "flood_km_class2", "flood_km_class3", "flood_coverage", "flood_source"]
# Overlapping hazard polygons resolve to the highest class: a road portion inside classes 2 and 3 counts once, as class 3.
FLOOD_PRECEDENCE = (3, 2, 1)


def values(value):
    if isinstance(value, str) and value.startswith("["):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
    return value if isinstance(value, list) else [value]


def positive_number(value):
    for item in values(value):
        match = re.search(r"\d+(?:\.\d+)?", str(item))
        if match and float(match[0]) > 0:
            number = float(match[0])
            return number * 1.609344 if "mph" in str(item).lower() else number
    return None


def directional_lanes(data, fallback):
    # Ordinary OSM lanes describe both directions; do not assign the full count to each.
    oneway = str(data.get("oneway", "False")).lower() in {"true", "yes", "1", "-1"}
    reversed_edge = str(data.get("reversed", "False")).lower() == "true"
    directional = positive_number(data.get(f"lanes:{'backward' if reversed_edge else 'forward'}"))
    if directional:
        return directional, "osm_directional_lanes"
    total = positive_number(data.get("lanes"))
    if total:
        return (total if oneway else max(total / 2, 1)), "osm_lanes"
    return fallback, "road_type_default_lanes"


def road_geometry(graph, source, target, data):
    raw = data.get("geometry")
    if raw:
        line, provenance = wkt.loads(raw), "osm_polyline"
    else:
        if not str(graph.graph.get("created_with", "")).startswith("OSMnx"):
            raise ValueError("Only OSMnx GraphML supports the documented missing-geometry reconstruction")
        # Matches OSMnx graph_to_gdfs(fill_edge_geometry=True): unsimplified single
        # segments have endpoint geometry; simplified multi-segment roads have WKT.
        line = LineString([(float(graph.nodes[node]["x"]), float(graph.nodes[node]["y"])) for node in (source, target)])
        provenance = "osm_endpoint_segment_osmnx_convention"
    if line.geom_type != "LineString" or not line.is_valid or line.length <= 0:
        return None, provenance
    coordinates = list(line.coords)
    start = Point(float(graph.nodes[source]["x"]), float(graph.nodes[source]["y"]))
    if start.distance(Point(coordinates[-1])) < start.distance(Point(coordinates[0])):
        coordinates.reverse()
    return LineString(coordinates), provenance


def flood_index(path, area):
    """Hazard polygon parts that touch the study area, their classes, and the dataset extent (its bounding box)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    polygons, levels, bounds = [], [], []
    for feature in data["features"]:
        level = float(feature["properties"]["Var"])
        if level not in {1, 2, 3}:
            raise ValueError("Unexpected source flood hazard class")
        geometry = shape(feature["geometry"])
        bounds.append(geometry.bounds)
        for part in geometry.geoms if geometry.geom_type == "MultiPolygon" else [geometry]:
            if area.intersects(part):
                polygons.append(part if part.is_valid else make_valid(part))
                levels.append(int(level))
    extent = box(min(b[0] for b in bounds), min(b[1] for b in bounds), max(b[2] for b in bounds), max(b[3] for b in bounds))
    return STRtree(polygons), levels, extent


def geodesic_km(geometry):
    """WGS84 geodesic length (km) of the line parts of a geometry; points from boundary touches add nothing."""
    if geometry.is_empty:
        return 0.0
    if geometry.geom_type == "LineString":
        return abs(GEOD.geometry_length(geometry)) / 1000
    if hasattr(geometry, "geoms"):
        return sum(geodesic_km(part) for part in geometry.geoms)
    return 0.0


def flood_portions(line, tree, levels, extent):
    """Length (km) of the line in each NOAH class, overlaps resolved by FLOOD_PRECEDENCE, and coverage status.

    Coverage is "mapped" inside the hazard dataset's extent (no mapped class there means no mapped hazard, not
    verified safety) and "unknown" outside it.
    """
    window = line.envelope.buffer(1e-6)
    hits = tree.query(line, predicate="intersects")
    portions, higher = {}, None
    for hazard in FLOOD_PRECEDENCE:
        parts = [tree.geometries[i].intersection(window) for i in hits if levels[i] == hazard]
        if not parts:
            continue
        region = unary_union(parts)
        kilometres = geodesic_km(line.intersection(region if higher is None else region.difference(higher)))
        if kilometres > 0:
            portions[hazard] = kilometres
        higher = region if higher is None else higher.union(region)
    total, length = sum(portions.values()), geodesic_km(line)
    if total > length:  # Floating-point overlap at polygon edges; never longer than the road.
        portions = {hazard: km * length / total for hazard, km in portions.items()}
    return portions, "mapped" if extent.covers(line) else "unknown"


def write_csv(path, fields, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def preprocess(boundary_path, graph_path, flood_path, assumptions):
    boundaries = json.loads(boundary_path.read_text(encoding="utf-8"))
    polygons = {feature["properties"]["city"]: shape(feature["geometry"]) for feature in boundaries["features"]}
    if set(polygons) != set(CITIES) or any(not polygon.is_valid for polygon in polygons.values()):
        raise ValueError("Exactly five valid study boundaries are required")
    area = prep(unary_union(list(polygons.values())))
    cities = {city: prep(polygon) for city, polygon in polygons.items()}
    types = assumptions["road_types"]
    graph = nx.read_graphml(graph_path, force_multigraph=True)
    print(f"Source graph loaded: {len(graph)} nodes, {graph.number_of_edges()} edges", flush=True)
    inside = {}
    for node_id, data in graph.nodes(data=True):
        point = Point(float(data["x"]), float(data["y"]))
        if area.covers(point):
            inside[str(node_id)] = next(city for city, polygon in cities.items() if polygon.covers(point))
    tree, levels, extent = flood_index(flood_path, area)
    segments, floods, used = [], [], set()
    excluded, fallbacks = Counter(), Counter()
    for source, target, key, data in graph.edges(keys=True, data=True):
        source, target = str(source), str(target)
        if source not in inside or target not in inside:
            excluded["endpoint_outside_five_cities"] += 1
            continue
        road_type = next((name for name in map(str, values(data.get("highway", "unclassified"))) if name in types), None)
        # Bus-only, emergency, and explicitly private/no-access roads are not public drivable links.
        if road_type is None or str(data.get("access", "")).lower() in {"private", "no"}:
            excluded["not_public_driving"] += 1
            continue
        line, geometry_source = road_geometry(graph, source, target, data)
        if line is None or not area.covers(line):
            excluded["geometry_outside_or_invalid"] += 1
            continue
        distance = abs(GEOD.geometry_length(line))
        if not math.isfinite(distance) or distance <= 0:
            excluded["zero_length"] += 1
            continue
        settings = types[road_type]
        speed = positive_number(data.get("maxspeed"))
        speed_source = "osm_maxspeed" if speed else "road_type_default_speed"
        speed = speed or settings["speed_kph"]
        lanes, lanes_source = directional_lanes(data, settings["lanes"])
        signal = "traffic_signals" in values(graph.nodes[target].get("highway", ""))
        # K = directional lanes x assumed base saturation flow x g/C (g/C assumed at a signalised approach).
        # Lanes are already per direction (directional_lanes), so they are not counted twice.
        capacity = lanes * assumptions["saturation_flow_vph_per_lane"] * (assumptions["assumed_effective_green_ratio"] if signal else 1)
        city = inside[source]
        road_id = f"{source}:{target}:{key}"
        portions, coverage = flood_portions(line, tree, levels, extent)
        fallbacks.update([speed_source, lanes_source, geometry_source])
        segments.append({
            "road_id": road_id, "source_node": source, "target_node": target, "city": city,
            "road_name": "; ".join(str(item) for item in values(data.get("name", "")) if str(item) != "nan"),
            "road_type": road_type, "distance_m": round(distance, 3), "geometry_source": geometry_source,
            "lanes": lanes, "lanes_source": lanes_source, "speed_kph": speed, "speed_source": speed_source,
            "free_flow_time_s": round(distance / (speed / 3.6), 6), "signal_at_target": int(signal),
            "road_capacity_vph": capacity,
            "capacity_source": "estimated_" + lanes_source + ("_signal_green_ratio" if signal else ""),
            "capacity_method": "directional_lanes_x_assumed_base_saturation_flow_x_gC",
            "capacity_uncertainty": "unvalidated: assumed base saturation flow; "
                                    + ("lanes from OSM" if lanes_source != "road_type_default_lanes" else "lanes from road-type default")
                                    + ("; assumed g/C" if signal else ""),
            "traffic_location": road_id,  # Each segment has its own hourly traffic series.
            "geometry_wkt": line.wkt,
        })
        floods.append({"road_id": road_id, **{f"flood_km_class{c}": round(portions.get(c, 0.0), 9) for c in (1, 2, 3)},
                       "flood_coverage": coverage,
                       "flood_source": "noah_5yr_hazard_class_lengths" if portions else
                                       ("no_mapped_hazard_not_verified_safe" if coverage == "mapped" else "outside_hazard_dataset_extent")})
        used.update((source, target))
    nodes = []
    for node_id in sorted(used):
        data = graph.nodes[node_id]
        signal = "traffic_signals" in values(data.get("highway", ""))
        street_count = int(data.get("street_count", graph.degree(node_id)))
        nodes.append({"node_id": node_id, "city": inside[node_id], "latitude": float(data["y"]), "longitude": float(data["x"]),
                      "street_count": street_count, "is_intersection": int(street_count >= 3), "traffic_signal": int(signal),
                      "signal_source": "osm_traffic_signals_tag" if signal else "signal_not_recorded_in_osm"})
    return nodes, segments, floods, excluded, fallbacks


def validate(nodes, segments, floods, types):
    node_ids = {node["node_id"] for node in nodes}
    ids = [segment["road_id"] for segment in segments]
    checks = {
        "unique_node_ids": len(node_ids) == len(nodes),
        "unique_road_ids": len(set(ids)) == len(ids),
        "segment_endpoints_exist": all(s["source_node"] in node_ids and s["target_node"] in node_ids for s in segments),
        "positive_distance_speed_capacity": all(s["distance_m"] > 0 and s["speed_kph"] > 0 and s["road_capacity_vph"] > 0 for s in segments),
        "known_road_types": all(s["road_type"] in types for s in segments),
        "flood_row_per_segment": [f["road_id"] for f in floods] == ids,
        "flood_portions_within_road": all(
            min(f[f"flood_km_class{c}"] for c in (1, 2, 3)) >= 0
            and sum(f[f"flood_km_class{c}"] for c in (1, 2, 3)) <= s["distance_m"] / 1000 * (1 + 1e-6) + 1e-6
            for f, s in zip(floods, segments)),
        "flood_coverage_declared": all(f["flood_coverage"] in {"mapped", "unknown"} for f in floods),
        "all_five_cities_present": {s["city"] for s in segments} == set(CITIES),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Static feature validation failed: {', '.join(failed)}")
    return checks


def main():
    assumptions = json.loads((ROOT / "configs/study_assumptions.json").read_text(encoding="utf-8"))
    nodes, segments, floods, excluded, fallbacks = preprocess(
        ROOT / "data/raw/study/five_city_boundaries.geojson", ROOT / "data/raw/spatial/metro_manila_road_graph.graphml",
        ROOT / "data/raw/spatial/flood_hazard.geojson", assumptions)
    checks = validate(nodes, segments, floods, assumptions["road_types"])
    FEATURES.mkdir(parents=True, exist_ok=True)
    write_csv(FEATURES / "road_nodes.csv", NODE_FIELDS, nodes)
    write_csv(FEATURES / "road_segments.csv", SEGMENT_FIELDS, segments)
    write_csv(FEATURES / "segment_flood.csv", FLOOD_FIELDS, floods)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "nodes": len(nodes), "segments": len(segments),
        "segments_by_city": dict(Counter(s["city"] for s in segments)),
        "segments_by_road_type": dict(Counter(s["road_type"] for s in segments)),
        "intersection_nodes": sum(n["is_intersection"] for n in nodes),
        "signal_nodes": sum(n["traffic_signal"] for n in nodes),
        "excluded_source_edges": dict(excluded),
        "value_sources": dict(fallbacks),
        "flood_km_by_class": {c: round(sum(f[f"flood_km_class{c}"] for f in floods), 3) for c in (1, 2, 3)},
        "flood_coverage": dict(Counter(f["flood_coverage"] for f in floods)),
        "flood_method": "Line-polygon intersection per NOAH class; overlaps resolved to the highest class; WGS84 geodesic lengths. "
                        "Mapped susceptibility, not current inundation or closures.",
        "flood_source_sha256": hashlib.sha256((ROOT / "data/raw/spatial/flood_hazard.geojson").read_bytes()).hexdigest(),
        "checks": checks,
    }
    quality = ROOT / "data/quality/feature_preprocessing.json"
    existing = json.loads(quality.read_text(encoding="utf-8")) if quality.exists() else {}
    quality.write_text(json.dumps({**existing, "static": report}, indent=2), encoding="utf-8")
    print(f"Static features: {len(nodes)} nodes, {len(segments)} segments -> {FEATURES.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
