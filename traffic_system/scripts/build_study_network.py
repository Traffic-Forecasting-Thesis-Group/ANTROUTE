"""Build the scoped, traceable ten-feature network from collected source files."""
import ast
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
from pyproj import Geod
from shapely import wkt
from shapely.geometry import LineString, Point, mapping, shape
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.strtree import STRtree

from src.routing.network import CITIES, RoadNetwork
from src.routing.weather import observed_effect
from src.routing.estimation import traffic_estimate, estimation_registry
from src.routing.historical_weather import build_weather

ROOT = Path(__file__).resolve().parents[1]
GEOD = Geod(ellps="WGS84")


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
    direction = "backward" if reversed_edge else "forward"
    directional = positive_number(data.get(f"lanes:{direction}"))
    if directional:
        return directional, "estimated_capacity_from_osm_directional_lanes"
    total = positive_number(data.get("lanes"))
    if total:
        return (total if oneway else max(total / 2, 1)), "estimated_capacity_from_osm_lanes"
    return fallback, "estimated_capacity_from_road_type_lanes"


def road_geometry(graph, source, target, data):
    raw = data.get("geometry")
    if raw:
        line = wkt.loads(raw)
        provenance = "osm_polyline"
    else:
        if not str(graph.graph.get("created_with", "")).startswith("OSMnx"):
            raise ValueError("Only OSMnx GraphML supports the documented missing-geometry reconstruction")
        # Matches OSMnx graph_to_gdfs(fill_edge_geometry=True). Unsimplified single
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
    data = json.loads(path.read_text(encoding="utf-8"))
    polygons, levels = [], []
    for feature in data["features"]:
        geometry = shape(feature["geometry"])
        level = float(feature["properties"]["Var"])
        if level not in {1, 2, 3}:
            raise ValueError("Unexpected source flood hazard class")
        parts = geometry.geoms if geometry.geom_type == "MultiPolygon" else [geometry]
        for part in parts:
            if area.intersects(part):
                if not part.is_valid:
                    from shapely import make_valid
                    part = make_valid(part)
                polygons.append(part)
                levels.append(level)
    print(f"Flood index ready: {len(polygons)} in-scope polygons", flush=True)
    return STRtree(polygons), levels


def build(boundary_path, graph_path, flood_path, assumption_path, weather_path=None):
    boundaries = json.loads(boundary_path.read_text(encoding="utf-8"))
    polygons = {feature["properties"]["city"]: shape(feature["geometry"]) for feature in boundaries["features"]}
    if set(polygons) != set(CITIES):
        raise ValueError("Exactly the five study boundaries are required")
    if any(not geometry.is_valid for geometry in polygons.values()):
        raise ValueError("Invalid source city boundary")
    area = unary_union(list(polygons.values()))
    prepared_area = prep(area)
    prepared_cities = {city: prep(polygon) for city, polygon in polygons.items()}
    assumptions = json.loads(assumption_path.read_text(encoding="utf-8"))
    bpr = assumptions.get("traffic_model") == "bpr"
    hourly = assumptions["hourly_volume_capacity_ratio"] if bpr else assumptions["hourly_congestion"]
    if len(hourly) != 24 or any(value < 0 for value in hourly):
        raise ValueError("Expected 24 nonnegative hourly scenario values")
    weather = json.loads(weather_path.read_text(encoding="utf-8")) if weather_path and weather_path.exists() else None
    graph = nx.read_graphml(graph_path, force_multigraph=True)
    print(f"Source graph loaded: {len(graph)} nodes", flush=True)
    inside = {}
    for node_id, data in graph.nodes(data=True):
        point = Point(float(data["x"]), float(data["y"]))
        if prepared_area.covers(point):
            city = next(city for city, polygon in prepared_cities.items() if polygon.covers(point))
            inside[str(node_id)] = (city, point)
    tree, levels = flood_index(flood_path, prepared_area)
    roads, used, labels = [], set(), {}
    excluded, sources = Counter(), Counter()
    types = assumptions["road_types"]
    for source, target, key, data in graph.edges(keys=True, data=True):
        source, target = str(source), str(target)
        if source not in inside or target not in inside:
            excluded["outside_endpoints"] += 1
            continue
        road_types = [str(value) for value in values(data.get("highway", "unclassified"))]
        road_type = next((name for name in road_types if name in types), None)
        # Bus-only, emergency, and explicitly private/no-access roads are not public drivable links.
        if road_type is None or str(data.get("access", "")).lower() in {"private", "no"}:
            excluded["not_public_driving"] += 1
            continue
        line, geometry_source = road_geometry(graph, source, target, data)
        if line is None or not prepared_area.covers(line):
            excluded["geometry_outside_or_invalid"] += 1
            continue
        distance = abs(GEOD.geometry_length(line))
        if not math.isfinite(distance) or distance <= 0:
            excluded["zero_length"] += 1
            continue
        settings = types[road_type]
        speed = positive_number(data.get("maxspeed"))
        speed_source = "estimated_free_flow_from_posted_osm_speed" if speed else "estimated_free_flow_from_road_type"
        speed = speed or settings["speed_kph"]
        lanes, capacity_source = directional_lanes(data, settings["lanes"])
        hazards = [levels[int(index)] for index in tree.query(line, predicate="intersects")]
        hazard = max(hazards, default=0) / 3
        free_flow = distance / (speed / 3.6)
        ratios = [value * settings["traffic_factor"] for value in hourly]
        if bpr:
            traffic = [traffic_estimate(free_flow, ratio, assumptions) for ratio in ratios]
            travel_times = [round(value[0], 6) for value in traffic]
            congestion = [round(value[1], 6) for value in traffic]
        else:
            congestion = [round(min(value, .9), 6) for value in ratios]
            travel_times = [round(max(free_flow / max(1 - value, .1), .001), 3) for value in congestion]
        effects = dict(assumptions["weather_scenarios"])
        city = inside[source][0]
        if weather and city in weather["cities"]:
            effects["current"] = observed_effect(weather["cities"][city], effects).model_dump(exclude_none=True)
        if bpr:
            # The real date/hour-specific effect is attached by the snapshot loader.
            effects["historical"] = effects["clear"]
        signal_at_target = "traffic_signals" in values(graph.nodes[target].get("highway", ""))
        capacity = (lanes * assumptions["saturation_flow_vph_per_lane"] *
                    (assumptions["assumed_effective_green_ratio"] if signal_at_target else 1)) if bpr else lanes * assumptions["capacity_per_lane_vph"]
        name = "; ".join(str(item) for item in values(data.get("name", "")))
        provenance = {
            "distance": geometry_source,
            "traffic_congestion": "estimated_bpr_from_assumed_demand" if bpr else "estimated_hourly_scenario",
            "travel_time": "estimated_bpr_" + speed_source if bpr else speed_source,
            "road_type": "osm_highway_tag",
            "time_of_day": "exact_hour_with_assumed_peak_policy" if bpr else "estimated_hourly_scenario",
            "weather": "historical_provider_grid_with_estimated_effects" if bpr else ("weatherstack_observation_with_estimated_effects" if weather else "estimated_scenarios"),
            "flood_risk": "noah_mapped_hazard_exposure" if hazards else "no_mapped_hazard_not_verified_safe",
            "road_capacity": capacity_source,
        }
        roads.append({
            "id": f"{source}:{target}:{key}", "source": source, "target": target,
            "distance_m": round(distance, 3), "road_type": road_type,
            "road_capacity_vph": capacity,
            "flood_risk": hazard, "traffic_congestion": congestion,
            "travel_time_s": travel_times,
            "time_of_day": [float(hour in assumptions["peak_hours"]) for hour in range(24)] if bpr else congestion,
            "weather": effects,
            "geometry": list(line.coords), "name": name, "feature_sources": provenance,
            "free_flow_time_s": free_flow if bpr else None,
            "volume_capacity_ratio": ratios if bpr else None,
        })
        used.update((source, target))
        for node_id in (source, target):
            if name:
                labels.setdefault(node_id, set()).add(name)
        sources.update(provenance.values())
    nodes = []
    for node_id in sorted(used):
        city, point = inside[node_id]
        data = graph.nodes[node_id]
        signal = "traffic_signals" in values(data.get("highway", ""))
        nodes.append({
            "id": node_id, "city": city, "latitude": point.y, "longitude": point.x,
            "is_intersection": int(data.get("street_count", graph.degree(node_id))) >= 3,
            "traffic_signal": signal,
            "label": f"{' / '.join(sorted(labels.get(node_id, []))[:3]) or 'Unnamed road'} · {city}",
            "feature_sources": {"number_of_intersections": "derived_osm_street_count",
                                "traffic_signal": "osm_signal_tag" if signal else "signal_not_recorded_in_osm"},
        })
    if not roads:
        raise ValueError("No valid roads remained in the study area")
    print(f"Validating network: {len(nodes)} nodes, {len(roads)} directed roads", flush=True)
    result = RoadNetwork.model_validate({
        "boundaries": {city: mapping(polygon) for city, polygon in polygons.items()},
        "nodes": nodes, "roads": roads,
        "cost_config": {"road_type_costs": {name: settings["cost"] for name, settings in types.items()}},
        "provenance": {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "mode": "historical_period_with_estimates" if bpr else "mapped_network_with_labeled_estimates",
            "period": {"start": "2025-06-01", "end": "2026-05-31"},
            "boundaries": [feature["properties"] for feature in boundaries["features"]],
            "graph_created_at": graph.graph.get("created_date"), "graph_source": "OpenStreetMap via OSMnx",
            "flood_source": "retained Project NOAH Metro Manila five-year flood hazard polygons; Var/3; maximum road intersection",
            "assumptions": assumptions, "assumptions_sha256": hashlib.sha256(assumption_path.read_bytes()).hexdigest(),
            "source_counts": dict(sources), "excluded_roads": dict(excluded),
            "weather": weather,
            "estimation_registry": estimation_registry(assumptions) if bpr else {},
            "limitations": [
                "Traffic uses an assumed V/C demand profile and BPR model, not historical traffic measurements.",
                "Capacity and travel time are estimates, including on roads with OSM lane/speed tags.",
                "Unrecorded signals are treated as absent; OSM signal coverage is incomplete.",
                "Flood exposure is a mapped hazard class, not measured water depth or current passability; bridges may be elevated.",
                "Weather effects are scenario assumptions; current observations carry their original timestamps.",
                "The time-of-day peak policy is distinct from congestion but correlated; objective weights are uncalibrated.",
                "September 2026 geometry, signals and classifications are held constant as an estimated surrogate for June 2025–May 2026.",
                "Historical weather grid assignment and speed effects are estimates, not road-level weather measurements.",
                "The source graph does not encode turn restrictions in this node-edge routing model.",
            ],
        },
    })
    return result


def main():
    output = ROOT / "data/processed"
    output.mkdir(exist_ok=True, parents=True)
    network = build(ROOT / "data/raw/study/five_city_boundaries.geojson",
                    ROOT / "data/raw/spatial/metro_manila_road_graph.graphml",
                    ROOT / "data/raw/spatial/flood_hazard.geojson",
                    ROOT / "configs/study_assumptions.json")
    weather_path = output / "historical_weather.json"
    weather_bundle = build_weather(network.boundaries, weather_path)
    network.provenance["historical_weather"] = {key: value for key, value in weather_bundle.items() if key != "records"}
    path = output / "study_network.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(network.model_dump_json(), encoding="utf-8")
    temporary.replace(path)
    policy = json.loads((ROOT / "configs/historical_study.json").read_text(encoding="utf-8"))
    artifacts = ["data/processed/study_network.json", "data/processed/historical_weather.json", "configs/study_assumptions.json", "configs/historical_study.json"]
    manifest = {"mode": "historical_period_with_estimates", "period": policy,
                "hours_per_city": weather_bundle["hours_per_city"],
                "artifacts": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in artifacts}}
    (output / "study_release.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    registry = network.provenance["estimation_registry"]
    (output / "estimation_registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    (output / "study_network_summary.json").write_text(json.dumps({
        "nodes": len(network.nodes), "roads": len(network.roads), "cities": dict(Counter(node.city for node in network.nodes)),
        "signal_nodes": sum(node.traffic_signal for node in network.nodes), "mode": manifest["mode"],
        "weather_hours_per_city": manifest["hours_per_city"], "period": policy,
    }, indent=2), encoding="utf-8")
    print(f"Published estimation release: {len(network.nodes)} nodes, {len(network.roads)} roads; 8760 provider-weather hours per city", flush=True)


if __name__ == "__main__":
    main()
