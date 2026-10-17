"""Filter a source GraphML network using supplied city boundaries and explicit features."""
import argparse
import json
from pathlib import Path

import networkx as nx
from shapely import wkt
from shapely.geometry import Point, mapping, shape
from shapely.ops import unary_union

from src.routing.network import CITIES, RoadNetwork


def prepare(graph_path: Path, boundary_path: Path, feature_path: Path) -> RoadNetwork:
    collection = json.loads(boundary_path.read_text(encoding="utf-8"))
    if collection.get("type") != "FeatureCollection":
        raise ValueError("Boundaries must be a GeoJSON FeatureCollection")
    features = collection["features"]
    names = [item["properties"]["city"] for item in features]
    if len(names) != 5 or set(names) != set(CITIES):
        raise ValueError("Supply one boundary feature for each of the five study cities")
    polygons = {item["properties"]["city"]: shape(item["geometry"]) for item in features}
    if any(p.geom_type not in {"Polygon", "MultiPolygon"} or not p.is_valid or p.is_empty for p in polygons.values()):
        raise ValueError("Every city boundary must be a valid polygon")
    area = unary_union(list(polygons.values()))
    supplied = json.loads(feature_path.read_text(encoding="utf-8"))
    # OSMnx GraphML exports are directed MultiDiGraphs with x/y and WKT geometry.
    graph = nx.read_graphml(graph_path, force_multigraph=True)
    if not graph.is_directed():
        raise ValueError("Source road graph must be directed")
    inside = {}
    for node_id, data in graph.nodes(data=True):
        point = Point(float(data["x"]), float(data["y"]))
        cities = [city for city, polygon in polygons.items() if polygon.covers(point)]
        if cities:
            inside[str(node_id)] = (cities[0], point)
    roads, used = [], set()
    for source, target, key, data in graph.edges(keys=True, data=True):
        source, target = str(source), str(target)
        if source not in inside or target not in inside:
            continue
        raw_geometry = data.get("geometry")
        if raw_geometry is None:
            # A straight line cannot establish that an unknown curved road stays in scope.
            raise ValueError(f"Road {source}:{target}:{key} lacks full geometry; supply it before import")
        line = wkt.loads(raw_geometry)
        if line.geom_type != "LineString":
            raise ValueError(f"Road {source}:{target}:{key} needs LineString geometry")
        if not area.covers(line):
            continue
        coordinates = list(line.coords)
        if Point(coordinates[-1]).distance(inside[source][1]) < Point(coordinates[0]).distance(inside[source][1]):
            coordinates.reverse()
        road_id = f"{source}:{target}:{key}"
        if road_id not in supplied["roads"]:
            raise ValueError(f"Missing ten-feature record for road {road_id}")
        record = dict(supplied["roads"][road_id])
        record.update(id=road_id, source=source, target=target, geometry=coordinates)
        roads.append(record)
        used.update((source, target))
    nodes = []
    for node_id in sorted(used):
        if node_id not in supplied["nodes"]:
            raise ValueError(f"Missing intersection/signal record for node {node_id}")
        city, point = inside[node_id]
        record = dict(supplied["nodes"][node_id])
        record.update(id=node_id, city=city, longitude=point.x, latitude=point.y)
        nodes.append(record)
    return RoadNetwork.model_validate({
        "boundaries": {city: mapping(polygon) for city, polygon in polygons.items()},
        "nodes": nodes, "roads": roads, "cost_config": supplied["cost_config"],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("graph", "boundaries", "features", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    network = prepare(args.graph, args.boundaries, args.features)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(network.model_dump_json(indent=2), encoding="utf-8")
    print(f"Validated study network: {len(network.nodes)} nodes, {len(network.roads)} roads")


if __name__ == "__main__":
    main()
