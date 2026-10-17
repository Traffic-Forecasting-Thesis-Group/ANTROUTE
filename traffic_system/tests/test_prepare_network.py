import json

import networkx as nx
import pytest
from shapely.geometry import LineString

from scripts.prepare_network import prepare


def write_inputs(tmp_path, network_data, *, missing_geometry=False):
    graph = nx.MultiDiGraph()
    for node in network_data["nodes"]:
        graph.add_node(node["id"], x=node["longitude"], y=node["latitude"])
    # Out-of-scope source material must not appear in the final network.
    graph.add_node("outside", x=-1, y=0.5)
    graph.add_edge("a", "outside", key=0, geometry="LINESTRING (0.2 0.5, -1 0.5)")
    for road in network_data["roads"]:
        attributes = {} if missing_geometry else {"geometry": LineString(road["geometry"]).wkt}
        graph.add_edge(road["source"], road["target"], key=0, **attributes)
    graph_path = tmp_path / "source.graphml"
    boundary_path = tmp_path / "boundaries.geojson"
    feature_path = tmp_path / "features.json"
    nx.write_graphml(graph, graph_path)
    boundary_path.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"city": city}, "geometry": geometry}
        for city, geometry in network_data["boundaries"].items()
    ]}), encoding="utf-8")
    feature_path.write_text(json.dumps({
        "nodes": {node["id"]: {key: node[key] for key in ("is_intersection", "traffic_signal")} for node in network_data["nodes"]},
        "roads": {f'{road["source"]}:{road["target"]}:0': {
            key: value for key, value in road.items() if key not in {"id", "source", "target", "geometry"}
        } for road in network_data["roads"]},
        "cost_config": network_data["cost_config"],
    }), encoding="utf-8")
    return graph_path, boundary_path, feature_path


def test_preparation_filters_outside_material_and_validates(network_data, tmp_path):
    paths = write_inputs(tmp_path, network_data)
    network = prepare(*paths)
    assert {node.id for node in network.nodes} == {"a", "b", "d"}
    assert {road.id for road in network.roads} == {"a:b:0", "b:d:0", "a:d:0"}


def test_missing_geometry_does_not_become_an_invented_straight_road(network_data, tmp_path):
    paths = write_inputs(tmp_path, network_data, missing_geometry=True)
    with pytest.raises(ValueError, match="lacks full geometry"):
        prepare(*paths)


def test_missing_feature_record_stops_preparation(network_data, tmp_path):
    paths = write_inputs(tmp_path, network_data)
    features = json.loads(paths[2].read_text(encoding="utf-8"))
    del features["roads"]["a:b:0"]
    paths[2].write_text(json.dumps(features), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing ten-feature record"):
        prepare(*paths)
