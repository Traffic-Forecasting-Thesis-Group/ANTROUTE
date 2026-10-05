"""Locating the CCTV intersections on the OSM road graph."""

import networkx as nx

from src.ingestion import spatial_topology as st


def road_graph(*crossings):
    """One 'EDSA' x '<other>' crossing per (node id, other street, lat, lon), unconnected."""
    G = nx.MultiDiGraph()
    for node, other, lat, lon in crossings:
        G.add_node(node, y=lat, x=lon)
        G.add_node(node + 1, y=lat + 0.001, x=lon)
        G.add_node(node + 2, y=lat, x=lon + 0.001)
        G.add_edge(node + 1, node, name="EDSA")
        G.add_edge(node, node + 2, name=other)
    return G


def test_a_same_name_crossing_far_from_the_camera_is_ignored(monkeypatch):
    monkeypatch.setattr(st, "INTERSECTIONS", {"EDSA-Aurora": (st.EDSA_ALIASES, ["aurora boulevard"])})
    # Pasay's Aurora Blvd (Tramo) comes first in the graph, as it did in the real OSM download.
    G = road_graph((100, "Aurora Boulevard", 14.5377, 121.0031), (200, "Aurora Boulevard", 14.6216, 121.0500))
    assert st.locate_key_intersections(G) == {"EDSA-Aurora": 200}


def test_no_match_near_the_camera_leaves_the_intersection_unlocated(monkeypatch):
    monkeypatch.setattr(st, "INTERSECTIONS", {"EDSA-Aurora": (st.EDSA_ALIASES, ["aurora boulevard"])})
    G = road_graph((100, "Aurora Boulevard", 14.5377, 121.0031))
    assert st.locate_key_intersections(G) == {}


def test_camera_flags_from_a_previous_run_are_cleared(monkeypatch):
    monkeypatch.setattr(st, "INTERSECTIONS", {"EDSA-Aurora": (st.EDSA_ALIASES, ["aurora boulevard"])})
    G = road_graph((100, "Aurora Boulevard", 14.5377, 121.0031), (200, "Aurora Boulevard", 14.6216, 121.0500))
    G.nodes[100].update(is_cctv_node=True, cctv_label="EDSA-Aurora")    # as saved in the old graphml
    st.locate_key_intersections(G)
    flagged = [n for n, d in G.nodes(data=True) if d.get("is_cctv_node")]
    assert flagged == [200] and G.nodes[200]["cctv_label"] == "EDSA-Aurora"


def test_every_camera_intersection_has_an_anchor():
    assert set(st.INTERSECTION_ANCHORS) == set(st.INTERSECTIONS)
