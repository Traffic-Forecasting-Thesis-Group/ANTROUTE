from datetime import datetime, timezone

import numpy as np
import pytest

from app import risk_routing
from app.risk_routing import RouteOutsideNetworkError, _Network, _congestion_label, _haversine_km, nearest_node
from app.route_engine import local_departure


def make_network(node_coords, risk_low=0.3, risk_high=0.6, has_risk=True):
    """A _Network with just enough filled in for the coordinate/labelling helpers --
    graph/risk rows/free_flow_seconds are never touched by nearest_node or
    _congestion_label, which only checks whether any risk window is loaded."""
    return _Network(
        graph=None,
        risk_by_window={"2026-05-25T17:30:00": None} if has_risk else {},
        free_flow_seconds=None,
        node_coords=node_coords,
        risk_low=risk_low,
        risk_high=risk_high,
        events=[],
        event_intersections={},
    )


def test_haversine_zero_distance_for_the_same_point():
    p = (14.6307, 121.0457)
    assert _haversine_km(p, p) == pytest.approx(0.0, abs=1e-9)


def test_haversine_matches_a_known_distance():
    # Manila (14.5995, 120.9842) to Quezon City (14.6760, 121.0437) is ~11 km.
    manila = (14.5995, 120.9842)
    qc = (14.6760, 121.0437)
    assert _haversine_km(manila, qc) == pytest.approx(11.0, abs=1.5)


def test_nearest_node_picks_the_closest_within_range():
    net = make_network({1: (14.60, 121.00), 2: (14.61, 121.00), 3: (20.0, 120.0)})
    assert nearest_node(net, (14.601, 121.001)) == 1


def test_nearest_node_rejects_a_point_far_from_every_node():
    net = make_network({1: (14.60, 121.00)})
    with pytest.raises(RouteOutsideNetworkError):
        nearest_node(net, (10.0, 125.0))  # hundreds of km away


def test_congestion_label_buckets_at_the_tercile_cutoffs():
    net = make_network({}, risk_low=0.3, risk_high=0.6)
    assert _congestion_label(net, 0.1) == "clear"
    assert _congestion_label(net, 0.3) == "clear"
    assert _congestion_label(net, 0.45) == "moderate"
    assert _congestion_label(net, 0.6) == "moderate"
    assert _congestion_label(net, 0.9) == "heavy"


def test_describe_window_names_the_recorded_moment():
    note = risk_routing.describe_window("2026-05-25T17:30:00", True, datetime(2026, 10, 5, 17, 32))
    assert note == "Traffic based on recorded data from Mon May 25, 5:30 PM"


def test_describe_window_flags_a_departure_outside_the_recorded_peaks():
    note = risk_routing.describe_window("2026-05-25T08:30:00", False, datetime(2026, 10, 5, 12, 0))
    assert note.startswith("No traffic data recorded around 12:00 PM")
    assert "Mon May 25, 8:30 AM" in note


def test_local_departure_converts_an_offset_time_to_manila_wall_clock():
    utc = datetime(2026, 10, 5, 9, 15, tzinfo=timezone.utc)
    assert local_departure(utc) == datetime(2026, 10, 5, 17, 15)


def test_local_departure_keeps_a_naive_time_as_manila_time():
    assert local_departure(datetime(2026, 10, 5, 7, 45)) == datetime(2026, 10, 5, 7, 45)


def test_local_departure_defaults_to_now():
    assert local_departure(None).tzinfo is None


class _ChainGraph:
    """Stand-in WeightedGraph for a straight path 0 -> 1 -> ... where edge i joins i and i+1."""

    def __init__(self, distances):
        self.distance = np.array(distances, dtype=float)

    def index_of(self, node_id):
        return node_id

    def edge_id(self, u, v):
        assert v == u + 1
        return u


def test_via_roads_names_the_longest_roads_in_driving_order():
    net = make_network({})
    net.edge_names = np.array(["Shaw Boulevard", None, "EDSA", "EDSA", "Side Street", "Ortigas Avenue"], dtype=object)
    wg = _ChainGraph([400, 50, 900, 600, 30, 700])
    assert risk_routing._via_roads(net, wg, list(range(7))) == "Shaw Boulevard, EDSA, Ortigas Avenue"


def test_via_roads_is_none_without_road_names():
    net = make_network({})
    assert risk_routing._via_roads(net, _ChainGraph([100]), [0, 1]) is None


def test_congestion_is_unknown_without_a_scored_risk_file():
    net = make_network({}, has_risk=False)
    assert _congestion_label(net, 0.0) == "unknown"


def test_available_is_false_without_the_road_network(tmp_path, monkeypatch):
    risk_routing._build_network.cache_clear()
    monkeypatch.setattr(risk_routing, "SPATIAL_DIR", tmp_path / "missing_spatial")
    assert risk_routing.available() is False
    risk_routing._build_network.cache_clear()
