import numpy as np
import pytest

from app import risk_routing
from app.risk_routing import RouteOutsideNetworkError, _Network, _congestion_label, _haversine_km, nearest_node


def make_network(node_coords, risk_low=0.3, risk_high=0.6):
    """A _Network with just enough filled in for the coordinate/labelling helpers --
    graph/wg/free_flow_seconds are never touched by nearest_node or _congestion_label."""
    return _Network(
        graph=None,
        wg_antroute=None,
        wg_baseline=None,
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


def test_available_is_false_without_a_scored_risk_file(tmp_path, monkeypatch):
    risk_routing._network.cache_clear()
    monkeypatch.setattr(risk_routing.settings, "risk_edges_path", str(tmp_path / "does_not_exist.csv"))
    assert risk_routing.available() is False
    risk_routing._network.cache_clear()
