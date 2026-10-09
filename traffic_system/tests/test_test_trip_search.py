"""
Searching a test-trip stop by name offers its exact point, so a trip planned from search
suggestions is recognised as the test trip (app/routers/places.py).
"""

from app.routers.places import _test_stop_results, _test_stops


def test_camera_stops_are_found_by_name_at_their_exact_points():
    shaw = _test_stop_results("edsa shaw")
    assert shaw and shaw[0]["name"] == "EDSA-Shaw"
    assert (shaw[0]["lat"], shaw[0]["lng"]) == (14.580878, 121.053531)    # the test sheet's point
    assert shaw[0]["address"] == "14.580878, 121.053531"


def test_abbreviations_and_partial_names_match():
    assert [s["name"] for s in _test_stop_results("Quezon Ave")] == ["EDSA-Quezon Ave"]
    assert _test_stop_results("ayala")[0]["name"].startswith("EDSA-Ayala")


def test_citywide_stops_lose_the_coordinates_in_their_label():
    names = [s["name"] for s in _test_stop_results("saint andrew")]
    assert "Saint Andrew Street" in names


def test_an_unrelated_query_adds_no_test_stop():
    assert _test_stop_results("SM Megamall") == []


def test_every_stop_appears_once():
    points = [(lat, lng) for _, _, lat, lng in _test_stops()]
    assert len(points) == len(set(points)) and len(points) >= 50
