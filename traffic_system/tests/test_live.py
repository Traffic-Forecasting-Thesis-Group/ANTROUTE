"""Live sources: HERE parsing, tiling, segment matching and the collector (no network access)."""
import gzip
import json
from datetime import datetime

import pytest
from shapely import wkt
from shapely.ops import transform

from scripts import collect_live
from src.live import here
from src.live.matching import _TO_METRES, Segment, coverage, match
from src.routing.condition_update import MANILA

# A short east-west street in Manila (lon, lat) and a parallel line 5 m north of it.
WEST, EAST, LAT = 120.9850, 120.9870, 14.5600
NORTH_5M = LAT + 5 / 111_320


def segment(road_id, points, road_type="primary"):
    line = transform(_TO_METRES, wkt.loads("LINESTRING (" + ", ".join(f"{x} {y}" for x, y in points) + ")"))
    return Segment(road_id, road_type, line.length, line)


def flow_result(points, speed=5.0, free_flow=10.0):
    return {"location": {"description": "Test Ave", "shape": {"links": [{"points": [{"lat": y, "lng": x} for x, y in points]}]}},
            "currentFlow": {"speed": speed, "freeFlow": free_flow, "jamFactor": 4.0, "confidence": 0.9, "traversability": "open"}}


def test_tiles_cover_bbox_without_gaps():
    parts = here.tiles((0.0, 0.0, 0.5, 0.3), 0.25)
    assert len(parts) == 4
    assert min(t[0] for t in parts) == 0.0 and max(t[2] for t in parts) == pytest.approx(0.5)
    assert all(t[2] - t[0] <= 0.25 + 1e-12 and t[3] - t[1] <= 0.25 + 1e-12 for t in parts)


def test_flow_and_incident_parsing_skip_unusable_results():
    payload = {"results": [flow_result([(WEST, LAT), (EAST, LAT)]), {"location": {}, "currentFlow": {"speed": 1, "freeFlow": 2}},
                           {"location": {"shape": {"links": [{"points": [{"lat": LAT, "lng": WEST}, {"lat": LAT, "lng": EAST}]}]}},
                            "currentFlow": {"speed": 3}}]}
    items = here.flow_items(payload)
    assert len(items) == 1 and items[0]["speed_mps"] == 5.0 and items[0]["lines"][0][0] == (WEST, LAT)
    incidents = here.incident_items({"results": [{
        "location": {"shape": {"links": [{"points": [{"lat": LAT, "lng": WEST}, {"lat": LAT, "lng": EAST}]}]}},
        "incidentDetails": {"id": "x1", "type": "roadClosure", "roadClosed": True, "description": {"value": "Closed"}}}]})
    assert incidents[0]["road_closed"] and incidents[0]["description"] == "Closed"


def test_missing_key_is_reported_without_a_request(monkeypatch):
    monkeypatch.delenv("HERE_API_KEY", raising=False)
    with pytest.raises(here.LiveSourceError, match="HERE_API_KEY"):
        here.fetch("flow", (0, 0, 1, 1))


def test_http_error_does_not_echo_the_key(monkeypatch):
    monkeypatch.setenv("HERE_API_KEY", "secret-key")

    class Response:
        status_code = 401

    class Session:
        def get(self, url, params, timeout):
            return Response()

    with pytest.raises(here.LiveSourceError) as error:
        here.fetch("flow", (0, 0, 0.1, 0.1), session=Session())
    assert "secret-key" not in str(error.value) and "401" in str(error.value)


def test_matching_requires_proximity_and_same_direction():
    eastbound = segment("east", [(WEST, LAT), (EAST, LAT)])
    westbound = segment("west", [(EAST, LAT), (WEST, LAT)])
    far = segment("far", [(WEST, LAT + 0.001), (EAST, LAT + 0.001)], road_type="residential")  # about 110 m away
    items = here.flow_items({"results": [flow_result([(WEST, NORTH_5M), (EAST, NORTH_5M)])]})
    matched = match([eastbound, westbound, far], items)
    assert matched == {"east": 0}
    assert match([westbound], items, directional=False) == {"west": 0}
    report = coverage([eastbound, westbound, far], matched)
    assert report["total"]["matched"] == 1 and report["by_road_type"]["residential"]["matched"] == 0


def test_collector_saves_snapshot_and_logs_failures(tmp_path):
    out = tmp_path / "live" / "raw"
    log_path = tmp_path / "live" / "collection_log.csv"
    moment = datetime(2026, 10, 21, 8, 15, tzinfo=MANILA)

    def failing():
        raise here.LiveSourceError("HERE incidents request failed: HTTP 503")

    rows = collect_live.snapshot(moment, out, log_path, None, None, True, 0.25, fetchers={
        "flow": lambda: {"results": [flow_result([(WEST, LAT), (EAST, LAT)])]},
        "incidents": failing,
        "weather": lambda: {"source": "Weatherstack", "cities": {"Manila": {"current": {"precip": 0}}}},
    })
    assert [r["status"] for r in rows] == ["ok", "error", "ok"]
    saved = out / "2026-10-21" / "0815_flow.json.gz"
    with gzip.open(saved, "rb") as handle:
        record = json.loads(handle.read())
    assert record["captured_at_manila"].startswith("2026-10-21T08:15") and len(record["payload"]["results"]) == 1
    assert not (out / "2026-10-21" / "0815_incidents.json.gz").exists()
    assert "HTTP 503" in log_path.read_text(encoding="utf-8")


def test_next_tick_is_aligned_to_the_interval():
    assert collect_live.next_tick(datetime(2026, 10, 21, 8, 7, 30, tzinfo=MANILA), 15).strftime("%H:%M") == "08:15"
    assert collect_live.next_tick(datetime(2026, 10, 21, 23, 50, tzinfo=MANILA), 15).strftime("%d %H:%M") == "22 00:00"
    assert collect_live.floor_tick(datetime(2026, 10, 21, 8, 16, 40, tzinfo=MANILA), 15).strftime("%H:%M") == "08:15"


# ---------------------------------------------------------------- live conditions
from src.live import conditions  # noqa: E402

FLOOD_SEVERITY = {"1": 1 / 3, "2": 2 / 3, "3": 1.0}
SCENARIOS = {"clear": {"severity": 0}, "rain": {"severity": 0.4}, "heavy_rain": {"severity": 0.8}}


def record(road_id, road_type="primary", lanes=2, flood_km=(0.0, 0.0, 0.0), city="Manila"):
    return conditions.SegmentRecord(road_id, city, road_type, 215.0, lanes, 3800.0, 15.0, flood_km)


def incident(kind, points, closed=False, text=""):
    return {"location": {"shape": {"links": [{"points": [{"lat": y, "lng": x} for x, y in points]}]}},
            "incidentDetails": {"id": kind, "type": kind, "roadClosed": closed, "description": {"value": text}}}


def weather(precip=0.0):
    return {"cities": {"Manila": {"current": {"precip": precip, "weather_descriptions": ["Clear"]}}}}


def build(flow_results, incident_results, precip=0.0, records=None, segments=None):
    segments = segments or [segment("east", [(WEST, LAT), (EAST, LAT)]),
                            segment("side", [(WEST, LAT + 0.003), (EAST, LAT + 0.003)], road_type="residential")]
    records = records or {"east": record("east"), "side": record("side", road_type="residential", lanes=1, flood_km=(0.1, 0.0, 0.0))}
    return conditions.build(records, segments, {"results": flow_results}, {"results": incident_results}, weather(precip),
                            FLOOD_SEVERITY, SCENARIOS)


def test_live_congestion_travel_time_and_city_load():
    out = build([flow_result([(WEST, NORTH_5M), (EAST, NORTH_5M)], speed=4.0, free_flow=10.0)], [])
    east, side = out["segments"]["east"], out["segments"]["side"]
    assert east["congestion_source"] == "live" and east["congestion"] == pytest.approx(0.6)
    assert east["travel_time_s"] == pytest.approx(215.0 / 4.0)
    # The uncovered residential road has no group of its own, so it takes the overall median.
    assert side["congestion_source"] == "group_estimate" and side["congestion"] == pytest.approx(0.6)
    assert side["travel_time_s"] == pytest.approx(15.0 / 0.4)
    assert east["time_of_day"] == pytest.approx(0.6) == side["time_of_day"]
    assert out["summary"]["live_matched"] == 1


def test_flood_report_replaces_noah_exposure_and_cuts_capacity():
    side = [(WEST, LAT + 0.003), (EAST, LAT + 0.003)]
    out = build([flow_result([(WEST, NORTH_5M), (EAST, NORTH_5M)])], [incident("other", side, text="Flooded road, gutter deep")])
    s = out["segments"]["side"]
    assert s["flood_source"] == "live_report" and s["flood_portions"] == [(0.215, 1.0)]
    assert s["capacity_factors"]["flood"] == 0.5 and s["capacity_vph"] == pytest.approx(3800 * 0.5)
    e = out["segments"]["east"]
    assert e["flood_source"] == "noah_hazard" and e["flood_portions"] == []
    assert out["segments"]["side"]["flood_source"] == "live_report"


def test_noah_exposure_kept_without_a_report():
    out = build([], [])
    assert out["segments"]["side"]["flood_portions"] == [(0.1, pytest.approx(1 / 3))]


def test_incident_closure_and_rain_change_capacity():
    east = [(WEST, LAT), (EAST, LAT)]
    out = build([], [incident("accident", east)], precip=8.0)
    e = out["segments"]["east"]
    assert e["capacity_factors"] == {"incident": 0.5, "flood": 1.0, "weather": 0.9}
    assert e["capacity_vph"] == pytest.approx(3800 * 0.5 * 0.9) and e["weather"] == "heavy_rain"
    closed = build([], [incident("roadClosure", east, closed=True)])
    assert closed["segments"]["east"]["closed"] and not closed["segments"]["side"]["closed"]


def test_missing_city_weather_is_an_error_not_clear():
    records = {"east": record("east", city="Pasig")}
    with pytest.raises(ValueError, match="Pasig"):
        build([], [], records=records, segments=[segment("east", [(WEST, LAT), (EAST, LAT)])])


# ---------------------------------------------------------------- saved captures and live routing
from datetime import timedelta  # noqa: E402

from scripts import calibrate_live  # noqa: E402
from src.live import snapshot as live_files  # noqa: E402
from src.routing.condition_update import Conditions, live_conditions, validate_conditions  # noqa: E402
from src.routing.dynamic_trip import simulate_trip  # noqa: E402
from src.routing.dynamic_weight import prepare_graph, segment_values  # noqa: E402
from src.routing.network import RoadNetwork  # noqa: E402

AT = datetime(2026, 10, 22, 8, 0, tzinfo=MANILA)


def save(raw, moment, kind):
    collect_live.write_snapshot(raw, moment, kind, {"results": []})


def test_captures_need_flow_and_incidents_and_recent_weather(tmp_path):
    raw = tmp_path / "raw"
    save(raw, AT, "flow"), save(raw, AT, "incidents"), save(raw, AT, "weather")
    save(raw, AT + timedelta(minutes=15), "flow")  # incidents failed at 08:15
    save(raw, AT + timedelta(minutes=30), "flow"), save(raw, AT + timedelta(minutes=30), "incidents")
    assert live_files.captures(raw) == [AT, AT + timedelta(minutes=30)]
    assert live_files.nearest_capture(AT + timedelta(minutes=14), raw) == AT  # never the later 08:30 capture
    with pytest.raises(live_files.LiveDataUnavailable):
        live_files.nearest_capture(AT + timedelta(minutes=20), raw, tolerance=timedelta(minutes=15))
    assert live_files.weather_at(AT + timedelta(minutes=30), raw)[0] == AT
    with pytest.raises(live_files.LiveDataUnavailable, match="weather"):
        live_files.weather_at(AT + timedelta(minutes=91), raw)


def live_item(travel_time_s=10.0, congestion=0.2, source="live", flood=None, incident=1.0, closed=False):
    return {"congestion": congestion, "congestion_source": source, "travel_time_s": travel_time_s, "time_of_day": 0.3,
            "weather": "rain", "weather_severity": 0.4, "flood_portions": flood or [],
            "flood_source": "live_report" if flood else "noah_hazard", "capacity_vph": 4000.0 * incident,
            "capacity_factors": {"incident": incident, "flood": 1.0, "weather": 0.95}, "closed": closed}


def fake_snapshot(monkeypatch, by_time):
    """by_time: {capture time: {road_id: live item}}; closures come from the items."""
    def snap(moment, raw=None):
        items = by_time[moment]
        return live_files.LiveSnapshot(moment, moment, items, {"live_matched": 1, "segments": len(items)},
                                       frozenset(road for road, item in items.items() if item["closed"]))

    def nearest(moment, raw=None, tolerance=live_files.CAPTURE_TOLERANCE):
        earlier = [t for t in by_time if moment - tolerance <= t <= moment]
        if not earlier:
            raise live_files.LiveDataUnavailable("none")
        return max(earlier)
    monkeypatch.setattr(live_files, "live_snapshot", snap)
    monkeypatch.setattr(live_files, "nearest_capture", nearest)


def test_live_conditions_score_every_road_from_the_capture(network, monkeypatch):
    fake_snapshot(monkeypatch, {AT: {"ab": live_item(travel_time_s=120, incident=0.5, flood=[(0.1, 1.0)]),
                                     "bd": live_item(source="group_estimate"), "ad": live_item(closed=True)}})
    conditions = live_conditions(AT)
    graph = prepare_graph(network, conditions)
    assert [edge.road.id for edges in graph.values() for edge in edges] == ["ab", "bd"]  # HERE closed a -> d
    nodes = {node.id: node for node in network.nodes}
    score = segment_values(network, network.roads[0], conditions, nodes)
    # Live travel time is used as observed: the rain multiplier of the fixture (2x) is not applied again.
    assert score.physical["travel_time_s"] == 120 and score.criteria["travel_time"] == 2.0
    assert score.criteria["time_profile"] == pytest.approx(0.1 * 0.3) and score.criteria["weather"] == pytest.approx(0.04)
    assert score.criteria["flood_exposure"] == pytest.approx(0.1) and score.criteria["capacity_burden"] == pytest.approx(0.1 / 2000)
    assert set(score.flags) == {"live_flood_report", "live_capacity_reduced", "exceeds_development_max:travel_time"}
    assert segment_values(network, network.roads[1], conditions, nodes).flags == ("live_group_estimate",)


def test_live_conditions_must_be_consistent(network):
    with pytest.raises(ValueError, match="live"):
        validate_conditions(network, Conditions(departure_date=AT.date(), time_of_day=8, weather="clear", live_at=AT))
    with pytest.raises(ValueError, match="own date and hour"):
        validate_conditions(network, Conditions(departure_date=AT.date(), time_of_day=9, weather="live", live_at=AT))
    validate_conditions(network, live_conditions(AT))  # Outside the historical period is fine in live mode.


def test_live_trip_replays_captures_and_reroutes_around_a_new_closure(network_data, monkeypatch):
    back = {**network_data["roads"][0], "id": "ba", "source": "b", "target": "a",
            "geometry": list(reversed(network_data["roads"][0]["geometry"]))}
    network_data["roads"].append(back)
    network = RoadNetwork.model_validate(network_data)
    later = AT + timedelta(minutes=15)
    normal = {"ab": live_item(travel_time_s=1200), "bd": live_item(), "ad": live_item(travel_time_s=3000), "ba": live_item()}
    fake_snapshot(monkeypatch, {AT: normal, later: {**normal, "bd": live_item(closed=True)}})
    result = simulate_trip(network, "a", "d", AT, "live", frozenset(), [], "antroute")
    assert [road.id for road in result.roads] == ["ab", "ba", "ad"]
    assert result.searches[1].trigger == "live capture 08:00 -> 08:15"
    with pytest.raises(ValueError, match="weather events"):
        from src.routing.dynamic_trip import TripEvent
        simulate_trip(network, "a", "d", AT, "live", frozenset(), [TripEvent(at_minute=1, weather="rain")], "antroute")


def test_live_split_and_hourly_sample():
    split = calibrate_live.load_split(json.loads(calibrate_live.POLICY.read_text(encoding="utf-8")))
    assert calibrate_live.period_of(datetime(2026, 11, 1, 9, tzinfo=MANILA), split) == "excluded"
    assert calibrate_live.period_of(datetime(2026, 10, 28, 23, 45, tzinfo=MANILA), split) == "development"
    sample = calibrate_live.hourly_captures([AT + timedelta(minutes=m) for m in (45, 0, 15, 60)])
    assert sample == [AT, AT + timedelta(minutes=60)]
    with pytest.raises(ValueError, match="chronological"):
        calibrate_live.load_split({"calibration_split": {name: {"start": "2026-10-22", "end": "2026-10-23"}
                                                         for name in calibrate_live.PERIODS}})
