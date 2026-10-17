import csv
import io
import json
from datetime import date

import pytest
from fastapi.testclient import TestClient

from src.routing import historical_traffic
from src.routing.condition_update import Conditions
from src.routing.datasets import FIELDS, coverage, csv_chunks, snapshot_rows
from src.routing.dynamic_weight import prepare_graph

ASSUMPTIONS = {"bpr_alpha": .15, "bpr_beta": 4, "weekend_demand_factor": .75, "hourly_demand_capacity_ratio": [1] * 24,
               "road_types": {"residential": {"traffic_factor": 1}, "primary": {"traffic_factor": 1.2}},
               "traffic_variation": {"seed": 42, "strength": .6, "day_coefficient": .2, "location_hour_coefficient": .25}}


@pytest.fixture
def table_network(network, tmp_path, monkeypatch):
    """Roads ab and bd share one location; ad uses a busier one. The table covers June 2, 08-09."""
    locations = {"ab": "san_juan__residential", "bd": "san_juan__residential", "ad": "san_juan__primary"}
    for road in network.roads:
        road.free_flow_time_s = 20
        road.traffic_location = locations[road.id]
        road.name = 'Road, "test"'
    path = tmp_path / historical_traffic.TABLE
    path.parent.mkdir(parents=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=historical_traffic.FIELDS)
        writer.writeheader()
        for location, road_type, congestion, factor in (("san_juan__residential", "residential", .2, 1.25),
                                                        ("san_juan__primary", "primary", .5, 2.0)):
            for hour in (8, 9):
                writer.writerow({"date": "2025-06-02", "hour": hour, "location_id": location, "city": "San Juan",
                                 "road_type": road_type, "day_type": "weekday", "traffic_congestion": congestion,
                                 "travel_time_factor": factor * (1 + hour - 8), "source": "test"})
    monkeypatch.setattr(historical_traffic, "ROOT", tmp_path)
    return network


def conditions(**changes):
    return Conditions(departure_date=date(2025, 6, 2), time_of_day=changes.pop("time_of_day", 8), weather="clear", **changes)


def test_table_generation_is_repeatable_and_varies_by_day_and_location():
    locations = [("a", "Manila", "residential"), ("b", "Manila", "primary")]
    rows = list(historical_traffic.table_rows(locations, ASSUMPTIONS, hours=72))
    assert rows == list(historical_traffic.table_rows(reversed(locations), ASSUMPTIONS, hours=72))
    assert set(rows[0]) == set(historical_traffic.FIELDS)
    assert not any("volume" in field for field in historical_traffic.FIELDS + FIELDS)
    eight = {(row["date"], row["location_id"]): row["traffic_congestion"] for row in rows if row["hour"] == 8}
    assert eight[("2025-06-02", "a")] != eight[("2025-06-03", "a")]
    assert eight[("2025-06-02", "a")] != eight[("2025-06-02", "b")]
    assert all(0 <= row["traffic_congestion"] < 1 and row["travel_time_factor"] >= 1 for row in rows)
    assert rows[0]["day_type"] == "weekend" and rows[-1]["day_type"] == "weekday"  # June 1, 2025 is a Sunday.


def test_routing_uses_dated_table_values(table_network):
    rows = {row["road_id"]: row for row in snapshot_rows(table_network, conditions())}
    assert rows["ab"]["traffic_congestion"] == .2 and rows["ab"]["travel_time"] == pytest.approx(25)
    assert rows["ad"]["traffic_congestion"] == .5 and rows["ad"]["travel_time"] == pytest.approx(40)
    later = {row["road_id"]: row for row in snapshot_rows(table_network, conditions(time_of_day=9))}
    assert later["ab"]["travel_time"] == pytest.approx(50)
    graph = prepare_graph(table_network, conditions(), traffic=historical_traffic.traffic_snapshot(date(2025, 6, 2), 8))
    for edges in graph.values():
        for edge in edges:
            assert rows[edge.road.id]["path_cost"] == pytest.approx(edge.cost)
    with pytest.raises(ValueError, match="No traffic record"):
        list(snapshot_rows(table_network, conditions(time_of_day=10)))


def test_closed_rows_are_retained_but_not_routable_and_csv_quotes_names(table_network):
    snapshot = conditions(closed_road_ids={"ab"})
    rows = list(snapshot_rows(table_network, snapshot))
    closed = next(row for row in rows if row["road_id"] == "ab")
    assert closed["closed"] and closed["path_cost"] is None
    assert all(edge.road.id != "ab" for edges in prepare_graph(table_network, snapshot).values() for edge in edges)
    decoded = list(csv.DictReader(io.StringIO("".join(csv_chunks(iter(rows))))))
    assert len(decoded) == len(table_network.roads)
    assert decoded[0]["road_name"] == 'Road, "test"'
    assert decoded[0]["traffic_classification"] == "estimated_not_measured"


def test_dataset_api_pagination_city_validation_export_map_and_route(table_network, monkeypatch):
    import app.main as service
    from app.study_policy import require_historical_release
    monkeypatch.setattr(service, "get_network", lambda: table_network)
    monkeypatch.setitem(service.app.dependency_overrides, require_historical_release, lambda: None)
    body = {"conditions": conditions().model_dump(mode="json")}
    with TestClient(service.app) as client:
        response = client.post("/dataset/snapshot?limit=1&offset=1", json=body)
        assert response.status_code == 200
        assert response.json()["total"] == 3 and len(response.json()["rows"]) == 1
        assert client.post("/dataset/snapshot", json={**body, "city": "Quezon City"}).status_code == 422
        assert client.post("/dataset/snapshot", json={**body, "simulation": {"seed": 1}}).status_code == 422
        bad = {"conditions": {**body["conditions"], "departure_date": "2026-06-01"}}
        assert client.post("/dataset/export", json=bad).status_code == 422
        exported = client.post("/dataset/export", json=body)
        assert len(list(csv.DictReader(io.StringIO(exported.text)))) == 3
        assert client.post("/dataset/map", json=body).json()["segment_count"] == 3
        route = client.post("/routes/plan", json={"origin": "a", "destination": "d", **body}).json()
        assert all(item["features"]["traffic_congestion"] in {.2, .5} for item in route["route"]["segment_features"])
        assert route["traffic"]["classification"] == "estimated_not_measured"
    summary = coverage(table_network)
    assert sum(summary["segments_by_city"].values()) == 3
    assert summary["traffic_locations"] == 2


def test_road_grid_weather_can_differ_inside_one_city(table_network, tmp_path, monkeypatch):
    from src.routing import historical_weather
    from src.routing.network import CITIES
    monkeypatch.setattr(historical_weather, "ROOT", tmp_path)
    path = tmp_path / "data/processed/historical_weather.json"
    clear = {"precip_mm": 0, "description": "Clear", "grid_cell": "clear"}
    rain = {"precip_mm": 10, "description": "Heavy rain", "grid_cell": "rain"}
    key = "2025-06-02T08:00"
    path.write_text(json.dumps({"source": "test", "spatial_method": "test", "records": {city: {key: clear} for city in CITIES},
        "node_grid_assignment": {"a": "clear", "b": "rain"},
        "records_by_grid": {"clear": {key: clear}, "rain": {key: rain}}}))
    table_network.provenance["assumptions"] = {"weather_scenarios": {
        "clear": {"cost": 0, "travel_time_multiplier": 1},
        "heavy_rain": {"cost": .8, "travel_time_multiplier": 1.2}}}
    for road in table_network.roads:
        road.weather["historical"] = road.weather["clear"]
    snapshot = Conditions(departure_date=date(2025, 6, 2), time_of_day=8, weather="historical")
    effects, _ = historical_weather.historical_snapshot(snapshot.departure_date, 8, table_network)
    assert effects["a"].cost == 0 and effects["b"].cost == .8
    rows = {row["road_id"]: row for row in snapshot_rows(table_network, snapshot)}
    assert rows["bd"]["travel_time"] == pytest.approx(25 * 1.2)
    assert rows["bd"]["weather_grid"] == "rain" and rows["ab"]["weather_grid"] == "clear"
