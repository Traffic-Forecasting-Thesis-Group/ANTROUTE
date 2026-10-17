import json
from datetime import date

import pytest

from src.routing.condition_update import Conditions, validate_conditions
from src.routing.estimation import estimation_registry
from src.routing.historical_traffic import bpr, demand_index
from src.routing.network import CITIES, FEATURES


def test_bpr_reference_formula_and_weekend_assumption():
    assumptions = {"bpr_alpha": .15, "bpr_beta": 4, "weekend_demand_factor": .75, "hourly_demand_capacity_ratio": [1] * 24,
                   "road_types": {"primary": {"traffic_factor": 1}},
                   "traffic_variation": {"seed": 0, "strength": 0, "day_coefficient": .2, "location_hour_coefficient": .25}}
    factor, congestion = bpr(1, assumptions)
    assert factor == pytest.approx(1.15)
    assert congestion == pytest.approx(1 - 1 / 1.15)
    weekday = demand_index("x", "Manila", "primary", date(2025, 6, 2), 8, assumptions)
    weekend = demand_index("x", "Manila", "primary", date(2025, 6, 1), 8, assumptions)
    assert weekday == 1 and weekend == .75


@pytest.mark.parametrize("day", [date(2025, 5, 31), date(2026, 6, 1)])
def test_outside_study_dates_are_rejected(network, day):
    with pytest.raises(ValueError, match="Departure date"):
        validate_conditions(network, Conditions(departure_date=day, time_of_day=8, weather="clear"))


def test_estimation_registry_covers_exactly_ten_features():
    from pathlib import Path
    assumptions = json.loads((Path(__file__).resolve().parents[1] / "configs/study_assumptions.json").read_text())
    registry = estimation_registry(assumptions)
    assert set(registry) == set(FEATURES)
    assert all(record["basis"] and record["source"] and record["classification"] for record in registry.values())
    assert "project assumptions" in registry["traffic_congestion"]["basis"]
    assert assumptions["weather_scenarios"]["rain"]["travel_time_multiplier"] == pytest.approx(1 / .925)


def test_historical_weather_is_date_specific(network, tmp_path, monkeypatch):
    from src.routing import historical_weather
    monkeypatch.setattr(historical_weather, "ROOT", tmp_path)
    path = tmp_path / "data/processed/historical_weather.json"
    path.parent.mkdir(parents=True)
    record = {"2025-06-01T08:00": {"precip_mm": 0, "description": "Clear", "grid_cell": "292_2420"},
              "2025-06-02T08:00": {"precip_mm": 9, "description": "Heavy rain", "grid_cell": "292_2420"}}
    path.write_text(json.dumps({"source": "test provider", "spatial_method": "test grid", "records": dict.fromkeys(CITIES, record)}))
    network.provenance["assumptions"] = {"weather_scenarios": {
        "clear": {"cost": 0, "travel_time_multiplier": 1},
        "heavy_rain": {"cost": .8, "travel_time_multiplier": 1.2}}}
    clear, first = historical_weather.historical_snapshot(date(2025, 6, 1), 8, network)
    rainy, second = historical_weather.historical_snapshot(date(2025, 6, 2), 8, network)
    assert clear["Manila"].travel_time_multiplier == 1
    assert rainy["Manila"].travel_time_multiplier == 1.2
    assert first["cities"]["Manila"]["localtime"] != second["cities"]["Manila"]["localtime"]
    with pytest.raises(ValueError, match="No historical weather"):
        historical_weather.historical_snapshot(date(2025, 6, 3), 8, network)


def test_api_rejects_outside_date_before_weather_lookup(network, monkeypatch):
    from fastapi.testclient import TestClient
    import app.main as service
    from app.study_policy import require_historical_release
    monkeypatch.setattr(service, "get_network", lambda: network)
    monkeypatch.setitem(service.app.dependency_overrides, require_historical_release, lambda: None)
    for road in network.roads:
        road.weather["historical"] = road.weather["clear"]
    with TestClient(service.app) as client:
        response = client.post("/routes/plan", json={"origin": "a", "destination": "d", "conditions": {
            "departure_date": "2026-06-01", "time_of_day": 8, "weather": "historical"}})
        assert response.status_code == 422
        assert "Departure date" in response.json()["detail"]
