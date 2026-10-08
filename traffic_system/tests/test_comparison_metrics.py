"""
GET /routes/comparison-metrics serves the thesis routing evaluation only (Section 3.9,
Appendix 3 Table 6) -- never the Congestion Risk Score forecast table, which has its own
endpoint. metrics.json is built here the way scripts/evaluate_routing.py builds it.
"""

import json
from dataclasses import asdict

import pytest
from fastapi.testclient import TestClient

from app import evaluation
from app.main import app
from src.routing.significance import compare_paired

# Ten paired test trips: ANTROUTE closer to Apple Maps on every metric. ETA errors in seconds.
TRIALS = {
    "route_optimality": ([96, 94, 97, 93, 95, 96, 92, 98, 94, 95], [85, 84, 88, 82, 86, 83, 85, 87, 84, 86]),
    "mae": ([120, 130, 110, 140, 125, 118, 135, 128, 122, 131], [270, 280, 260, 290, 275, 268, 285, 278, 272, 281]),
    "rmse": ([170, 180, 160, 190, 175, 168, 185, 178, 172, 181], [370, 380, 360, 390, 375, 368, 385, 378, 372, 381]),
    "mse": ([32400] * 5 + [33000] * 5, [138000] * 5 + [139000] * 5),
    "mape": ([6.5, 7.0, 6.2, 7.4, 6.8, 6.6, 7.1, 6.9, 6.7, 7.0], [14.5, 15.0, 14.2, 15.4, 14.8, 14.6, 15.1, 14.9, 14.7, 15.0]),
    "r_squared": ([0.9, 0.92, 0.91, 0.89, 0.93, 0.9, 0.91, 0.92, 0.9, 0.91], [0.74, 0.73, 0.75, 0.72, 0.76, 0.74, 0.73, 0.75, 0.74, 0.74]),
}


@pytest.fixture
def evaluated(tmp_path, monkeypatch):
    rows = [{"scenario_type": "overall", **asdict(compare_paired(m, p, b))} for m, (p, b) in TRIALS.items()]
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"significance": rows}, default=float), encoding="utf-8")
    monkeypatch.setattr(evaluation, "ROUTING_METRICS", path)
    evaluation._cache.clear()
    return path


def test_the_comparison_shows_the_thesis_metrics_against_improved_aco(evaluated):
    body = TestClient(app).get("/routes/comparison-metrics").json()
    assert body["source"] == "routing" and body["baseline_name"] == "Improved ACO (Cheng 2023)"
    assert [r["metric"] for r in body["metrics"]] == ["Route Optimality", "MAE", "RMSE", "MSE", "MAPE", "R²"]
    by = {r["metric"]: r for r in body["metrics"]}
    assert by["Route Optimality"]["antroute"] == "95.0%" and by["Route Optimality"]["baseline"] == "85.0%"
    assert by["MAE"]["antroute"] == "2.1 min" and by["MAE"]["baseline"] == "4.6 min"   # seconds shown in minutes
    assert by["R²"]["antroute"] == "0.91"
    assert body["route_optimality"] == {"antroute": 95.0, "baseline": 85.0, "n_trials": 10}


def test_improvement_follows_equations_7_and_8(evaluated):
    by = {r["metric"]: r for r in TestClient(app).get("/routes/comparison-metrics").json()["metrics"]}
    # Equation 8 (higher is better): (95 - 85) / 85
    assert by["Route Optimality"]["improvement_pct"] == pytest.approx(11.8, abs=0.05)
    # Equation 7 (lower is better): (baseline - proposed) / baseline, positive = ANTROUTE better
    mae_p, mae_b = sum(TRIALS["mae"][0]) / 10, sum(TRIALS["mae"][1]) / 10
    assert by["MAE"]["improvement_pct"] == pytest.approx(round((mae_b - mae_p) / mae_b * 100, 1))
    assert all(r["improvement_pct"] > 0 for r in by.values())          # better on every metric here
    paired = [r for m, r in by.items() if m != "R²"]                    # R² has no per-trip pairing
    assert all(r["significant"] and r["p_value"] < 0.05 for r in paired)
    assert by["R²"]["p_value"] is None


TRIP = {
    "trip_id": "E01", "scenario_type": "incident_exposed", "window": "2026-05-25T18:20:00",
    "stop_nodes": "21834288 8462613410",
    "route_optimality_antroute": 86.4, "route_optimality_baseline": 65.5,
    "mae_antroute": 1014.5, "mae_baseline": 1176.8, "rmse_antroute": 1014.5, "rmse_baseline": 1176.8,
    "mse_antroute": 1029124.3, "mse_baseline": 1384922.6, "mape_antroute": 76.9, "mape_baseline": 67.6,
    "r_squared_antroute": None, "r_squared_baseline": None,   # one leg: R² undefined
}


@pytest.fixture
def per_trip(tmp_path, monkeypatch):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"significance": [], "per_trip": [TRIP]}), encoding="utf-8")
    monkeypatch.setattr(evaluation, "ROUTING_METRICS", path)
    evaluation._cache.clear()


def test_a_planned_test_trip_gets_its_own_metrics(per_trip):
    result = evaluation.get_trip_evaluation([21834288, 8462613410], "2026-03-02T18:20:00")
    assert result.status == "evaluated" and result.trip_id == "E01"
    assert result.route_optimality.antroute == 86.4 and result.route_optimality.baseline == 65.5
    by = {r.metric: r for r in result.metrics}
    assert "R²" not in by                                       # not shown for a single leg
    assert by["MAE"].antroute == "16.9 min" and by["MAE"].improvement_pct == pytest.approx(13.8, abs=0.05)
    assert by["MAPE"].improvement_pct < 0                       # baseline better on this trip


def test_same_stops_at_another_time_points_to_the_evaluated_time(per_trip):
    result = evaluation.get_trip_evaluation([21834288, 8462613410], "2026-03-02T07:30:00")
    assert result.status == "other_time" and result.metrics == [] and result.route_optimality is None
    assert "6:20 PM" in result.message


def test_a_trip_that_was_not_tested_gets_no_borrowed_numbers(per_trip):
    result = evaluation.get_trip_evaluation([1, 2], "2026-05-25T18:20:00")
    assert result.status == "not_evaluated" and result.metrics == [] and result.route_optimality is None


def test_eta_errors_are_pooled_over_legs_not_averaged_over_trips(tmp_path, monkeypatch):
    rows = [{"scenario_type": "overall", **asdict(compare_paired(m, p, b))} for m, (p, b) in TRIALS.items()]
    pooled = {
        "antroute": {"overall": {"mae": 600.0, "rmse": 720.0, "mse": 518400.0, "mape": 30.0, "r_squared": 0.55}},
        "baseline": {"overall": {"mae": 900.0, "rmse": 960.0, "mse": 921600.0, "mape": 40.0, "r_squared": 0.40}},
    }
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"significance": rows, "pooled": pooled}, default=float), encoding="utf-8")
    monkeypatch.setattr(evaluation, "ROUTING_METRICS", path)
    evaluation._cache.clear()
    by = {r["metric"]: r for r in TestClient(app).get("/routes/comparison-metrics").json()["metrics"]}
    assert by["Route Optimality"]["antroute"] == "95.0%"                 # still the mean over trips
    assert by["MAE"]["antroute"] == "10.0 min" and by["MAE"]["baseline"] == "15.0 min"
    assert by["R²"]["antroute"] == "0.55" and by["R²"]["p_value"] is None  # no per-trip pairing
    assert by["MAE"]["p_value"] is not None                              # Wilcoxon stays per trip


def test_without_the_evaluation_it_says_so_instead_of_showing_another_table(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "ROUTING_METRICS", tmp_path / "missing.json")
    evaluation._cache.clear()
    response = TestClient(app).get("/routes/comparison-metrics")
    assert response.status_code == 404
    assert "evaluate_routing.py" in response.json()["detail"]
