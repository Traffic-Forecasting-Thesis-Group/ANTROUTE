"""
GET /routes/comparison-metrics serves the thesis routing evaluation only (Section 3.9,
Appendix 3 Table 6) -- never the Congestion Risk Score forecast table, which has its own
endpoint. metrics.json is built here the way scripts/evaluate_routing.py builds it: Route
Optimality against Google Maps, congestion and event exposure, route length descriptive, and
no ETA metric.
"""

import json
from dataclasses import asdict

import pytest
from fastapi.testclient import TestClient

from app import evaluation
from app.main import app
from src.routing.significance import compare_paired

# Ten paired test trips: ANTROUTE better on every metric.
TRIALS = {
    "route_optimality": ([96, 94, 97, 93, 95, 96, 92, 98, 94, 95], [85, 84, 88, 82, 86, 83, 85, 87, 84, 86]),
    "congestion_exposure": ([4.1, 3.9, 4.4, 3.8, 4.0, 4.2, 3.7, 4.3, 4.0, 3.6], [6.0, 5.8, 6.4, 5.9, 6.1, 6.2, 5.7, 6.3, 6.0, 5.6]),
    "event_exposure": ([0.0, 0.1, 0.0, 0.2, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0], [0.5, 0.6, 0.4, 0.9, 0.7, 0.5, 0.6, 0.8, 0.4, 0.5]),
    "avoidable_event_exposure": ([0.0] * 9 + [0.1], [0.3, 0.4, 0.2, 0.5, 0.3, 0.4, 0.2, 0.6, 0.3, 0.4]),
}
MEAN_SD = {"overall": {"distance_m": {"antroute": {"mean": 8400.0, "sd": 1.0}, "baseline": {"mean": 7900.0, "sd": 1.0}}}}
REFERENCE = {
    "reference": "Google Maps",
    "access_methods": ["Google Maps web (desktop), Depart at"],
    "traffic_models": ["typical_range_upper"],
    "departure_dates": {"first": "2026-10-19", "last": "2026-10-19", "n_distinct": 1},
    "queried_on": {"first": "2026-10-14", "last": "2026-10-15", "n_distinct": 2},
    "departure_slots": ["Mon 07:30", "Mon 17:10"],
}
MODEL = {"checkpoint": "/content/drive/MyDrive/MMDA_CHECKPOINTS/stgnn_edge_v8.pt", "use_text": True}


def write_report(path, scenarios=("overall",), **extra):
    rows = [
        {"scenario_type": scenario, **asdict(compare_paired(m, p, b))}
        for scenario in scenarios
        for m, (p, b) in TRIALS.items()
    ]
    path.write_text(json.dumps({"significance": rows, **extra}, default=float), encoding="utf-8")


@pytest.fixture
def evaluated(tmp_path, monkeypatch):
    path = tmp_path / "metrics.json"
    write_report(path, per_trial_mean_sd=MEAN_SD, reference=REFERENCE, model=MODEL)
    monkeypatch.setattr(evaluation, "ROUTING_METRICS", path)
    evaluation._cache.clear()
    return path


def test_the_comparison_shows_the_thesis_metrics_against_improved_aco(evaluated):
    body = TestClient(app).get("/routes/comparison-metrics").json()
    assert body["source"] == "routing" and body["baseline_name"] == "Improved ACO (Cheng 2023)"
    assert [r["metric"] for r in body["metrics"]] == [
        "Route Optimality", "Congestion exposure", "Event exposure", "Avoidable event exposure",
        "Route length (descriptive)",
    ]
    by = {r["metric"]: r for r in body["metrics"]}
    assert by["Route Optimality"]["antroute"] == "95.0%" and by["Route Optimality"]["baseline"] == "85.0%"
    assert by["Congestion exposure"]["antroute"] == "4.00" and by["Congestion exposure"]["baseline"] == "6.00"
    assert by["Route length (descriptive)"]["antroute"] == "8.40 km"
    assert body["route_optimality"] == {"antroute": 95.0, "baseline": 85.0, "n_trials": 10}


def test_no_eta_metric_is_shown(evaluated):
    body = TestClient(app).get("/routes/comparison-metrics").json()
    shown = {r["metric"] for r in body["metrics"]}
    assert not shown & {"MAE", "RMSE", "MSE", "MAPE", "R²"}
    assert "eta_calibrated" not in body


def test_the_description_records_the_google_maps_reference_and_the_model(evaluated):
    description = TestClient(app).get("/routes/comparison-metrics").json()["description"]
    assert "Google Maps (Google Maps web (desktop), Depart at)" in description
    assert "typical_range_upper" in description and "Mon 07:30" in description
    assert "2026-10-14 to 2026-10-15" in description                  # when the lookups were made
    assert "stgnn_edge_v8.pt (text branch on)" in description
    assert "Apple" not in description


def test_improvement_follows_equations_7_and_8(evaluated):
    by = {r["metric"]: r for r in TestClient(app).get("/routes/comparison-metrics").json()["metrics"]}
    # Equation 8 (higher is better): (95 - 85) / 85
    assert by["Route Optimality"]["improvement_pct"] == pytest.approx(11.8, abs=0.05)
    # Equation 7 (lower is better): (baseline - proposed) / baseline, positive = ANTROUTE better
    assert by["Congestion exposure"]["improvement_pct"] == pytest.approx(33.3, abs=0.05)
    tested = [by[m] for m in ("Route Optimality", "Congestion exposure", "Event exposure", "Avoidable event exposure")]
    assert all(r["significant"] and r["p_value"] < 0.05 for r in tested)
    length = by["Route length (descriptive)"]
    assert length["improvement_pct"] is None and length["p_value"] is None   # descriptive, never tested


def test_an_unrecorded_reference_is_said_to_be_unrecorded(tmp_path, monkeypatch):
    path = tmp_path / "metrics.json"
    write_report(path)
    monkeypatch.setattr(evaluation, "ROUTING_METRICS", path)
    evaluation._cache.clear()
    description = TestClient(app).get("/routes/comparison-metrics").json()["description"]
    assert "not recorded" in description and "checkpoint behind these risk scores is not recorded" in description


TRIP = {
    "trip_id": "E01", "scenario_type": "incident_exposed", "window": "2026-05-25T18:20:00",
    "stop_nodes": "21834288 8462613410",
    "route_optimality_antroute": 86.4, "route_optimality_baseline": 65.5,
    "congestion_exposure_antroute": 3.2, "congestion_exposure_baseline": 4.0,
    "event_exposure_antroute": 0.0, "event_exposure_baseline": 0.75,
    "distance_m_antroute": 5200.0, "distance_m_baseline": 4800.0,
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
    assert by["Congestion exposure"].improvement_pct == pytest.approx(20.0)
    assert by["Event exposure"].antroute == "0.00" and by["Event exposure"].improvement_pct == pytest.approx(100.0)
    assert by["Route length (descriptive)"].antroute == "5.20 km"
    assert "Google Maps" in result.message


def test_same_stops_at_another_time_points_to_the_evaluated_time(per_trip):
    result = evaluation.get_trip_evaluation([21834288, 8462613410], "2026-03-02T07:30:00")
    assert result.status == "other_time" and result.metrics == [] and result.route_optimality is None
    assert "6:20 PM" in result.message


def test_a_trip_that_was_not_tested_gets_no_borrowed_numbers(per_trip):
    result = evaluation.get_trip_evaluation([1, 2], "2026-05-25T18:20:00")
    assert result.status == "not_evaluated" and result.metrics == [] and result.route_optimality is None


def test_without_the_evaluation_it_says_so_instead_of_showing_another_table(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "ROUTING_METRICS", tmp_path / "missing.json")
    evaluation._cache.clear()
    response = TestClient(app).get("/routes/comparison-metrics")
    assert response.status_code == 404
    assert "evaluate_routing.py" in response.json()["detail"] and "Google Maps" in response.json()["detail"]


def test_each_scenario_gets_its_own_table(tmp_path, monkeypatch):
    path = tmp_path / "metrics.json"
    write_report(path, scenarios=("overall", "recommended"))
    monkeypatch.setattr(evaluation, "ROUTING_METRICS", path)
    evaluation._cache.clear()
    body = TestClient(app).get("/routes/comparison-metrics").json()
    assert [(s["scenario"], s["label"], s["n_trials"]) for s in body["scenarios"]] == [
        ("overall", "All scenarios", 10),
        ("recommended", "Recommended", 10),
    ]  # scenarios with no trials are left out
    assert body["scenarios"][0]["metrics"] == body["metrics"]
