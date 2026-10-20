import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app import study_policy


def test_revised_period_explicitly_allows_estimates():
    status = study_policy.study_status()
    assert status["requirements"]["start_inclusive"] == "2025-06-01T00:00:00+08:00"
    assert status["requirements"]["end_exclusive"] == "2026-06-01T00:00:00+08:00"
    assert status["requirements"]["dynamic_resolution"] == "hourly"
    assert status["requirements"]["allow_estimates"] is True
    assert status["required_hours_per_entity"] == 8760
    assert status["mode"] == "historical_period_with_estimates"


def test_missing_release_cannot_be_unlocked_by_a_metadata_label(network, monkeypatch, tmp_path):
    import app.main as service
    policy = Path(study_policy.ROOT / "configs/historical_study.json").read_text()
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/historical_study.json").write_text(policy)
    monkeypatch.setattr(study_policy, "ROOT", tmp_path)
    monkeypatch.setattr(service, "get_network", lambda: network)
    network.provenance["mode"] = "historical_period_with_estimates"
    with TestClient(app) as client:
        result = client.post("/routes/plan", json={"origin": "a", "destination": "d",
            "conditions": {"time_of_day": 8, "weather": "clear"}})
        assert result.status_code == 503
        assert client.get("/study/status").json()["ready"] is False
        assert client.post("/conditions/weather/refresh").status_code == 410


FIXTURE_SPLIT = {"development": {"start": "2025-06-01", "end": "2025-12-31"},
                 "tuning": {"start": "2026-01-01", "end": "2026-02-28"}, "test": {"start": "2026-03-01", "end": "2026-05-31"}}


def release(tmp_path, policy, artifacts=None, **manifest_changes):
    """A fake release in tmp_path; artifact files hold '{}' and hashes are given (default: wrong)."""
    from src.routing.network import CITIES
    (tmp_path / "configs").mkdir(exist_ok=True)
    (tmp_path / "data/processed").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs/historical_study.json").write_text(json.dumps(policy))
    artifacts = artifacts or {name: "wrong-hash" for name in study_policy.RELEASE_ARTIFACTS}
    for name in artifacts:
        if name != "configs/historical_study.json":
            (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / name).write_text("{}")
    manifest = {"mode": "historical_period_with_estimates", "contract_version": 2, "period": policy,
                "hours_per_city": dict.fromkeys(CITIES, 8760), "artifacts": artifacts, **manifest_changes}
    (tmp_path / "data/processed/study_release.json").write_text(json.dumps(manifest))


def test_undeclared_split_blocks_calibration_and_routing(monkeypatch, tmp_path):
    policy = json.loads((study_policy.ROOT / "configs/historical_study.json").read_text())
    release(tmp_path, policy)
    monkeypatch.setattr(study_policy, "ROOT", tmp_path)
    status = study_policy.study_status()
    assert not status["ready"] and "not declared" in status["message"] and "blocked" in status["message"]


def test_stale_artifact_hash_or_old_contract_blocks_release(monkeypatch, tmp_path):
    policy = {**json.loads((study_policy.ROOT / "configs/historical_study.json").read_text()), "calibration_split": FIXTURE_SPLIT}
    release(tmp_path, policy)
    monkeypatch.setattr(study_policy, "ROOT", tmp_path)
    assert not study_policy.study_status()["ready"]
    assert "changed" in study_policy.study_status()["message"]
    release(tmp_path, policy, contract_version=1)
    assert "contract version 1" in study_policy.study_status()["message"]


def test_traffic_table_content_digest_is_verified(monkeypatch, tmp_path):
    from src.routing import historical_traffic
    from datetime import date
    policy = {**json.loads((study_policy.ROOT / "configs/historical_study.json").read_text()), "calibration_split": FIXTURE_SPLIT}
    table = tmp_path / historical_traffic.TABLE
    table.parent.mkdir(parents=True)
    historical_traffic.write_table(table, [(date(2025, 6, 1), 0, ["r"], [.1], [1.1], [11.0])])
    monkeypatch.setattr(historical_traffic, "ROOT", tmp_path)
    monkeypatch.setattr(study_policy, "TRAFFIC_ROOT", tmp_path)
    index = historical_traffic.table_index()
    assert study_policy.traffic_table_verified(index)
    # Same size, different content: a size check alone would pass.
    data = table.read_bytes()
    table.write_bytes(data.replace(b"0.100000", b"0.900000"))
    assert len(table.read_bytes()) == len(data)
    assert not study_policy.traffic_table_verified(index)
