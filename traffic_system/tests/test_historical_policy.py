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


def test_stale_artifact_hash_blocks_release(monkeypatch, tmp_path):
    policy = json.loads((study_policy.ROOT / "configs/historical_study.json").read_text())
    (tmp_path / "configs").mkdir()
    (tmp_path / "data/processed").mkdir(parents=True)
    (tmp_path / "configs/historical_study.json").write_text(json.dumps(policy))
    from src.routing.network import CITIES
    artifacts = {name: "wrong-hash" for name in study_policy.RELEASE_ARTIFACTS}
    for name in artifacts:
        if name != "configs/historical_study.json":
            (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / name).write_text("{}")
    manifest = {"mode": "historical_period_with_estimates", "period": policy,
                "hours_per_city": dict.fromkeys(CITIES, 8760), "artifacts": artifacts}
    (tmp_path / "data/processed/study_release.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(study_policy, "ROOT", tmp_path)
    assert not study_policy.study_status()["ready"]
    assert "changed" in study_policy.study_status()["message"]
