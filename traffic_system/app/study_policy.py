"""Verify the authorized estimation release; never relabel it as observed traffic."""
import hashlib
import json
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from fastapi import HTTPException

from src.routing.network import CITIES

ROOT = Path(__file__).resolve().parents[1]
RELEASE_ARTIFACTS = ("data/processed/study_network.json", "data/processed/historical_weather.json",
                     "data/processed/temporal/traffic_hourly.csv",
                     "configs/study_assumptions.json", "configs/historical_study.json")


@lru_cache(maxsize=8)
def _hash(path, modified_ns, size):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def study_status():
    requirements = json.loads((ROOT / "configs/historical_study.json").read_text(encoding="utf-8"))
    start = datetime.fromisoformat(requirements["start_inclusive"])
    end = datetime.fromisoformat(requirements["end_exclusive"])
    hours = int((end - start).total_seconds() / 3600)
    reason, ready = "Estimation release has not been built or verified", False
    try:
        manifest = json.loads((ROOT / "data/processed/study_release.json").read_text(encoding="utf-8"))
        if not requirements["allow_estimates"] or manifest["mode"] != "historical_period_with_estimates" or manifest["period"] != requirements:
            raise ValueError("Release mode or period does not match the authorized policy")
        if manifest["hours_per_city"] != {city: hours for city in CITIES}:
            raise ValueError("Incomplete provider-weather hour coverage")
        if set(manifest["artifacts"]) != set(RELEASE_ARTIFACTS):
            raise ValueError("Release artifact set is incomplete")
        for name, expected in manifest["artifacts"].items():
            path = ROOT / name
            stat = path.stat()
            if _hash(str(path), stat.st_mtime_ns, stat.st_size) != expected:
                raise ValueError("Release artifact changed; rebuild the estimation release")
        configured = Path(os.getenv("STUDY_NETWORK_PATH", str(ROOT / "data/processed/study_network.json"))).resolve()
        if configured != (ROOT / "data/processed/study_network.json").resolve():
            raise ValueError("Configured network is not the verified estimation release")
        ready, reason = True, "Routing enabled with explicitly documented estimates; historical traffic is modeled, not measured."
    except (OSError, ValueError, KeyError) as exc:
        if isinstance(exc, ValueError):
            reason = str(exc)
    return {"ready": ready, "requirements": requirements, "required_hours_per_entity": hours,
            "mode": "historical_period_with_estimates", "message": reason, "audit": None}


def require_historical_release():
    status = study_status()
    if not status["ready"]:
        raise HTTPException(503, status["message"])
