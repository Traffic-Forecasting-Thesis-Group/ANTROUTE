"""Verify the authorized estimation release; never relabel it as observed traffic."""
import hashlib
import json
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from fastapi import HTTPException

from src.routing.calibration import CalibrationBlocked, load_split
from src.routing.criteria import CONTRACT_VERSION
from src.routing.historical_traffic import INDEX as TRAFFIC_INDEX, ROOT as TRAFFIC_ROOT, TABLE as TRAFFIC_TABLE, table_index
from src.routing.network import CITIES

ROOT = Path(__file__).resolve().parents[1]
# The multi-gigabyte traffic table is covered by its index, which records the table's size and content SHA-256;
# the content digest itself is verified once per file version (see traffic_table_verified).
RELEASE_ARTIFACTS = ("data/processed/study_network.json", "data/processed/historical_weather.json",
                     "data/processed/calibration.json", TRAFFIC_INDEX,
                     "configs/study_assumptions.json", "configs/historical_study.json")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=8)
def _hash(path, modified_ns, size):
    return file_sha256(path)


def traffic_table_verified(index) -> bool:
    """Content SHA-256 of the traffic table against its index, computed once per (path, mtime, size)."""
    table = TRAFFIC_ROOT / TRAFFIC_TABLE
    stat = table.stat()
    return _hash(str(table), stat.st_mtime_ns, stat.st_size) == index["sha256"]


def study_status():
    requirements = json.loads((ROOT / "configs/historical_study.json").read_text(encoding="utf-8"))
    start = datetime.fromisoformat(requirements["start_inclusive"])
    end = datetime.fromisoformat(requirements["end_exclusive"])
    hours = int((end - start).total_seconds() / 3600)
    reason, ready = "Estimation release has not been built or verified", False
    try:
        load_split(requirements)
        manifest = json.loads((ROOT / "data/processed/study_release.json").read_text(encoding="utf-8"))
        if manifest.get("contract_version") != CONTRACT_VERSION:
            raise ValueError(f"Release was built for contract version {manifest.get('contract_version', 1)}; "
                             f"rebuild for version {CONTRACT_VERSION}")
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
        if not traffic_table_verified(table_index()):
            raise ValueError("Hourly traffic table content does not match its index digest; run scripts.preprocess_traffic")
        configured = Path(os.getenv("STUDY_NETWORK_PATH", str(ROOT / "data/processed/study_network.json"))).resolve()
        if configured != (ROOT / "data/processed/study_network.json").resolve():
            raise ValueError("Configured network is not the verified estimation release")
        ready, reason = True, "Routing enabled with explicitly documented estimates; historical traffic is an assumed scenario, not measured."
    except CalibrationBlocked as blocked:
        reason = str(blocked)
    except (OSError, ValueError, KeyError) as exc:
        if isinstance(exc, ValueError):
            reason = str(exc)
    return {"ready": ready, "requirements": requirements, "required_hours_per_entity": hours,
            "mode": "historical_period_with_estimates", "contract_version": CONTRACT_VERSION, "message": reason, "audit": None}


def require_historical_release():
    status = study_status()
    if not status["ready"]:
        raise HTTPException(503, status["message"])
