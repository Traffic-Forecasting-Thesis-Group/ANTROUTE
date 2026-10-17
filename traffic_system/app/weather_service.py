"""Explicit weather refresh, age checks, and read-only per-request snapshots."""
import json
import math
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException

from src.routing.network import CITIES, RoadNetwork
from src.routing.weather import fetch_current_weather, observed_effect, weather_condition

ROOT = Path(__file__).resolve().parents[1]
LOCK = threading.Lock()


def snapshot_path():
    return Path(os.getenv("WEATHER_SNAPSHOT_PATH", str(ROOT / "data/raw/study/current_weather.json")))


def age_seconds(snapshot):
    try:
        retrieved = [datetime.fromisoformat(item["retrieved_at"]) for item in snapshot["cities"].values()]
        return max((datetime.now(timezone.utc) - instant).total_seconds() for instant in retrieved)
    except (KeyError, ValueError, TypeError):
        return float("inf")


def read_snapshot(network: RoadNetwork):
    path = snapshot_path()
    try:
        snapshot = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        snapshot = network.provenance.get("weather")
    if not snapshot or set(snapshot.get("cities", {})) != set(CITIES) or not math.isfinite(age_seconds(snapshot)):
        raise HTTPException(503, "Current weather observations are unavailable. Refresh weather or choose a scenario.")
    return snapshot


def public_snapshot(snapshot):
    return {"source": "Weatherstack", "age_seconds": max(age_seconds(snapshot), 0),
            "stale": age_seconds(snapshot) > 3600,
            "cities": {city: {
                "retrieved_at": item["retrieved_at"],
                "localtime": item.get("location", {}).get("localtime"),
                "description": item["current"].get("weather_descriptions", []),
                "precip_mm": item["current"].get("precip"),
                "condition": weather_condition(item),
            } for city, item in snapshot["cities"].items()}}


def current_effects(network):
    snapshot = read_snapshot(network)
    if age_seconds(snapshot) > 3600:
        raise HTTPException(503, "Current weather snapshot is older than one hour. Refresh weather or choose a scenario.")
    scenarios = network.provenance.get("assumptions", {}).get("weather_scenarios")
    if not scenarios:
        raise HTTPException(503, "Weather conversion settings are missing from the network")
    effects = {city: observed_effect(item, scenarios) for city, item in snapshot["cities"].items()}
    return effects, public_snapshot(snapshot)


def refresh_weather(network):
    with LOCK:
        try:
            existing = read_snapshot(network)
            if age_seconds(existing) < 300:
                return public_snapshot(existing)
        except HTTPException:
            pass
        try:
            snapshot = fetch_current_weather(network.boundaries)
            path = snapshot_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
            temporary.replace(path)
        except (OSError, ValueError, RuntimeError):
            raise HTTPException(502, "Weather refresh failed. Check the server's Weatherstack key, quota, and network connection.") from None
        return public_snapshot(snapshot)
