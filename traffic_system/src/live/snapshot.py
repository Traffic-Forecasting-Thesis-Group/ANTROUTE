"""Saved live captures as routing inputs.

A capture is one collector tick with both a HERE flow and a HERE incidents file. Its weather is the latest
saved weather observation at or before the capture, no older than WEATHER_MAX_AGE; otherwise the capture is
unavailable (weather is never assumed clear). live_snapshot() turns one capture into per-road live inputs
(src/live/conditions.py) and caches the result, so every road of one search reads the same snapshot.
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

from . import conditions as live_conditions
from .matching import load_segments
from ..routing.condition_update import MANILA

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/live/raw"
FEATURES = ROOT / "data/processed/features"
ASSUMPTIONS = ROOT / "configs/study_assumptions.json"
WEATHER_MAX_AGE = timedelta(minutes=90)
CAPTURE_TOLERANCE = timedelta(minutes=15)


class LiveDataUnavailable(ValueError):
    """No usable saved capture (flow, incidents and recent weather) for the requested moment."""


@dataclass(frozen=True)
class LiveSnapshot:
    captured_at: datetime  # Asia/Manila
    weather_at: datetime
    segments: dict  # road_id -> live inputs (conditions.build)
    summary: dict
    closed: frozenset[str]


def _moment(folder: str, name: str) -> datetime:
    return datetime.strptime(f"{folder} {name[:4]}", "%Y-%m-%d %H%M").replace(tzinfo=MANILA)


def _files(kind: str, raw: Path) -> dict[datetime, Path]:
    return {_moment(path.parent.name, path.name): path for path in raw.glob(f"*/*_{kind}.json.gz")}


def captures(raw: Path = RAW) -> list[datetime]:
    """Capture times with flow and incidents saved, oldest first."""
    flow, incidents = _files("flow", raw), _files("incidents", raw)
    return sorted(flow.keys() & incidents.keys())


def nearest_capture(moment: datetime, raw: Path = RAW, tolerance: timedelta = CAPTURE_TOLERANCE) -> datetime:
    """The latest capture at or before moment, within tolerance; a later capture is never used for an earlier time."""
    moment = moment.astimezone(MANILA)
    earlier = [t for t in captures(raw) if moment - tolerance <= t <= moment]
    if not earlier:
        raise LiveDataUnavailable(f"No live capture within {tolerance} before {moment:%Y-%m-%d %H:%M}")
    return earlier[-1]


def read(path: Path) -> dict:
    with gzip.open(path, "rb") as handle:
        return json.loads(handle.read().decode("utf-8"))


def weather_at(moment: datetime, raw: Path = RAW) -> tuple[datetime, dict]:
    observations = [t for t in _files("weather", raw) if moment - WEATHER_MAX_AGE <= t <= moment]
    if not observations:
        raise LiveDataUnavailable(f"No weather observation within {WEATHER_MAX_AGE} before {moment:%Y-%m-%d %H:%M}")
    latest = max(observations)
    return latest, read(_files("weather", raw)[latest])["payload"]


@lru_cache(maxsize=1)
def _static():
    records = live_conditions.load_records(FEATURES / "road_segments.csv", FEATURES / "segment_flood.csv")
    segments = load_segments(FEATURES / "road_segments.csv")
    assumptions = json.loads(ASSUMPTIONS.read_text(encoding="utf-8"))
    flood = {k: float(v) for k, v in assumptions["flood_class_severity"]["values"].items()}
    return records, segments, flood, assumptions["weather_scenarios"]


def _build(captured_at: datetime, raw: Path) -> LiveSnapshot:
    flow, incidents = _files("flow", raw)[captured_at], _files("incidents", raw)[captured_at]
    observed_at, weather = weather_at(captured_at, raw)
    records, segments, flood, scenarios = _static()
    try:
        built = live_conditions.build(records, segments, read(flow)["payload"], read(incidents)["payload"], weather,
                                      flood, scenarios)
    except ValueError as error:  # For example a city without a weather observation.
        raise LiveDataUnavailable(str(error)) from None
    closed = frozenset(road_id for road_id, item in built["segments"].items() if item["closed"])
    return LiveSnapshot(captured_at, observed_at, built["segments"], built["summary"], closed)


def live_snapshot(captured_at: datetime, raw: Path = RAW) -> LiveSnapshot:
    """Live inputs of one saved capture (an exact capture time; see nearest_capture). Saved captures never change,
    so a built snapshot is cached; a missing one raises and is looked up again next time."""
    return _cached(captured_at.astimezone(MANILA).replace(second=0, microsecond=0), raw)


@lru_cache(maxsize=8)
def _cached(captured_at: datetime, raw: Path) -> LiveSnapshot:
    if captured_at not in _files("flow", raw) or captured_at not in _files("incidents", raw):
        raise LiveDataUnavailable(f"No saved live capture at {captured_at:%Y-%m-%d %H:%M}")
    return _build(captured_at, raw)
