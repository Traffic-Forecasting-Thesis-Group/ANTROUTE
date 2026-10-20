"""One immutable condition snapshot per ACO run; closures are constraints."""
import hashlib
import json
from typing import Annotated
from datetime import date, datetime, timedelta, timezone

from pydantic import ConfigDict, Field

from .network import RoadNetwork, StudyModel

# Asia/Manila has had no daylight saving since 1978, so the fixed +08:00 offset is exact for the study period.
MANILA = timezone(timedelta(hours=8), "Asia/Manila")


class Conditions(StudyModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)
    departure_date: date = date(2025, 6, 1)
    time_of_day: Annotated[int, Field(strict=True, ge=0, le=23)]
    weather: str = Field(min_length=1)
    closed_road_ids: frozenset[str] = frozenset()


def manila_time(moment: datetime) -> datetime:
    """A timezone-aware simulation datetime in Asia/Manila; naive datetimes are rejected, not guessed."""
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("Simulation datetimes must be timezone-aware (Asia/Manila)")
    return moment.astimezone(MANILA)


def snapshot_id(conditions: Conditions, calibration_id: str) -> str:
    """Identity of one immutable condition snapshot under one calibration."""
    payload = {"date": conditions.departure_date.isoformat(), "hour": conditions.time_of_day, "weather": conditions.weather,
               "closed": sorted(conditions.closed_road_ids), "calibration": calibration_id}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def day_category(day: date) -> str:
    """Declared day categories of the time profile: Saturday and Sunday are weekend (holidays not modelled)."""
    return "weekend" if day.weekday() >= 5 else "weekday"


def validate_conditions(network: RoadNetwork, conditions: Conditions) -> None:
    if not date(2025, 6, 1) <= conditions.departure_date <= date(2026, 5, 31):
        raise ValueError("Departure date must be between June 1, 2025 and May 31, 2026")
    unknown = conditions.closed_road_ids - {road.id for road in network.roads}
    if unknown:
        raise ValueError(f"Unknown closed road IDs: {', '.join(sorted(unknown))}")
    if conditions.weather not in network.roads[0].weather:
        raise ValueError("Weather condition is not available in this network")
