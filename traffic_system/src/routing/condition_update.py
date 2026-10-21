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
    # Live mode: the saved capture (Asia/Manila) whose HERE traffic, incidents and weather score every road.
    # weather is then "live", and date and hour are the capture's own.
    live_at: datetime | None = None


def manila_time(moment: datetime) -> datetime:
    """A timezone-aware simulation datetime in Asia/Manila; naive datetimes are rejected, not guessed."""
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("Simulation datetimes must be timezone-aware (Asia/Manila)")
    return moment.astimezone(MANILA)


def snapshot_id(conditions: Conditions, calibration_id: str) -> str:
    """Identity of one immutable condition snapshot under one calibration."""
    payload = {"date": conditions.departure_date.isoformat(), "hour": conditions.time_of_day, "weather": conditions.weather,
               "closed": sorted(conditions.closed_road_ids), "calibration": calibration_id}
    if conditions.live_at is not None:
        payload["live_at"] = conditions.live_at.isoformat()
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def day_category(day: date) -> str:
    """Declared day categories of the time profile: Saturday and Sunday are weekend (holidays not modelled)."""
    return "weekend" if day.weekday() >= 5 else "weekday"


LIVE_WEATHER = "live"


def live_conditions(captured_at: datetime, closed_road_ids: frozenset[str] = frozenset()) -> Conditions:
    """Snapshot of one saved live capture; the roads HERE reports closed are added when the graph is built."""
    captured_at = manila_time(captured_at)
    return Conditions(departure_date=captured_at.date(), time_of_day=captured_at.hour, weather=LIVE_WEATHER,
                      closed_road_ids=closed_road_ids, live_at=captured_at)


def validate_conditions(network: RoadNetwork, conditions: Conditions) -> None:
    if conditions.live_at is not None or conditions.weather == LIVE_WEATHER:
        if conditions.live_at is None or conditions.weather != LIVE_WEATHER:
            raise ValueError("Live conditions need both a capture time and weather 'live'")
        live = manila_time(conditions.live_at)
        if (live.date(), live.hour) != (conditions.departure_date, conditions.time_of_day):
            raise ValueError("Live conditions must use the capture's own date and hour")
        unknown = conditions.closed_road_ids - {road.id for road in network.roads}
        if unknown:
            raise ValueError(f"Unknown closed road IDs: {', '.join(sorted(unknown))}")
        return
    if not date(2025, 6, 1) <= conditions.departure_date <= date(2026, 5, 31):
        raise ValueError("Departure date must be between June 1, 2025 and May 31, 2026")
    unknown = conditions.closed_road_ids - {road.id for road in network.roads}
    if unknown:
        raise ValueError(f"Unknown closed road IDs: {', '.join(sorted(unknown))}")
    if conditions.weather not in network.roads[0].weather:
        raise ValueError("Weather condition is not available in this network")
