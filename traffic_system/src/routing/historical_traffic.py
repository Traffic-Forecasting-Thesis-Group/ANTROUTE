"""Hourly traffic table keyed by date, hour and city/road-type location.

The table plays the role weatherstack_hourly.csv plays for weather, but its values are
estimated: no hourly traffic observations exist for this period. Congestion and travel
time come from the published BPR form applied to an assumed demand-to-capacity index;
no vehicle counts are stored or produced.
"""
import csv
import hashlib
import math
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TABLE = "data/processed/temporal/traffic_hourly.csv"
FIELDS = ["date", "hour", "location_id", "city", "road_type", "day_type", "traffic_congestion", "travel_time_factor", "source"]
START, HOURS = datetime(2025, 6, 1), 8760


def _noise(seed, *parts):
    """Stable value in [-1, 1]; independent of Python's randomized hash and row order."""
    key = "|".join(map(str, (seed, *parts))).encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") / (2**64 - 1) * 2 - 1


def demand_index(location, city, road_type, day: date, hour, assumptions):
    """Assumed demand-to-capacity index for one location-hour (not a vehicle count).

    Base hourly profile * road-type factor * weekend factor, with repeatable daily (per city)
    and three-hour (per location) variation. Coefficients are uncalibrated assumptions.
    """
    variation = assumptions["traffic_variation"]
    index = assumptions["hourly_demand_capacity_ratio"][hour] * assumptions["road_types"][road_type]["traffic_factor"]
    if day.weekday() >= 5:
        index *= assumptions["weekend_demand_factor"]
    seed = variation["seed"]
    knot, fraction = divmod(day.toordinal() * 24 + hour, 3)  # Absolute hours keep continuity across midnight.
    location_hour = ((1 - fraction / 3) * _noise(seed, "location", location, knot)
                     + fraction / 3 * _noise(seed, "location", location, knot + 1))
    return index * math.exp(variation["strength"] * (variation["day_coefficient"] * _noise(seed, "day", city, day.isoformat())
                                                     + variation["location_hour_coefficient"] * location_hour))


def bpr(index, assumptions):
    """BPR travel-time factor 1 + alpha * x^beta and congestion index 1 - 1/factor."""
    factor = 1 + assumptions["bpr_alpha"] * index ** assumptions["bpr_beta"]
    return factor, 1 - 1 / factor


def table_rows(locations, assumptions, hours=HOURS):
    """locations: iterable of (location_id, city, road_type)."""
    locations = sorted(locations)
    for offset in range(hours):
        instant = START + timedelta(hours=offset)
        day, hour = instant.date(), instant.hour
        for location, city, road_type in locations:
            factor, congestion = bpr(demand_index(location, city, road_type, day, hour, assumptions), assumptions)
            yield {"date": day.isoformat(), "hour": hour, "location_id": location, "city": city, "road_type": road_type,
                   "day_type": "weekend" if day.weekday() >= 5 else "weekday",
                   "traffic_congestion": round(congestion, 6), "travel_time_factor": round(factor, 6),
                   "source": "estimated_bpr_assumed_demand"}


@lru_cache(maxsize=1)
def _load(path, modified):
    records = {}
    with Path(path).open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = f"{row['date']}T{int(row['hour']):02d}"
            records.setdefault(key, {})[row["location_id"]] = (float(row["traffic_congestion"]), float(row["travel_time_factor"]))
    return records


def traffic_snapshot(day: date, hour: int) -> dict[str, tuple[float, float]]:
    """{location_id: (traffic_congestion, travel_time_factor)} for one study hour."""
    path = ROOT / TABLE
    snapshot = _load(str(path), path.stat().st_mtime_ns).get(f"{day.isoformat()}T{hour:02d}")
    if snapshot is None:
        raise ValueError(f"No traffic record for {day.isoformat()} {hour:02d}:00")
    return snapshot


def traffic_for(network, conditions):
    """Dated traffic snapshot when the network is linked to the hourly table, else None."""
    if not any(road.traffic_location for road in network.roads):
        return None
    return traffic_snapshot(conditions.departure_date, conditions.time_of_day)
