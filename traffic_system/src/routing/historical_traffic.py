"""Hourly traffic table with one row per road segment per hour: an assumed congestion scenario.

The table plays the role weatherstack_hourly.csv plays for weather, but its values are not
observations or independent demand estimates: no hourly traffic observations or demand inputs
exist for this period. Congestion and travel time come from the published BPR form applied to a
seeded, assumed demand-to-capacity index; no vehicle counts or flows are stored or produced, and
index x capacity is never presented as demand.

The full year is 8760 hours x every segment (hundreds of millions of rows), so the CSV is
written once with a byte-offset index; routing reads only the requested hour.
"""
import hashlib
import json
import math
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

from .network import CITIES

ROOT = Path(__file__).resolve().parents[2]
TABLE = "data/processed/temporal/traffic_hourly.csv"
INDEX = "data/processed/temporal/traffic_hourly.index.json"
FIELDS = ["date", "hour", "road_id", "traffic_congestion", "travel_time_factor", "travel_time_s"]
START, HOURS = datetime(2025, 6, 1), 8760
# Neighbouring roads share an area term so congestion is spatially coherent (~1.1 km cells).
CELL_DEGREES = 0.01


def _uniform(seed, *stream, size):
    """Repeatable values in [-1, 1] for one named random stream."""
    return np.random.default_rng([seed, *stream]).uniform(-1, 1, size)


def _knot_noise(seed, stream, size):
    """Three-hour knots linearly interpolated, so values change smoothly from hour to hour."""
    cache = {}

    def at(absolute_hour):
        knot, fraction = divmod(absolute_hour, 3)
        for key in (knot, knot + 1):
            if key not in cache:
                cache[key] = _uniform(seed, stream, key, size=size)
        for key in [key for key in cache if key < knot]:
            del cache[key]
        return (1 - fraction / 3) * cache[knot] + fraction / 3 * cache[knot + 1]
    return at


def demand_indices(roads, assumptions, hours=HOURS):
    """Yield (date, hour, x): x[i] is the assumed demand-to-capacity index of roads[i] (not a vehicle count).

    roads: sequence of {"road_id", "city", "road_type", "longitude", "latitude"}, sorted by road_id.
    x = hourly profile * road-type factor * weekend factor
        * exp(strength * (day*city_day + area*cell_3h + road*road_static + location_hour*road_3h))
    Coefficients are uncalibrated assumptions that add scenario diversity, not empirical accuracy.
    """
    ids = [road["road_id"] for road in roads]
    if ids != sorted(ids) or len(set(ids)) != len(ids):
        raise ValueError("Roads must be unique and sorted by road_id")
    variation = assumptions["traffic_variation"]
    seed, strength = variation["seed"], variation["strength"]
    count = len(roads)
    base = np.array([assumptions["road_types"][road["road_type"]]["traffic_factor"] for road in roads], dtype=np.float64)
    city = np.array([CITIES.index(road["city"]) for road in roads])
    cells = [(math.floor(road["longitude"] / CELL_DEGREES), math.floor(road["latitude"] / CELL_DEGREES)) for road in roads]
    cell_ids = {cell: index for index, cell in enumerate(sorted(set(cells)))}
    cell = np.array([cell_ids[item] for item in cells])
    day_noise = _uniform(seed, 1, size=(len(CITIES), hours // 24 + 1))
    static = strength * variation["road_coefficient"] * _uniform(seed, 3, size=count)
    area_at = _knot_noise(seed, 2, len(cell_ids))
    local_at = _knot_noise(seed, 4, count)
    for offset in range(hours):
        instant = START + timedelta(hours=offset)
        day, hour = instant.date(), instant.hour
        absolute = day.toordinal() * 24 + hour  # Absolute hours keep continuity across midnight.
        scale = assumptions["hourly_demand_capacity_ratio"][hour]
        if day.weekday() >= 5:
            scale *= assumptions["weekend_demand_factor"]
        exponent = static + strength * (variation["day_coefficient"] * day_noise[city, offset // 24]
                                        + variation["area_coefficient"] * area_at(absolute)[cell]
                                        + variation["location_hour_coefficient"] * local_at(absolute))
        yield day, hour, scale * base * np.exp(exponent)


def bpr_time(free_flow_time, demand_vph, capacity_vph, alpha=0.15, beta=4.0):
    """BPR traversal time t0 * (1 + alpha * (q/K)^beta) for independently specified demand q and capacity K.

    q and K must share direction, vehicle units and time interval. No independent demand exists in this
    study's inputs, so routing does not call this; it is kept for scenarios that supply q.
    """
    if capacity_vph <= 0 or demand_vph < 0 or free_flow_time <= 0:
        raise ValueError("BPR needs positive free-flow time and capacity and nonnegative demand")
    return free_flow_time * (1 + alpha * (demand_vph / capacity_vph) ** beta)


def bpr(index, assumptions):
    """BPR travel-time factor 1 + alpha * x^beta and congestion index 1 - 1/factor (scalars or arrays)."""
    factor = 1 + assumptions["bpr_alpha"] * index ** assumptions["bpr_beta"]
    return factor, 1 - 1 / factor


def estimated_hours(roads, assumptions, hours=HOURS):
    """Yield (date, hour, road_ids, congestion, travel_time_factor, travel_time_s) for every study hour."""
    ids = [road["road_id"] for road in roads]
    free_flow = np.array([road["free_flow_time_s"] for road in roads], dtype=np.float64)
    for day, hour, index in demand_indices(roads, assumptions, hours):
        factor, congestion = bpr(index, assumptions)
        yield day, hour, ids, congestion, factor, free_flow * factor


def weekday_profile(roads, assumptions, development_days):
    """Mean weekday (congestion, travel_time_factor) per road and hour over development days only: {road_id: [(c, f)] * 24}.

    Stored on each road as a 24-hour fallback; routing uses the dated hourly table.
    """
    development_days = set(development_days)
    sums, counts = np.zeros((2, 24, len(roads))), np.zeros(24)
    for day, hour, index in demand_indices(roads, assumptions):
        if day in development_days and day.weekday() < 5:
            factor, congestion = bpr(index, assumptions)
            sums[0, hour] += congestion
            sums[1, hour] += factor
            counts[hour] += 1
    means = sums / counts[None, :, None]
    return {road["road_id"]: [(float(means[0, hour, i]), float(means[1, hour, i])) for hour in range(24)]
            for i, road in enumerate(roads)}


def index_path(table_path):
    return Path(table_path).with_name(Path(INDEX).name)


def write_table(path, hourly, description=None):
    """Write the hourly table and its byte-offset index; validate every hour while writing.

    hourly: iterable of (date, hour, road_ids, congestion, travel_time_factor, travel_time_s).
    """
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    digest, offsets, position, rows, roads = hashlib.sha256(), {}, 0, 0, None
    with temporary.open("wb") as stream:
        def emit(text):
            nonlocal position
            data = text.encode("utf-8")
            stream.write(data)
            digest.update(data)
            position += len(data)
        emit(",".join(FIELDS) + "\n")
        for day, hour, ids, congestion, factor, travel_time in hourly:
            key = f"{day.isoformat()}T{hour:02d}"
            congestion, factor, travel_time = (np.asarray(values, dtype=np.float64) for values in (congestion, factor, travel_time))
            if key in offsets:
                raise ValueError(f"Duplicate traffic hour: {key}")
            if roads is None:
                roads = len(ids)
                if len(set(ids)) != roads:
                    raise ValueError("Duplicate road IDs in the traffic table")
            if len(ids) != roads:
                raise ValueError(f"Incomplete road coverage at {key}")
            if not (np.isfinite(congestion).all() and np.isfinite(travel_time).all()
                    and ((congestion >= 0) & (congestion < 1)).all() and (factor >= 1).all()):
                raise ValueError(f"Traffic value out of range at {key}")
            prefix = f"{day.isoformat()},{hour},"
            start = position
            emit("".join(f"{prefix}{road},{c:.6f},{f:.6f},{t:.3f}\n"
                         for road, c, f, t in zip(ids, congestion.tolist(), factor.tolist(), travel_time.tolist())))
            offsets[key] = [start, position - start]
            rows += len(ids)
    temporary.replace(path)
    index = {"table": path.name, "columns": FIELDS, "rows": rows, "roads": roads or 0, "hours": len(offsets),
             "bytes": position, "sha256": digest.hexdigest(),
             "generated_at": datetime.now(timezone.utc).isoformat(),
             "classification": "assumed_congestion_scenario", **(description or {}), "offsets": offsets}
    index_path(path).write_text(json.dumps(index), encoding="utf-8")
    return {key: value for key, value in index.items() if key != "offsets"}


@lru_cache(maxsize=2)
def _index(path, modified):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def table_index():
    path = ROOT / INDEX
    if not path.exists():
        raise ValueError("Hourly traffic table is missing; run scripts.preprocess_traffic")
    index = _index(str(path), path.stat().st_mtime_ns)
    table = ROOT / TABLE
    if not table.exists() or table.stat().st_size != index["bytes"]:
        raise ValueError("Hourly traffic table does not match its index; run scripts.preprocess_traffic")
    return index


@lru_cache(maxsize=48)
def _read_hour(path, modified, start, length):
    with Path(path).open("rb") as stream:
        stream.seek(start)
        text = stream.read(length).decode("utf-8")
    records = {}
    for line in text.splitlines():
        _, _, road, congestion, factor, _ = line.split(",")
        records[road] = (float(congestion), float(factor))
    return records


def traffic_snapshot(day: date, hour: int) -> dict[str, tuple[float, float]]:
    """{road_id: (traffic_congestion, travel_time_factor)} for one study hour."""
    index = table_index()
    span = index["offsets"].get(f"{day.isoformat()}T{hour:02d}")
    if span is None:
        raise ValueError(f"No traffic record for {day.isoformat()} {hour:02d}:00")
    path = ROOT / TABLE
    return _read_hour(str(path), path.stat().st_mtime_ns, *span)


def traffic_for(network, conditions):
    """Dated traffic snapshot when the network is linked to the hourly table, else None."""
    if not any(road.traffic_location for road in network.roads):
        return None
    return traffic_snapshot(conditions.departure_date, conditions.time_of_day)
