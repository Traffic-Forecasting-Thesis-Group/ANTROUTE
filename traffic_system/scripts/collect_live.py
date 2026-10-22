"""Collect live data for the study area: HERE traffic flow and incidents, and current weather.

Every --interval minutes (aligned to the clock, Asia/Manila) the collector saves one snapshot:
  data/live/raw/<YYYY-MM-DD>/<HHMM>_flow.json.gz       HERE flow (speeds, free-flow speeds, jam factor)
  data/live/raw/<YYYY-MM-DD>/<HHMM>_incidents.json.gz  HERE incidents (including road closures)
  data/live/raw/<YYYY-MM-DD>/<HHMM>_weather.json.gz    Weatherstack current weather, once per clock hour (first tick
                                                       of the hour; retried on the next tick if it failed)
and appends one row per request to data/live/collection_log.csv. A failed request is logged and retried
at the next tick; it is never replaced by invented values. Raw responses are kept unchanged so that
matching, calibration and evaluation can be repeated later from the same files.

Usage (from traffic_system, with HERE_API_KEY and WEATHERSTACK_API_KEY in .env):
  python -m scripts.collect_live                 run continuously (Ctrl+C to stop)
  python -m scripts.collect_live --once          take one snapshot now and exit (for Task Scheduler);
                                                 the time is rounded down to the interval, weather on the hour
  python -m scripts.collect_live --once --force-weather
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from shapely.geometry import shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.live import here  # noqa: E402
from src.routing.condition_update import MANILA  # noqa: E402
from src.routing.weather import fetch_current_weather  # noqa: E402

BOUNDARIES = ROOT / "data/raw/study/five_city_boundaries.geojson"
LOG_FIELDS = ["captured_at_manila", "captured_at_utc", "kind", "status", "items", "bytes", "file", "error"]


def study_area(padding_degrees: float = 0.005):
    collection = json.loads(BOUNDARIES.read_text(encoding="utf-8"))
    geometries = {feature["properties"]["city"]: feature["geometry"] for feature in collection["features"]}
    west, south, east, north = unary_union([shape(g) for g in geometries.values()]).bounds
    bbox = (west - padding_degrees, south - padding_degrees, east + padding_degrees, north + padding_degrees)
    return bbox, geometries


def write_snapshot(out: Path, moment: datetime, kind: str, payload: dict) -> tuple[Path, int]:
    folder = out / moment.strftime("%Y-%m-%d")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{moment.strftime('%H%M')}_{kind}.json.gz"
    record = {"kind": kind, "captured_at_manila": moment.isoformat(),
              "captured_at_utc": moment.astimezone(timezone.utc).isoformat(), "payload": payload}
    data = json.dumps(record, ensure_ascii=False).encode("utf-8")
    temporary = path.with_suffix(".tmp")
    with gzip.open(temporary, "wb") as handle:
        handle.write(data)
    temporary.replace(path)
    return path, path.stat().st_size


def log(log_path: Path, row: dict) -> None:
    new = not log_path.exists()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in LOG_FIELDS})


def snapshot(moment: datetime, out: Path, log_path: Path, bbox, geometries, with_weather: bool,
             tile_degrees: float, fetchers=None) -> list[dict]:
    """Capture the sources due at this moment; each source succeeds or fails on its own."""
    fetchers = fetchers or {
        "flow": lambda: here.fetch("flow", bbox, tile_degrees),
        "incidents": lambda: here.fetch("incidents", bbox, tile_degrees),
        "weather": lambda: fetch_current_weather(geometries),
    }
    rows = []
    for kind in ("flow", "incidents", "weather"):
        if kind == "weather" and not with_weather:
            continue
        row = {"captured_at_manila": moment.isoformat(), "captured_at_utc": moment.astimezone(timezone.utc).isoformat(),
               "kind": kind}
        try:
            payload = fetchers[kind]()
            path, size = write_snapshot(out, moment, kind, payload)
            items = len(payload.get("results", [])) if kind != "weather" else len(payload.get("cities", {}))
            row.update(status="ok", items=items, bytes=size, file=str(path.relative_to(out.parent.parent)))
        except Exception as error:  # Logged and retried at the next tick; never filled in.
            row.update(status="error", error=f"{type(error).__name__}: {error}"[:300])
        log(log_path, row)
        rows.append(row)
    return rows


def floor_tick(now: datetime, interval_minutes: int) -> datetime:
    """The tick a scheduled run belongs to (a run that starts a little late keeps its slot)."""
    start = now.replace(second=0, microsecond=0)
    return start - timedelta(minutes=(start.hour * 60 + start.minute) % interval_minutes)


def next_tick(now: datetime, interval_minutes: int) -> datetime:
    start = now.replace(second=0, microsecond=0)
    minutes = (start.hour * 60 + start.minute) // interval_minutes * interval_minutes + interval_minutes
    return start.replace(hour=0, minute=0) + timedelta(minutes=minutes)


def weather_saved_for_hour(moment: datetime, out: Path) -> bool:
    """Whether a weather observation is already saved for this Asia/Manila clock hour."""
    folder = out / moment.strftime("%Y-%m-%d")
    return any(folder.glob(f"{moment:%H}??_weather.json.gz"))


def collect_tick(moment: datetime, out: Path = ROOT / "data/live/raw", tile_degrees: float = 0.5,
                 force_weather: bool = False) -> list[dict]:
    """One collector tick: HERE flow and incidents, and weather when this hour has no observation yet.

    Weather is requested once per clock hour (about 24 Weatherstack calls a day): on the first tick of the hour,
    and again on a later tick of the same hour only if the earlier request failed.
    """
    load_dotenv(ROOT / ".env")
    here.api_key()  # Raises LiveSourceError when the key is missing.
    bbox, geometries = study_area()
    weather_due = force_weather or not weather_saved_for_hour(moment, out)
    return snapshot(moment, out, out.parent / "collection_log.csv", bbox, geometries, weather_due, tile_degrees)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--interval", type=int, default=15, help="minutes between traffic snapshots")
    parser.add_argument("--tile-degrees", type=float, default=0.5, help="largest bbox side per HERE request")
    parser.add_argument("--out", type=Path, default=ROOT / "data/live/raw")
    parser.add_argument("--once", action="store_true", help="take one snapshot now and exit")
    parser.add_argument("--force-weather", action="store_true", help="with --once, also request weather off the hour")
    args = parser.parse_args(argv)
    load_dotenv(ROOT / ".env")
    try:
        here.api_key()  # Fail early when the key is missing.
    except here.LiveSourceError as error:
        sys.exit(f"{error}. Add HERE_API_KEY=<your key> to traffic_system/.env and run again.")
    def take(moment: datetime, force_weather: bool = False) -> None:
        rows = collect_tick(moment, args.out, args.tile_degrees, force_weather)
        print(moment.strftime("%Y-%m-%d %H:%M"), " ".join(f"{r['kind']}={r['status']}" for r in rows), flush=True)

    if args.once:
        take(floor_tick(datetime.now(MANILA), args.interval), force_weather=args.force_weather)
        return
    print(f"Collecting every {args.interval} min into {args.out} (Ctrl+C to stop)", flush=True)
    try:
        while True:
            tick = next_tick(datetime.now(MANILA), args.interval)
            time.sleep(max(0.0, (tick - datetime.now(MANILA)).total_seconds()))
            take(tick)
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
