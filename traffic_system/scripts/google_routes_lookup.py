"""
Fill a Google Maps worklist (scripts/google_maps_worklist.py) from the Google Routes API, and
record how each figure was obtained -- the same columns a person filling it by hand records.

    python scripts/google_routes_lookup.py --worklist outputs/google_eval/lookups.csv --dry-run
    python scripts/google_routes_lookup.py --worklist outputs/google_eval/lookups.csv
    python scripts/google_maps_worklist.py --template outputs/google_eval/template.csv \\
        --out outputs/google_eval/lookups.csv --apply

Per worklist row, one computeRoutes request (Routes API v2):

    travelMode          DRIVE
    routingPreference   TRAFFIC_AWARE_OPTIMAL   (Google's traffic-aware route and time)
    trafficModel        BEST_GUESS
    departureTime       the row's departure_slot (weekday and time, Manila) on the first such
                        date after today -- Google predicts traffic only for future departures
    intermediates       the system's waypoints, as via points (pass-through, no stopover), so
                        Google times the system's route rather than substituting its own;
                        none for google_own_route, which is Google's own fastest route

and writes duration_seconds plus google_access_method, google_departure_date,
google_queried_on and google_traffic_model. Every slot uses one departure date, so the whole
run describes the same week. The worklist is saved after every row, so an interrupted run
resumes where it stopped; rows that already have a duration are skipped.

The key is read from GOOGLE_ROUTES_API_KEY (environment, or traffic_system/.env). It is
never written to any output. Each request is billed to the key's Google Cloud project.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from src.routing.route_trials import REFERENCE_FIELDS  # noqa: E402

ENDPOINT = "https://routes.googleapis.com/directions/v2:computeRoutes"
FIELD_MASK = "routes.duration,routes.distanceMeters"
LOCAL_TZ = ZoneInfo("Asia/Manila")
ACCESS_METHOD = "Google Routes API v2 computeRoutes (DRIVE, TRAFFIC_AWARE_OPTIMAL, waypoints as via points)"
TRAFFIC_MODEL = "BEST_GUESS"
WEEKDAYS = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}


def api_key() -> str:
    key = os.environ.get("GOOGLE_ROUTES_API_KEY", "").strip()
    env = REPO_ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "GOOGLE_ROUTES_API_KEY":
                key = value.strip().strip('"').strip("'")
    if not key:
        raise SystemExit("GOOGLE_ROUTES_API_KEY is not set (environment or traffic_system/.env)")
    return key


def departure(slot: str, after: date) -> datetime:
    """The first date strictly after `after` falling on the slot's weekday, at the slot's time (Manila)."""
    day_name, _, clock = str(slot).strip().partition(" ")
    if day_name not in WEEKDAYS or ":" not in clock:
        raise ValueError(f"departure_slot {slot!r} is not like 'Mon 17:10'")
    hour, minute = (int(x) for x in clock.split(":"))
    ahead = (WEEKDAYS[day_name] - after.weekday()) % 7 or 7
    day = after + timedelta(days=ahead)
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=LOCAL_TZ)


def latlng(text: str) -> dict:
    lat, lng = (float(x) for x in str(text).split(","))
    return {"location": {"latLng": {"latitude": lat, "longitude": lng}}}


def request_body(row: dict, depart: datetime) -> dict:
    body = {
        "origin": latlng(row["start"]),
        "destination": latlng(row["end"]),
        "travelMode": "DRIVE",
        "routingPreference": "TRAFFIC_AWARE_OPTIMAL",
        "trafficModel": TRAFFIC_MODEL,
        "departureTime": depart.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "computeAlternativeRoutes": False,
    }
    stops = [p.strip() for p in str(row.get("stops") or "").split("|") if p.strip()]
    if stops:
        body["intermediates"] = [{**latlng(p), "via": True} for p in stops]
    return body


def compute_route(body: dict, key: str, retries: int = 4, field_mask: str = FIELD_MASK) -> dict:
    data = json.dumps(body).encode("utf-8")
    for attempt in range(retries + 1):
        req = urllib.request.Request(ENDPOINT, data=data, method="POST", headers={
            "Content-Type": "application/json", "X-Goog-Api-Key": key, "X-Goog-FieldMask": field_mask,
        })
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", "replace")
            if err.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"HTTP {err.code}: {detail[:400]}") from None
        except urllib.error.URLError as err:
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"network error: {err}") from None
    raise RuntimeError("unreachable")


def seconds(duration: str) -> float:
    """Routes API durations are strings like '1234s'."""
    return float(str(duration).rstrip("s"))


def lookup_key(row: dict) -> tuple:
    """What makes two lookups the same request: endpoints, waypoints and departure slot."""
    return tuple(str(row.get(c, "")).strip() for c in ("departure_slot", "start", "stops", "end"))


def answered(paths: List[Path]) -> Dict[tuple, dict]:
    """Already-answered lookups in other worklists, by lookup_key, with their provenance."""
    out: Dict[tuple, dict] = {}
    for path in paths:
        for row in pd.read_csv(path, dtype=str, keep_default_na=False).to_dict("records"):
            if str(row.get("duration_seconds", "")).strip() and all(str(row.get(f, "")).strip() for f in REFERENCE_FIELDS):
                out.setdefault(lookup_key(row), row)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--worklist", type=Path, required=True, nargs="+")
    p.add_argument("--dry-run", action="store_true", help="print the first request and stop; nothing is sent")
    p.add_argument("--limit", type=int, default=None, help="at most this many requests (for a trial run)")
    p.add_argument("--pause", type=float, default=0.2, help="seconds between requests")
    p.add_argument("--reuse", type=Path, nargs="*", default=[],
                   help="filled worklists whose identical lookups (same endpoints, waypoints, slot and "
                        "departure date) are copied instead of requested again")
    a = p.parse_args()
    cache = answered(a.reuse)
    reused = 0

    today = datetime.now(LOCAL_TZ).date()
    key = None if a.dry_run else api_key()
    sent = 0
    for path in a.worklist:
        work = pd.read_csv(path, dtype=str, keep_default_na=False)
        for field in ("duration_seconds", *REFERENCE_FIELDS):
            if field not in work:
                work[field] = ""
        todo = [i for i in work.index if not work.at[i, "duration_seconds"].strip()]
        print(f"{path}: {len(todo)} of {len(work)} lookups to do")
        for i in todo:
            if a.limit is not None and sent >= a.limit:
                break
            row = work.loc[i].to_dict()
            depart = departure(row["departure_slot"], today)
            hit = cache.get(lookup_key(row))
            if hit is not None and hit["google_departure_date"] == depart.date().isoformat():
                for field in ("duration_seconds", *REFERENCE_FIELDS):
                    work.at[i, field] = hit[field]
                work.to_csv(path, index=False)
                reused += 1
                continue
            body = request_body(row, depart)
            if a.dry_run:
                print(json.dumps(body, indent=2))
                return
            sent += 1   # every attempt counts toward --limit, failed or not
            try:
                result = compute_route(body, key)
            except RuntimeError as err:
                if str(err).startswith(("HTTP 400", "HTTP 401", "HTTP 403")):
                    # A key, API or request problem repeats on every row: stop rather than
                    # send the rest of the worklist into the same error.
                    raise SystemExit(f"stopped: {row['lookup']} {row['trip_legs']}: {err}") from None
                print(f"  {row['lookup']} {row['trip_legs']}: FAILED -- {err}")
                continue
            routes = result.get("routes") or []
            if not routes or "duration" not in routes[0]:
                print(f"  {row['lookup']} {row['trip_legs']}: no route returned -- left blank")
                continue
            work.at[i, "duration_seconds"] = f"{seconds(routes[0]['duration']):.0f}"
            work.at[i, "google_access_method"] = ACCESS_METHOD
            work.at[i, "google_departure_date"] = depart.date().isoformat()
            work.at[i, "google_queried_on"] = today.isoformat()
            work.at[i, "google_traffic_model"] = TRAFFIC_MODEL
            work.to_csv(path, index=False)   # resumable
            print(f"  {row['lookup']:17s} {row['trip_legs']:28s} {row['departure_slot']} -> "
                  f"{depart:%Y-%m-%d %H:%M}  {work.at[i, 'duration_seconds']} s")
            time.sleep(a.pause)
    print(f"\n{sent} request(s) sent, {reused} lookup(s) reused from earlier worklists")


if __name__ == "__main__":
    main()
