"""Fetch the five OSM boundaries and optional Weatherstack current observations.

Credentials stay in .env/environment; errors never echo request URLs or secrets.
Public Nominatim requests are sequential, cached on disk, and spaced >1 second apart.
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
from shapely.geometry import shape

from src.routing.network import CITIES
from src.routing.weather import fetch_current_weather

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data/raw/study"


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def fetch_boundaries(destination=SOURCE_DIR, refresh=False):
    destination.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "ANTROUTE/2.0 (five-city academic routing study; one-time boundary collection)"
    features = []
    for city in CITIES:
        cache = destination / f"boundary_{city.lower().replace(' ', '_')}.json"
        if cache.exists() and not refresh:
            records = json.loads(cache.read_text(encoding="utf-8"))
        else:
            response = session.get("https://nominatim.openstreetmap.org/search", params={
                "q": f"{city}, Metro Manila, Philippines", "format": "jsonv2",
                "polygon_geojson": 1, "countrycodes": "ph", "limit": 5,
            }, timeout=40)
            if response.status_code != 200:
                raise RuntimeError(f"Boundary request failed for {city}: HTTP {response.status_code}")
            records = response.json()
            cache.write_text(json.dumps(records), encoding="utf-8")
            time.sleep(1.1)
        candidates = [record for record in records
                      if record.get("geojson", {}).get("type") in {"Polygon", "MultiPolygon"}
                      and record.get("type") == "administrative"
                      and 14.4 < float(record["lat"]) < 14.8
                      and 120.8 < float(record["lon"]) < 121.3]
        if not candidates:
            raise ValueError(f"No valid Metro Manila city boundary returned for {city}")
        record = candidates[0]
        polygon = shape(record["geojson"])
        if not polygon.is_valid:
            raise ValueError(f"Source boundary invalid for {city}")
        features.append({"type": "Feature", "properties": {
            "city": city, "source": "OpenStreetMap via Nominatim",
            "osm_type": record["osm_type"], "osm_id": record["osm_id"],
            "display_name": record["display_name"], "licence": record.get("licence", ""),
            "retrieved_at": datetime.fromtimestamp(cache.stat().st_mtime, timezone.utc).isoformat(),
        }, "geometry": record["geojson"]})
        print(f"Boundary ready: {city}", flush=True)
    path = destination / "five_city_boundaries.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")
    return path


def fetch_weather(boundary_path, destination=SOURCE_DIR):
    load_dotenv(ROOT / ".env")
    boundaries = json.loads(boundary_path.read_text(encoding="utf-8"))
    observations = fetch_current_weather({feature["properties"]["city"]: feature["geometry"] for feature in boundaries["features"]})
    path = destination / "current_weather.json"
    path.write_text(json.dumps(observations, indent=2), encoding="utf-8")
    print("Weather observations ready for all five cities", flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather", action="store_true", help="Fetch five current weather observations using the configured key")
    parser.add_argument("--refresh-boundaries", action="store_true")
    args = parser.parse_args()
    if args.weather:
        raise SystemExit("Current-weather collection is disabled for this historical study. Obtain verified hourly observations for May 2021–May 2026; see docs/historical_data_requirements.md.")
    try:
        path = fetch_boundaries(refresh=args.refresh_boundaries)
        if args.weather:
            fetch_weather(path)
    except requests.RequestException:
        raise SystemExit("Public source connection failed (request details withheld).") from None
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
