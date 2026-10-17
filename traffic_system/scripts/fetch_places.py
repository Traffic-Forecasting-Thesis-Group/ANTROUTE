"""Collect named OSM places (barangays, districts, landmarks) inside the five study cities.

One-time Overpass download via OSMnx; search stays local afterwards. Places are search
aliases only: each one is snapped to its nearest routable road node and adds no feature data.
"""
import csv
import json
from pathlib import Path

import osmnx as ox
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely.prepared import prep

from src.routing.network import CITIES

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/raw/study/places_gazetteer.csv"
TAGS = {
    "place": ["suburb", "quarter", "neighbourhood", "village", "hamlet", "locality", "square"],
    "boundary": ["administrative"],
    "amenity": ["school", "college", "university", "hospital", "clinic", "townhall", "place_of_worship",
                "marketplace", "bus_station", "police", "fire_station", "library", "courthouse", "community_centre"],
    "shop": ["mall", "department_store", "supermarket"],
    "railway": ["station"],
    "public_transport": ["station"],
    "tourism": ["attraction", "museum", "hotel"],
    "leisure": ["park", "stadium", "sports_centre"],
    "office": ["government"],
    "landuse": ["commercial", "retail"],
}


def category(row):
    if row.get("boundary") == "administrative":
        return "barangay" if str(row.get("admin_level")) == "10" else None
    for key in TAGS:
        if key != "boundary" and isinstance(row.get(key), str) and row[key] in TAGS[key]:
            return row[key].replace("_", " ")
    return None


def main():
    boundaries = json.loads((ROOT / "data/raw/study/five_city_boundaries.geojson").read_text(encoding="utf-8"))
    cities = {feature["properties"]["city"]: shape(feature["geometry"]) for feature in boundaries["features"]}
    if set(cities) != set(CITIES):
        raise SystemExit("Exactly the five study boundaries are required")
    prepared = {city: prep(polygon) for city, polygon in cities.items()}
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(ROOT / "cache/osmnx")
    # Detailed outlines exceed Overpass request limits; the hull is narrowed per city below.
    features = ox.features_from_polygon(unary_union(list(cities.values())).convex_hull, TAGS)
    rows, seen = [], set()
    for (element, osm_id), row in features.iterrows():
        name, kind = row.get("name"), category(row)
        if not isinstance(name, str) or not name.strip() or kind is None:
            continue
        point = row.geometry.representative_point()
        city = next((city for city, polygon in prepared.items() if polygon.covers(point)), None)
        key = (name.strip().casefold(), city)
        # Several mapped parts of one place (building + area) collapse to the first point.
        if city is None or key in seen:
            continue
        seen.add(key)
        rows.append({"name": name.strip(), "category": kind, "city": city,
                     "latitude": round(point.y, 7), "longitude": round(point.x, 7),
                     "osm_element": element, "osm_id": osm_id})
    rows.sort(key=lambda item: (item["city"], item["name"]))
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} named places to {OUTPUT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
