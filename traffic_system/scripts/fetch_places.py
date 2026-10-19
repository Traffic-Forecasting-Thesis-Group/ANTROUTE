"""Collect named OSM places (barangays, districts, landmarks) inside the five study cities.

Reads a downloaded OpenStreetMap extract, so no live API is queried:
    https://download.geofabrik.de/asia/philippines-latest.osm.pbf
The public Overpass endpoint is reserved for human-operated tools, so it is not used here.
Places are search aliases only: each one is snapped to its nearest routable road node and adds no feature data.

Run: python -m scripts.fetch_places --extract path/to/philippines-latest.osm.pbf
"""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import osmium
from osmium.geom import WKBFactory
from shapely import wkb
from shapely.geometry import MultiPoint, Point, box, shape
from shapely.ops import unary_union
from shapely.prepared import prep

from src.routing.network import CITIES

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/raw/study/places_gazetteer.csv"
FIELDS = ["name", "category", "city", "latitude", "longitude", "osm_element", "osm_id"]
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


def category(tags):
    if tags.get("boundary") == "administrative":
        return "barangay" if str(tags.get("admin_level")) == "10" else None
    for key in TAGS:
        if key != "boundary" and tags.get(key) in TAGS[key]:
            return tags[key].replace("_", " ")
    return None


def representative_point(item, factory):
    """A point inside the place; None when the extract lacks its geometry."""
    try:
        if item.is_node():
            return Point(item.location.lon, item.location.lat)
        if item.is_area():
            return wkb.loads(factory.create_multipolygon(item), hex=True).representative_point()
        if item.is_way():
            points = [(node.lon, node.lat) for node in item.nodes if node.location.valid()]
            return MultiPoint(points).centroid if points else None
    except (osmium.InvalidLocationError, RuntimeError, ValueError):
        return None
    return None  # Relations are read through their assembled area.


def collect(extract_path, cities):
    prepared = {city: prep(polygon) for city, polygon in cities.items()}
    bounds = prep(box(*unary_union(list(cities.values())).bounds))
    factory, rows, seen = WKBFactory(), [], set()
    processor = (osmium.FileProcessor(str(extract_path)).with_locations().with_areas()
                 .with_filter(osmium.filter.KeyFilter(*TAGS)))
    for item in processor:
        tags = dict(item.tags)
        name, kind = tags.get("name", "").strip(), category(tags)
        if not name or kind is None:
            continue
        point = representative_point(item, factory)
        if point is None or point.is_empty or not bounds.covers(point):
            continue
        city = next((city for city, polygon in prepared.items() if polygon.covers(point)), None)
        key = (name.casefold(), city)
        # Several mapped parts of one place (building + area) collapse to the first point.
        if city is None or key in seen:
            continue
        seen.add(key)
        element = "node" if item.is_node() else "way" if item.is_way() else "area"
        osm_id = item.orig_id() if item.is_area() else item.id
        rows.append({"name": name, "category": kind, "city": city, "latitude": round(point.y, 7),
                     "longitude": round(point.x, 7), "osm_element": element, "osm_id": osm_id})
    return sorted(rows, key=lambda row: (row["city"], row["name"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--extract", required=True, type=Path, help="OpenStreetMap .osm.pbf extract covering Metro Manila")
    arguments = parser.parse_args()
    boundaries = json.loads((ROOT / "data/raw/study/five_city_boundaries.geojson").read_text(encoding="utf-8"))
    cities = {feature["properties"]["city"]: shape(feature["geometry"]) for feature in boundaries["features"]}
    if set(cities) != set(CITIES):
        raise SystemExit("Exactly the five study boundaries are required")
    rows = collect(arguments.extract, cities)
    if not rows:
        raise SystemExit("No named places found inside the five cities; check the extract")
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    OUTPUT.with_suffix(".source.json").write_text(json.dumps({
        "source": "OpenStreetMap extract (Geofabrik)", "extract": arguments.extract.name,
        "licence": "ODbL 1.0, https://www.openstreetmap.org/copyright",
        "generated_at": datetime.now(timezone.utc).isoformat(), "places": len(rows)}, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} named places to {OUTPUT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
