import json
import math
import re
import unicodedata

import httpx
from fastapi import APIRouter, Query

from app.cache import redis_client

router = APIRouter(prefix="/places", tags=["places"])

NOMINATIM_HEADERS = {
    # Nominatim's usage policy requires a real identifying User-Agent — swap
    # the contact info below for something real before this goes beyond
    # local dev/thesis demo use:
    # https://operations.osmfoundation.org/policies/nominatim/
    # NOTE: HTTP headers must be plain ASCII — no em dashes or other
    # non-ASCII characters inside the actual string value below.
    "User-Agent": "ANTRoute/1.0 (thesis project - contact: angelrose.palazo@gmail.com)",
}

# Metro Manila bounding box (west, north, east, south).
MM_WEST, MM_NORTH, MM_EAST, MM_SOUTH = 120.90, 14.78, 121.16, 14.35
METRO_MANILA_VIEWBOX = f"{MM_WEST},{MM_NORTH},{MM_EAST},{MM_SOUTH}"

# The box also catches edges of Cavite/Rizal/Bulacan, so results must also
# name a Metro Manila city in their address.
METRO_MANILA_AREAS = (
    "manila", "quezon city", "caloocan", "las pinas", "makati", "malabon",
    "mandaluyong", "marikina", "muntinlupa", "navotas", "paranaque", "pasay",
    "pasig", "san juan", "taguig", "valenzuela", "pateros",
    "metro manila", "national capital region",
)
_AREA_KEYS = ("city", "town", "municipality", "city_district", "county", "state", "state_district", "region")

MAX_RESULTS = 8
CACHE_TTL_SECONDS = 60 * 60 * 24

_ABBREVIATIONS = {
    r"\bsta\.?\b": "santa",
    r"\bsto\.?\b": "santo",
    r"\bbrgy\.?\b": "barangay",
    r"\bave\.?\b": "avenue",
    r"\bst\.?\b": "street",
    r"\bblvd\.?\b": "boulevard",
}


def _normalize_query(query: str) -> str:
    normalized = query
    for pattern, replacement in _ABBREVIATIONS.items():
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    return normalized


def _plain(text: str) -> str:
    """Lowercase and strip accents so 'Las Piñas' matches 'las pinas'."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()

def _in_metro_manila(item: dict) -> bool:
    try:
        lat, lon = float(item["lat"]), float(item["lon"])
    except (KeyError, ValueError):
        return False
    if not (MM_SOUTH <= lat <= MM_NORTH and MM_WEST <= lon <= MM_EAST):
        return False

    address = item.get("address") or {}
    values = [_plain(str(address[k])) for k in _AREA_KEYS if address.get(k)]
    if not values:
        return True  # no address info to check, trust the bounding box
    return any(area in value for value in values for area in METRO_MANILA_AREAS)


def _build_labels(item: dict) -> tuple[str, str, str]:
    """Returns (name, short address, full label shown in the input)."""
    address = item.get("address") or {}

    name = (item.get("name") or "").strip()
    if not name:
        name = item.get("display_name", "").split(",")[0].strip()

    road = address.get("road")
    if road and address.get("house_number"):
        road = f"{address['house_number']} {road}"
    area = (
        address.get("suburb")
        or address.get("neighbourhood")
        or address.get("quarter")
        or address.get("city_district")
    )
    city = address.get("city") or address.get("town") or address.get("municipality")

    parts: list[str] = []
    for part in (road, area, city):
        if part and part.lower() != name.lower() and part.lower() not in [p.lower() for p in parts]:
            parts.append(part)

    short_address = ", ".join(parts)
    full_label = ", ".join([name] + parts)
    return name, short_address, full_label


async def _nominatim_search(query: str) -> list[dict]:
    params = {
        "q": query,
        "format": "json",
        "limit": 15,  # extra, since out-of-area results get filtered out
        "addressdetails": 1,
        "countrycodes": "ph",
        "viewbox": METRO_MANILA_VIEWBOX,
        "bounded": 1,  # only inside the box, not just a ranking hint
        "accept-language": "en",
    }
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(
            "https://nominatim.openstreetmap.org/search",
            params=params,
            headers=NOMINATIM_HEADERS,
        )
    data = response.json()
    return data if isinstance(data, list) else []


def _distance_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    h = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def _rank_results(results: list[dict], query: str, lat: float | None, lng: float | None) -> list[dict]:
    """
    Best name match first (exact, starts with, contains all words), then the
    nearest to the user when a location is given, otherwise Nominatim's order.
    """
    q = _plain(_normalize_query(query)).strip()
    tokens = q.split()
    has_location = lat is not None and lng is not None

    ranked = []
    for position, item in enumerate(results):
        name = _plain(item["name"])
        if name == q:
            match = 0
        elif name.startswith(q):
            match = 1
        elif tokens and all(t in name for t in tokens):
            match = 2
        else:
            match = 3

        distance = None
        if has_location:
            distance = round(_distance_km(lat, lng, item["lat"], item["lng"]), 1)

        ranked.append((match, distance or 0.0, position, {**item, "distance_km": distance}))

    ranked.sort(key=lambda r: (r[0], r[1], r[2]))
    return [r[3] for r in ranked]


def _drop_last_word(query: str) -> str | None:
    words = query.strip().split()
    if len(words) <= 1:
        return None
    return " ".join(words[:-1])


async def _get_results(query: str) -> list[dict]:
    # v2: older cached entries could include places outside Metro Manila.
    cache_key = f"places:search:v2:{query.strip().lower()}"
    cached = await redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    normalized_query = _normalize_query(query)

    try:
        raw_results = await _nominatim_search(normalized_query)

        # Only one relaxed retry (for typos), and only when nothing was found.
        if not raw_results:
            relaxed = _drop_last_word(normalized_query)
            if relaxed:
                raw_results = await _nominatim_search(relaxed)
    except (httpx.HTTPError, ValueError):
        return []

    results = []
    seen_labels = set()
    for item in raw_results:
        if not _in_metro_manila(item):
            continue

        name, address, full_label = _build_labels(item)
        if full_label.lower() in seen_labels:
            continue
        seen_labels.add(full_label.lower())

        results.append({
            "id": str(item["place_id"]),
            "lat": float(item["lat"]),
            "lng": float(item["lon"]),
            "name": name,
            "address": address,
            "formatted_address": full_label,
        })

    await redis_client.set(cache_key, json.dumps(results), ex=CACHE_TTL_SECONDS)
    return results


@router.get("/search")
async def search_places(
    query: str = Query(..., min_length=2),
    lat: float | None = None,
    lng: float | None = None,
):
    # lat/lng (optional) is where the user is. It is only used to rank, so the
    # cached results are shared by everyone.
    results = await _get_results(query)
    return {"results": _rank_results(results, query, lat, lng)[:MAX_RESULTS]}


@router.get("/reverse")
async def reverse_geocode(lat: float, lng: float):
    cache_key = f"places:reverse:v2:{round(lat, 5)}:{round(lng, 5)}"
    cached = await redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    params = {
        "lat": lat,
        "lon": lng,
        "format": "json",
        "zoom": 18,
        "addressdetails": 1,
        "accept-language": "en",
    }

    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(
            "https://nominatim.openstreetmap.org/reverse",
            params=params,
            headers=NOMINATIM_HEADERS,
        )
    data = response.json()

    address = data.get("address", {})
    road = address.get("road")
    if road and address.get("house_number"):
        road = f"{address['house_number']} {road}"
    place_name = data.get("name")  # building or place at this exact spot, if any
    area = address.get("neighbourhood") or address.get("suburb") or address.get("quarter")
    city = address.get("city") or address.get("municipality") or address.get("town")

    parts: list[str] = []
    for part in (place_name, road, area, city):
        if part and all(part.lower() not in p.lower() for p in parts):
            parts.append(part)
    label = ", ".join(parts) or data.get("display_name")

    result = {"label": label}
    await redis_client.set(cache_key, json.dumps(result), ex=CACHE_TTL_SECONDS)
    return result