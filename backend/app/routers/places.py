import json
import re

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

METRO_MANILA_VIEWBOX = "120.90,14.78,121.20,14.35"

CACHE_TTL_SECONDS = 60 * 60 * 24  

_POSTAL_CODE_RE = re.compile(r"^\d{4,5}$")

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


def _split_display_name(display_name: str) -> tuple[str, str]:
    parts = [p.strip() for p in display_name.split(",")]
    name = parts[0] if parts else display_name

    rest = [
        p for p in parts[1:]
        if p and p.lower() != "philippines" and not _POSTAL_CODE_RE.match(p)
    ]
    address = ", ".join(rest[:2])
    return name, address


async def _nominatim_search(query: str, use_bias: bool) -> list[dict]:
    params = {
        "q": query,
        "format": "json",
        "limit": 10,
        "addressdetails": 1,
        "countrycodes": "ph",
    }
    if use_bias:
        params["viewbox"] = METRO_MANILA_VIEWBOX

    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(
            "https://nominatim.openstreetmap.org/search",
            params=params,
            headers=NOMINATIM_HEADERS,
        )
    data = response.json()
    return data if isinstance(data, list) else []


def _drop_last_word(query: str) -> str | None:
    words = query.strip().split()
    if len(words) <= 1:
        return None
    return " ".join(words[:-1])


@router.get("/search")
async def search_places(query: str = Query(..., min_length=2)):
    cache_key = f"places:search:{query.strip().lower()}"
    cached = await redis_client.get(cache_key)
    if cached:
        return {"results": json.loads(cached)}

    normalized_query = _normalize_query(query)

    raw_results = await _nominatim_search(normalized_query, use_bias=True)

    if len(raw_results) < 3:
        broader = await _nominatim_search(normalized_query, use_bias=False)
        seen_ids = {item["place_id"] for item in raw_results}
        raw_results += [item for item in broader if item["place_id"] not in seen_ids]

    attempts = 0
    relaxed_query = normalized_query
    while len(raw_results) == 0 and attempts < 2:
        relaxed_query = _drop_last_word(relaxed_query)
        if not relaxed_query:
            break
        raw_results = await _nominatim_search(relaxed_query, use_bias=True)
        attempts += 1

    results = []
    for item in raw_results[:10]:
        name, address = _split_display_name(item["display_name"])
        results.append({
            "id": str(item["place_id"]),
            "lat": float(item["lat"]),
            "lng": float(item["lon"]),
            "name": name,
            "address": address,
            "formatted_address": item["display_name"],
        })

    await redis_client.set(cache_key, json.dumps(results), ex=CACHE_TTL_SECONDS)
    return {"results": results}


@router.get("/reverse")
async def reverse_geocode(lat: float, lng: float):
    cache_key = f"places:reverse:{round(lat, 5)}:{round(lng, 5)}"
    cached = await redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    params = {
        "lat": lat,
        "lon": lng,
        "format": "json",
        "zoom": 18, 
        "addressdetails": 1,
    }

    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(
            "https://nominatim.openstreetmap.org/reverse",
            params=params,
            headers=NOMINATIM_HEADERS,
        )
    data = response.json()

    address = data.get("address", {})
    specific = ", ".join(
        filter(
            None,
            [
                address.get("house_number"),
                address.get("road"),
                address.get("neighbourhood") or address.get("suburb"),
                address.get("city") or address.get("municipality") or address.get("town"),
            ],
        )
    )
    label = specific or data.get("display_name")

    result = {"label": label}
    await redis_client.set(cache_key, json.dumps(result), ex=CACHE_TTL_SECONDS)
    return result