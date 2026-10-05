import json
import logging
from typing import List, Tuple

import httpx

from app.cache import redis_client

logger = logging.getLogger("uvicorn.error")

# Free public OSRM servers, tried in order. Neither has an uptime guarantee;
# self-host OSRM later if you need it reliable.
OSRM_ENDPOINTS = [
    "https://router.project-osrm.org/route/v1/driving",
    "https://routing.openstreetmap.de/routed-car/route/v1/driving",
]

CACHE_TTL_SECONDS = 60 * 60 * 6


async def fetch_paths(points: List[Tuple[float, float]]) -> List[List[Tuple[float, float]]]:
    """
    Returns road paths for the given (lat, lng) points, each path a list of
    (lat, lng). With exactly 2 points OSRM can return up to 3 alternative
    roads; with more stops it returns 1. Returns [] if every server fails, so
    the caller can fall back to a straight line.
    """
    if len(points) < 2:
        return []

    coords = ";".join(f"{lng},{lat}" for lat, lng in points)
    cache_key = f"osrm:{coords}"

    try:
        cached = await redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception as exc:
        logger.warning("OSRM cache read failed: %r", exc)

    params = {
        "alternatives": "3" if len(points) == 2 else "false",
        "overview": "simplified",
        "geometries": "geojson",
    }

    for base_url in OSRM_ENDPOINTS:
        try:
            async with httpx.AsyncClient(timeout=6.0) as client:
                response = await client.get(f"{base_url}/{coords}", params=params)
            if response.status_code != 200:
                logger.warning("OSRM %s: HTTP %s %s", base_url, response.status_code, response.text[:150])
                continue
            data = response.json()
            if data.get("code") != "Ok":
                logger.warning("OSRM %s: code=%s %s", base_url, data.get("code"), data.get("message"))
                continue

            paths = [
                [(lat, lng) for lng, lat in route["geometry"]["coordinates"]]
                for route in data["routes"]
            ]
            try:
                await redis_client.set(cache_key, json.dumps(paths), ex=CACHE_TTL_SECONDS)
            except Exception as exc:
                logger.warning("OSRM cache write failed: %r", exc)
            return paths
        except Exception as exc:
            logger.warning("OSRM %s failed: %r", base_url, exc)

    return []