"""HERE Traffic API v7 client: live flow and incidents for the study area.

Flow results carry, for each directed stretch, a shape (lat/lng points) and currentFlow with speed and
freeFlow in m/s, jamFactor 0-10, confidence 0-1 and traversability. Incident results carry a shape and
incidentDetails (type, roadClosed, start/end time). Errors never echo the request URL, which holds the key.
"""
from __future__ import annotations

import math
import os

import requests

BASE_URL = "https://data.traffic.hereapi.com/v7"


class LiveSourceError(RuntimeError):
    """A live source is not configured, unreachable, or returned an unusable response."""


def api_key() -> str:
    key = os.getenv("HERE_API_KEY", "")
    if not key:
        raise LiveSourceError("HERE_API_KEY is not configured")
    return key


def tiles(bbox: tuple[float, float, float, float], max_degrees: float) -> list[tuple[float, float, float, float]]:
    """Split (west, south, east, north) into tiles no larger than max_degrees on either side."""
    west, south, east, north = bbox
    columns = max(1, math.ceil((east - west) / max_degrees))
    rows = max(1, math.ceil((north - south) / max_degrees))
    width, height = (east - west) / columns, (north - south) / rows
    return [(west + i * width, south + j * height, west + (i + 1) * width, south + (j + 1) * height)
            for j in range(rows) for i in range(columns)]


def _request(endpoint: str, bbox, key: str, session, timeout: float) -> dict:
    params = {"in": "bbox:{:.6f},{:.6f},{:.6f},{:.6f}".format(*bbox), "locationReferencing": "shape", "apiKey": key}
    try:
        response = session.get(f"{BASE_URL}/{endpoint}", params=params, timeout=timeout)
    except requests.RequestException as error:
        raise LiveSourceError(f"HERE {endpoint} connection failed ({type(error).__name__})") from None
    if response.status_code != 200:
        raise LiveSourceError(f"HERE {endpoint} request failed: HTTP {response.status_code}")
    try:
        data = response.json()
    except ValueError:
        raise LiveSourceError(f"HERE {endpoint} returned invalid JSON") from None
    if not isinstance(data.get("results"), list):
        raise LiveSourceError(f"HERE {endpoint} response has no results list")
    return data


def fetch(endpoint: str, bbox, max_tile_degrees: float = 0.25, session=None, timeout: float = 30) -> dict:
    """Fetch 'flow' or 'incidents' for the bbox, tile by tile, and combine the results."""
    if endpoint not in ("flow", "incidents"):
        raise ValueError("endpoint must be 'flow' or 'incidents'")
    key, session = api_key(), session or requests.Session()
    combined = {"endpoint": endpoint, "bbox": list(bbox), "tiles": [], "source_updated": [], "results": []}
    for tile in tiles(bbox, max_tile_degrees):
        data = _request(endpoint, tile, key, session, timeout)
        combined["tiles"].append(list(tile))
        combined["source_updated"].append(data.get("sourceUpdated"))
        combined["results"].extend(data["results"])
    return combined


def _lines(location: dict) -> list[list[tuple[float, float]]]:
    """Polylines (lon, lat) of a result's shape links; empty when the shape is missing."""
    lines = []
    for link in ((location or {}).get("shape") or {}).get("links") or []:
        points = [(float(p["lng"]), float(p["lat"])) for p in link.get("points") or [] if "lat" in p and "lng" in p]
        if len(points) >= 2:
            lines.append(points)
    return lines


def flow_items(payload: dict) -> list[dict]:
    """One item per flow result with a usable shape and speed."""
    items = []
    for result in payload.get("results", []):
        flow, lines = result.get("currentFlow") or {}, _lines(result.get("location"))
        speed, free_flow = flow.get("speed"), flow.get("freeFlow")
        if not lines or speed is None or not free_flow:
            continue
        items.append({
            "lines": lines, "description": (result.get("location") or {}).get("description"),
            "speed_mps": float(speed), "speed_uncapped_mps": float(flow.get("speedUncapped", speed)),
            "free_flow_mps": float(free_flow), "jam_factor": flow.get("jamFactor"),
            "confidence": flow.get("confidence"), "traversability": flow.get("traversability", "open"),
        })
    return items


def incident_items(payload: dict) -> list[dict]:
    """One item per incident with a usable shape."""
    items = []
    for result in payload.get("results", []):
        details, lines = result.get("incidentDetails") or {}, _lines(result.get("location"))
        if not lines:
            continue
        description = details.get("description")
        items.append({
            "lines": lines, "id": details.get("id"), "type": details.get("type"),
            "road_closed": bool(details.get("roadClosed")), "criticality": details.get("criticality"),
            "start": details.get("startTime"), "end": details.get("endTime"),
            "description": description.get("value") if isinstance(description, dict) else description,
        })
    return items
