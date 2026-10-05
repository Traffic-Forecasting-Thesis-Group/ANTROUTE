import logging
import math
from typing import List, Optional, Tuple

from app import risk_routing

logger = logging.getLogger("uvicorn.error")

# TODO: replace with PostGIS-backed places table if you want fully
# offline fallback. This dictionary is only used now when a destination has
# no real coordinates attached
KNOWN_PLACES = {
    "current location": (14.5995, 120.9842),
    "pup, manila mabini campus": (14.5892, 120.9829),
    "sm manila": (14.5926, 120.9810),
    "moa arena": (14.5352, 120.9829),
    "mendiola": (14.5993, 120.9873),
}
DEFAULT_COORDS = (14.5995, 120.9842)

VIA_ROADS = ["EDSA Southbound", "Taft Avenue", "Quirino Avenue", "Roxas Boulevard", "Macapagal Blvd"]

# TODO: replace with a real lookup against your MMDA/News API/GDELT event
# feed (the one behind FeedScreen). This is a stand-in so route responses
# can demonstrate the event-aware behavior end-to-end.
ACTIVE_EVENTS = [
    {"keyword": "moa", "note": "Event detour applied: concert near MOA reroutes via Macapagal Blvd"},
]

# Placeholder variants for ANTROUTE when the real pipeline cannot run.
# TODO: replace with the real model call.
#
# There is deliberately no baseline placeholder. The baseline exists to be compared against,
# and an invented baseline route would make that comparison meaningless, so when the real
# baseline cannot run the request says so instead (BaselineUnavailableError).
ANTROUTE_VARIANTS = [
    {"label": "Best route", "km_mult": 1.0, "speed_kmh": 32, "congestion": "moderate"},
    {"label": "Least traffic", "km_mult": 0.92, "speed_kmh": 22, "congestion": "clear"},
    {"label": "Shortest distance", "km_mult": 0.83, "speed_kmh": 18, "congestion": "heavy"},
]


class BaselineUnavailableError(Exception):
    """The baseline cannot produce a route for this request; the message says why."""


def geocode(place_name: str) -> Tuple[float, float]:
    return KNOWN_PLACES.get(place_name.strip().lower(), DEFAULT_COORDS)


def haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = a
    lat2, lon2 = b
    r = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def find_event_note(destination_names: List[str]) -> Optional[str]:
    joined = " ".join(destination_names).lower()
    for event in ACTIVE_EVENTS:
        if event["keyword"] in joined:
            return event["note"]
    return None


def resolve_points(
    origin_name: str,
    destinations: List[dict],
    origin_coords_override: Optional[Tuple[float, float]] = None,
) -> List[Tuple[float, float]]:
    """Origin followed by each destination, as (lat, lng)."""
    points = [origin_coords_override if origin_coords_override else geocode(origin_name)]
    for dest in destinations:
        if dest.get("lat") is not None and dest.get("lng") is not None:
            points.append((dest["lat"], dest["lng"]))
        else:
            points.append(geocode(dest["name"]))
    return points


def plan_real_routes_or_none(points: List[Tuple[float, float]], model: str) -> Optional[List[dict]]:
    """
    The real CNN+LSTM -> RADR STGNN -> MLP Decoder -> Dynamic Weight Engine -> ACO
    pipeline, when every stop falls inside the monitored k-hop subgraph and a scored
    risk_edges.csv is available locally. None otherwise, so the caller can fall back
    to the distance-only placeholder instead of failing the request -- most searched
    places in Metro Manila are outside the 8-camera network's current coverage.
    """
    if not risk_routing.available():
        return None
    try:
        return risk_routing.plan_real_routes(points[0], points[1:], model)
    except risk_routing.RouteOutsideNetworkError as exc:
        if model == "baseline":
            raise BaselineUnavailableError(f"No baseline route: {exc}") from exc
        logger.info("Falling back to the placeholder: %s", exc)
        return None
    except ValueError as exc:
        # The baseline's own failures: no route with the fallback off, or an unreachable
        # destination. ANTROUTE's ValueErrors propagate as before.
        if model == "baseline":
            raise BaselineUnavailableError(str(exc)) from exc
        raise


def plan_routes(
    origin_name: str,
    destinations: List[dict],
    origin_coords_override: Optional[Tuple[float, float]] = None,
    model: str = "antroute",
) -> List[dict]:
    """
    Routes from the real model pipeline when every stop is inside the road network.

    `model` selects ANTROUTE (risk- and event-aware ACO) or the baseline (Improved ACO,
    Cheng 2023). When the real pipeline cannot run, ANTROUTE falls back to a distance-only
    placeholder (haversine + fixed multipliers) so the app still responds; the baseline
    raises BaselineUnavailableError instead of inventing a route.
    """

    points = resolve_points(origin_name, destinations, origin_coords_override)

    real_routes = plan_real_routes_or_none(points, model)
    if real_routes is not None:
        return real_routes
    if model == "baseline":
        raise BaselineUnavailableError(
            "No baseline route: the scored road network (risk_edges.csv) is not available on "
            "this server, so the baseline cannot run."
        )

    destination_names = [d["name"] for d in destinations]
    total_km = sum(haversine_km(points[i], points[i + 1]) for i in range(len(points) - 1))

    base_km = round(total_km * 1.35, 1) if total_km > 0 else 1.0
    event_note = find_event_note(destination_names)

    routes = []
    for i, v in enumerate(ANTROUTE_VARIANTS):
        distance_km = round(base_km * v["km_mult"], 1)
        duration_min = max(5, round((distance_km / v["speed_kmh"]) * 60))
        routes.append({
            "label": v["label"],
            "via": VIA_ROADS[i % len(VIA_ROADS)],
            "duration_min": duration_min,
            "distance_km": distance_km,
            "congestion_level": v["congestion"],
            "event_note": event_note if i == 0 else None,
        })

    return routes