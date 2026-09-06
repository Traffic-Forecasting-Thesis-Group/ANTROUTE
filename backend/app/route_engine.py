import math
from typing import List, Optional, Tuple

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


def plan_routes(
    origin_name: str,
    destinations: List[dict], 
    origin_coords_override: Optional[Tuple[float, float]] = None,
) -> List[dict]:
    """
    TODO: this is a placeholder computation (haversine distance + fixed
    variant multipliers), NOT the real CNN-LSTM + STGFormer + ACO/Dijkstra/
    A*/Q-learning pipeline. Swap this function's internals for the real
    model call once it's ready — the request/response contract (see
    schemas_route.py) is designed to stay the same either way, so the
    frontend won't need to change.
    """

    origin_coords = origin_coords_override if origin_coords_override else geocode(origin_name)

    total_km = 0.0
    current = origin_coords
    destination_names = []
    for dest in destinations:
        destination_names.append(dest["name"])
        if dest.get("lat") is not None and dest.get("lng") is not None:
            dest_coords = (dest["lat"], dest["lng"])
        else:
            dest_coords = geocode(dest["name"])
        total_km += haversine_km(current, dest_coords)
        current = dest_coords

    base_km = round(total_km * 1.35, 1) if total_km > 0 else 1.0
    event_note = find_event_note(destination_names)

    variants = [
        {"label": "Best route", "km_mult": 1.0, "speed_kmh": 32, "congestion": "moderate"},
        {"label": "Least traffic", "km_mult": 0.92, "speed_kmh": 22, "congestion": "clear"},
        {"label": "Shortest distance", "km_mult": 0.83, "speed_kmh": 18, "congestion": "heavy"},
    ]

    routes = []
    for i, v in enumerate(variants):
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
