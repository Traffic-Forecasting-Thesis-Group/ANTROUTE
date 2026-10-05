from fastapi import APIRouter

from app.evaluation import get_comparison_metrics
from app.osrm import fetch_paths
from app.route_engine import plan_routes, resolve_points
from app.schemas_route import ComparisonMetricsResponse, RoutePlanRequest, RoutePlanResponse

router = APIRouter(prefix="/routes", tags=["routes"])


@router.post("/plan", response_model=RoutePlanResponse)
async def plan(payload: RoutePlanRequest):
    destinations = [{"name": d.name, "lat": d.lat, "lng": d.lng} for d in payload.destinations]

    origin_coords_override = None
    if payload.origin_lat is not None and payload.origin_lng is not None:
        origin_coords_override = (payload.origin_lat, payload.origin_lng)

    routes = plan_routes(payload.origin, destinations, origin_coords_override, payload.model)

    # Routes already carrying a real path came from the actual graph pipeline (the
    # nodes it computed cost over) -- drawing a different, independently-computed
    # OSRM line for those would mean the map doesn't match what was scored. OSRM is
    # only a stand-in for the placeholder routes, which never had real geometry.
    if not all(route.get("path") for route in routes):
        points = resolve_points(payload.origin, destinations, origin_coords_override)
        paths = await fetch_paths(points)
        offset = 1 if payload.model == "baseline" else 0
        for i, route in enumerate(routes):
            if route.get("path"):
                continue
            path = paths[(i + offset) % len(paths)] if paths else points
            route["path"] = [{"lat": lat, "lng": lng} for lat, lng in path]

    return RoutePlanResponse(routes=routes)


@router.get("/comparison-metrics", response_model=ComparisonMetricsResponse)
async def comparison_metrics():
    return get_comparison_metrics()