from fastapi import APIRouter

from app.evaluation import get_comparison_metrics
from app.osrm import fetch_paths
from app.route_engine import BaselineUnavailableError, plan_routes, resolve_points
from app.schemas_route import ComparisonMetricsResponse, RoutePlanRequest, RoutePlanResponse

router = APIRouter(prefix="/routes", tags=["routes"])


@router.post("/plan", response_model=RoutePlanResponse)
async def plan(payload: RoutePlanRequest):
    destinations = [{"name": d.name, "lat": d.lat, "lng": d.lng} for d in payload.destinations]

    origin_coords_override = None
    if payload.origin_lat is not None and payload.origin_lng is not None:
        origin_coords_override = (payload.origin_lat, payload.origin_lng)

    try:
        routes = plan_routes(payload.origin, destinations, origin_coords_override, payload.model)
    except BaselineUnavailableError as exc:
        # A 200 with no routes, not an error status: the app requests ANTROUTE and the
        # baseline side by side, and "the baseline has no route here" is a result to show
        # next to ANTROUTE's, not a failure that should take ANTROUTE's results down with it.
        return RoutePlanResponse(routes=[], notice=str(exc))

    # Routes already carrying a real path came from the actual graph pipeline (the
    # nodes it computed cost over) -- drawing a different, independently-computed
    # OSRM line for those would mean the map doesn't match what was scored. OSRM is
    # only a stand-in for ANTROUTE's placeholder routes, which never had real geometry.
    if not all(route.get("path") for route in routes):
        points = resolve_points(payload.origin, destinations, origin_coords_override)
        paths = await fetch_paths(points)
        for i, route in enumerate(routes):
            if route.get("path"):
                continue
            path = paths[i % len(paths)] if paths else points
            route["path"] = [{"lat": lat, "lng": lng} for lat, lng in path]

    return RoutePlanResponse(routes=routes)


@router.get("/comparison-metrics", response_model=ComparisonMetricsResponse)
async def comparison_metrics():
    return get_comparison_metrics()