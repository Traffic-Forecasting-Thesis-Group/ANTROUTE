"""Shared computation for Celery jobs and the existing /routes/plan endpoint."""

from starlette.concurrency import run_in_threadpool

from app.schemas_route import RoutePlanRequest, RoutePlanResponse


def _calculate_routes(payload: RoutePlanRequest):
    # Loading the graph/model modules belongs in the compute process or thread.
    from app.route_engine import BaselineUnavailableError, plan_routes, resolve_points

    destinations = [{"name": d.name, "lat": d.lat, "lng": d.lng} for d in payload.destinations]
    origin_coords = None
    if payload.origin_lat is not None and payload.origin_lng is not None:
        origin_coords = (payload.origin_lat, payload.origin_lng)
    try:
        routes = plan_routes(payload.origin, destinations, origin_coords, payload.model)
    except BaselineUnavailableError as exc:
        return [], str(exc), []
    points = []
    if not all(route.get("path") for route in routes):
        points = resolve_points(payload.origin, destinations, origin_coords)
    return routes, None, points


async def compute_route_plan(payload: RoutePlanRequest, *, offload: bool = True) -> RoutePlanResponse:
    if offload:
        routes, notice, points = await run_in_threadpool(_calculate_routes, payload)
    else:
        routes, notice, points = _calculate_routes(payload)
    if notice is not None:
        return RoutePlanResponse(routes=[], notice=notice)

    # Preserve graph geometry. OSRM is only used for existing placeholder routes.
    if points:
        from app.osrm import fetch_paths

        paths = await fetch_paths(points)
        for i, route in enumerate(routes):
            if route.get("path"):
                continue
            path = paths[i % len(paths)] if paths else points
            route["path"] = [{"lat": lat, "lng": lng} for lat, lng in path]
    return RoutePlanResponse(routes=routes)
