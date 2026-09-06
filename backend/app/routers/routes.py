from fastapi import APIRouter

from app.route_engine import plan_routes
from app.schemas_route import RoutePlanRequest, RoutePlanResponse

router = APIRouter(prefix="/routes", tags=["routes"])


@router.post("/plan", response_model=RoutePlanResponse)
async def plan(payload: RoutePlanRequest):
    destinations = [{"name": d.name, "lat": d.lat, "lng": d.lng} for d in payload.destinations]

    origin_coords_override = None
    if payload.origin_lat is not None and payload.origin_lng is not None:
        origin_coords_override = (payload.origin_lat, payload.origin_lng)

    routes = plan_routes(payload.origin, destinations, origin_coords_override)
    return RoutePlanResponse(routes=routes)
