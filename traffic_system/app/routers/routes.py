from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool

from app import risk_routing
from app.evaluation import MetricsUnavailableError, get_comparison_metrics
from app.risk_routing import LOCAL_TZ
from app.route_engine import BaselineUnavailableError, RoutingUnavailableError, local_departure, plan_routes
from app.routers.places import _get_results
from app.schemas_route import ComparisonMetricsResponse, RoutePlanRequest, RoutePlanResponse

router = APIRouter(prefix="/routes", tags=["routes"])


async def _resolve(name: str, lat: Optional[float], lng: Optional[float]) -> Tuple[float, float]:
    """
    Coordinates for one stop. The app sends them whenever the user picked a search
    suggestion or used GPS; text typed without picking one is geocoded the same way
    the search box does, so a trip never silently starts from somewhere else.
    """
    if lat is not None and lng is not None:
        return lat, lng
    results = await _get_results(name) if len(name.strip()) >= 2 else []
    if not results:
        raise HTTPException(
            status_code=422,
            detail=f"Couldn't find \"{name}\" in Metro Manila. Pick a place from the suggestions.",
        )
    return results[0]["lat"], results[0]["lng"]


@router.post("/plan", response_model=RoutePlanResponse)
async def plan(payload: RoutePlanRequest):
    points: List[Tuple[float, float]] = [await _resolve(payload.origin, payload.origin_lat, payload.origin_lng)]
    for d in payload.destinations:
        points.append(await _resolve(d.name, d.lat, d.lng))

    depart_at = local_departure(payload.depart_at)
    departure_time = depart_at.replace(tzinfo=ZoneInfo(LOCAL_TZ)).isoformat(timespec="minutes")
    # The ACO search is CPU-bound (and the first call waits for the network to load),
    # so keep it off the event loop -- the app sends its ANTRoute and baseline requests
    # in parallel.
    try:
        routes, traffic_note = await run_in_threadpool(plan_routes, points, payload.model, depart_at)
    except BaselineUnavailableError as exc:
        # A 200 with no routes, not an error status: the app requests ANTROUTE and the
        # baseline side by side, and "the baseline has no route here" is a result to show
        # next to ANTROUTE's, not a failure that should take ANTROUTE's results down with it.
        return RoutePlanResponse(routes=[], notice=str(exc), departure_time=departure_time)
    except RoutingUnavailableError:
        raise HTTPException(status_code=503, detail="Routing data isn't loaded on the server.")
    except risk_routing.RouteOutsideNetworkError:
        raise HTTPException(
            status_code=422,
            detail="One of these places is too far from the Metro Manila road network to route.",
        )
    except risk_routing.NoRouteError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return RoutePlanResponse(routes=routes, departure_time=departure_time, traffic_note=traffic_note)


@router.get("/comparison-metrics", response_model=ComparisonMetricsResponse)
async def comparison_metrics():
    # Reading risk_edges.csv (~100MB) takes a few seconds the first time; keep it off the
    # event loop.
    try:
        return await run_in_threadpool(get_comparison_metrics)
    except MetricsUnavailableError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
