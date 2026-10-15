from datetime import datetime
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool

from app import recorded, risk_routing
from app.evaluation import (
    MetricsUnavailableError,
    get_comparison_metrics,
    get_forecast_metrics,
    get_trip_evaluation,
)
from app.risk_routing import LOCAL_TZ
from app.route_engine import (
    BaselineUnavailableError,
    RoutingUnavailableError,
    UnknownWindowError,
    local_departure,
    plan_routes,
)
from app.what_if import WhatIfUnavailableError, load_what_if
from app.routers.places import _get_results
from app.schemas_route import (
    ComparisonMetricsResponse,
    ComparisonTripDetail,
    ComparisonTripsResponse,
    ConditionUpdateDetail,
    ConditionUpdatesResponse,
    RecordedInputs,
    RecordedSessionsResponse,
    RoutePlanRequest,
    RoutePlanResponse,
    TripEvaluationResponse,
    WhatIfResponse,
)

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
    simulation = payload.recorded_window is not None
    if simulation:
        # Simulation mode leaves at the recorded window's own time, on its own day.
        try:
            depart_at = datetime.fromisoformat(payload.recorded_window)
        except ValueError:
            raise HTTPException(status_code=422, detail=f"{payload.recorded_window!r} is not a recorded window.")
    departure_time = depart_at.replace(tzinfo=ZoneInfo(LOCAL_TZ)).isoformat(timespec="minutes")
    # The ACO search is CPU-bound (and the first call waits for the network to load),
    # so keep it off the event loop -- the app sends its ANTRoute and baseline requests
    # in parallel.
    try:
        routes, traffic_note = await run_in_threadpool(
            plan_routes, points, payload.model, depart_at, payload.recorded_window
        )
    except BaselineUnavailableError as exc:
        # A 200 with no routes, not an error status: the app requests ANTROUTE and the
        # baseline side by side, and "the baseline has no route here" is a result to show
        # next to ANTROUTE's, not a failure that should take ANTROUTE's results down with it.
        return RoutePlanResponse(routes=[], notice=str(exc), departure_time=departure_time)
    except UnknownWindowError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RoutingUnavailableError:
        raise HTTPException(status_code=503, detail="Routing data isn't loaded on the server.")
    except risk_routing.RouteOutsideNetworkError:
        raise HTTPException(
            status_code=422,
            detail="One of these places is too far from the Metro Manila road network to route.",
        )
    except risk_routing.NoRouteError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return RoutePlanResponse(
        routes=routes, departure_time=departure_time, traffic_note=traffic_note,
        recorded_inputs=await _panel(routes, payload.recorded_window, depart_at), simulation=simulation,
    )


async def _panel(routes: List[dict], recorded_window: Optional[str], depart_at: datetime) -> Optional[dict]:
    """The "Inputs used" panel for the window the routes were planned on, if any."""
    window = recorded_window
    if window is None:
        window = next((r["inputs"]["window"] for r in routes if r.get("inputs")), None)
    if window is None:
        window, _ = risk_routing.window_for_departure(depart_at)
    return await run_in_threadpool(recorded.inputs_for_window, window) if window else None


@router.get("/recorded-sessions", response_model=RecordedSessionsResponse)
async def recorded_sessions():
    """Every recorded day and session, for simulation mode. May 25 is held out (split "test")."""
    if not risk_routing.available():
        raise HTTPException(status_code=503, detail="Routing data isn't loaded on the server.")
    sessions = await run_in_threadpool(recorded.recorded_windows)
    return RecordedSessionsResponse(
        sessions=sessions,
        note="Recorded AM and PM peaks only; there is no live feed. Held-out sessions were never seen by the "
             "model in training or validation.",
    )


@router.get("/recorded-inputs", response_model=RecordedInputs)
async def recorded_inputs(window: str):
    """The recorded inputs of one window (a RecordedWindow.start): cameras, incidents, weather, flood."""
    if not risk_routing.available():
        raise HTTPException(status_code=503, detail="Routing data isn't loaded on the server.")
    try:
        return await run_in_threadpool(recorded.inputs_for_window, window)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"{window!r} is not a recorded window.")


@router.get("/what-if", response_model=WhatIfResponse)
async def what_if():
    """Precomputed what-if cases: the checkpoint with rain, flood, incidents or cameras changed."""
    try:
        return await run_in_threadpool(load_what_if)
    except WhatIfUnavailableError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/comparison-trips", response_model=ComparisonTripsResponse)
async def comparison_trips(scenario: str = "overall", update: str = "event"):
    """
    ANTRoute against the Improved ACO baseline trip by trip (research question 3), with how
    the evaluation was obtained and the measured recalculation response time of both systems.
    """
    try:
        return await run_in_threadpool(cmp.get_comparison_trips, scenario, update)
    except cmp.ComparisonUnavailableError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/comparison-trips/{trip_id}", response_model=ComparisonTripDetail)
async def comparison_trip_detail(trip_id: str):
    """One evaluated trip with both systems' routes drawn."""
    try:
        return await run_in_threadpool(cmp.get_comparison_trip_detail, trip_id)
    except cmp.ComparisonUnavailableError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/condition-updates", response_model=ConditionUpdatesResponse)
async def condition_updates(update: str = "event", system: str = "antroute"):
    """
    Dynamic adaptability (research question 2) for one update kind: recalculation response
    time, how often and how much the route changed, and route quality before and after the
    update. Read from the offline harness -- nothing is replanned in the request.
    """
    try:
        return await run_in_threadpool(cu.get_condition_updates, update, system)
    except cu.ConditionUpdatesUnavailableError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/condition-updates/{trip_id}", response_model=ConditionUpdateDetail)
async def condition_update_detail(trip_id: str, update: str = "event", system: str = "antroute"):
    """One case with the original and recalculated routes drawn, for the before/after maps."""
    try:
        return await run_in_threadpool(cu.get_condition_update_detail, update, trip_id, system)
    except cu.ConditionUpdatesUnavailableError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/comparison-metrics", response_model=ComparisonMetricsResponse)
async def comparison_metrics():
    # Reading risk_edges.csv (~100MB) takes a few seconds the first time; keep it off the
    # event loop.
    try:
        return await run_in_threadpool(get_comparison_metrics)
    except MetricsUnavailableError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/trip-evaluation", response_model=TripEvaluationResponse)
async def trip_evaluation(payload: RoutePlanRequest):
    """
    The routing evaluation of the trip in `payload` (the same body as /routes/plan; `model`
    is ignored): its own Route Optimality and congestion and event exposure when its stops and
    departure time match one of the test trips timed in Google Maps, otherwise a status saying
    why not.
    """
    if not risk_routing.available():
        raise HTTPException(status_code=503, detail="Routing data isn't loaded on the server.")
    points: List[Tuple[float, float]] = [await _resolve(payload.origin, payload.origin_lat, payload.origin_lng)]
    for d in payload.destinations:
        points.append(await _resolve(d.name, d.lat, d.lng))
    try:
        stops = await run_in_threadpool(risk_routing.snap_stops, points)
        window, _ = risk_routing.window_for_departure(local_departure(payload.depart_at))
        return await run_in_threadpool(get_trip_evaluation, stops, window)
    except risk_routing.RouteOutsideNetworkError:
        raise HTTPException(
            status_code=422,
            detail="One of these places is too far from the Metro Manila road network to evaluate.",
        )
    except MetricsUnavailableError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/forecast-metrics", response_model=ComparisonMetricsResponse)
async def forecast_metrics():
    """Congestion Risk Score diagnostics on risk_edges.csv's test split. Not the thesis
    routing evaluation, which is /routes/comparison-metrics."""
    try:
        return await run_in_threadpool(get_forecast_metrics)
    except MetricsUnavailableError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
