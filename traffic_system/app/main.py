"""Route requests use a fresh, consistent snapshot of the dynamic inputs.

Two releases: the historical estimation release (dates June 2025 - May 2026) and the live release, scored from
the saved HERE traffic, incident and weather captures. Conditions with weather "live" use the live release;
without live_at they take the latest saved capture.
"""
import json
import os
from collections import OrderedDict
from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4
from functools import lru_cache
from pathlib import Path
from typing import Literal
from datetime import date, datetime, time
from time import perf_counter
from itertools import chain, islice

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import Field, ValidationError, model_validator
from dotenv import load_dotenv

from app import places
from app.study_policy import live_status, require_historical_release, require_live_release, study_status

from src.routing.aco_routing import (AntColonyConfig, NoFeasibleRoute, SearchExhausted, SearchTimeLimit, ant_colony_route,
                                      distance_reference, node_positions)
from src.routing.contributions import route_contributions
from src.routing.explanation import explain
from src.routing.baseline_cheng import AHP_WEIGHTS, REFERENCE, ChengConfig, ahp, cheng_graph, cheng_route
from src.routing.dynamic_trip import Algorithm, ReplanConfig, TripEvent, simulate_trip
from src.routing.condition_update import LIVE_WEATHER, MANILA, Conditions, live_conditions, snapshot_id, validate_conditions
from src.routing.criteria import CONTRACT_VERSION, CriterionInputError
from src.routing.weather import MissingWeatherObservation
from src.routing.dynamic_weight import closed_roads, prepare_graph, segment_values
from src.routing.datasets import coverage, snapshot_rows, csv_chunks
from src.routing.metrics import evaluate
from src.routing.network import CITIES, FEATURES, City, RoadNetwork, StudyModel, load_network
from src.routing.historical_weather import historical_snapshot
from src.routing.historical_traffic import traffic_for

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


@asynccontextmanager
async def lifespan(_app):
    # Controlled startup: verify the release once, including the traffic table's content digest (cached per file version).
    study_status()
    # Live mode: collect a HERE capture every 15 minutes while the API runs (app/live_collector.py).
    from app.live_collector import collector
    if serving_mode() == "live":
        collector.start()
    yield
    collector.stop()


app = FastAPI(title="ANTROUTE ACO Study", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@lru_cache(maxsize=1)
def _load(path: str, modified_ns: int, size: int) -> RoadNetwork:
    return load_network(Path(path))


def get_network() -> RoadNetwork:
    default = Path(__file__).resolve().parents[1] / "data/processed/study_network.json"
    path = Path(os.getenv("STUDY_NETWORK_PATH", str(default))).resolve()
    try:
        stat = path.stat()
        return _load(str(path), stat.st_mtime_ns, stat.st_size)
    except (OSError, ValidationError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Study network unavailable. Prepare a validated five-city network with all ten features. "
                   "See traffic_system/docs/network_data.md.",
        ) from exc


def get_live_network() -> RoadNetwork:
    path = (Path(__file__).resolve().parents[1] / "data/processed/live_network.json").resolve()
    try:
        stat = path.stat()
        return _load_live(str(path), stat.st_mtime_ns, stat.st_size)
    except (OSError, ValidationError, ValueError) as exc:
        raise HTTPException(503, "Live network unavailable. Run python -m scripts.calibrate_live.") from exc


@lru_cache(maxsize=1)
def _load_live(path: str, modified_ns: int, size: int) -> RoadNetwork:
    return load_network(Path(path))


Mode = Literal["live", "historical"]


def serving_mode(requested: str | None = None) -> str:
    """Which release a page uses: the client's choice (the data tab) when given, else ANTROUTE_MODE (live by default)."""
    return requested or os.getenv("ANTROUTE_MODE", "live")


def serving_network(requested: str | None = None) -> RoadNetwork:
    return get_live_network() if serving_mode(requested) == "live" else get_network()


def _release(conditions: Conditions) -> tuple[RoadNetwork, Conditions]:
    """The release and the resolved snapshot for a request: live conditions without live_at take the latest capture."""
    # The release checks honour FastAPI dependency overrides, as when they were route dependencies.
    gate = lambda check: app.dependency_overrides.get(check, check)()
    if conditions.weather != LIVE_WEATHER and conditions.live_at is None:
        gate(require_historical_release)
        return get_network(), conditions
    gate(require_live_release)
    from src.live.snapshot import LiveDataUnavailable, captures, nearest_capture
    try:
        moment = nearest_capture(conditions.live_at) if conditions.live_at is not None else captures()[-1]
        resolved = live_conditions(moment, conditions.closed_road_ids)
    except (LiveDataUnavailable, IndexError, ValueError) as exc:
        raise HTTPException(503, f"Live snapshot unavailable: {exc}") from None
    return get_live_network(), resolved


# ACO parameters are fixed by the study, not set per request.
ACO = AntColonyConfig()


class RouteRequest(StudyModel):
    origin: str = Field(min_length=1)
    destination: str = Field(min_length=1)
    conditions: Conditions


class BaselineRequest(StudyModel):
    origin: str = Field(min_length=1)
    destination: str = Field(min_length=1)
    conditions: Conditions
    baseline: ChengConfig = Field(default_factory=ChengConfig)


class TripRequest(StudyModel):
    origin: str = Field(min_length=1)
    destination: str | None = Field(default=None, min_length=1)
    destinations: list[str] = Field(default_factory=list, max_length=5)  # Stops visited in order.
    conditions: Conditions
    departure_minute: int = Field(default=0, ge=0, le=59, strict=True)
    events: list[TripEvent] = Field(default_factory=list, max_length=50)
    algorithm: Algorithm = "antroute"
    baseline: ChengConfig = Field(default_factory=ChengConfig)
    replan: ReplanConfig = Field(default_factory=ReplanConfig)

    @model_validator(mode="after")
    def one_way_to_give_stops(self):
        if bool(self.destination) == bool(self.destinations):
            raise ValueError("Give either destination or a non-empty destinations list")
        return self

    @property
    def stops(self) -> list[str]:
        return self.destinations or [self.destination]


ALGORITHMS = {
    "antroute": "ANTROUTE: CRITIC-weighted ten-feature ACO, re-planned from fresh pheromone when conditions change",
    "cheng": "Cheng (2023) improved ACO baseline with eq. 9 dynamic optimization",
}


class SnapshotRequest(StudyModel):
    conditions: Conditions
    city: City | None = None


TRAFFIC_DISCLOSURE = {"classification": "assumed_congestion_scenario",
                      "source": "data/processed/temporal/traffic_hourly.csv",
                      "note": "Hourly congestion and travel-time factors per road segment from an assumed, seeded demand-to-capacity "
                              "index passed through the BPR function. A modeled scenario: no vehicle counts or traffic measurements."}
LIVE_DISCLOSURE = {"classification": "live_observed",
                   "source": "HERE Traffic API v7 flow and incidents, Weatherstack current weather (data/live/raw)",
                   "note": "Live speeds where HERE reports them; other roads take the median live congestion of matched roads "
                           "of the same city and road type (flagged live_group_estimate)."}


def _disclosure(conditions: Conditions) -> dict:
    return LIVE_DISCLOSURE if conditions.live_at is not None else TRAFFIC_DISCLOSURE


@app.get("/dataset/info")
def dataset_info(mode: Mode | None = None):
    live = serving_mode(mode) == "live"
    return {**coverage(serving_network(mode)), "traffic": LIVE_DISCLOSURE if live else TRAFFIC_DISCLOSURE,
            "features": FEATURES, "period": study_status()["requirements"]}


@app.post("/dataset/snapshot")
def dataset_snapshot(request: SnapshotRequest, offset: int = Query(default=0, ge=0),
                        limit: int = Query(default=100, ge=1, le=1000)):
    network, conditions = _release(request.conditions)
    try:
        rows = list(islice(snapshot_rows(network, conditions, request.city), offset, offset + limit))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    cities = {node.id: node.city for node in network.nodes}
    total = sum(1 for road in network.roads if request.city is None or cities[road.source] == request.city)
    return {"rows": rows, "total": total, "offset": offset, "limit": limit,
            "traffic": _disclosure(conditions)}


@app.post("/dataset/export")
def dataset_export(request: SnapshotRequest):
    network, conditions = _release(request.conditions)
    try:
        rows = iter(snapshot_rows(network, conditions, request.city))
        first = next(rows, None)  # Validate inputs and weather before sending response headers.
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    stamp = (f"live_{conditions.live_at:%Y-%m-%d_%H%M}" if conditions.live_at is not None
             else f"{conditions.departure_date}_{conditions.time_of_day:02d}")
    filename = f"antroute_{stamp}.csv"
    return StreamingResponse(csv_chunks(chain([] if first is None else [first], rows)),
                             media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{filename}"'})


# One color per study city for the coverage map; distinct from the route, baseline and explanation colors.
CITY_COLORS = {"Manila": "#0f766e", "San Juan": "#65a30d", "Mandaluyong": "#b45309", "Makati": "#db2777", "Pasig": "#6366f1"}


class CoverageRequest(StudyModel):
    city: City | None = None


@app.post("/dataset/map")
def dataset_map(request: CoverageRequest):
    """Every road segment included in the study, grouped by the city it starts in."""
    network = serving_network()
    nodes = {node.id: node for node in network.nodes}
    bands = {city: {"label": city, "color": CITY_COLORS[city], "lines": []} for city in CITIES
             if request.city is None or city == request.city}
    for road in network.roads:
        city = nodes[road.source].city
        if city in bands:
            bands[city]["lines"].append(road.geometry)
    return {"bands": list(bands.values()), "segment_count": sum(len(band["lines"]) for band in bands.values())}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/study/status")
def historical_study_status():
    return study_status()


@app.get("/study/results")
def sop_results(mode: Mode | None = None):
    """Latest SOP experiment summary written by scripts.evaluate_sop (per-run rows omitted), for the chosen mode."""
    live = serving_mode(mode) == "live"
    path = Path(__file__).resolve().parents[1] / ("outputs/live_sop_results.json" if live else "outputs/sop_results.json")
    if not path.exists():
        raise HTTPException(404, "No SOP results yet. From traffic_system run: python -m scripts.evaluate_sop"
                                 + (" --live run (after the live test period)" if live else ""))
    results = json.loads(path.read_text(encoding="utf-8"))
    results.pop("runs", None)
    return results


@app.get("/network")
def network_info(include_nodes: bool = True, mode: Mode | None = None):
    network = serving_network(mode)
    return {
        "cities": CITIES,
        "features": FEATURES,
        "nodes": [node.model_dump() for node in network.nodes] if include_nodes else [],
        "weather_conditions": sorted(network.roads[0].weather),
        "road_ids": [road.id for road in network.roads] if include_nodes else [],
        "cost_config": network.cost_config.model_dump(),
        "boundaries": network.boundaries,
        "node_count": len(network.nodes), "road_count": len(network.roads),
        "serving_mode": serving_mode(mode),
        "data_quality": {
            "mode": network.provenance.get("mode", "supplied_feature_data"),
            "limitations": network.provenance.get("limitations", []),
            "built_at": network.provenance.get("built_at"),
            "estimation_registry": network.provenance.get("estimation_registry", {}),
        },
    }


@app.get("/places/search")
def search_places(query: str = Query(min_length=2, max_length=120)):
    return {"results": places.search_nodes(serving_network(), query)}


@app.get("/places/nearest")
def snap_node(latitude: float = Query(ge=-90, le=90), longitude: float = Query(ge=-180, le=180)):
    return places.nearest_node(serving_network(), latitude, longitude)


@app.get("/roads/nearest")
def snap_road(latitude: float = Query(ge=-90, le=90), longitude: float = Query(ge=-180, le=180)):
    return places.nearest_road(serving_network(), latitude, longitude)


@app.get("/conditions/weather")
def weather_status(departure_date: date = date(2025, 6, 1), time_of_day: int = Query(default=8, ge=0, le=23)):
    if not date(2025, 6, 1) <= departure_date <= date(2026, 5, 31):
        raise HTTPException(422, "Date is outside June 2025–May 2026")
    try:
        return historical_snapshot(departure_date, time_of_day, get_network())[1]
    except (OSError, ValueError, KeyError):
        raise HTTPException(503, "Historical weather snapshot is unavailable") from None


@app.post("/conditions/weather/refresh")
def weather_refresh():
    raise HTTPException(410, "Live weather is outside June 2025–May 2026. Select a historical date/hour to load the archived provider records.")


@app.get("/conditions/live")
def live_conditions_status():
    """Live collection and release status, the in-app collector's state, and the latest capture's coverage."""
    from app.live_collector import collector
    return {**live_status(), "collector": collector.state}


def _route_inputs(network: RoadNetwork, conditions: Conditions):
    try:
        validate_conditions(network, conditions)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if conditions.live_at is not None:
        from src.live.snapshot import LiveDataUnavailable, live_snapshot
        try:
            live = live_snapshot(conditions.live_at)
        except LiveDataUnavailable as exc:
            raise HTTPException(503, f"Live snapshot unavailable: {exc}") from None
        return None, {"captured_at": live.captured_at.isoformat(), "weather_at": live.weather_at.isoformat(),
                      **{key: value for key, value in live.summary.items()}}, None
    weather_by_city, observation = None, None
    if conditions.weather == "historical":
        try:
            weather_by_city, observation = historical_snapshot(conditions.departure_date, conditions.time_of_day, network)
        except (OSError, ValueError, KeyError):
            raise HTTPException(503, "Historical weather snapshot is unavailable") from None
    try:
        traffic = traffic_for(network, conditions)
    except (OSError, ValueError):
        raise HTTPException(503, "Hourly traffic table is unavailable") from None
    return weather_by_city, observation, traffic


class SearchOutcomeError(HTTPException):
    def __init__(self, status: int, outcome: str, message: str):
        super().__init__(status_code=status, detail={"outcome": outcome, "message": message})


def _search(solve):
    """Distinct outcomes: a route (possibly time-limited), no feasible route, time limit without a route,
    search exhaustion, unavailable data and invalid requests."""
    try:
        return solve()
    except NoFeasibleRoute as exc:
        raise SearchOutcomeError(404, "no_feasible_route", str(exc)) from exc
    except SearchTimeLimit as exc:
        raise SearchOutcomeError(504, "search_time_limit", str(exc)) from exc
    except SearchExhausted as exc:
        raise SearchOutcomeError(422, "search_exhausted", str(exc)) from exc
    except (CriterionInputError, MissingWeatherObservation) as exc:
        raise SearchOutcomeError(503, "data_unavailable", str(exc)) from exc
    except ValueError as exc:
        raise SearchOutcomeError(422, "invalid_request", str(exc)) from exc


# Recent decisions, so /routes/explain explains the selected route and snapshot instead of searching again.
DECISIONS: "OrderedDict[str, dict]" = OrderedDict()
MAX_DECISIONS = 64


def _remember(network, algorithm, mode, seed, origin, destination, road_ids, conditions, native_cost, common_cost,
              stop_reason, visits, ant_routes) -> dict:
    calibration = network.cost_config.calibration_id
    record = {"search_id": uuid4().hex, "algorithm": algorithm, "mode": mode, "seed": seed, "origin": origin,
              "destination": destination, "road_ids": road_ids, "snapshot_id": snapshot_id(conditions, calibration),
              "conditions": conditions.model_dump(mode="json"), "calibration_id": calibration,
              "graph_built_at": network.provenance.get("built_at"), "contract_version": CONTRACT_VERSION,
              "native_objective": "ten_criterion_critic_cost" if algorithm == "antroute" else "cheng_comprehensive_cost",
              "native_cost": native_cost, "common_evaluation_cost": common_cost, "stop_reason": stop_reason}
    DECISIONS[record["search_id"]] = {**record, "_conditions": conditions, "_visits": visits, "_ant_routes": ant_routes}
    while len(DECISIONS) > MAX_DECISIONS:
        DECISIONS.popitem(last=False)
    return record


def _route_response(network, roads, origin, conditions, weather_by_city, observation, traffic):
    """Route geometry and criteria. path_cost is always the common ten-criterion cost, for any algorithm."""
    coordinates = []
    nodes = {node.id: node for node in network.nodes}
    for road in roads:
        coordinates.extend(road.geometry if not coordinates else road.geometry[1:])
    segments = [segment_values(network, road, conditions, nodes, weather_by_city, traffic) for road in roads]
    return {
        "route": {
            "node_ids": [origin] + [road.target for road in roads],
            "road_ids": [road.id for road in roads],
            "geometry": coordinates,
            "road_names": [road.name or road.id for road in roads],
            "feature_sources": {road.id: road.feature_sources for road in roads},
            "segment_features": [{"road_id": road.id, "path_cost": score.cost, "physical": score.physical,
                                  "criteria": score.criteria, "scaled": score.scaled, "flags": list(score.flags)}
                                 for road, score in zip(roads, segments)],
        },
        "contributions": route_contributions(network, roads, segments),
        "conditions": conditions.model_dump(mode="json"),
        "weather_observation": observation,
        "data_quality": network.provenance.get("mode", "supplied_feature_data"),
        "estimation_registry": network.provenance.get("estimation_registry", {}),
        "traffic": _disclosure(conditions),
    }


@app.post("/routes/plan")
def plan_route(request: RouteRequest):
    received = perf_counter()
    network, conditions = _release(request.conditions)
    weather_by_city, observation, traffic = _route_inputs(network, conditions)
    started = perf_counter()  # Search budget includes graph preparation; input loading is not search time.
    result = _search(lambda: ant_colony_route(prepare_graph(network, conditions, weather_by_city, traffic),
                                              request.origin, request.destination, ACO, node_positions(network),
                                              distance_reference(network, "antroute"), started))
    roads = [edge.road for edge in result.roads]
    response = _route_response(network, roads, request.origin, conditions, weather_by_city, observation, traffic)
    common = sum(segment["path_cost"] for segment in response["route"]["segment_features"])
    decision = _remember(network, "antroute", "controlled", ACO.seed, request.origin, request.destination, [road.id for road in roads],
                         conditions, sum(edge.cost for edge in result.roads), common, result.stop_reason,
                         result.visits, result.ant_routes)
    return {**response, "aco": ACO.model_dump(), "decision": decision,
            "evaluation": {**evaluate(result, ACO), "response_time_ms": (perf_counter() - received) * 1000}}


class ExplainRequest(StudyModel):
    search_id: str = Field(min_length=1, max_length=64)


@app.post("/routes/explain")
def explain_route(request: ExplainRequest):
    """Why the selected ANTROUTE decision left out shorter or cheaper roads, on that decision's immutable snapshot.

    The reference routes (Dijkstra) explain; they never plan, and no ACO search is run here.
    """
    record = DECISIONS.get(request.search_id)
    if record is None:
        raise SearchOutcomeError(404, "decision_not_found", "Unknown or expired search_id; plan the route again")
    if record["algorithm"] != "antroute":
        raise SearchOutcomeError(422, "invalid_request", "Explanations cover ANTROUTE's ten-criterion decisions")
    conditions = record["_conditions"]
    network = _release(conditions)[0]
    if network.cost_config.calibration_id != record["calibration_id"]:
        raise SearchOutcomeError(409, "calibration_changed", "The release changed since this decision; plan the route again")
    conditions = record["_conditions"]
    weather_by_city, _, traffic = _route_inputs(network, conditions)
    nodes = {node.id: node for node in network.nodes}
    graph = prepare_graph(network, conditions, weather_by_city, traffic)
    edges = {edge.road.id: edge for roads in graph.values() for edge in roads}
    chosen = [edges[road_id] for road_id in record["road_ids"]]
    colony = SimpleNamespace(visits=record["_visits"], ant_routes=record["_ant_routes"])
    explained = explain(network, graph, chosen, colony, record["origin"], record["destination"],
                        lambda road: segment_values(network, road, conditions, nodes, weather_by_city, traffic))
    public = {key: value for key, value in record.items() if not key.startswith("_")}
    return {**explained, "decision": public}


@app.post("/routes/baseline")
def plan_baseline_route(request: BaselineRequest):
    """Cheng (2023) on the same eligibility, traffic and weather as /routes/plan, with its own native cost."""
    received = perf_counter()
    network, conditions = _release(request.conditions)
    weather_by_city, observation, traffic = _route_inputs(network, conditions)
    started = perf_counter()
    result = _search(lambda: cheng_route(cheng_graph(network, conditions, weather_by_city, traffic),
                                         request.origin, request.destination, request.baseline, node_positions(network),
                                         distance_reference(network, "cheng"), None, started))
    roads = [edge.road for edge in result.roads]
    response = _route_response(network, roads, request.origin, conditions, weather_by_city, observation, traffic)
    common = sum(segment["path_cost"] for segment in response["route"]["segment_features"])
    decision = _remember(network, "cheng", request.baseline.mode, request.baseline.seed, request.origin, request.destination,
                         [road.id for road in roads], conditions, sum(edge.cost for edge in result.roads), common,
                         result.stop_reason, result.visits, result.ant_routes)
    computed = ahp()
    return {**response, "decision": decision,
            "baseline": {**REFERENCE, "settings": request.baseline.model_dump(),
                         "ahp_weights": dict(zip(("path_length", "travel_time", "traffic_flow"), AHP_WEIGHTS)),
                         "ahp_weights_computed": list(computed.weights), "ahp_consistency_ratio": computed.consistency_ratio,
                         "references": network.provenance.get("cheng_references"),
                         "comprehensive_cost": decision["native_cost"]},
            "evaluation": {**evaluate(result, request.baseline), "path_cost": common,
                           "response_time_ms": (perf_counter() - received) * 1000}}


@app.post("/routes/trip")
def simulate_route_trip(request: TripRequest):
    """Plan, drive and re-plan as the hour, weather, closures or (live) the saved capture change during the trip."""
    received = perf_counter()
    network, conditions = _release(request.conditions)
    _route_inputs(network, conditions)  # Rejects invalid departure conditions before simulating.
    if conditions.live_at is not None:
        departure = conditions.live_at  # Live trips replay the saved captures from this one.
    else:
        departure = datetime.combine(conditions.departure_date, time(conditions.time_of_day, request.departure_minute),
                                     tzinfo=MANILA)
    result = _search(lambda: simulate_trip(
        network, request.origin, request.stops, departure, conditions.weather,
        conditions.closed_road_ids, request.events, request.algorithm, ACO, request.baseline, request.replan))
    coordinates = []
    for road in result.roads:
        coordinates.extend(road.geometry if not coordinates else road.geometry[1:])
    evaluation = result.evaluation()
    initial = result.searches[0]
    decision = _remember(network, request.algorithm, request.baseline.mode if request.algorithm == "cheng" else "controlled",
                         (request.baseline if request.algorithm == "cheng" else ACO).seed, request.origin, request.stops[0],
                         result.departure_plan, result.departure_conditions, initial.route_cost, None, initial.stop_reason,
                         result.departure_visits, result.departure_ant_routes)
    return {
        "algorithm": {"name": request.algorithm, "description": ALGORITHMS[request.algorithm],
                      "aco": ACO.model_dump(), "baseline": request.baseline.model_dump(),
                      "replan": request.replan.model_dump()},
        "decision": decision,
        "route": {
            "node_ids": [request.origin] + [road.target for road in result.roads],
            "road_ids": [road.id for road in result.roads],
            "geometry": coordinates,
            "road_names": [road.name or road.id for road in result.roads],
            "segments": [{"road_id": road.id, "entered_at_minute": entered / 60, "travel_time_s": seconds, "path_cost": cost}
                         for road, entered, seconds, cost in zip(result.roads, result.entered_at_s, result.travel_time_s, result.path_costs)],
        },
        "trip": {"departure": departure.isoformat(), "travel_time_s": sum(result.travel_time_s),
                 "stops": request.stops, "stop_arrival_minutes": [seconds / 60 for seconds in result.stop_arrivals_s],
                 "events": [event.model_dump(mode="json") for event in request.events],
                 "changes_without_replanning": result.unsearched_changes},
        "conditions": conditions.model_dump(mode="json"),
        "evaluation": {**evaluation, "response_time_ms": (perf_counter() - received) * 1000},
        "contributions": route_contributions(network, result.roads, result.segment_features),
        "data_quality": network.provenance.get("mode", "supplied_feature_data"),
        "traffic": _disclosure(conditions),
    }
