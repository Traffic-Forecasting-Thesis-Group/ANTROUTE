"""Route requests use a fresh, consistent snapshot of the three dynamic inputs."""
import json
import os
from functools import lru_cache
from pathlib import Path
from datetime import date, datetime, time
from time import perf_counter
from itertools import chain, islice

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import Field, ValidationError, model_validator
from dotenv import load_dotenv

from app import places
from app.study_policy import require_historical_release, study_status

from src.routing.aco_routing import AntColonyConfig, NoFeasibleRoute, SearchExhausted, ant_colony_route
from src.routing.contributions import route_contributions
from src.routing.explanation import explain
from src.routing.baseline_cheng import REFERENCE, ChengConfig, ahp, cheng_graph, cheng_route
from src.routing.dynamic_trip import Algorithm, ReplanConfig, TripEvent, simulate_trip
from src.routing.condition_update import Conditions, validate_conditions
from src.routing.dynamic_weight import prepare_graph, segment_values
from src.routing.datasets import coverage, snapshot_rows, csv_chunks
from src.routing.metrics import evaluate
from src.routing.network import CITIES, FEATURES, City, RoadNetwork, StudyModel, load_network
from src.routing.historical_weather import historical_snapshot
from src.routing.historical_traffic import traffic_for

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
app = FastAPI(title="ANTROUTE ACO Study")
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


TRAFFIC_DISCLOSURE = {"classification": "estimated_not_measured",
                      "source": "data/processed/temporal/traffic_hourly.csv",
                      "note": "Hourly congestion and travel-time factors per city and road type from an assumed demand index and the BPR function. No vehicle counts."}


@app.get("/dataset/info", dependencies=[Depends(require_historical_release)])
def dataset_info():
    return {**coverage(get_network()), "traffic": TRAFFIC_DISCLOSURE,
            "features": FEATURES, "period": study_status()["requirements"]}


@app.post("/dataset/snapshot", dependencies=[Depends(require_historical_release)])
def dataset_snapshot(request: SnapshotRequest, offset: int = Query(default=0, ge=0),
                        limit: int = Query(default=100, ge=1, le=1000)):
    network = get_network()
    try:
        rows = list(islice(snapshot_rows(network, request.conditions, request.city), offset, offset + limit))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    cities = {node.id: node.city for node in network.nodes}
    total = sum(1 for road in network.roads if request.city is None or cities[road.source] == request.city)
    return {"rows": rows, "total": total, "offset": offset, "limit": limit,
            "traffic": TRAFFIC_DISCLOSURE}


@app.post("/dataset/export", dependencies=[Depends(require_historical_release)])
def dataset_export(request: SnapshotRequest):
    try:
        rows = iter(snapshot_rows(get_network(), request.conditions, request.city))
        first = next(rows, None)  # Validate inputs and weather before sending response headers.
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    filename = f"antroute_{request.conditions.departure_date}_{request.conditions.time_of_day:02d}.csv"
    return StreamingResponse(csv_chunks(chain([] if first is None else [first], rows)),
                             media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{filename}"'})


# One color per study city for the coverage map; distinct from the route, baseline and explanation colors.
CITY_COLORS = {"Manila": "#0f766e", "San Juan": "#65a30d", "Mandaluyong": "#b45309", "Makati": "#db2777", "Pasig": "#6366f1"}


class CoverageRequest(StudyModel):
    city: City | None = None


@app.post("/dataset/map")
def dataset_map(request: CoverageRequest):
    """Every road segment included in the study, grouped by the city it starts in."""
    network = get_network()
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
def sop_results():
    """Latest SOP experiment summary written by scripts.evaluate_sop (per-run rows omitted)."""
    path = Path(__file__).resolve().parents[1] / "outputs/sop_results.json"
    if not path.exists():
        raise HTTPException(404, "No SOP results yet. From traffic_system run: python -m scripts.evaluate_sop")
    results = json.loads(path.read_text(encoding="utf-8"))
    results.pop("runs", None)
    return results


@app.get("/network")
def network_info(include_nodes: bool = True):
    network = get_network()
    return {
        "cities": CITIES,
        "features": FEATURES,
        "nodes": [node.model_dump() for node in network.nodes] if include_nodes else [],
        "weather_conditions": sorted(network.roads[0].weather),
        "road_ids": [road.id for road in network.roads] if include_nodes else [],
        "cost_config": network.cost_config.model_dump(),
        "boundaries": network.boundaries,
        "node_count": len(network.nodes), "road_count": len(network.roads),
        "data_quality": {
            "mode": network.provenance.get("mode", "supplied_feature_data"),
            "limitations": network.provenance.get("limitations", []),
            "built_at": network.provenance.get("built_at"),
            "estimation_registry": network.provenance.get("estimation_registry", {}),
        },
    }


@app.get("/places/search")
def search_places(query: str = Query(min_length=2, max_length=120)):
    return {"results": places.search_nodes(get_network(), query)}


@app.get("/places/nearest")
def snap_node(latitude: float = Query(ge=-90, le=90), longitude: float = Query(ge=-180, le=180)):
    return places.nearest_node(get_network(), latitude, longitude)


@app.get("/roads/nearest")
def snap_road(latitude: float = Query(ge=-90, le=90), longitude: float = Query(ge=-180, le=180)):
    return places.nearest_road(get_network(), latitude, longitude)


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


def _route_inputs(network: RoadNetwork, conditions: Conditions):
    try:
        validate_conditions(network, conditions)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
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


def _search(solve):
    try:
        return solve()
    except NoFeasibleRoute as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (SearchExhausted, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _route_response(network, roads, origin, conditions, weather_by_city, observation, traffic):
    """Route geometry and features. path_cost is always the ten-feature cost, for any algorithm."""
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
            "segment_features": [{"road_id": road.id, "path_cost": cost, "features": features}
                                 for road, (_, features, cost) in zip(roads, segments)],
        },
        "contributions": route_contributions(network, roads, [(raw, features) for raw, features, _ in segments]),
        "conditions": conditions.model_dump(mode="json"),
        "weather_observation": observation,
        "data_quality": network.provenance.get("mode", "supplied_feature_data"),
        "estimation_registry": network.provenance.get("estimation_registry", {}),
        "traffic": TRAFFIC_DISCLOSURE,
    }


@app.post("/routes/plan")
def plan_route(request: RouteRequest, _historical_release: None = Depends(require_historical_release)):
    network = get_network()
    weather_by_city, observation, traffic = _route_inputs(network, request.conditions)
    # Data loading/validation is input collection. Timed work begins with graph preparation.
    started = perf_counter()
    positions = {node.id: (node.longitude, node.latitude) for node in network.nodes}
    result = _search(lambda: ant_colony_route(prepare_graph(network, request.conditions, weather_by_city, traffic),
                                              request.origin, request.destination, ACO, positions))
    elapsed_ms = (perf_counter() - started) * 1000
    return {**_route_response(network, [edge.road for edge in result.roads], request.origin, request.conditions,
                              weather_by_city, observation, traffic),
            "aco": ACO.model_dump(),
            "evaluation": evaluate(result, ACO, elapsed_ms)}


@app.post("/routes/explain")
def explain_route(request: RouteRequest, _historical_release: None = Depends(require_historical_release)):
    """Why the departure plan left out shorter or cheaper roads (reference routes explain, never plan)."""
    network = get_network()
    weather_by_city, _, traffic = _route_inputs(network, request.conditions)
    positions = {node.id: (node.longitude, node.latitude) for node in network.nodes}
    nodes = {node.id: node for node in network.nodes}
    graph = prepare_graph(network, request.conditions, weather_by_city, traffic)
    colony = _search(lambda: ant_colony_route(graph, request.origin, request.destination, ACO, positions))
    return explain(network, graph, colony.roads, colony, request.origin, request.destination,
                   lambda road: segment_values(network, road, request.conditions, nodes, weather_by_city, traffic))


@app.post("/routes/baseline")
def plan_baseline_route(request: BaselineRequest, _historical_release: None = Depends(require_historical_release)):
    """Cheng (2023) on the same graph, closures, traffic and weather as /routes/plan."""
    network = get_network()
    weather_by_city, observation, traffic = _route_inputs(network, request.conditions)
    started = perf_counter()
    positions = {node.id: (node.longitude, node.latitude) for node in network.nodes}
    result = _search(lambda: cheng_route(cheng_graph(network, request.conditions, weather_by_city, traffic),
                                         request.origin, request.destination, request.baseline, positions))
    elapsed_ms = (perf_counter() - started) * 1000
    response = _route_response(network, [edge.road for edge in result.roads], request.origin, request.conditions,
                               weather_by_city, observation, traffic)
    weights = ahp()
    return {**response,
            "baseline": {**REFERENCE, "settings": request.baseline.model_dump(),
                         "ahp_weights": dict(zip(("path_length", "travel_time", "traffic_flow"), weights.weights)),
                         "ahp_consistency_ratio": weights.consistency_ratio,
                         "comprehensive_cost": sum(edge.cost for edge in result.roads)},
            "evaluation": {
                "distance_travel_m": sum(edge.road.distance_m for edge in result.roads),
                "path_cost": sum(segment["path_cost"] for segment in response["route"]["segment_features"]),
                "computational_time_ms": elapsed_ms,
                "convergence_rule": {
                    "rule": "fixed_iterations",
                    "max_iterations": request.baseline.max_iterations,
                    "iterations": len(result.best_cost_by_iteration),
                    "best_found_at_iteration": result.best_found_at,
                    "iterations_to_convergence": result.best_found_at,
                    "stop_reason": "iteration_limit",
                    "best_cost_by_iteration": result.best_cost_by_iteration,
                },
            }}


@app.post("/routes/trip")
def simulate_route_trip(request: TripRequest, _historical_release: None = Depends(require_historical_release)):
    """Plan, drive and re-plan as the hour, weather or closures change during the trip."""
    network = get_network()
    _route_inputs(network, request.conditions)  # Rejects invalid departure conditions before simulating.
    departure = datetime.combine(request.conditions.departure_date,
                                 time(request.conditions.time_of_day, request.departure_minute))
    result = _search(lambda: simulate_trip(
        network, request.origin, request.stops, departure, request.conditions.weather,
        request.conditions.closed_road_ids, request.events, request.algorithm, ACO, request.baseline, request.replan))
    coordinates = []
    for road in result.roads:
        coordinates.extend(road.geometry if not coordinates else road.geometry[1:])
    return {
        "algorithm": {"name": request.algorithm, "description": ALGORITHMS[request.algorithm],
                      "aco": ACO.model_dump(), "baseline": request.baseline.model_dump(),
                      "replan": request.replan.model_dump()},
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
        "conditions": request.conditions.model_dump(mode="json"),
        "evaluation": result.evaluation(),
        "contributions": route_contributions(network, result.roads, result.segment_features),
        "data_quality": network.provenance.get("mode", "supplied_feature_data"),
        "traffic": TRAFFIC_DISCLOSURE,
    }
