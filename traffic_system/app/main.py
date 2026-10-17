"""Route requests use a fresh, consistent snapshot of the three dynamic inputs."""
import os
from functools import lru_cache
from pathlib import Path
from datetime import date
from time import perf_counter

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import Field, ValidationError
from dotenv import load_dotenv

from app import places
from app.study_policy import require_historical_release, study_status

from src.routing.aco_routing import AntColonyConfig, NoFeasibleRoute, SearchExhausted, ant_colony_route
from src.routing.condition_update import Conditions, validate_conditions
from src.routing.dynamic_weight import prepare_graph
from src.routing.metrics import evaluate
from src.routing.network import CITIES, FEATURES, RoadNetwork, StudyModel, load_network
from src.routing.historical_weather import historical_snapshot

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


class RouteRequest(StudyModel):
    origin: str = Field(min_length=1)
    destination: str = Field(min_length=1)
    conditions: Conditions
    aco: AntColonyConfig = Field(default_factory=AntColonyConfig)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/study/status")
def historical_study_status():
    return study_status()


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


@app.post("/routes/plan")
def plan_route(request: RouteRequest, _historical_release: None = Depends(require_historical_release)):
    network = get_network()
    try:
        validate_conditions(network, request.conditions)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    weather_by_city, observation = None, None
    if request.conditions.weather == "historical":
        try:
            weather_by_city, observation = historical_snapshot(request.conditions.departure_date, request.conditions.time_of_day, network)
        except (OSError, ValueError, KeyError):
            raise HTTPException(503, "Historical weather snapshot is unavailable") from None
    # Data loading/validation is input collection. Timed work begins with graph preparation.
    started = perf_counter()
    try:
        graph = prepare_graph(network, request.conditions, weather_by_city)
        positions = {node.id: (node.longitude, node.latitude) for node in network.nodes}
        result = ant_colony_route(graph, request.origin, request.destination, request.aco, positions)
    except NoFeasibleRoute as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SearchExhausted as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    elapsed_ms = (perf_counter() - started) * 1000
    coordinates = []
    for edge in result.roads:
        geometry = edge.road.geometry
        coordinates.extend(geometry if not coordinates else geometry[1:])
    return {
        "route": {
            "node_ids": [request.origin] + [edge.road.target for edge in result.roads],
            "road_ids": [edge.road.id for edge in result.roads],
            "geometry": coordinates,
            "road_names": [edge.road.name or edge.road.id for edge in result.roads],
            "feature_sources": {edge.road.id: edge.road.feature_sources for edge in result.roads},
        },
        "conditions": request.conditions.model_dump(mode="json"),
        "aco": request.aco.model_dump(),
        "evaluation": evaluate(result, request.aco, elapsed_ms),
        "weather_observation": observation,
        "data_quality": network.provenance.get("mode", "supplied_feature_data"),
        "estimation_registry": network.provenance.get("estimation_registry", {}),
    }
