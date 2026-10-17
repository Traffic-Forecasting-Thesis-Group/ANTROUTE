"""Route requests use a fresh, consistent snapshot of the three dynamic inputs."""
import os
from functools import lru_cache
from pathlib import Path
from datetime import date
from time import perf_counter
from itertools import chain, islice

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import Field, ValidationError
from dotenv import load_dotenv

from app import places
from app.study_policy import require_historical_release, study_status

from src.routing.aco_routing import AntColonyConfig, NoFeasibleRoute, SearchExhausted, ant_colony_route
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


class RouteRequest(StudyModel):
    origin: str = Field(min_length=1)
    destination: str = Field(min_length=1)
    conditions: Conditions
    aco: AntColonyConfig = Field(default_factory=AntColonyConfig)


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


@app.post("/dataset/map", dependencies=[Depends(require_historical_release)])
def dataset_map(request: SnapshotRequest):
    network = get_network()
    roads = {road.id: road for road in network.roads}
    bands = [{"label": label, "color": color, "lines": []} for label, color in [
        ("0–0.1", "#16865b"), ("0.1–0.3", "#75a83b"), ("0.3–0.5", "#d79922"),
        ("0.5–0.7", "#df6535"), ("0.7–1", "#a32935"), ("Closed", "#111827")]]
    try:
        for row in snapshot_rows(network, request.conditions, request.city):
            band = 5 if row["closed"] else sum(row["traffic_congestion"] >= threshold for threshold in (.1, .3, .5, .7))
            bands[band]["lines"].append(roads[row["road_id"]].geometry)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"bands": bands, "segment_count": sum(len(band["lines"]) for band in bands),
            "classification": "Estimated congestion index from the hourly traffic table, not observed traffic", "traffic": TRAFFIC_DISCLOSURE}


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
    try:
        traffic = traffic_for(network, request.conditions)
    except (OSError, ValueError):
        raise HTTPException(503, "Hourly traffic table is unavailable") from None
    # Data loading/validation is input collection. Timed work begins with graph preparation.
    started = perf_counter()
    try:
        graph = prepare_graph(network, request.conditions, weather_by_city, traffic)
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
    nodes = {node.id: node for node in network.nodes}
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
            "segment_features": [{"road_id": edge.road.id, "path_cost": edge.cost,
                "features": segment_values(network, edge.road, request.conditions,
                    nodes, weather_by_city, traffic)[0]}
                for edge in result.roads],
        },
        "conditions": request.conditions.model_dump(mode="json"),
        "aco": request.aco.model_dump(),
        "evaluation": evaluate(result, request.aco, elapsed_ms),
        "weather_observation": observation,
        "data_quality": network.provenance.get("mode", "supplied_feature_data"),
        "estimation_registry": network.provenance.get("estimation_registry", {}),
        "traffic": TRAFFIC_DISCLOSURE,
    }
