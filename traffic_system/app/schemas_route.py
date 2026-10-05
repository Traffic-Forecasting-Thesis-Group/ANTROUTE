from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel


class DestinationInput(BaseModel):
    name: str
    lat: Optional[float] = None
    lng: Optional[float] = None


class RoutePlanRequest(BaseModel):
    origin: str
    origin_lat: Optional[float] = None
    origin_lng: Optional[float] = None
    destinations: List[DestinationInput]
    model: Literal["antroute", "baseline"] = "antroute"
    # When the driver leaves, ISO 8601 (an offset is expected; without one it is read as
    # Manila time). Omitted means leaving now.
    depart_at: Optional[datetime] = None


class PathPoint(BaseModel):
    lat: float
    lng: float


class RouteOption(BaseModel):
    label: str
    via: str
    duration_min: int
    distance_km: float
    congestion_level: str
    event_note: Optional[str] = None
    path: List[PathPoint] = []


class RoutePlanResponse(BaseModel):
    routes: List[RouteOption]
    departure_time: str  # the departure planned for, Manila time with offset
    # Which recorded traffic the routes were planned on, or that none is loaded.
    traffic_note: str


class ComparisonMetricRow(BaseModel):
    metric: str
    antroute: str
    baseline: str
    improvement_pct: int
    higher_is_better: bool = False


class OptimalityPct(BaseModel):
    antroute: float
    baseline: float


class ComparisonMetricsResponse(BaseModel):
    route_optimality_pct: OptimalityPct
    metrics: List[ComparisonMetricRow]