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
    # Which algorithm produced this route: "antroute", or for the baseline "iaco",
    # "shortest_distance" (IACO found no route) or "mixed" (multi-stop, some legs each).
    algorithm: Optional[str] = None
    # Why a baseline route is not IACO's own, when it is not.
    fallback_reason: Optional[str] = None
    # Mean predicted congestion risk along the route, on one yardstick for both systems.
    mean_risk: Optional[float] = None


class RoutePlanResponse(BaseModel):
    routes: List[RouteOption]
    departure_time: str  # the departure planned for, Manila time with offset
    # Which recorded traffic the routes were planned on, or that none is loaded.
    traffic_note: str = ""
    # Set when `routes` is empty for a reason the user should see, e.g. no baseline route.
    notice: Optional[str] = None


class ComparisonMetricRow(BaseModel):
    metric: str
    antroute: str
    baseline: str
    # Relative difference, positive when ANTROUTE is better (thesis Equations 7-8).
    improvement_pct: Optional[float] = None
    higher_is_better: bool = False
    # Wilcoxon signed-rank p-value, on the routing evaluation only.
    p_value: Optional[float] = None
    significant: Optional[bool] = None


class ComparisonMetricsResponse(BaseModel):
    # "routing": ANTROUTE vs the baseline Improved ACO, from scripts/evaluate_routing.py.
    # "forecast": the congestion model vs an always-Medium forecaster, until that has run.
    # The app labels each by name and must not present one as the other.
    source: Literal["routing", "forecast"]
    title: str
    baseline_name: str
    description: str
    metrics: List[ComparisonMetricRow]