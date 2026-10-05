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
    optimize_stop_order: bool = True
    model: Literal["antroute", "baseline"] = "antroute"


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
    # None only on the placeholder routes, which no algorithm produced.
    algorithm: Optional[str] = None
    # Why a baseline route is not IACO's own, when it is not.
    fallback_reason: Optional[str] = None
    # Mean predicted congestion risk along the route, on one yardstick for both systems.
    mean_risk: Optional[float] = None


class RoutePlanResponse(BaseModel):
    routes: List[RouteOption]
    # Set when `routes` is empty for a reason the user should see, e.g. no baseline route.
    notice: Optional[str] = None


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
    # What these numbers compare ANTROUTE's congestion forecast against. This is a forecasting
    # reference, not the routing baseline (Improved ACO), and the app must not present one as
    # the other.
    baseline_name: str = "Always 'Medium'"