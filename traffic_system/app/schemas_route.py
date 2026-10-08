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
    # Share of the route's length whose risk came from the model (predicted on the road, or on a
    # predicted road within 1.5 km); the rest is the window's median. Below 0.5 the
    # congestion_level is "unknown" rather than a level read off that median.
    risk_coverage: Optional[float] = None


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


class RouteOptimalitySummary(BaseModel):
    """Mean Route Optimality (%) per system over the evaluated test trips (Equation 1)."""
    antroute: float
    baseline: float
    n_trials: int


class TripEvaluationResponse(BaseModel):
    """
    The routing evaluation for one planned trip (POST /routes/trip-evaluation): its own
    Route Optimality and ETA errors against Apple Maps, when it is one of the evaluated test
    trips. Only those have Apple Maps travel times, so any other trip has no evaluation.
    """
    # "evaluated": this trip, at this departure time, is a test trip; metrics are its own.
    # "other_time": the same stops were evaluated, but departing at evaluated_window's time.
    # "not_evaluated": these stops are not among the test trips.
    status: Literal["evaluated", "other_time", "not_evaluated"]
    message: str
    trip_id: Optional[str] = None
    scenario_type: Optional[str] = None
    evaluated_window: Optional[str] = None
    baseline_name: str = "Improved ACO (Cheng 2023)"
    route_optimality: Optional[RouteOptimalitySummary] = None
    metrics: List[ComparisonMetricRow] = []


class ComparisonMetricsResponse(BaseModel):
    # "routing": ANTROUTE vs the baseline Improved ACO, from scripts/evaluate_routing.py --
    #            the only source GET /routes/comparison-metrics serves.
    # "forecast": congestion-model diagnostics, GET /routes/forecast-metrics only.
    source: Literal["routing", "forecast"]
    title: str
    baseline_name: str
    description: str
    metrics: List[ComparisonMetricRow]
    route_optimality: Optional[RouteOptimalitySummary] = None