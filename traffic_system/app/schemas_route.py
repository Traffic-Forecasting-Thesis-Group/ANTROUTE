from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

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
    # Simulation mode: plan on this recorded window (a RecordedWindow.start from
    # /routes/recorded-sessions) instead of the window matching depart_at.
    recorded_window: Optional[str] = None


class CameraReading(BaseModel):
    """One camera in one recorded window."""
    camera: str
    node_id: int
    lat: float
    lon: float
    model_risk: Optional[float] = None       # the model's mean risk on the camera's edges, 0-1
    human_score: Optional[float] = None      # the human congestion label as 0 / 0.5 / 1
    human_label: Optional[Literal["Light", "Medium", "Heavy"]] = None
    vehicle_count: Optional[float] = None    # mean YOLO vehicle count over the window
    distance_m: Optional[int] = None         # from the route, on a route's inputs


class IncidentReading(BaseModel):
    """One MMDA incident in force at a recorded window's moment."""
    location: str
    intersection: str
    kind: str
    lanes: int
    strength: float                          # base impact x linear decay, 0-1
    reported_at: str
    minutes_ago: Optional[int] = None
    node_id: Optional[int] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    distance_m: Optional[int] = None         # from the route, on a route's inputs


class WeatherReading(BaseModel):
    available: bool
    where: Optional[str] = None
    hour: Optional[str] = None
    temp_c: Optional[float] = None
    precip_mm: Optional[float] = None
    precip_mm_max: Optional[float] = None
    humidity_pct: Optional[float] = None
    cells: Optional[int] = None
    source: Optional[str] = None
    note: Optional[str] = None


class RouteInputs(BaseModel):
    """
    The recorded inputs along one route, for comparing alternatives: the cameras and incidents
    on it, the weather and flood along it, and the vehicle counts at its cameras. Always
    "recorded": read from one window of risk_edges.csv, not a live feed.
    """
    recorded: bool = True
    window: str
    cameras: List[CameraReading] = []
    incidents: List[IncidentReading] = []
    weather: WeatherReading
    flood: Dict[str, Any] = {}
    vehicle_count_mean: Optional[float] = None


class WindowInfo(BaseModel):
    window: str
    window_end: Optional[str] = None
    day: str
    session: Literal["AM", "PM"]
    label: str
    split: str                               # train / val / test / infer
    held_out: bool                           # True for the test split (May 25)
    split_note: str = ""


class RecordedInputs(BaseModel):
    """The "Inputs used" panel: every recorded input of one window, over the whole map."""
    recorded: bool = True
    matched_window: WindowInfo
    cameras: List[CameraReading] = []
    incidents: List[IncidentReading] = []
    weather: WeatherReading
    flood: Dict[str, Any] = {}
    vehicle_counts_available: bool = False


class RecordedWindow(BaseModel):
    start: str
    end: Optional[str] = None


class RecordedSession(BaseModel):
    day: str
    session: Literal["AM", "PM"]
    label: str
    split: str
    held_out: bool
    note: str = ""
    labelled_edges: int = 0
    windows: List[RecordedWindow]


class RecordedSessionsResponse(BaseModel):
    sessions: List[RecordedSession]
    note: str


class WhatIfCase(BaseModel):
    key: str
    label: str
    description: str
    camera_mean: float                       # mean absolute change in risk on camera edges
    camera_max: float
    other_mean: float                        # ... on every other edge
    other_max: float
    other_moved: float                       # share of other edges whose risk moved by more than 0.001


class WhatIfResponse(BaseModel):
    """
    Precomputed what-if cases: one input of the real checkpoint changed at a time, in the same
    batch, and how far the risk moved. Re-inference by the checkpoint offline, not live.
    """
    model: str
    windows: int
    note: str
    cases: List[WhatIfCase]
    route_effect: Optional[Dict[str, Any]] = None


class PathPoint(BaseModel):
    lat: float
    lng: float


class RouteOption(BaseModel):
    label: str
    via: str
    # The ETA engine's travel time. Not a thesis metric and never scored against a reference,
    # so it always travels with duration_note ("estimated, not evaluated").
    duration_min: int
    duration_note: str = "estimated, not evaluated"
    distance_km: float
    congestion_level: str
    # "Estimated from weather, road type and traffic reports" when most of the route is in the assumed risk tier,
    # so the level comes from the non-visual estimate rather than the camera-fed model.
    congestion_note: Optional[str] = None
    event_note: Optional[str] = None
    path: List[PathPoint] = []
    # Which algorithm produced this route: "antroute", or for the baseline "iaco",
    # "shortest_distance" (IACO found no route) or "mixed" (multi-stop, some legs each).
    algorithm: Optional[str] = None
    # Why a baseline route is not IACO's own, when it is not.
    fallback_reason: Optional[str] = None
    # Mean predicted congestion risk along the route, on one yardstick for both systems.
    mean_risk: Optional[float] = None
    # Share of the route's length in the observed or propagated risk tier, i.e. whose risk traces
    # back to the camera-fed decoder. Below 0.5 the congestion_level comes with congestion_note.
    risk_coverage: Optional[float] = None
    # Share of the route's length per risk tier (src/routing/dynamic_weight.py TIERS):
    # "observed" (camera subgraph, decoder prediction), "propagated" (within 1.5 km of it) and
    # "assumed" (farther: estimated from weather, flood, incidents, clock and road class).
    risk_sources: Optional[Dict[str, float]] = None
    # The recorded inputs along this route (cameras, incidents, weather, flood, vehicle counts).
    inputs: Optional[RouteInputs] = None


class RoutePlanResponse(BaseModel):
    routes: List[RouteOption]
    departure_time: str  # the departure planned for, Manila time with offset
    # Which recorded traffic the routes were planned on, or that none is loaded.
    traffic_note: str = ""
    # Set when `routes` is empty for a reason the user should see, e.g. no baseline route.
    notice: Optional[str] = None
    # The recorded inputs of the window the routes were planned on (the "Inputs used" panel).
    recorded_inputs: Optional[RecordedInputs] = None
    # True when the user chose the recorded day and session (simulation mode).
    simulation: bool = False


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


class ScenarioEvaluation(BaseModel):
    """The evaluation table for one scenario of Section 3.9, or "overall" for every trip."""
    scenario: str
    label: str
    n_trials: int
    metrics: List[ComparisonMetricRow]


class TripEvaluationResponse(BaseModel):
    """
    The routing evaluation for one planned trip (POST /routes/trip-evaluation): its own
    Route Optimality against Google Maps and its congestion and event exposure, when it is one
    of the evaluated test trips. Only those have Google Maps travel times, so any other trip
    has no evaluation.
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
    # The same table per scenario of Section 3.9, "overall" first; routing evaluation only.
    scenarios: List[ScenarioEvaluation] = []