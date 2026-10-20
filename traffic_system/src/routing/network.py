"""Explicit input features and strict five-city geometry validation."""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from shapely.geometry import LineString, Point, shape
from shapely.ops import unary_union
from shapely.prepared import prep

from .criteria import CONTRACT_VERSION, CRITERIA, check_calibration

CITIES = ("San Juan", "Manila", "Mandaluyong", "Makati", "Pasig")
City = Literal["San Juan", "Manila", "Mandaluyong", "Makati", "Pasig"]
Unit = Annotated[float, Field(ge=0, le=1)]
Positive = Annotated[float, Field(gt=0)]
NonNegative = Annotated[float, Field(ge=0)]
FEATURES = CRITERIA  # The ten routing criteria of contract version 2 (see criteria.py).
DAY_CATEGORIES = ("weekday", "weekend")
FloodClass = Literal["1", "2", "3"]


class StudyModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Node(StudyModel):
    id: str = Field(min_length=1)
    city: City
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    is_intersection: bool
    traffic_signal: bool
    label: str = ""
    feature_sources: dict[str, str] = Field(default_factory=dict)


class WeatherEffect(StudyModel):
    """Assumed effect of one weather category: severity a and travel-time multiplier (not calibrated)."""
    severity: Unit
    travel_time_multiplier: float = Field(ge=1)


class Road(StudyModel):
    id: str = Field(min_length=1)
    source: str
    target: str
    distance_m: Positive
    road_type: str
    road_capacity_vph: Positive  # K: directional capacity of this segment.
    # Length (km) of the segment inside each mapped NOAH hazard class, overlaps resolved to one class per portion.
    flood_km_by_class: dict[FloodClass, NonNegative] = Field(default_factory=dict)
    flood_coverage: Literal["mapped", "unknown"] = "mapped"
    # 24-hour fallback profiles used only when the dated hourly traffic table is not linked.
    traffic_congestion: list[Unit] = Field(min_length=24, max_length=24)
    travel_time_s: list[Positive] = Field(min_length=24, max_length=24)
    # h: development-fitted time-profile severity by day category and local hour.
    time_profile: dict[Literal["weekday", "weekend"], list[Unit]]
    weather: dict[str, WeatherEffect] = Field(min_length=1)
    # WGS84 longitude/latitude coordinates, in source-to-target order.
    geometry: list[tuple[float, float]] = Field(min_length=2)
    name: str = ""
    feature_sources: dict[str, str] = Field(default_factory=dict)
    free_flow_time_s: Positive | None = None
    # Key into the hourly traffic table (one series per road segment); None uses the 24-hour profiles above.
    traffic_location: str | None = None

    @model_validator(mode="after")
    def check_portions_and_profile(self):
        if set(self.time_profile) != set(DAY_CATEGORIES) or any(len(values) != 24 for values in self.time_profile.values()):
            raise ValueError(f"Road {self.id} needs a 24-hour time profile for weekday and weekend")
        if sum(self.flood_km_by_class.values()) > self.distance_m / 1000 * (1 + 1e-6) + 1e-9:
            raise ValueError(f"Road {self.id} flood portions are longer than the road")
        return self


class CostConfig(StudyModel):
    """Fixed calibration of contract version 2. Never changed by requests, closures or replanning."""
    contract_version: Literal[2] = CONTRACT_VERSION
    weights: dict[str, Unit]
    development_max: dict[str, NonNegative]  # Routing scale of each criterion, from development data only.
    road_class_severity: dict[str, Unit] = Field(min_length=1)  # p by OSM road class
    flood_class_severity: dict[FloodClass, Unit]  # Assumed scoring scenario for NOAH classes.
    calibration_id: str = ""  # SHA-256 of the calibration artifact that produced these values.

    @model_validator(mode="after")
    def check_weights(self):
        check_calibration(self.weights, self.development_max)
        if set(self.flood_class_severity) != {"1", "2", "3"}:
            raise ValueError("Flood class severity must map NOAH classes 1, 2 and 3")
        return self


class RoadNetwork(StudyModel):
    # Each key contains the sourced and validated city Polygon/MultiPolygon geometry.
    boundaries: dict[City, dict]
    nodes: list[Node] = Field(min_length=2)
    roads: list[Road] = Field(min_length=1)
    cost_config: CostConfig
    provenance: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_scope_and_features(self):
        if set(self.boundaries) != set(CITIES):
            raise ValueError("Boundaries must contain exactly the five study cities")
        polygons = {}
        for city, geometry in self.boundaries.items():
            try:
                polygon = shape(geometry)
            except Exception as exc:
                raise ValueError(f"Invalid boundary for {city}") from exc
            if polygon.geom_type not in {"Polygon", "MultiPolygon"} or polygon.is_empty or not polygon.is_valid:
                raise ValueError(f"Valid polygon boundary required for {city}")
            west, south, east, north = polygon.bounds
            if not (-180 <= west <= east <= 180 and -90 <= south <= north <= 90):
                raise ValueError("Boundaries must use WGS84 longitude/latitude")
            polygons[city] = polygon
        area = prep(unary_union(list(polygons.values())))
        polygons = {city: prep(polygon) for city, polygon in polygons.items()}
        nodes = {node.id: node for node in self.nodes}
        if len(nodes) != len(self.nodes):
            raise ValueError("Node IDs must be unique")
        if len({road.id for road in self.roads}) != len(self.roads):
            raise ValueError("Road IDs must be unique")
        for node in self.nodes:
            if not polygons[node.city].covers(Point(node.longitude, node.latitude)):
                raise ValueError(f"Node {node.id} is outside its declared study city")
        weather_names = set(self.roads[0].weather)
        for road in self.roads:
            if road.source not in nodes or road.target not in nodes:
                raise ValueError(f"Road {road.id} references an unknown node")
            if road.road_type not in self.cost_config.road_class_severity:
                raise ValueError(f"Road type {road.road_type} has no declared class severity")
            if set(road.weather) != weather_names or any(not name.strip() for name in road.weather):
                raise ValueError("All roads must support the same nonempty weather condition names")
            for coordinates, node_id in ((road.geometry[0], road.source), (road.geometry[-1], road.target)):
                node = nodes[node_id]
                if abs(coordinates[0] - node.longitude) > 1e-9 or abs(coordinates[1] - node.latitude) > 1e-9:
                    raise ValueError(f"Road {road.id} geometry does not match its endpoints")
            line = LineString(road.geometry)
            if not line.is_valid or line.length <= 0 or not area.covers(line):
                raise ValueError(f"Road {road.id} geometry leaves the five-city study area or is invalid")
        return self


def load_network(path: Path) -> RoadNetwork:
    return RoadNetwork.model_validate_json(path.read_text(encoding="utf-8"))
