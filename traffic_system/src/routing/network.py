"""Explicit input features and strict five-city geometry validation."""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from shapely.geometry import LineString, Point, shape
from shapely.ops import unary_union
from shapely.prepared import prep

CITIES = ("San Juan", "Manila", "Mandaluyong", "Makati", "Pasig")
City = Literal["San Juan", "Manila", "Mandaluyong", "Makati", "Pasig"]
Unit = Annotated[float, Field(ge=0, le=1)]
Positive = Annotated[float, Field(gt=0)]
FEATURES = (
    "distance", "traffic_congestion", "travel_time", "number_of_intersections",
    "traffic_signal", "road_type", "time_of_day", "weather", "flood_risk", "road_capacity",
)


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
    cost: Unit
    travel_time_multiplier: float = Field(ge=1)
    flood_risk: Unit | None = None


class Road(StudyModel):
    id: str = Field(min_length=1)
    source: str
    target: str
    distance_m: Positive
    road_type: str
    road_capacity_vph: Positive
    flood_risk: Unit
    traffic_congestion: list[Unit] = Field(min_length=24, max_length=24)
    travel_time_s: list[Positive] = Field(min_length=24, max_length=24)
    time_of_day: list[Unit] = Field(min_length=24, max_length=24)
    weather: dict[str, WeatherEffect] = Field(min_length=1)
    # WGS84 longitude/latitude coordinates, in source-to-target order.
    geometry: list[tuple[float, float]] = Field(min_length=2)
    name: str = ""
    feature_sources: dict[str, str] = Field(default_factory=dict)
    free_flow_time_s: Positive | None = None
    volume_capacity_ratio: list[Annotated[float, Field(ge=0)]] | None = Field(default=None, min_length=24, max_length=24)


class CostConfig(StudyModel):
    weights: dict[str, Positive] = Field(default_factory=lambda: dict.fromkeys(FEATURES, 0.1))
    distance_scale_m: Positive = 1000
    travel_time_scale_s: Positive = 600
    capacity_scale_vph: Positive = 4000
    road_type_costs: dict[str, Unit] = Field(min_length=1)

    @model_validator(mode="after")
    def check_weights(self):
        if set(self.weights) != set(FEATURES):
            raise ValueError("Weights must contain exactly the ten study features")
        if abs(sum(self.weights.values()) - 1) > 1e-9:
            raise ValueError("Feature weights must sum to one")
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
            if road.road_type not in self.cost_config.road_type_costs:
                raise ValueError(f"Road type {road.road_type} has no configured cost")
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
