"""The versioned mathematical contract: ten additive routing criteria per directed road segment.

Three kinds of value are kept apart:
  physical   measurements in their own units (metres, seconds, vehicles/hour) for simulation and display;
  criteria   the ten additive criteria below, which sum along a route and are unchanged when a road is
             split at a node that is neither an intersection nor a signal;
  scaled     criterion / development maximum, the values the fixed weights multiply.

For one directed segment with length L (km), traversal time T (min), congestion severity g, road-class
severity p, time-profile severity h, weather severity a, flood portions (km, severity), directional
capacity K (veh/h) and encounters I, S at its downstream node:

  distance          L
  congestion        L * g
  travel_time       T
  intersections     I
  signals           S
  road_type         L * p
  time_profile      L * h
  weather           L * a
  flood_exposure    sum(portion_km * portion_severity)
  capacity_burden   L / K          (provisional study mapping, not a published road penalty)

  scaled[j] = criterion[j] / development_max[j]
  edge_cost = sum_j weight[j] * scaled[j]          route_cost = sum of edge costs

Scaled values and costs are not clipped: a value above the development maximum is flagged, not capped.
Calibration, routing, exports and explanations all call criteria() and scale(), so they share one definition.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

CONTRACT_VERSION = 2
CRITERIA = ("distance", "congestion", "travel_time", "intersections", "signals",
            "road_type", "time_profile", "weather", "flood_exposure", "capacity_burden")
UNITS = {
    "distance": "km", "congestion": "km x congestion severity", "travel_time": "min",
    "intersections": "encounters", "signals": "encounters", "road_type": "km x class severity",
    "time_profile": "km x profile severity", "weather": "km x weather severity",
    "flood_exposure": "km x flood class severity", "capacity_burden": "km per veh/h",
}
LABELS = {
    "distance": "Distance", "congestion": "Congestion", "travel_time": "Travel time", "intersections": "Intersections",
    "signals": "Traffic signals", "road_type": "Road type", "time_profile": "Time profile", "weather": "Weather",
    "flood_exposure": "Flood exposure", "capacity_burden": "Capacity burden",
}
# Criteria that are a severity times length; their route-level severity is a length-weighted mean.
LENGTH_WEIGHTED = {"congestion", "road_type", "time_profile", "weather", "flood_exposure"}


class CriterionInputError(ValueError):
    """An input to the criterion contract is missing, non-finite or outside its declared range."""


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise CriterionInputError(f"{name} must be a finite number")
    return float(value)


def _unit(name: str, value: float) -> float:
    value = _finite(name, value)
    if not 0 <= value <= 1:
        raise CriterionInputError(f"{name} must be within [0, 1]")
    return value


@dataclass(frozen=True)
class SegmentInputs:
    """Physical inputs of one directed segment under one condition snapshot."""
    length_m: float
    traversal_s: float  # Including congestion and weather effects.
    congestion: float  # g in [0, 1]
    class_severity: float  # p in [0, 1]
    profile_severity: float  # h in [0, 1]
    weather_severity: float  # a in [0, 1]
    flood_portions: tuple[tuple[float, float], ...]  # (km, severity in [0, 1]) for each mapped hazard portion.
    capacity_vph: float  # K > 0, this direction
    intersection: int  # I: 1 when the downstream node is an intersection approach
    signal: int  # S: 1 when the downstream node has a traffic signal
    flags: tuple[str, ...] = field(default=())  # Missing-data and provenance flags carried to outputs.

    def validate(self) -> None:
        length = _finite("length_m", self.length_m)
        if length <= 0:
            raise CriterionInputError("length_m must be positive")
        if _finite("traversal_s", self.traversal_s) <= 0:
            raise CriterionInputError("traversal_s must be positive")
        if _finite("capacity_vph", self.capacity_vph) <= 0:
            raise CriterionInputError("capacity_vph must be positive")
        for name in ("congestion", "class_severity", "profile_severity", "weather_severity"):
            _unit(name, getattr(self, name))
        covered = 0.0
        for kilometres, severity in self.flood_portions:
            if _finite("flood portion km", kilometres) < 0:
                raise CriterionInputError("flood portion lengths must be nonnegative")
            _unit("flood portion severity", severity)
            covered += kilometres
        if covered > length / 1000 * (1 + 1e-6) + 1e-9:
            raise CriterionInputError("flood portions are longer than the segment")
        for name in ("intersection", "signal"):
            if getattr(self, name) not in (0, 1):
                raise CriterionInputError(f"{name} must be 0 or 1 per traversed approach")


def criteria(inputs: SegmentInputs) -> dict[str, float]:
    """The ten additive criteria of one segment. Pure: no I/O and no calibration values."""
    inputs.validate()
    km = inputs.length_m / 1000
    values = {
        "distance": km,
        "congestion": km * inputs.congestion,
        "travel_time": inputs.traversal_s / 60,
        "intersections": float(inputs.intersection),
        "signals": float(inputs.signal),
        "road_type": km * inputs.class_severity,
        "time_profile": km * inputs.profile_severity,
        "weather": km * inputs.weather_severity,
        "flood_exposure": sum(kilometres * severity for kilometres, severity in inputs.flood_portions),
        "capacity_burden": km / inputs.capacity_vph,
    }
    for name, value in values.items():
        if not math.isfinite(value) or value < 0:
            raise CriterionInputError(f"criterion {name} must be finite and nonnegative")
    return values


def check_calibration(weights: dict[str, float], maxima: dict[str, float]) -> None:
    """Weights in [0, 1] summing to one; a positive finite maximum for every positively weighted criterion."""
    if set(weights) != set(CRITERIA) or set(maxima) != set(CRITERIA):
        raise CriterionInputError("Weights and development maxima must name exactly the ten criteria")
    for name in CRITERIA:
        _unit(f"weight {name}", weights[name])
        maximum = _finite(f"development maximum {name}", maxima[name])
        if maximum < 0:
            raise CriterionInputError(f"development maximum {name} must be nonnegative")
        if weights[name] > 0 and maximum <= 0:
            raise CriterionInputError(f"criterion {name} has positive weight but no positive development maximum")
    if abs(sum(weights.values()) - 1) > 1e-9:
        raise CriterionInputError("Weights must sum to one")


@dataclass(frozen=True)
class Scaled:
    values: dict[str, float | None]  # criterion / development maximum; None when the maximum is zero
    contributions: dict[str, float]  # weight * scaled; zero for zero-weight criteria
    cost: float
    exceedances: tuple[str, ...]  # criteria above their development maximum (scaled > 1), not clipped


def scale(values: dict[str, float], weights: dict[str, float], maxima: dict[str, float]) -> Scaled:
    """scaled[j] = criterion[j] / development_max[j]; edge cost = sum_j weight[j] * scaled[j]."""
    scaled, contributions, over = {}, {}, []
    for name in CRITERIA:
        maximum = maxima[name]
        scaled[name] = values[name] / maximum if maximum > 0 else None
        if weights[name] > 0:
            if scaled[name] is None:
                raise CriterionInputError(f"criterion {name} has positive weight but no positive development maximum")
            contributions[name] = weights[name] * scaled[name]
        else:
            contributions[name] = 0.0
        if scaled[name] is not None and scaled[name] > 1 + 1e-12:
            over.append(name)
    return Scaled(scaled, contributions, sum(contributions.values()), tuple(over))
