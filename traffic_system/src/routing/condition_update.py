"""One immutable condition snapshot per ACO run; closures are constraints."""
from typing import Annotated
from datetime import date

from pydantic import Field

from .network import RoadNetwork, StudyModel


class Conditions(StudyModel):
    departure_date: date = date(2025, 6, 1)
    time_of_day: Annotated[int, Field(strict=True, ge=0, le=23)]
    weather: str = Field(min_length=1)
    closed_road_ids: frozenset[str] = frozenset()


def validate_conditions(network: RoadNetwork, conditions: Conditions) -> None:
    if not date(2025, 6, 1) <= conditions.departure_date <= date(2026, 5, 31):
        raise ValueError("Departure date must be between June 1, 2025 and May 31, 2026")
    unknown = conditions.closed_road_ids - {road.id for road in network.roads}
    if unknown:
        raise ValueError(f"Unknown closed road IDs: {', '.join(sorted(unknown))}")
    if conditions.weather not in network.roads[0].weather:
        raise ValueError("Weather condition is not available in this network")
