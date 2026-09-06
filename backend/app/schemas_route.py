from typing import List, Optional

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


class RouteOption(BaseModel):
    label: str  
    via: str
    duration_min: int
    distance_km: float
    congestion_level: str 
    event_note: Optional[str] = None


class RoutePlanResponse(BaseModel):
    routes: List[RouteOption]
