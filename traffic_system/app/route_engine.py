from datetime import datetime
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo

from app import risk_routing


class RoutingUnavailableError(Exception):
    """The road network files aren't on this server, so nothing can be routed."""


def local_departure(depart_at: Optional[datetime]) -> datetime:
    """
    The departure as naive Manila wall-clock time -- the clock the recorded risk
    windows are written in. The app sends an offset-aware time; one without an offset
    is taken as already local. None means leaving now.
    """
    tz = ZoneInfo(risk_routing.LOCAL_TZ)
    if depart_at is None:
        return datetime.now(tz).replace(tzinfo=None, microsecond=0)
    if depart_at.tzinfo is None:
        return depart_at
    return depart_at.astimezone(tz).replace(tzinfo=None)


def plan_routes(
    points: List[Tuple[float, float]],
    model: str = "antroute",
    depart_at: Optional[datetime] = None,
) -> Tuple[List[dict], str]:
    """
    Routes through `points` (origin first, as (lat, lng)) from the real CNN+LSTM ->
    RADR STGNN -> MLP Decoder -> Dynamic Weight Engine -> ACO pipeline. `model`
    selects ANTRoute (risk-aware) or the baseline (same search, lambda=0).

    `depart_at` is naive Manila time (see local_departure); routes use the congestion
    recorded at that time of day. Also returns a line naming that recorded traffic,
    or saying none is loaded.

    Raises RoutingUnavailableError, risk_routing.RouteOutsideNetworkError or
    risk_routing.NoRouteError -- never a made-up route.
    """
    if not risk_routing.available():
        raise RoutingUnavailableError
    window, traffic_note = risk_routing.window_for_departure(depart_at or local_departure(None))
    routes = risk_routing.plan_real_routes(points[0], points[1:], model, window)
    return routes, traffic_note
