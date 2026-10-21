"""Simulated trips: plan at departure, drive, and protect a valid route when conditions change at a node.

Both algorithms share this simulator, its inputs, update opportunities and scoring. At each node arrival:
  1. apply the events that are due (closures, reopenings, weather);
  2. form a new immutable snapshot (hour from the Asia/Manila trip clock, weather, closures);
  3. rebuild eligibility and the algorithm's native edge costs;
  4. rescore the incumbent remainder under the new native objective (invalid if it uses a closed road);
  5. when the trigger policy fires, run a bounded ACO search with fresh pheromone;
  6. compare candidate and incumbent under the same new native objective;
  7. adopt the candidate only if the incumbent is invalid or the candidate improves beyond tolerance.
A valid incumbent is kept after a worse candidate, a tie, a time limit or a failed search.

Trigger policies (ReplanConfig.policy):
  controlled_common        every snapshot change is an update opportunity for both algorithms (SOP experiments);
  antroute_cost_threshold  full-system ANTROUTE: a road's native cost changed by >= threshold, or a road closed/reopened;
  cheng_reference_printed  full-system Cheng reference: eq. 9 as printed on roads whose travel time changed, warm pheromone.
Live trips (weather "live") replay the saved live captures: the snapshot at each node is the latest capture at
or before the trip clock (held up to LIVE_HOLD, the information a driver would have), so a new 15-minute capture
is a condition change like a new hour.
Every driven road is scored with the ten-criterion cost and travel time of the moment it is entered (the common
evaluation cost), whichever native cost the algorithm routes by. The vehicle does not re-enter driven nodes
unless no other route remains.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from time import perf_counter
from typing import Literal

from pydantic import Field

from .aco_kernel import NoFeasibleRoute, SearchExhausted, SearchResult
from .aco_routing import AntColonyConfig, ant_colony_route, distance_reference, node_positions
from .baseline_cheng import ChengConfig, cheng_route, cheng_snapshot, rescale_pheromone
from .condition_update import LIVE_WEATHER, Conditions, live_conditions, manila_time, snapshot_id
from .dynamic_weight import SegmentScore, WeightedRoad, prepare_graph, segment_values
from .historical_traffic import traffic_for
from .historical_weather import historical_snapshot
from .network import Road, RoadNetwork, StudyModel

Algorithm = Literal["antroute", "cheng"]
LIVE_HOLD = timedelta(minutes=60)  # A live trip fails if collection has no capture this recent.
Policy = Literal["controlled_common", "antroute_cost_threshold", "cheng_reference_printed"]


class TripEvent(StudyModel):
    at_minute: float = Field(ge=0, le=1440)
    close_road_ids: frozenset[str] = frozenset()
    reopen_road_ids: frozenset[str] = frozenset()
    weather: str | None = None


class ReplanConfig(StudyModel):
    policy: Policy = "controlled_common"
    threshold: float = Field(default=0.10, gt=0, le=10)  # antroute_cost_threshold only.
    # A candidate replaces a valid incumbent only when cost < incumbent - (abs + rel * incumbent).
    adopt_abs_tol: float = Field(default=0, ge=0)
    adopt_rel_tol: float = Field(default=1e-9, ge=0, lt=1)


def changed_roads(old_costs: dict[str, float], new_costs: dict[str, float], threshold: float) -> int:
    """Roads whose native cost changed by >= threshold, plus roads closed or reopened."""
    changed = sum(1 for road_id, old in old_costs.items()
                  if road_id in new_costs and abs(new_costs[road_id] - old) >= threshold * old)
    return changed + len(old_costs.keys() ^ new_costs.keys())


@dataclass
class SearchRecord:
    kind: Literal["initial", "reroute"]
    at_node: str
    at_minute: float
    trigger: str
    snapshot_id: str
    affected_roads: int
    iterations: int
    iterations_to_convergence: int | None  # N_conv; None when the search did not converge.
    best_found_at: int  # Iteration that first found this search's final best route.
    stop_reason: str  # converged, iteration_limit, time_limit, search_exhausted
    route_cost: float | None  # Candidate cost in the algorithm's native objective.
    incumbent_cost: float | None  # Remainder of the current route under the same new objective; None if invalid.
    decision: str  # initial, adopted_incumbent_invalid, adopted_better, kept_incumbent, kept_incumbent_search_failed
    search_time_ms: float  # ACO search only.
    routing_time_ms: float  # Recalculation: snapshot rebuild, rescoring and search for this routing.
    best_cost_by_iteration: list[float | None]


@dataclass
class TripResult:
    roads: list[Road] = field(default_factory=list)
    entered_at_s: list[float] = field(default_factory=list)
    travel_time_s: list[float] = field(default_factory=list)
    path_costs: list[float] = field(default_factory=list)
    segment_features: list[SegmentScore] = field(default_factory=list)  # Score of each road when entered.
    searches: list[SearchRecord] = field(default_factory=list)
    unsearched_changes: list[dict] = field(default_factory=list)
    monitoring_time_ms: float = 0.0  # Condition checks that did not lead to a search.
    stop_arrivals_s: list[float] = field(default_factory=list)
    detection_delays_s: list[float] = field(default_factory=list)  # Simulated time from an event to its detection at a node.
    # The departure decision (leg 1), kept so it can be explained without searching again.
    departure_conditions: Conditions | None = None
    departure_plan: list[str] = field(default_factory=list)
    departure_visits: dict[str, int] = field(default_factory=dict)
    departure_ant_routes: int = 0

    def evaluation(self) -> dict:
        """Measures of the trip as driven. Service durations (ms) are separate from simulated journey time (s)."""
        initial = self.searches[0]
        converged = [search.iterations_to_convergence for search in self.searches if search.iterations_to_convergence is not None]
        reroutes = [search for search in self.searches if search.kind == "reroute"]
        planning = sum(search.routing_time_ms for search in self.searches if search.kind == "initial")
        recalculation = sum(search.routing_time_ms for search in reroutes)
        return {
            "distance_travel_m": sum(road.distance_m for road in self.roads),
            "path_cost": sum(self.path_costs),  # Common evaluation cost: ten-criterion cost when each road was entered.
            "search_time_ms": sum(search.search_time_ms for search in self.searches),
            "initial_search_time_ms": initial.search_time_ms,
            "computational_time_ms": planning + recalculation + self.monitoring_time_ms,
            "initial_routing_time_ms": planning,
            "dynamic_rerouting_time_ms": recalculation,
            "recalculation_time_ms": recalculation,
            "reroutes": len(reroutes),
            "routes_adopted": sum(1 for search in reroutes if search.decision.startswith("adopted")),
            "mean_rerouting_time_ms": recalculation / len(reroutes) if reroutes else None,
            "detection_delay_s": self.detection_delays_s,
            "convergence": {
                "iterations_to_convergence": initial.iterations_to_convergence,
                "converged": initial.iterations_to_convergence is not None,
                "stop_reason": initial.stop_reason,
                "first_best_found_iteration": initial.best_found_at,
                "mean_iterations_to_convergence": sum(converged) / len(converged) if converged else None,
                "searches_not_converged": len(self.searches) - len(converged),
                "searches": [asdict(search) for search in self.searches],
            },
        }


def _changes(old: Conditions, new: Conditions) -> str:
    parts = []
    if old.live_at != new.live_at and new.live_at is not None:
        parts.append(f"live capture {old.live_at:%H:%M} -> {new.live_at:%H:%M}")
    elif (old.departure_date, old.time_of_day) != (new.departure_date, new.time_of_day):
        parts.append(f"hour {old.time_of_day:02d}:00 -> {new.time_of_day:02d}:00")
    if old.weather != new.weather:
        parts.append(f"weather {old.weather} -> {new.weather}")
    if old.closed_road_ids != new.closed_road_ids:
        parts.append("closures changed")
    return ", ".join(parts)


def simulate_trip(network: RoadNetwork, origin: str, destination: str | Sequence[str], departure: datetime, weather: str,
                  closures: frozenset[str], events: list[TripEvent], algorithm: Algorithm,
                  aco: AntColonyConfig = AntColonyConfig(), baseline: ChengConfig = ChengConfig(),
                  replan: ReplanConfig = ReplanConfig()) -> TripResult:
    """destination: one node, or the stops to visit in order (fixed order). Each leg is planned when it starts with
    fresh pheromone; the trip clock, events and conditions carry across legs. departure must be timezone-aware."""
    stops = [destination] if isinstance(destination, str) else list(destination)
    if not stops:
        raise ValueError("At least one destination is required")
    if any(a == b for a, b in zip([origin, *stops], stops)):
        raise ValueError("Each stop must differ from the one before it")
    if weather == LIVE_WEATHER and any(event.weather for event in events):
        raise ValueError("Live trips take their weather from the saved captures; weather events do not apply")
    if replan.policy == "cheng_reference_printed" and algorithm != "cheng":
        raise ValueError("The printed eq. 9 reference policy applies to the Cheng baseline only")
    departure = manila_time(departure)
    nodes = {node.id: node for node in network.nodes}
    positions = node_positions(network)
    reference = distance_reference(network, algorithm)
    calibration = network.cost_config.calibration_id
    config = baseline if algorithm == "cheng" else aco
    if algorithm == "cheng" and replan.policy == "cheng_reference_printed" and baseline.mode != "cheng_reference_printed":
        config = baseline.model_copy(update={"mode": "cheng_reference_printed"})
    inputs: dict[tuple, tuple] = {}
    pending = sorted(events, key=lambda event: event.at_minute)
    state_weather, state_closed = weather, set(closures)
    clock = departure
    result = TripResult()

    def conditions_now() -> Conditions:
        if state_weather == LIVE_WEATHER:
            from ..live.snapshot import nearest_capture
            return live_conditions(nearest_capture(clock, tolerance=LIVE_HOLD), frozenset(state_closed))
        return Conditions(departure_date=clock.date(), time_of_day=clock.hour, weather=state_weather,
                          closed_road_ids=frozenset(state_closed))

    def snapshot_inputs(conditions: Conditions):
        key = (conditions.departure_date, conditions.time_of_day, conditions.weather, conditions.live_at)
        if conditions.live_at is not None:
            return None, None  # Live inputs come from the capture itself.
        if key not in inputs:
            weather_by_city = None
            if conditions.weather == "historical":
                weather_by_city = historical_snapshot(conditions.departure_date, conditions.time_of_day, network)[0]
            inputs[key] = weather_by_city, traffic_for(network, conditions)
        return inputs[key]

    def build(conditions: Conditions):
        """Graph of native edge costs (eligible roads only) and the per-road values the trigger compares."""
        weather_by_city, traffic = snapshot_inputs(conditions)
        if algorithm == "cheng":
            return cheng_snapshot(network, conditions, weather_by_city, traffic)
        graph = prepare_graph(network, conditions, weather_by_city, traffic)
        return graph, {edge.road.id: edge.cost for roads in graph.values() for edge in roads}

    def run(graph, start: str, driven: set[str], pheromone, started: float) -> SearchResult:
        view = {node: [edge for edge in roads if edge.road.target not in driven] for node, roads in graph.items()}

        def solve(adjacency):
            if algorithm == "cheng":
                return cheng_route(adjacency, start, destination, config, positions, reference, pheromone, started)
            return ant_colony_route(adjacency, start, destination, config, positions, reference, started)
        try:
            return solve(view)
        except NoFeasibleRoute:
            if not driven:
                raise
            return solve(graph)  # Only a route back through driven nodes remains.

    def rescore(graph, remainder: list[Road]) -> float | None:
        """Incumbent remainder cost under the new native objective; None when a road is no longer eligible."""
        costs = {edge.road.id: edge.cost for roads in graph.values() for edge in roads}
        if any(road.id not in costs for road in remainder):
            return None
        return sum(costs[road.id] for road in remainder)

    def record(kind, start, trigger, conditions, affected, found, stop, candidate, incumbent, decision, search_ms, routing_ms):
        result.searches.append(SearchRecord(
            kind, start, (clock - departure).total_seconds() / 60, trigger, snapshot_id(conditions, calibration), affected,
            found.iterations_completed if found else 0, found.converged_at if found else None,
            found.first_best_found_iteration if found else 0, stop, candidate, incumbent, decision, search_ms, routing_ms,
            found.best_cost_by_iteration if found else []))

    def apply_events():
        nonlocal state_weather
        minute = (clock - departure).total_seconds() / 60
        while pending and pending[0].at_minute <= minute:
            event = pending.pop(0)
            state_closed.update(event.close_road_ids)
            state_closed.difference_update(event.reopen_road_ids)
            state_weather = event.weather or state_weather
            result.detection_delays_s.append((minute - event.at_minute) * 60)

    current = origin
    conditions = graph = values = None
    for leg, destination in enumerate(stops):
        started = perf_counter()
        apply_events()
        latest = conditions_now()
        if latest != conditions:
            conditions = latest
            graph, values = build(conditions)
        trigger = "departure" if leg == 0 else f"leg {leg + 1} starts at stop {leg}"
        found = run(graph, current, set(), None, started)  # A failed initial search fails the trip.
        record("initial", current, trigger, conditions, 0, found, found.stop_reason, sum(edge.cost for edge in found.roads),
               None, "initial", found.search_time_ms, (perf_counter() - started) * 1000)
        plan, pheromone = [edge.road for edge in found.roads], found.pheromone
        if leg == 0:
            result.departure_conditions, result.departure_plan = conditions, [road.id for road in plan]
            result.departure_visits, result.departure_ant_routes = found.visits, found.ant_routes
        visited = [current]
        while current != destination:
            apply_events()
            latest = conditions_now()
            if latest != conditions:
                started = perf_counter()
                new_graph, new_values = build(latest)
                trigger = _changes(conditions, latest)
                if replan.policy == "cheng_reference_printed":
                    affected = len(rescale_pheromone(pheromone, values, new_values))
                elif replan.policy == "antroute_cost_threshold":
                    affected = changed_roads(values, new_values, replan.threshold)
                else:
                    affected = len(values.keys() ^ new_values.keys()) + sum(
                        1 for road_id, old in values.items() if road_id in new_values and new_values[road_id] != old)
                conditions, graph, values = latest, new_graph, new_values
                incumbent = rescore(graph, plan)
                if affected or incumbent is None:
                    try:
                        # Fresh pheromone except in the printed eq. 9 reference mode (cheng_route decides).
                        found = run(graph, current, set(visited[:-1]), pheromone, started)
                    except (NoFeasibleRoute, SearchExhausted) as error:
                        if incumbent is None:
                            raise  # No valid route remains: the trip fails as a whole.
                        stop = "search_exhausted" if not isinstance(error, NoFeasibleRoute) else "no_feasible_route"
                        record("reroute", current, trigger, conditions, affected, None, stop, None, incumbent,
                               "kept_incumbent_search_failed", 0.0, (perf_counter() - started) * 1000)
                    else:
                        candidate = sum(edge.cost for edge in found.roads)
                        better = incumbent is None or candidate < incumbent - (replan.adopt_abs_tol + replan.adopt_rel_tol * incumbent)
                        decision = ("adopted_incumbent_invalid" if incumbent is None else "adopted_better") if better else "kept_incumbent"
                        if better:
                            plan, pheromone = [edge.road for edge in found.roads], found.pheromone
                        record("reroute", current, trigger, conditions, affected, found, found.stop_reason, candidate, incumbent,
                               decision, found.search_time_ms, (perf_counter() - started) * 1000)
                else:
                    result.unsearched_changes.append({"at_node": current, "at_minute": (clock - departure).total_seconds() / 60,
                                                      "change": trigger})
                    result.monitoring_time_ms += (perf_counter() - started) * 1000
            road = plan.pop(0)
            weather_by_city, traffic = snapshot_inputs(conditions)
            score = segment_values(network, road, conditions, nodes, weather_by_city, traffic)
            result.roads.append(road)
            result.entered_at_s.append((clock - departure).total_seconds())
            result.travel_time_s.append(score.physical["travel_time_s"])
            result.path_costs.append(score.cost)
            result.segment_features.append(score)
            clock += timedelta(seconds=score.physical["travel_time_s"])
            current = road.target
            visited.append(current)
            if len(result.roads) > len(network.roads) * len(stops):
                raise RuntimeError("Trip simulation exceeded the number of roads in the network")
        result.stop_arrivals_s.append((clock - departure).total_seconds())
    return result
