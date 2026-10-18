"""Simulated trips: plan at departure, drive, and react when conditions change at a node.

The two algorithms share this simulator, its inputs, check points and scoring, so only the
way each adapts to a change differs:
  antroute  ANTROUTE: re-plans from the current node with fresh pheromone when a road's
            ten-feature cost changed by >= threshold (default 10%), or a road closed or reopened.
  cheng     Cheng (2023) baseline: eq. 9 on every road whose travel time changed, then a full
            NC_max re-search from the current node with the rescaled pheromone.

Conditions are checked on arrival at each node. The hour follows the trip clock and brings the
hourly traffic, time of day and historical weather with it; scheduled events close or reopen
roads or switch the weather. Every driven road is scored with the ten-feature cost and travel
time of the moment it is entered, whichever cost the algorithm routes by. The vehicle does not
re-enter nodes it has driven through unless no other route remains.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from time import perf_counter
from typing import Literal

from pydantic import Field

from .aco_routing import AntColonyConfig, NoFeasibleRoute, ant_colony_route
from .baseline_cheng import ChengConfig, cheng_route, cheng_snapshot, rescale_pheromone
from .condition_update import Conditions
from .dynamic_weight import WeightedRoad, prepare_graph, segment_values
from .historical_traffic import traffic_for
from .historical_weather import historical_snapshot
from .network import Road, RoadNetwork, StudyModel

Algorithm = Literal["antroute", "cheng"]


class TripEvent(StudyModel):
    at_minute: float = Field(ge=0, le=1440)
    close_road_ids: frozenset[str] = frozenset()
    reopen_road_ids: frozenset[str] = frozenset()
    weather: str | None = None


class ReplanConfig(StudyModel):
    threshold: float = Field(default=0.10, gt=0, le=10)


def changed_roads(old_costs: dict[str, float], new_costs: dict[str, float], threshold: float) -> int:
    """ANTROUTE's re-plan trigger: roads whose cost changed by >= threshold, closed or reopened."""
    changed = sum(1 for road_id, old in old_costs.items()
                  if road_id in new_costs and abs(new_costs[road_id] - old) >= threshold * old)
    return changed + len(old_costs.keys() ^ new_costs.keys())


@dataclass
class SearchRecord:
    kind: Literal["initial", "reroute"]
    at_node: str
    at_minute: float
    trigger: str
    affected_roads: int
    iterations: int
    iterations_to_convergence: int  # Iteration that first found this search's final best route.
    stop_reason: str
    route_cost: float  # In the algorithm's own objective.
    routing_time_ms: float  # Graph build, change handling and search for this routing.
    best_cost_by_iteration: list[float | None]


@dataclass
class TripResult:
    roads: list[Road] = field(default_factory=list)
    entered_at_s: list[float] = field(default_factory=list)
    travel_time_s: list[float] = field(default_factory=list)
    path_costs: list[float] = field(default_factory=list)
    searches: list[SearchRecord] = field(default_factory=list)
    unsearched_changes: list[dict] = field(default_factory=list)
    monitoring_time_ms: float = 0.0  # Condition checks that did not lead to re-routing.
    stop_arrivals_s: list[float] = field(default_factory=list)

    def evaluation(self) -> dict:
        """The SOP measures for the trip as driven.

        Initial routing time = planning each leg when it starts (one leg: the routing at departure).
        Computational time = initial routing + dynamic re-routing + condition checks without a re-route.
        Convergence rate is measured as iterations to convergence of the routing at departure.
        """
        initial = self.searches[0]
        reroutes = [search for search in self.searches if search.kind == "reroute"]
        planning = sum(search.routing_time_ms for search in self.searches if search.kind == "initial")
        rerouting = sum(search.routing_time_ms for search in reroutes)
        return {
            "distance_travel_m": sum(road.distance_m for road in self.roads),
            "path_cost": sum(self.path_costs),
            "computational_time_ms": planning + rerouting + self.monitoring_time_ms,
            "initial_routing_time_ms": planning,
            "dynamic_rerouting_time_ms": rerouting,
            "reroutes": len(reroutes),
            "mean_rerouting_time_ms": rerouting / len(reroutes) if reroutes else None,
            "convergence": {
                "iterations_to_convergence": initial.iterations_to_convergence,
                "mean_iterations_to_convergence": sum(search.iterations_to_convergence for search in self.searches) / len(self.searches),
                "searches": [asdict(search) for search in self.searches],
            },
        }


def _changes(old: Conditions, new: Conditions) -> str:
    parts = []
    if (old.departure_date, old.time_of_day) != (new.departure_date, new.time_of_day):
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
    """destination: one node, or the stops to visit in order. Each leg is routed when it starts,
    with fresh pheromone; the trip clock, events and conditions carry across legs."""
    stops = [destination] if isinstance(destination, str) else list(destination)
    if not stops:
        raise ValueError("At least one destination is required")
    if any(a == b for a, b in zip([origin, *stops], stops)):
        raise ValueError("Each stop must differ from the one before it")
    nodes = {node.id: node for node in network.nodes}
    positions = {node.id: (node.longitude, node.latitude) for node in network.nodes}
    inputs: dict[tuple, tuple] = {}
    pending = sorted(events, key=lambda event: event.at_minute)
    state_weather, state_closed = weather, set(closures)
    clock = departure
    result = TripResult()

    def conditions_now() -> Conditions:
        return Conditions(departure_date=clock.date(), time_of_day=clock.hour, weather=state_weather,
                          closed_road_ids=frozenset(state_closed))

    def snapshot_inputs(conditions: Conditions):
        key = (conditions.departure_date, conditions.time_of_day, conditions.weather)
        if key not in inputs:
            weather_by_city = None
            if conditions.weather == "historical":
                weather_by_city = historical_snapshot(conditions.departure_date, conditions.time_of_day, network)[0]
            inputs[key] = weather_by_city, traffic_for(network, conditions)
        return inputs[key]

    def build(conditions: Conditions):
        """Graph the algorithm routes on, and the per-road values its trigger compares."""
        weather_by_city, traffic = snapshot_inputs(conditions)
        if algorithm == "cheng":
            return cheng_snapshot(network, conditions, weather_by_city, traffic)
        graph = prepare_graph(network, conditions, weather_by_city, traffic)
        return graph, {edge.road.id: edge.cost for roads in graph.values() for edge in roads}

    def search(graph: dict[str, list[WeightedRoad]], start: str, trigger: str, affected: int,
               visited: list[str], pheromone: dict[str, float] | None, preparation_ms: float, kind: str = "reroute"):
        started = perf_counter()
        driven = set(visited[:-1])
        view = {node: [edge for edge in roads if edge.road.target not in driven] for node, roads in graph.items()}

        def run(adjacency):
            if algorithm == "cheng":
                return cheng_route(adjacency, start, destination, baseline, positions, pheromone)
            return ant_colony_route(adjacency, start, destination, aco, positions)
        try:
            found = run(view)
        except NoFeasibleRoute:
            if not driven:
                raise
            found = run(graph)  # Only a route back through driven nodes remains.
        elapsed = (perf_counter() - started) * 1000
        stop = "iteration_limit" if algorithm == "cheng" or not found.converged else "convergence_rule"
        result.searches.append(SearchRecord(
            kind, start, (clock - departure).total_seconds() / 60,
            trigger, affected, len(found.best_cost_by_iteration), found.best_found_at, stop,
            sum(edge.cost for edge in found.roads), preparation_ms + elapsed, found.best_cost_by_iteration))
        return [edge.road for edge in found.roads], found.pheromone

    def apply_events():
        nonlocal state_weather
        minute = (clock - departure).total_seconds() / 60
        while pending and pending[0].at_minute <= minute:
            event = pending.pop(0)
            state_closed.update(event.close_road_ids)
            state_closed.difference_update(event.reopen_road_ids)
            state_weather = event.weather or state_weather

    current = origin
    conditions = graph = values = None
    for leg, destination in enumerate(stops):
        # Plan this leg from the stop just reached (or the origin) under the conditions of now.
        started = perf_counter()
        apply_events()
        latest = conditions_now()
        if latest != conditions:
            conditions = latest
            graph, values = build(conditions)
        trigger = "departure" if leg == 0 else f"leg {leg + 1} starts at stop {leg}"
        plan, pheromone = search(graph, current, trigger, 0, [current], None, (perf_counter() - started) * 1000, "initial")
        visited = [current]
        while current != destination:
            apply_events()
            minute = (clock - departure).total_seconds() / 60
            latest = conditions_now()
            affected, trigger, check_ms = 0, "", 0.0
            if latest != conditions:
                started = perf_counter()
                new_graph, new_values = build(latest)
                trigger = _changes(conditions, latest)
                if algorithm == "cheng":
                    affected = len(rescale_pheromone(pheromone, values, new_values))
                else:
                    affected = changed_roads(values, new_values, replan.threshold)
                check_ms = (perf_counter() - started) * 1000
                if not affected:
                    result.unsearched_changes.append({"at_node": current, "at_minute": minute, "change": trigger})
                    result.monitoring_time_ms += check_ms
                conditions, graph, values = latest, new_graph, new_values
            if not affected and plan and plan[0].id in conditions.closed_road_ids:
                affected, trigger = 1, "next planned road closed"
            if affected:
                plan, pheromone = search(graph, current, trigger, affected, visited, pheromone, check_ms)
            road = plan.pop(0)
            weather_by_city, traffic = snapshot_inputs(conditions)
            raw, _, cost = segment_values(network, road, conditions, nodes, weather_by_city, traffic)
            result.roads.append(road)
            result.entered_at_s.append((clock - departure).total_seconds())
            result.travel_time_s.append(raw["travel_time"])
            result.path_costs.append(cost)
            clock += timedelta(seconds=raw["travel_time"])
            current = road.target
            visited.append(current)
            if len(result.roads) > len(network.roads) * len(stops):
                raise RuntimeError("Trip simulation exceeded the number of roads in the network")
        result.stop_arrivals_s.append((clock - departure).total_seconds())
    return result
