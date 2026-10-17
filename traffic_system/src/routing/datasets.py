"""Tabular exports of the same ten-feature values the router uses for one date and hour."""
import csv
import io
from collections import Counter

import networkx as nx

from .condition_update import validate_conditions
from .dynamic_weight import segment_values
from .historical_traffic import traffic_for
from .historical_weather import historical_snapshot, historical_grid_assignments, historical_grid_records
from .network import FEATURES, CITIES

FIELDS = ["timestamp", "road_id", "road_name", "city", "source_node", "target_node", *FEATURES,
          "weather_condition", "weather_grid", "precip_mm", "traffic_location", "closed", "path_cost",
          "traffic_classification"]


def snapshot_rows(network, conditions, city=None):
    validate_conditions(network, conditions)
    traffic = traffic_for(network, conditions)
    nodes = {node.id: node for node in network.nodes}
    effects, grids, records = None, {}, {}
    if conditions.weather == "historical":
        effects, _ = historical_snapshot(conditions.departure_date, conditions.time_of_day, network)
        grids = historical_grid_assignments()
        records = historical_grid_records(conditions.departure_date, conditions.time_of_day)
    timestamp = f"{conditions.departure_date.isoformat()}T{conditions.time_of_day:02d}:00:00+08:00"
    for road in network.roads:
        source = nodes[road.source]
        if city is not None and source.city != city:
            continue
        raw, _, cost = segment_values(network, road, conditions, nodes, effects, traffic)
        closed = road.id in conditions.closed_road_ids
        grid = grids.get(road.source, "")
        record = records.get(grid, {})
        yield {"timestamp": timestamp, "road_id": road.id, "road_name": road.name,
               "city": source.city, "source_node": road.source, "target_node": road.target,
               **{name: round(value, 8) if isinstance(value, float) else value for name, value in raw.items()},
               "weather_condition": conditions.weather, "weather_grid": grid,
               "precip_mm": record.get("precip_mm", ""), "traffic_location": road.traffic_location or "",
               "closed": closed, "path_cost": None if closed else round(cost, 10),
               "traffic_classification": "estimated_not_measured"}


def csv_chunks(rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS)
    writer.writeheader()
    for index, row in enumerate(rows, 1):
        writer.writerow(row)
        if index % 256 == 0:
            yield buffer.getvalue()
            buffer.seek(0)
            buffer.truncate(0)
    if buffer.tell():
        yield buffer.getvalue()


def coverage(network):
    graph = nx.DiGraph()
    graph.add_nodes_from(node.id for node in network.nodes)
    graph.add_edges_from((road.source, road.target) for road in network.roads)
    components = list(nx.strongly_connected_components(graph))
    largest = max(components, key=len)
    nodes = {node.id: node for node in network.nodes}
    counts = Counter(nodes[road.source].city for road in network.roads)
    weather = network.provenance.get("historical_weather", {})
    return {"directed_segments": len(network.roads), "nodes": len(nodes),
            "segments_by_city": {city: counts[city] for city in CITIES},
            "largest_strongly_connected_nodes": len(largest),
            "strongly_connected_components": len(components),
            "hours": 8760, "potential_segment_hour_rows": len(network.roads) * 8760,
            "retained_weather_grids": weather.get("grid_count"),
            "assigned_weather_grids": weather.get("assigned_grid_count"),
            "excluded_source_segments": network.provenance.get("excluded_roads", {}),
            "complete_road_coverage_verified": False,
            "traffic_locations": len({road.traffic_location for road in network.roads if road.traffic_location}),
            "units": {"distance": "m", "travel_time": "s", "road_capacity": "equivalent vehicles/hour",
                      "weather": "estimated categorical cost [0,1]", "flood_risk": "static hazard proxy [0,1]"},
            "note": "All retained public-driving segments are available for snapshots, including disconnected components. Counts do not establish map completeness."}
