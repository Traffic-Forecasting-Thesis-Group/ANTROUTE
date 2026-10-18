"""Assemble the routing network and release from the preprocessed feature tables.

Order: scripts.preprocess_road_features -> scripts.preprocess_traffic -> scripts.build_study_network
"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from shapely import wkt
from shapely.geometry import mapping, shape

from app.study_policy import RELEASE_ARTIFACTS
from src.routing.condition_update import Conditions
from src.routing.critic import CriticAccumulator
from src.routing.dynamic_weight import segment_values
from src.routing.estimation import estimation_registry
from src.routing.historical_traffic import TABLE as TRAFFIC_TABLE, traffic_snapshot
from src.routing.historical_weather import build_weather, historical_snapshot
from src.routing.network import CITIES, FEATURES, CostConfig, RoadNetwork

ROOT = Path(__file__).resolve().parents[1]
# Every 15th day moves the weekday by one, so the sample covers each weekday and weekends.
CRITIC_DAY_STEP = 15


def read_csv(path):
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def typical_profiles(traffic_path):
    """Mean weekday congestion and travel-time factor per location and hour.

    Stored on each road as a 24-hour fallback; routing uses the dated hourly table.
    """
    sums = defaultdict(lambda: [[0.0, 0.0, 0] for _ in range(24)])
    for row in read_csv(traffic_path):
        if row["day_type"] == "weekday":
            cell = sums[row["location_id"]][int(row["hour"])]
            cell[0] += float(row["traffic_congestion"])
            cell[1] += float(row["travel_time_factor"])
            cell[2] += 1
    return {location: [(congestion / count, factor / count) for congestion, factor, count in hours]
            for location, hours in sums.items()}


def build(features_dir, boundary_path, assumption_path, traffic_path, quality_path=None):
    """Assemble the routing network from the preprocessed feature tables only."""
    boundaries = json.loads(boundary_path.read_text(encoding="utf-8"))
    polygons = {feature["properties"]["city"]: shape(feature["geometry"]) for feature in boundaries["features"]}
    if set(polygons) != set(CITIES):
        raise ValueError("Exactly the five study boundaries are required")
    assumptions = json.loads(assumption_path.read_text(encoding="utf-8"))
    profiles = typical_profiles(traffic_path)
    floods = {row["road_id"]: row for row in read_csv(features_dir / "segment_flood.csv")}
    roads, labels, sources = [], {}, Counter()
    for row in read_csv(features_dir / "road_segments.csv"):
        profile = profiles[row["traffic_location"]]
        free_flow = float(row["free_flow_time_s"])
        flood = floods[row["road_id"]]
        provenance = {
            "distance": row["geometry_source"],
            "traffic_congestion": "traffic_hourly_table_estimated_bpr",
            "travel_time": f"{row['speed_source']}_free_flow_x_traffic_hourly_table_factor",
            "road_type": "osm_highway_tag",
            "time_of_day": "exact_hour_with_assumed_peak_policy",
            "weather": "historical_provider_grid_with_estimated_effects",
            "flood_risk": flood["flood_source"],
            "road_capacity": row["capacity_source"],
        }
        roads.append({
            "id": row["road_id"], "source": row["source_node"], "target": row["target_node"],
            "distance_m": float(row["distance_m"]), "road_type": row["road_type"],
            "road_capacity_vph": float(row["road_capacity_vph"]), "flood_risk": float(flood["flood_risk"]),
            "traffic_congestion": [round(congestion, 6) for congestion, _ in profile],
            "travel_time_s": [round(free_flow * factor, 6) for _, factor in profile],
            "time_of_day": [float(hour in assumptions["peak_hours"]) for hour in range(24)],
            # The real date/hour-specific weather effect is attached by the snapshot loader.
            "weather": {**assumptions["weather_scenarios"], "historical": assumptions["weather_scenarios"]["clear"]},
            "geometry": list(wkt.loads(row["geometry_wkt"]).coords), "name": row["road_name"],
            "feature_sources": provenance, "free_flow_time_s": free_flow, "traffic_location": row["traffic_location"],
        })
        for node_id in (row["source_node"], row["target_node"]):
            if row["road_name"]:
                labels.setdefault(node_id, set()).add(row["road_name"])
        sources.update(provenance.values())
    nodes = [{
        "id": row["node_id"], "city": row["city"], "latitude": float(row["latitude"]), "longitude": float(row["longitude"]),
        "is_intersection": row["is_intersection"] == "1", "traffic_signal": row["traffic_signal"] == "1",
        "label": f"{' / '.join(sorted(labels.get(row['node_id'], []))[:3]) or 'Unnamed road'} · {row['city']}",
        "feature_sources": {"number_of_intersections": "derived_osm_street_count", "traffic_signal": row["signal_source"]},
    } for row in read_csv(features_dir / "road_nodes.csv")]
    static_report = {}
    if quality_path is not None and quality_path.exists():
        static_report = json.loads(quality_path.read_text(encoding="utf-8")).get("static", {})
    print(f"Validating network: {len(nodes)} nodes, {len(roads)} directed roads", flush=True)
    return RoadNetwork.model_validate({
        "boundaries": {city: mapping(polygon) for city, polygon in polygons.items()},
        "nodes": nodes, "roads": roads,
        "cost_config": {"road_type_costs": {name: settings["cost"] for name, settings in assumptions["road_types"].items()}},
        "provenance": {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "mode": "historical_period_with_estimates",
            "period": {"start": "2025-06-01", "end": "2026-05-31"},
            "boundaries": [feature["properties"] for feature in boundaries["features"]],
            "graph_source": "OpenStreetMap via OSMnx, preprocessed into data/processed/features",
            "flood_source": "Project NOAH five-year flood hazard polygons; class/3; maximum road intersection",
            "traffic_source": f"{TRAFFIC_TABLE} (estimated, not observed)",
            "feature_tables": {name: hashlib.sha256((features_dir / name).read_bytes()).hexdigest()
                               for name in ("road_nodes.csv", "road_segments.csv", "segment_flood.csv")},
            "assumptions": assumptions, "assumptions_sha256": hashlib.sha256(assumption_path.read_bytes()).hexdigest(),
            "source_counts": dict(sources), "excluded_roads": static_report.get("excluded_source_edges", {}),
            "estimation_registry": estimation_registry(assumptions),
            "limitations": [
                "Traffic congestion and travel time come from an estimated hourly table (assumed demand index and BPR), not traffic measurements or vehicle counts.",
                "Capacity and free-flow speed are estimates, including on roads with OSM lane/speed tags.",
                "Unrecorded signals are treated as absent; OSM signal coverage is incomplete.",
                "Flood exposure is a mapped hazard class, not measured water depth or current passability; bridges may be elevated.",
                "Weather effects are scenario assumptions applied to historical provider records.",
                "The time-of-day peak policy is distinct from congestion but correlated. Feature weights come from CRITIC over estimated feature values; they are not calibrated against observed route choices.",
                "September 2026 geometry, signals and classifications are held constant as an estimated surrogate for June 2025–May 2026.",
                "Historical weather grid assignment and speed effects are estimates, not road-level weather measurements.",
                "The source graph does not encode turn restrictions in this node-edge routing model.",
            ],
        },
    })


def critic_sample_hours():
    day = date(2025, 6, 1)
    while day <= date(2026, 5, 31):
        for hour in range(24):
            yield day, hour
        day += timedelta(days=CRITIC_DAY_STEP)


def apply_critic(network: RoadNetwork) -> None:
    """Replace the provisional equal weights with CRITIC weights over the sampled segment-hours."""
    nodes = {node.id: node for node in network.nodes}
    accumulator = CriticAccumulator(FEATURES)
    hours = list(critic_sample_hours())
    for index, (day, hour) in enumerate(hours, 1):
        conditions = Conditions(departure_date=day, time_of_day=hour, weather="historical")
        weather, traffic = historical_snapshot(day, hour, network)[0], traffic_snapshot(day, hour)
        accumulator.add([[features[name] for name in FEATURES] for _, features, _ in
                         (segment_values(network, road, conditions, nodes, weather, traffic) for road in network.roads)])
        if index % 100 == 0 or index == len(hours):
            print(f"CRITIC sample: {index}/{len(hours)} hours", flush=True)
    result = accumulator.result()
    network.cost_config = CostConfig(**{**network.cost_config.model_dump(), "weights": result["weights"]})
    network.provenance["weighting"] = {
        "method": "CRITIC (Diakoulaki, Mavrotas and Papayannakis, 1995)",
        "reference": "https://doi.org/10.1016/0305-0548(94)00059-H",
        "sample": f"All {len(network.roads)} segments at every hour of every {CRITIC_DAY_STEP}th day from 2025-06-01 "
                  f"({len(hours)} hours), historical provider weather",
        **result,
    }


def main():
    output = ROOT / "data/processed"
    network = build(output / "features", ROOT / "data/raw/study/five_city_boundaries.geojson",
                    ROOT / "configs/study_assumptions.json", ROOT / TRAFFIC_TABLE, ROOT / "data/quality/feature_preprocessing.json")
    weather_path = output / "historical_weather.json"
    weather_bundle = build_weather(network.boundaries, weather_path, network.nodes)
    network.provenance["historical_weather"] = {key: value for key, value in weather_bundle.items()
                                               if key not in {"records", "records_by_grid", "node_grid_assignment"}}
    apply_critic(network)
    path = output / "study_network.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(network.model_dump_json(), encoding="utf-8")
    temporary.replace(path)
    policy = json.loads((ROOT / "configs/historical_study.json").read_text(encoding="utf-8"))
    manifest = {"mode": "historical_period_with_estimates", "period": policy,
                "hours_per_city": weather_bundle["hours_per_city"],
                "artifacts": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in RELEASE_ARTIFACTS}}
    (output / "study_release.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    registry = network.provenance["estimation_registry"]
    (output / "estimation_registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    (output / "study_network_summary.json").write_text(json.dumps({
        "nodes": len(network.nodes), "roads": len(network.roads), "cities": dict(Counter(node.city for node in network.nodes)),
        "signal_nodes": sum(node.traffic_signal for node in network.nodes), "mode": manifest["mode"],
        "weather_hours_per_city": manifest["hours_per_city"], "period": policy,
    }, indent=2), encoding="utf-8")
    print(f"Published estimation release: {len(network.nodes)} nodes, {len(network.roads)} roads; 8760 weather and traffic hours", flush=True)


if __name__ == "__main__":
    main()
