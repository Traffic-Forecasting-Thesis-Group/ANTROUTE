"""Assemble the routing network and release from the preprocessed feature tables.

Order: scripts.preprocess_road_features -> scripts.preprocess_traffic -> scripts.build_study_network
"""
import csv
import hashlib
import json
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from shapely import wkt
from shapely.geometry import mapping, shape

from app.study_policy import RELEASE_ARTIFACTS, file_sha256
from src.routing.condition_update import Conditions
from src.routing.critic import CriticAccumulator
from src.routing.dynamic_weight import segment_values
from src.routing.estimation import estimation_registry
from src.routing.historical_traffic import TABLE as TRAFFIC_TABLE, traffic_snapshot, weekday_profile
from src.routing.historical_weather import build_weather, historical_snapshot
from src.routing.network import CITIES, FEATURES, CostConfig, RoadNetwork
from scripts.preprocess_traffic import segment_inputs

ROOT = Path(__file__).resolve().parents[1]
# Every 15th day moves the weekday by one, so the sample covers each weekday and weekends.
CRITIC_DAY_STEP = 15


def read_csv(path):
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def time_of_day_profile(road_type, assumptions):
    """How busy this type of road typically is at each hour, 0-1: hourly profile x road-type factor, each scaled to its maximum."""
    profile = assumptions["hourly_demand_capacity_ratio"]
    factors = {name: settings["traffic_factor"] for name, settings in assumptions["road_types"].items()}
    return [round(profile[hour] / max(profile) * factors[road_type] / max(factors.values()), 6) for hour in range(24)]


def build(features_dir, boundary_path, assumption_path, quality_path=None):
    """Assemble the routing network from the preprocessed feature tables only."""
    boundaries = json.loads(boundary_path.read_text(encoding="utf-8"))
    polygons = {feature["properties"]["city"]: shape(feature["geometry"]) for feature in boundaries["features"]}
    if set(polygons) != set(CITIES):
        raise ValueError("Exactly the five study boundaries are required")
    assumptions = json.loads(assumption_path.read_text(encoding="utf-8"))
    segments = read_csv(features_dir / "road_segments.csv")
    print(f"Weekday traffic profiles for {len(segments)} segments", flush=True)
    profiles = weekday_profile(segment_inputs(features_dir / "road_segments.csv"), assumptions)
    floods = {row["road_id"]: row for row in read_csv(features_dir / "segment_flood.csv")}
    roads, labels, sources = [], {}, Counter()
    for row in segments:
        profile = profiles[row["road_id"]]
        free_flow = float(row["free_flow_time_s"])
        flood = floods[row["road_id"]]
        provenance = {
            "distance": row["geometry_source"],
            "traffic_congestion": "traffic_hourly_table_estimated_bpr",
            "travel_time": f"{row['speed_source']}_free_flow_x_traffic_hourly_table_factor",
            "road_type": "osm_highway_tag",
            "time_of_day": "assumed_hourly_profile_x_road_type_factor",
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
            "time_of_day": time_of_day_profile(row["road_type"], assumptions),
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
        "cost_config": {"road_type_costs": {name: settings["cost"] for name, settings in assumptions["road_types"].items()},
                        "weather_flood_exposure_share": assumptions.get("weather_flood_exposure_share", 0.5)},
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
                "Traffic congestion and travel time come from an estimated hourly table with one series per road segment (assumed demand index and BPR), not traffic measurements or vehicle counts.",
                "Capacity and free-flow speed are estimates, including on roads with OSM lane/speed tags.",
                "Unrecorded signals are treated as absent; OSM signal coverage is incomplete.",
                "Flood exposure is a mapped hazard class, not measured water depth or current passability; bridges may be elevated.",
                "Weather effects are scenario assumptions applied to historical provider records.",
                "Time of day uses the same assumed hourly profile and road-type factors as the traffic estimate, so it is correlated with congestion. Feature weights come from CRITIC over estimated feature values; they are not calibrated against observed route choices.",
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


def critic_within_hours(hourly_rows):
    """CRITIC within each hour, averaged over the hours in which each feature varies.

    Routing compares roads within one date/hour snapshot, so CRITIC compares the same
    alternatives: within each hour, the alternatives are the road segments. A feature that is
    the same on every road in an hour (weather in a dry hour) cannot rank roads then, so its
    weight is the mean over the hours in which it does vary ("active" hours); the means are
    rescaled to sum to one. Hours where every feature is constant are skipped.
    hourly_rows: iterable of [segments x features] rows, one block per sampled hour.
    """
    totals, active = dict.fromkeys(FEATURES, 0.0), dict.fromkeys(FEATURES, 0)
    ranges, used = {name: [1.0, 0.0] for name in FEATURES}, 0
    for rows in hourly_rows:
        accumulator = CriticAccumulator(FEATURES)
        accumulator.add(rows)
        try:
            result = accumulator.result()
        except ValueError:
            continue
        used += 1
        for name, value in result["weights"].items():
            if result["contrast_std"][name] > 1e-12:
                totals[name] += value
                active[name] += 1
                ranges[name] = [min(ranges[name][0], value), max(ranges[name][1], value)]
    if not used:
        raise ValueError("CRITIC is undefined: every feature is constant in every sampled hour")
    means = {name: totals[name] / active[name] if active[name] else 0.0 for name in FEATURES}
    scale = sum(means.values())
    return {"weights": {name: means[name] / scale for name in FEATURES}, "hours_used": used,
            "active_hours": active, "hourly_weight_range": {name: ranges[name] if active[name] else [0.0, 0.0] for name in FEATURES}}


def apply_critic(network: RoadNetwork) -> None:
    """Replace the provisional equal weights with CRITIC weights computed within each sampled hour."""
    nodes = {node.id: node for node in network.nodes}
    hours = list(critic_sample_hours())

    def hourly_rows():
        for index, (day, hour) in enumerate(hours, 1):
            conditions = Conditions(departure_date=day, time_of_day=hour, weather="historical")
            weather, traffic = historical_snapshot(day, hour, network)[0], traffic_snapshot(day, hour)
            yield [[features[name] for name in FEATURES] for _, features, _ in
                   (segment_values(network, road, conditions, nodes, weather, traffic) for road in network.roads)]
            if index % 100 == 0 or index == len(hours):
                print(f"CRITIC sample: {index}/{len(hours)} hours", flush=True)
    result = critic_within_hours(hourly_rows())
    network.cost_config = CostConfig(**{**network.cost_config.model_dump(), "weights": result["weights"]})
    network.provenance["weighting"] = {
        "method": "CRITIC (Diakoulaki, Mavrotas and Papayannakis, 1995), computed within each hour",
        "reference": "https://doi.org/10.1016/0305-0548(94)00059-H",
        "sample": f"Within each of {len(hours)} sampled hours (every hour of every {CRITIC_DAY_STEP}th day from 2025-06-01, "
                  f"historical provider weather), the alternatives are all {len(network.roads)} segments; "
                  "each feature's weight is its mean over the hours in which it varies between roads, rescaled to sum to one",
        **result,
    }


def main():
    output = ROOT / "data/processed"
    network = build(output / "features", ROOT / "data/raw/study/five_city_boundaries.geojson",
                    ROOT / "configs/study_assumptions.json", ROOT / "data/quality/feature_preprocessing.json")
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
                "artifacts": {name: file_sha256(ROOT / name) for name in RELEASE_ARTIFACTS}}
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
