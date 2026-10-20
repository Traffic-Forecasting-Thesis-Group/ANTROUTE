"""Assemble the routing network (contract version 2) and its development-only calibration.

Order: scripts.preprocess_road_features -> scripts.preprocess_traffic -> scripts.build_study_network

Every fitted value (fallback traffic profiles, time profiles, hourly CRITIC weights, development maxima and
Cheng references) uses the declared development period only. Until configs/historical_study.json declares the
development/tuning/test split, the build stops with CalibrationBlocked and publishes nothing.
"""
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from shapely import wkt
from shapely.geometry import mapping, shape

from app.study_policy import RELEASE_ARTIFACTS, file_sha256
from src.routing.baseline_cheng import AHP_WEIGHTS, estimated_flow_vph
from src.routing.calibration import CalibrationBlocked, ProfileFit, class_severity, days, hourly_critic, load_split
from src.routing.condition_update import Conditions, day_category
from src.routing.criteria import CONTRACT_VERSION, CRITERIA, CriterionInputError, UNITS, criteria
from src.routing.dynamic_weight import segment_inputs
from src.routing.estimation import estimation_registry
from src.routing.historical_traffic import TABLE as TRAFFIC_TABLE, traffic_snapshot, weekday_profile
from src.routing.historical_weather import build_weather, historical_snapshot
from src.routing.network import CITIES, RoadNetwork
from src.routing.weather import MissingWeatherObservation
from scripts.preprocess_traffic import segment_inputs as traffic_inputs

ROOT = Path(__file__).resolve().parents[1]
# Every 15th development day moves the weekday by one, so the CRITIC sample covers each weekday and weekends.
CRITIC_DAY_STEP = 15
PROFILE_DAY_STEP = 1  # Time profiles use every development day.
PROVISIONAL = "provisional-uncalibrated"


def read_csv(path):
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def flood_severity(assumptions):
    return {hazard: float(value) for hazard, value in assumptions["flood_class_severity"]["values"].items()}


def provisional_cost_config(road_class_severity, flood_class_severity):
    """Equal weights and unit maxima so criteria can be computed during calibration. Never published or routed on."""
    return {"weights": dict.fromkeys(CRITERIA, 0.1), "development_max": dict.fromkeys(CRITERIA, 1.0),
            "road_class_severity": road_class_severity, "flood_class_severity": flood_class_severity,
            "calibration_id": PROVISIONAL}


def build(features_dir, boundary_path, assumption_path, development_days, quality_path=None):
    """Network from the preprocessed feature tables, with a provisional cost config and empty time profiles."""
    boundaries = json.loads(boundary_path.read_text(encoding="utf-8"))
    polygons = {feature["properties"]["city"]: shape(feature["geometry"]) for feature in boundaries["features"]}
    if set(polygons) != set(CITIES):
        raise ValueError("Exactly the five study boundaries are required")
    assumptions = json.loads(assumption_path.read_text(encoding="utf-8"))
    segments = read_csv(features_dir / "road_segments.csv")
    print(f"Development weekday fallback profiles for {len(segments)} segments", flush=True)
    profiles = weekday_profile(traffic_inputs(features_dir / "road_segments.csv"), assumptions, development_days)
    floods = {row["road_id"]: row for row in read_csv(features_dir / "segment_flood.csv")}
    speeds = defaultdict(list)
    for row in segments:
        speeds[row["road_type"]].append(float(row["speed_kph"]))
    roads, labels, sources = [], {}, Counter()
    for row in segments:
        profile = profiles[row["road_id"]]
        free_flow = float(row["free_flow_time_s"])
        flood = floods[row["road_id"]]
        provenance = {
            "distance": row["geometry_source"],
            "congestion": "assumed_congestion_scenario_hourly_table",
            "travel_time": f"{row['speed_source']}_free_flow_x_scenario_bpr_factor",
            "road_type": "osm_highway_tag_speed_median_mapping",
            "time_profile": "development_fit_group_day_category_hour",
            "weather": "historical_provider_grid_with_assumed_effects",
            "flood_exposure": flood["flood_source"],
            "capacity_burden": row["capacity_source"],
        }
        roads.append({
            "id": row["road_id"], "source": row["source_node"], "target": row["target_node"],
            "distance_m": float(row["distance_m"]), "road_type": row["road_type"],
            "road_capacity_vph": float(row["road_capacity_vph"]),
            "flood_km_by_class": {hazard: float(flood[f"flood_km_class{hazard}"]) for hazard in ("1", "2", "3")
                                  if float(flood[f"flood_km_class{hazard}"]) > 0},
            "flood_coverage": flood["flood_coverage"],
            "traffic_congestion": [round(congestion, 6) for congestion, _ in profile],
            "travel_time_s": [round(free_flow * factor, 6) for _, factor in profile],
            "time_profile": {"weekday": [0.0] * 24, "weekend": [0.0] * 24},  # Fitted by fit_time_profiles.
            # The date/hour-specific weather effect is attached by the snapshot loader.
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
        "feature_sources": {"intersections": "derived_osm_street_count", "signals": row["signal_source"]},
    } for row in read_csv(features_dir / "road_nodes.csv")]
    static_report = {}
    if quality_path is not None and quality_path.exists():
        static_report = json.loads(quality_path.read_text(encoding="utf-8")).get("static", {})
    severity = class_severity(speeds)
    print(f"Validating network: {len(nodes)} nodes, {len(roads)} directed roads", flush=True)
    return RoadNetwork.model_validate({
        "boundaries": {city: mapping(polygon) for city, polygon in polygons.items()},
        "nodes": nodes, "roads": roads,
        "cost_config": provisional_cost_config(severity, flood_severity(assumptions)),
        "provenance": {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "contract_version": CONTRACT_VERSION,
            "mode": "historical_period_with_estimates",
            "period": {"start": "2025-06-01", "end": "2026-05-31"},
            "boundaries": [feature["properties"] for feature in boundaries["features"]],
            "graph_source": "OpenStreetMap via OSMnx, preprocessed into data/processed/features",
            "flood_source": "Project NOAH five-year flood hazard polygons; km per class, overlaps to the highest class",
            "traffic_source": f"{TRAFFIC_TABLE} (assumed congestion scenario, not observed)",
            "feature_tables": {name: hashlib.sha256((features_dir / name).read_bytes()).hexdigest()
                               for name in ("road_nodes.csv", "road_segments.csv", "segment_flood.csv")},
            "assumptions": assumptions, "assumptions_sha256": hashlib.sha256(assumption_path.read_bytes()).hexdigest(),
            "road_class_severity": {"values": severity, **{k: v for k, v in assumptions["road_class_severity"].items()}},
            "source_counts": dict(sources), "excluded_roads": static_report.get("excluded_source_edges", {}),
            "estimation_registry": estimation_registry(assumptions),
            "limitations": [
                "Congestion and travel time come from an assumed congestion scenario (seeded demand-to-capacity index and BPR) per road segment, not traffic measurements, vehicle counts or independent demand estimates.",
                "Capacity uses an assumed 1,900 veh/h/lane base saturation flow and lane fallbacks; capacity burden L/K is a provisional study mapping.",
                "Unrecorded signals are treated as absent; OSM signal coverage is incomplete.",
                "Flood exposure is mapped NOAH hazard susceptibility with an assumed class severity, not measured water depth, current inundation or passability.",
                "Weather severities and speed multipliers are assumptions applied to historical provider records; they are not calibrated.",
                "The time profile is fitted on the same assumed congestion scenario as congestion, so the two criteria are correlated.",
                "Weights come from hourly CRITIC over development data; they are not calibrated against observed route choices.",
                "September 2026 geometry, signals and classifications are held constant as an estimated surrogate for June 2025–May 2026.",
                "Historical weather grid assignment and speed effects are estimates, not road-level weather measurements.",
                "The source graph does not encode turn restrictions in this node-edge routing model.",
            ],
        },
    })


def fit_time_profiles(network, development):
    """Fit h by road group, day category and hour from every development hour of the traffic table."""
    fit = ProfileFit(groups={}, road_types={})
    nodes = {node.id: node for node in network.nodes}
    linked = [road for road in network.roads if road.traffic_location]
    for road in linked:
        fit.groups[road.id] = f"{nodes[road.source].city}|{road.road_type}"
        fit.road_types[road.id] = road.road_type
    for number, day in enumerate(days(development, PROFILE_DAY_STEP), 1):
        category = day_category(day)
        for hour in range(24):
            traffic = traffic_snapshot(day, hour)
            fit.add(category, hour, {road.id: traffic[road.traffic_location][0] for road in linked})
        if number % 30 == 0:
            print(f"Time profiles: {number} development days", flush=True)
    profiles, fallbacks = fit.profiles()
    for road in linked:
        road.time_profile = profiles[road.id]
    return {"groups": len(set(fit.groups.values())), "fallback_levels_used": fallbacks, "day_step": PROFILE_DAY_STEP,
            "day_categories": "weekday Mon-Fri, weekend Sat-Sun (holidays not modelled)"}


def critic_sample(development):
    """(day, hour) of every hour of every CRITIC_DAY_STEP-th development day."""
    for day in days(development, CRITIC_DAY_STEP):
        for hour in range(24):
            yield day, hour


def hour_inputs(network, nodes, day, hour, assumptions):
    """Criterion rows for every road in one hour plus Cheng's physical values, or (None, reason).

    Eligibility policy: an hour is eligible only when the weather and traffic snapshots exist and every road's
    criteria are valid. Otherwise the whole hour is excluded and the reason recorded; nothing is zero-filled.
    """
    conditions = Conditions(departure_date=day, time_of_day=hour, weather="historical")
    try:
        weather = historical_snapshot(day, hour, network)[0]
        traffic = traffic_snapshot(day, hour)
    except (MissingWeatherObservation, KeyError, ValueError) as error:
        return None, None, f"input_unavailable: {error}"
    rows, cheng = [], []
    try:
        for road in network.roads:
            inputs, physical = segment_inputs(network, road, conditions, nodes, weather, traffic)
            values = criteria(inputs)
            rows.append([values[name] for name in CRITERIA])
            cheng.append((physical["distance_m"], physical["travel_time_s"],
                          estimated_flow_vph(physical["congestion"], physical["capacity_vph"],
                                             assumptions["bpr_alpha"], assumptions["bpr_beta"])))
    except CriterionInputError as error:
        return None, None, f"invalid_input: {error}"
    return rows, cheng, None


def calibrate(network, split, assumptions):
    """Hourly CRITIC weights, development maxima and Cheng references from development hours only."""
    nodes = {node.id: node for node in network.nodes}
    maxima = dict.fromkeys(CRITERIA, 0.0)
    references = [0.0, 0.0, 0.0]
    sample = list(critic_sample(split["development"]))

    def hours():
        for number, (day, hour) in enumerate(sample, 1):
            rows, cheng, reason = hour_inputs(network, nodes, day, hour, assumptions)
            if rows is not None:
                for j, name in enumerate(CRITERIA):
                    maxima[name] = max(maxima[name], max(row[j] for row in rows))
                for k in range(3):
                    references[k] = max(references[k], max(item[k] for item in cheng))
            if number % 100 == 0 or number == len(sample):
                print(f"CRITIC sample: {number}/{len(sample)} hours", flush=True)
            yield f"{day.isoformat()}T{hour:02d}", rows, reason
    result = hourly_critic(hours())
    longest = max(road.distance_m for road in network.roads)
    return {
        **result,
        "development_max": maxima, "units": UNITS,
        "sample": f"Every hour of every {CRITIC_DAY_STEP}th development day from {split['development'][0].isoformat()}; "
                  f"alternatives are all {len(network.roads)} directed segments in each hour",
        "cheng_references": {"distance_m": references[0], "travel_time_s": references[1], "flow_vph": references[2],
                             "flow_provenance": "simulation_adaptation_inverse_bpr_of_assumed_congestion",
                             "ahp_weights": list(AHP_WEIGHTS)},
        "distance_reference_m": {"antroute": maxima["distance"] * 1000, "cheng": longest},
    }


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main():
    policy = json.loads((ROOT / "configs/historical_study.json").read_text(encoding="utf-8"))
    try:
        split = load_split(policy)
    except CalibrationBlocked as blocked:
        print(f"BLOCKED: {blocked}", file=sys.stderr, flush=True)
        sys.exit(2)
    output = ROOT / "data/processed"
    assumptions_path = ROOT / "configs/study_assumptions.json"
    development_days = list(days(split["development"]))
    network = build(output / "features", ROOT / "data/raw/study/five_city_boundaries.geojson", assumptions_path,
                    development_days, ROOT / "data/quality/feature_preprocessing.json")
    weather_bundle = build_weather(network.boundaries, output / "historical_weather.json", network.nodes)
    network.provenance["historical_weather"] = {key: value for key, value in weather_bundle.items()
                                               if key not in {"records", "records_by_grid", "node_grid_assignment"}}
    profile_summary = fit_time_profiles(network, split["development"])
    assumptions = network.provenance["assumptions"]
    calibration = calibrate(network, split, assumptions)
    calibration.update({
        "contract_version": CONTRACT_VERSION, "generated_at": datetime.now(timezone.utc).isoformat(), "code_commit": git_commit(),
        "split": {name: [start.isoformat(), end.isoformat()] for name, (start, end) in split.items()},
        "road_class_severity": network.provenance["road_class_severity"],
        "flood_class_severity": assumptions["flood_class_severity"], "time_profile": profile_summary,
        "input_hashes": {"assumptions": network.provenance["assumptions_sha256"], **network.provenance["feature_tables"],
                         "traffic_index": file_sha256(ROOT / "data/processed/temporal/traffic_hourly.index.json"),
                         "historical_weather_source": weather_bundle["source_sha256"]},
    })
    calibration_path = output / "calibration.json"
    calibration_path.write_text(json.dumps(calibration, indent=1), encoding="utf-8")
    calibration_id = file_sha256(calibration_path)
    network.cost_config = network.cost_config.model_validate({
        **network.cost_config.model_dump(), "weights": calibration["weights"],
        "development_max": calibration["development_max"], "calibration_id": calibration_id})
    network.provenance["weighting"] = {
        "method": "Hourly CRITIC (Diakoulaki, Mavrotas and Papayannakis, 1995) over development hours; fixed weight = mean over the common valid-hour set, zeros kept",
        "reference": "https://doi.org/10.1016/0305-0548(94)00059-H", "calibration_id": calibration_id,
        "valid_hours": len(calibration["valid_hours"]), "excluded_hours": len(calibration["excluded_hours"]),
        "weights": calibration["weights"], "development_max": calibration["development_max"]}
    network.provenance["cheng_references"] = calibration["cheng_references"]
    network.provenance["distance_reference_m"] = calibration["distance_reference_m"]
    path = output / "study_network.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(network.model_dump_json(), encoding="utf-8")
    temporary.replace(path)
    manifest = {"mode": "historical_period_with_estimates", "contract_version": CONTRACT_VERSION, "period": policy,
                "code_commit": calibration["code_commit"], "calibration_id": calibration_id,
                "hours_per_city": weather_bundle["hours_per_city"],
                "artifacts": {name: file_sha256(ROOT / name) for name in RELEASE_ARTIFACTS}}
    (output / "study_release.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    registry = network.provenance["estimation_registry"]
    (output / "estimation_registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    (output / "study_network_summary.json").write_text(json.dumps({
        "nodes": len(network.nodes), "roads": len(network.roads), "cities": dict(Counter(node.city for node in network.nodes)),
        "signal_nodes": sum(node.traffic_signal for node in network.nodes), "mode": manifest["mode"],
        "contract_version": CONTRACT_VERSION, "calibration_id": calibration_id,
        "weather_hours_per_city": manifest["hours_per_city"], "period": policy,
    }, indent=2), encoding="utf-8")
    print(f"Published release (contract v{CONTRACT_VERSION}, calibration {calibration_id[:12]}): {len(network.nodes)} nodes, "
          f"{len(network.roads)} roads; {len(calibration['valid_hours'])} valid CRITIC hours, "
          f"{len(calibration['excluded_hours'])} excluded", flush=True)


if __name__ == "__main__":
    main()
