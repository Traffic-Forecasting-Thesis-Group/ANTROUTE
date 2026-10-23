"""Build the live release: the routing network scored from saved live captures, calibrated on the live development week.

Order: scripts.collect_live (development week saved) -> scripts.calibrate_live -> scripts.tune_search --live
       -> scripts.evaluate_sop freeze --live -> scripts.evaluate_sop run --live

Everything fitted here uses development captures only (configs/live_study.json):
  hours      the first saved capture of every development hour; an hour without a usable capture (flow,
             incidents and weather within 90 minutes) is excluded and its reason recorded, never filled in
  weights    hourly CRITIC over all directed segments of each hour; fixed weight = mean over the valid hours
  maxima     the largest value of each criterion over the valid development hours (the routing scale)
  Cheng      the largest length, live travel time and inverse-BPR flow over the same hours (fixed references)
  peaks      per day category (weekday/weekend), the two most prominent peaks of the mean hourly live city load and
             the steepest rise into each; no clock hours are assumed (the dynamic experiment departs one collection
             interval before each onset)
Road-class severity is the static OSM speed mapping of the historical release; it does not change live.

--provisional calibrates on every capture saved so far, whatever its period, to check the pipeline before the
development week is complete. A provisional release is labelled as such and the evaluation refuses it.

Writes data/processed/live_calibration.json, data/processed/live_network.json and data/processed/live_release.json.
Run from traffic_system: python -m scripts.calibrate_live [--provisional]
"""
import argparse
import hashlib
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.study_policy import file_sha256  # noqa: E402
from scripts.build_study_network import build, git_commit  # noqa: E402
from src.live.snapshot import LiveDataUnavailable, captures, live_snapshot  # noqa: E402
from src.routing.baseline_cheng import AHP_WEIGHTS, estimated_flow_vph  # noqa: E402
from src.routing.calibration import hourly_critic  # noqa: E402
from src.routing.condition_update import live_conditions  # noqa: E402
from src.routing.criteria import CONTRACT_VERSION, CRITERIA, CriterionInputError, UNITS, criteria  # noqa: E402
from src.routing.dynamic_weight import segment_inputs  # noqa: E402
from src.routing.estimation import estimation_registry  # noqa: E402

POLICY = ROOT / "configs/live_study.json"
OUTPUT = ROOT / "data/processed"
CALIBRATION = OUTPUT / "live_calibration.json"
NETWORK = OUTPUT / "live_network.json"
RELEASE = OUTPUT / "live_release.json"
PERIODS = ("development", "tuning", "excluded", "test")
LIVE_SOURCES = {
    "congestion": "here_flow_live_speed_over_free_flow_or_group_median",
    "travel_time": "here_flow_live_speed_or_group_median",
    "time_profile": "live_city_load_length_weighted_congestion",
    "weather": "weatherstack_current_city_observation_with_assumed_severity",
    "flood_exposure": "noah_hazard_or_here_live_flood_report",
    "capacity_burden": "base_capacity_x_live_incident_flood_weather_factors",
}
PEAKS = ("first", "second")  # The two most prominent daily load peaks, in clock order.


def load_split(policy: dict) -> dict[str, tuple[date, date]]:
    declared = policy["calibration_split"]
    split = {name: (date.fromisoformat(declared[name]["start"]), date.fromisoformat(declared[name]["end"])) for name in PERIODS}
    previous = None
    for name in PERIODS:
        start, end = split[name]
        if start > end or (previous is not None and start <= previous):
            raise ValueError("Live periods must be chronological and must not overlap")
        previous = end
    return split


def period_of(moment: datetime, split) -> str | None:
    return next((name for name, (start, end) in split.items() if start <= moment.date() <= end), None)


def hourly_captures(moments) -> list[datetime]:
    """The first capture of each clock hour."""
    first = {}
    for moment in sorted(moments):
        first.setdefault((moment.date(), moment.hour), moment)
    return list(first.values())


def day_category(day: date) -> str:
    return "weekend" if day.weekday() >= 5 else "weekday"


def peak_onsets(loads: dict[tuple[str, int], list[float]], lead_minutes: int) -> dict:
    """Peaks and onsets of the mean live load per day category, found in the data with no assumed clock hours.

    loads: {(category, hour): [mean of the city loads of each development hour]}. The measured hours form a circular
    daily curve (midnight joins the previous evening). Peaks are its local maxima; the two with the largest
    topographic prominence are kept, in clock order. A peak's onset is the measured hour with the largest rise from
    the hour before, between the lowest point since the other peak and the peak. The departure is lead_minutes (one
    collection interval) before the onset hour, so a trip crosses into the rise. A curve with fewer than two peaks
    leaves the missing ones out; their trips are then recorded as unavailable."""
    from scipy.signal import find_peaks
    result = {}
    for category in ("weekday", "weekend"):
        means = {hour: sum(v) / len(v) for (c, hour), v in loads.items() if c == category and v}
        hours = sorted(means)
        count, onsets = len(hours), {}
        if count >= 3:
            curve = [means[h] for h in hours] * 3  # Tiled so peaks and rises wrap around midnight.
            found, properties = find_peaks(curve, prominence=0)
            middle = [(p, i) for i, p in zip(found, properties["prominences"]) if count <= i < 2 * count]
            kept = sorted(i for _, i in sorted(middle, reverse=True)[:len(PEAKS)])
            for number, (name, top) in enumerate(zip(PEAKS, kept)):
                other = kept[number - 1] if len(kept) > 1 else top - count
                if other >= top:
                    other -= count
                trough = min(range(other + 1, top + 1), key=lambda i: curve[i])
                steps = [(curve[i] - curve[i - 1], i) for i in range(trough + 1, top + 1)
                         if (hours[i % count] - hours[(i - 1) % count]) % 24 == 1]  # Consecutive measured hours only.
                if steps:
                    rise, index = max(steps)
                    hour = hours[index % count]
                    start = (hour * 60 - lead_minutes) % (24 * 60)
                    onsets[name] = {"peak_hour": hours[top % count], "prominence": float(properties["prominences"][list(found).index(top)]),
                                    "onset_hour": hour, "rise": rise, "departure": f"{start // 60:02d}:{start % 60:02d}"}
        result[category] = {"hourly_mean_load": {str(h): means[h] for h in hours},
                            "samples": {str(h): len(loads[(category, h)]) for h in hours}, "onsets": onsets}
    return {"method": "Per day category, the two most prominent local maxima of the circular mean hourly live city load "
                      "(length-weighted mean live congestion, averaged over the five cities), in clock order; each onset "
                      "is the steepest hourly rise between the preceding low point and the peak; departure one "
                      f"collection interval ({lead_minutes} min) before the onset hour. No clock hours are assumed.",
            **result}


def live_network():
    """The study network with live provenance; dynamic inputs are read from the captures, not stored on roads."""
    network = build(OUTPUT / "features", ROOT / "data/raw/study/five_city_boundaries.geojson",
                    ROOT / "configs/study_assumptions.json", None, ROOT / "data/quality/feature_preprocessing.json")
    for road in network.roads:
        road.traffic_location = None
        road.feature_sources = {**road.feature_sources, **LIVE_SOURCES}
    network.provenance.update({
        "mode": "live", "period": json.loads(POLICY.read_text(encoding="utf-8"))["collection"],
        "traffic_source": "HERE Traffic API v7 flow and incidents, saved every 15 minutes (data/live/raw)",
        "limitations": [
            "HERE reports live speeds mainly on main roads; a road without a live HERE line takes the median live congestion of matched roads of the same city and road type (flagged live_group_estimate).",
            "Time of day is the live load of the road's city (length-weighted mean live congestion), not a fitted profile.",
            "Flood exposure is the NOAH hazard exposure unless HERE reports flooding on the road now; then the whole road counts at the highest severity.",
            "Live capacity multiplies the base capacity (1,900 veh/h/lane x lanes x g/C) by assumed incident, flood and rain factors (configs/live_assumptions.json).",
            "Weather is one current observation per city with assumed severities; it is not road-level.",
            "Weights come from hourly CRITIC over the live development week; they are not calibrated against observed route choices.",
            "The Cheng baseline flow is the inverse BPR of live congestion times live capacity, not a vehicle count.",
            "Road geometry, signals and classes come from the September 2026 OpenStreetMap inventory.",
        ],
    })
    return network


def hour_rows(network, nodes, moment, assumptions):
    try:
        conditions = live_conditions(moment)
        live_snapshot(moment)
        rows, cheng = [], []
        for road in network.roads:
            inputs, physical = segment_inputs(network, road, conditions, nodes)
            values = criteria(inputs)
            rows.append([values[name] for name in CRITERIA])
            cheng.append((physical["distance_m"], physical["travel_time_s"],
                          estimated_flow_vph(physical["congestion"], physical["capacity_vph"],
                                             assumptions["bpr_alpha"], assumptions["bpr_beta"])))
    except LiveDataUnavailable as error:
        return None, None, f"input_unavailable: {error}"
    except CriterionInputError as error:
        return None, None, f"invalid_input: {error}"
    return rows, cheng, None


def calibrate(network, sample, assumptions) -> dict:
    nodes = {node.id: node for node in network.nodes}
    maxima = dict.fromkeys(CRITERIA, 0.0)
    references = [0.0, 0.0, 0.0]
    coverage = []
    loads = {}

    def hours():
        for number, moment in enumerate(sample, 1):
            rows, cheng, reason = hour_rows(network, nodes, moment, assumptions)
            if rows is not None:
                for j, name in enumerate(CRITERIA):
                    maxima[name] = max(maxima[name], max(row[j] for row in rows))
                for k in range(3):
                    references[k] = max(references[k], max(item[k] for item in cheng))
                summary = live_snapshot(moment).summary
                coverage.append(summary["live_matched"] / summary["segments"])
                if summary["city_load"]:
                    loads.setdefault((day_category(moment.date()), moment.hour), []).append(
                        sum(summary["city_load"].values()) / len(summary["city_load"]))
            print(f"Live CRITIC sample: {number}/{len(sample)} hours ({moment:%Y-%m-%d %H:%M})"
                  f"{'' if rows is not None else ' excluded: ' + reason}", flush=True)
            yield moment.isoformat(), rows, reason
    result = hourly_critic(hours())
    return {
        **result, "development_max": maxima, "units": UNITS,
        "sample": f"First saved capture of each development hour; alternatives are all {len(network.roads)} directed segments",
        "live_coverage_share": {"min": min(coverage), "mean": sum(coverage) / len(coverage), "max": max(coverage)},
        "peak_onsets": peak_onsets(loads, json.loads(POLICY.read_text(encoding="utf-8"))["collection"]["traffic_interval_minutes"]),
        "cheng_references": {"distance_m": references[0], "travel_time_s": references[1], "flow_vph": references[2],
                             "flow_provenance": "simulation_adaptation_inverse_bpr_of_live_congestion",
                             "ahp_weights": list(AHP_WEIGHTS)},
        "distance_reference_m": {"antroute": maxima["distance"] * 1000,
                                 "cheng": max(road.distance_m for road in network.roads)},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provisional", action="store_true", help="calibrate on every saved capture (pipeline check only)")
    args = parser.parse_args(argv)
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    split = load_split(policy)
    saved = captures()
    if args.provisional:
        moments = hourly_captures(saved)
    else:
        moments = hourly_captures(t for t in saved if period_of(t, split) == "development")
        last_day = split["development"][1]
        if not saved or max(saved).date() <= last_day:
            sys.exit(f"The live development week ends {last_day}; {len(moments)} development hours are saved so far. "
                     "Run again after it ends (or pass --provisional to check the pipeline).")
    if not moments:
        sys.exit("No saved live captures; run python -m scripts.collect_live first.")
    network = live_network()
    assumptions = network.provenance["assumptions"]
    calibration = calibrate(network, moments, assumptions)
    calibration.update({
        "status": "provisional_pipeline_check" if args.provisional else "development_calibration",
        "contract_version": CONTRACT_VERSION, "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit(), "split": {name: [s.isoformat(), e.isoformat()] for name, (s, e) in split.items()},
        "road_class_severity": network.provenance["road_class_severity"],
        "flood_class_severity": assumptions["flood_class_severity"],
        "live_assumptions": json.loads((ROOT / "configs/live_assumptions.json").read_text(encoding="utf-8")),
        "input_hashes": {"live_assumptions": file_sha256(ROOT / "configs/live_assumptions.json"),
                         "live_study": file_sha256(POLICY), **network.provenance["feature_tables"]},
    })
    CALIBRATION.write_text(json.dumps(calibration, indent=1), encoding="utf-8")
    calibration_id = ("provisional-" if args.provisional else "") + file_sha256(CALIBRATION)
    network.cost_config = network.cost_config.model_validate({
        **network.cost_config.model_dump(), "weights": calibration["weights"],
        "development_max": calibration["development_max"], "calibration_id": calibration_id})
    network.provenance["weighting"] = {
        "method": "Hourly CRITIC (Diakoulaki, Mavrotas and Papayannakis, 1995) over live development hours; fixed weight = mean over the valid hours, zeros kept",
        "calibration_id": calibration_id, "status": calibration["status"],
        "valid_hours": len(calibration["valid_hours"]), "excluded_hours": len(calibration["excluded_hours"]),
        "weights": calibration["weights"], "development_max": calibration["development_max"]}
    network.provenance["cheng_references"] = calibration["cheng_references"]
    network.provenance["distance_reference_m"] = calibration["distance_reference_m"]
    # Live criteria come from HERE and Weatherstack captures, so the registry describes those, not the historical scenario.
    network.provenance["estimation_registry"] = estimation_registry(assumptions, live=True)
    temporary = NETWORK.with_suffix(".tmp")
    temporary.write_text(network.model_dump_json(), encoding="utf-8")
    temporary.replace(NETWORK)
    RELEASE.write_text(json.dumps({
        "mode": "live", "status": calibration["status"], "contract_version": CONTRACT_VERSION,
        "calibration_id": calibration_id, "code_commit": calibration["code_commit"],
        "split": calibration["split"], "valid_hours": len(calibration["valid_hours"]),
        "artifacts": {path.name: file_sha256(path) for path in (CALIBRATION, NETWORK)},
        "network_sha256": hashlib.sha256(NETWORK.read_bytes()).hexdigest(),
    }, indent=2), encoding="utf-8")
    weights = ", ".join(f"{name} {calibration['weights'][name]:.3f}" for name in CRITERIA)
    print(f"Live release ({calibration['status']}, calibration {calibration_id[:24]}): "
          f"{len(calibration['valid_hours'])} valid hours, {len(calibration['excluded_hours'])} excluded\nWeights: {weights}")


if __name__ == "__main__":
    main()
