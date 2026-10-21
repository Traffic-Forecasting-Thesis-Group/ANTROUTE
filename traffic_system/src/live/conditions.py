"""Turn one live snapshot (HERE flow and incidents, current weather) into per-segment live inputs.

For every directed segment:
  congestion       1 - live speed / free-flow speed (matched), else the group median (flagged group_estimate)
  travel_time_s    length / live speed (matched), else free-flow time / (1 - group congestion)
  time_of_day      live load of the segment's city: length-weighted mean congestion of its matched roads now
  weather          category and severity of the city's current observation
  flood_portions   NOAH hazard portions (km, severity); a live flood report replaces them with the whole
                   segment at the report severity
  capacity_vph     base capacity x incident factor x flood factor x weather factor (live effective capacity)
  closed           a HERE road closure or a closed flow stretch
Assumed values come from configs/live_assumptions.json and are reported with every snapshot.
"""
from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from .here import flow_items, incident_items
from .matching import Segment, match
from ..routing.weather import weather_condition

ROOT = Path(__file__).resolve().parents[2]
ASSUMPTIONS = ROOT / "configs/live_assumptions.json"


@dataclass(frozen=True)
class SegmentRecord:
    road_id: str
    city: str
    road_type: str
    length_m: float
    lanes: float
    capacity_vph: float
    free_flow_time_s: float
    flood_km: tuple[float, float, float]  # NOAH class 1, 2, 3


def load_records(segments_csv, flood_csv) -> dict[str, SegmentRecord]:
    flood = {}
    with open(flood_csv, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            flood[row["road_id"]] = tuple(float(row[f"flood_km_class{k}"] or 0) for k in (1, 2, 3))
    records = {}
    with open(segments_csv, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            records[row["road_id"]] = SegmentRecord(
                row["road_id"], row["city"], row["road_type"], float(row["distance_m"]), float(row["lanes"] or 1),
                float(row["road_capacity_vph"]), float(row["free_flow_time_s"]), flood.get(row["road_id"], (0.0, 0.0, 0.0)))
    return records


def load_assumptions() -> dict:
    return json.loads(ASSUMPTIONS.read_text(encoding="utf-8"))


def usable_flow(items: list[dict], assumptions: dict) -> list[dict]:
    return [item for item in items if item["free_flow_mps"] >= assumptions["min_free_flow_mps"]]


def _is_flood(item: dict, rule: dict) -> bool:
    text = f"{item.get('type') or ''} {item.get('description') or ''}".lower()
    return item.get("type") in rule["types"] or any(word in text for word in rule["words"])


def build(records: dict[str, SegmentRecord], segments: list[Segment], flow: dict, incidents: dict, weather: dict,
          flood_class_severity: dict, weather_scenarios: dict, assumptions: dict | None = None) -> dict:
    a = assumptions or load_assumptions()
    all_flow = flow_items(flow)
    flow_lines = usable_flow(all_flow, a)
    matched = match(segments, flow_lines)
    events = incident_items(incidents)
    closure_lines = [e for e in events if e["road_closed"]]
    closure_lines += [{"lines": f["lines"]} for f in all_flow if str(f["traversability"]).lower() == "closed"]
    flood_lines = [e for e in events if _is_flood(e, a["flood_report"])]
    blocking = [e for e in events if not e["road_closed"] and e["type"] in a["capacity_factors"]["lane_blocking_incident_types"]]
    closed = set(match(segments, closure_lines, min_share=a["closure_min_share"])) if closure_lines else set()
    flooded = set(match(segments, flood_lines, min_share=a["incident_min_share"])) if flood_lines else set()
    blocked = set(match(segments, blocking, min_share=a["incident_min_share"])) if blocking else set()

    # Live congestion on matched roads; group medians for the rest.
    live = {}
    for road_id, index in matched.items():
        item = flow_lines[index]
        live[road_id] = (min(max(1 - item["speed_mps"] / item["free_flow_mps"], 0.0), a["max_congestion"]),
                         max(item["speed_mps"], a["min_speed_mps"]))
    by_group, by_type, by_city_weighted = {}, {}, {}
    for road_id, (g, _) in live.items():
        r = records[road_id]
        by_group.setdefault((r.city, r.road_type), []).append(g)
        by_type.setdefault(r.road_type, []).append(g)
        total = by_city_weighted.setdefault(r.city, [0.0, 0.0])
        total[0] += g * r.length_m
        total[1] += r.length_m
    all_values = [g for g, _ in live.values()]
    overall = statistics.median(all_values) if all_values else 0.0
    city_load = {city: s / w for city, (s, w) in by_city_weighted.items() if w > 0}
    overall_load = (sum(s for s, _ in by_city_weighted.values()) / sum(w for _, w in by_city_weighted.values())
                    if by_city_weighted else 0.0)

    categories = {city: weather_condition(observation) for city, observation in weather["cities"].items()}
    caps = a["capacity_factors"]
    out = {}
    for road_id, r in records.items():
        if road_id in live:
            g, speed = live[road_id]
            travel_time, source = r.length_m / speed, "live"
        else:
            values = by_group.get((r.city, r.road_type)) or by_type.get(r.road_type)
            g = statistics.median(values) if values else overall
            travel_time, source = r.free_flow_time_s / max(1 - g, 1 - a["max_congestion"]), "group_estimate"
        category = categories.get(r.city)
        if category is None:
            raise ValueError(f"No current weather observation for {r.city}")
        if road_id in flooded:
            portions, flood_source = [(r.length_m / 1000, a["flood_report"]["severity"])], "live_report"
        else:
            portions = [(km, float(flood_class_severity[str(k)])) for k, km in zip((1, 2, 3), r.flood_km) if km > 0]
            flood_source = "noah_hazard"
        incident_factor = max((r.lanes - 1) / r.lanes, caps["single_lane_floor"]) if road_id in blocked else 1.0
        flood_factor = caps["flood_reported"] if road_id in flooded else 1.0
        weather_factor = caps["weather"][category]
        out[road_id] = {
            "congestion": g, "congestion_source": source, "travel_time_s": travel_time,
            "time_of_day": city_load.get(r.city, overall_load),
            "weather": category, "weather_severity": weather_scenarios[category]["severity"],
            "flood_portions": portions, "flood_source": flood_source,
            "capacity_vph": r.capacity_vph * incident_factor * flood_factor * weather_factor,
            "capacity_factors": {"incident": incident_factor, "flood": flood_factor, "weather": weather_factor},
            "closed": road_id in closed,
        }
    summary = {"segments": len(records), "live_matched": len(live), "group_estimates": len(records) - len(live),
               "closed": len(closed), "flood_reported": len(flooded), "lane_blocking": len(blocked),
               "city_load": city_load, "weather": categories}
    return {"segments": out, "summary": summary}
