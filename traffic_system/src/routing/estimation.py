"""Published functional forms with separately disclosed project assumptions."""
from datetime import date


def traffic_estimate(free_flow_s: float, vc_ratio: float, assumptions: dict, day: date | None = None):
    vc_ratio *= assumptions.get("demand_scale", 1.0)
    if day is not None and day.weekday() >= 5:
        vc_ratio *= assumptions["weekend_demand_factor"]
    factor = 1 + assumptions["bpr_alpha"] * vc_ratio ** assumptions["bpr_beta"]
    return free_flow_s * factor, 1 - 1 / factor


def estimation_registry(assumptions):
    sources = assumptions["sources"]
    return {
        "distance": {"classification": "calculated", "basis": "WGS84 geodesic length of OSM geometry; 2026 map is a temporal surrogate.", "source": sources["map"]},
        "traffic_congestion": {"classification": "estimated", "basis": "1 - 1/(1 + alpha*(V/C)^beta). V/C profiles, road-type factors and weekend factor are project assumptions, not measurements.", "source": sources["bpr"]},
        "travel_time": {"classification": "estimated", "basis": "t = (length / free-flow speed) * (1 + 0.15*(V/C)^4) / (1 - rain speed reduction). Posted speed is only a free-flow proxy; fallback speeds are assumptions.", "source": sources["bpr"]},
        "number_of_intersections": {"classification": "calculated", "basis": "Count traversed nodes with OSM street_count >= 3; exclude origin; current topology is a temporal surrogate.", "source": sources["map"]},
        "traffic_signal": {"classification": "mapped_or_estimated", "basis": "OSM traffic_signals tags; missing tags are assumed absent, not confirmed absent. Historical installations are unknown.", "source": sources["map"]},
        "road_type": {"classification": "mapped", "basis": "OSM highway category; category costs and temporal stability are project assumptions.", "source": sources["map"]},
        "time_of_day": {"classification": "calculated", "basis": "Exact Asia/Manila request hour. A separate binary peak-hours cost uses assumed hours 07,08,17,18,19; it is not copied from congestion.", "source": "Project policy in configs/study_assumptions.json (no empirical peak-hour validation)"},
        "weather": {"classification": "historical_provider_record_with_estimated_effect", "basis": "Nearest 0.05-degree provider grid to each city representative point; source measurement provenance is unverified. Rain speed reductions 7.5%/11.5% are reference-range midpoints, not local calibration.", "source": sources["rain"]},
        "flood_risk": {"classification": "estimated_hazard_proxy", "basis": "Maximum NOAH class intersecting the road / 3. No mapped overlap is not proof of safety; no hourly flood observations or bridge elevation correction.", "source": sources["flood"]},
        "road_capacity": {"classification": "estimated", "basis": "c = directional lanes * 1900 * g/C. Assume g/C=0.5 at a tagged downstream signal, otherwise 1. Lane fallbacks and signal timings are unmeasured; unsignalized use is a proxy.", "source": sources["capacity"]},
    }
