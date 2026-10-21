"""What each of the ten criteria (contract version 2) is made of, its classification and its source.

Each "source" names what the study actually uses: the data provider (OpenStreetMap, HERE, Weatherstack, Project NOAH)
or the project's own assumption file. Values transferred from published references are stated as assumptions in the
basis text, not linked as if they were local evidence.
"""


def estimation_registry(assumptions, live: bool = False):
    sources = assumptions["sources"]
    registry = {
        "distance": {"classification": "calculated", "basis": "L = WGS84 geodesic length of the OSM geometry in km; the 2026 map is a temporal surrogate.", "source": sources["map"]},
        "congestion": {"classification": "assumed_congestion_scenario", "basis": "L x g, where g = 1 - 1/(1 + 0.15*x^4) per road segment and hour from data/processed/temporal/traffic_hourly.csv and x is a seeded, assumed demand-to-capacity index (hourly profile, road-type factor, weekend factor, repeatable daily per-city, three-hour per-area, static per-road and three-hour per-road variation). The BPR form and its coefficients 0.15 and 4 are standard values adopted as assumptions. Project assumptions, not measurements, vehicle counts or independently estimated demand.", "source": sources["assumptions"]},
        "travel_time": {"classification": "assumed_congestion_scenario", "basis": "T (min) = free-flow time x BPR travel-time factor of the same assumed scenario x assumed weather multiplier. Posted OSM speed is only a free-flow proxy; fallback speeds are assumptions.", "source": sources["assumptions"]},
        "intersections": {"classification": "calculated", "basis": "I = 1 when the segment's downstream node has OSM street_count >= 3: one encounter per traversed approach; geometry vertices are not nodes. Current topology is a temporal surrogate.", "source": sources["map"]},
        "signals": {"classification": "mapped_or_estimated", "basis": "S = 1 when the downstream node has an OSM traffic_signals tag: one encounter per traversed approach. Missing tags are assumed absent, not confirmed absent.", "source": sources["map"]},
        "road_type": {"classification": "assumed_preference", "basis": "L x p, p_k = 1 - median free-flow speed of OSM class k / the largest class median over development roads. A stated preference for faster classes, not a published penalty.", "source": sources["map"]},
        "time_profile": {"classification": "fitted_on_assumed_scenario", "basis": "L x h, h = mean congestion of the road's group (city x road type) for the day category (weekday/weekend) and local hour over development dates only; falls back to road type, then all roads. The clock hour itself is not a penalty.", "source": sources["assumptions"]},
        "weather": {"classification": "historical_provider_record_with_assumed_effect", "basis": "L x a, a = assumed severity of the hourly category from the nearest 0.05-degree Weatherstack grid (1-hour precipitation: >= 7.5 mm heavy rain, > 0 rain). Kept separate from flood exposure. Severities and the rain speed reductions (7.5% rain, 11.5% heavy rain, the midpoints of published foreign ranges) are assumptions, not calibrated for Metro Manila.", "source": sources["weather"]},
        "flood_exposure": {"classification": "mapped_hazard_with_assumed_severity", "basis": "Sum over portions of km inside each NOAH five-year hazard class x assumed class severity (class/3 scenario; alternatives equal and convex). Overlaps resolve to the highest class. Mapped susceptibility, not current inundation or closure; no bridge elevation correction.", "source": sources["flood"]},
        "capacity_burden": {"classification": "provisional_study_mapping", "basis": "L / K, K = directional lanes x assumed 1,900 veh/h/lane base saturation flow (a standard value adopted as an assumption, not validated locally) x g/C (0.5 assumed at a mapped signal, else 1). Inverse capacity is a provisional study mapping, not a universal published road penalty; lanes and signal timings are unvalidated.", "source": sources["assumptions"]},
    }
    if live:
        registry.update({
            "congestion": {"classification": "live_provider_observation", "basis": "L x g, g = 1 - live speed / free-flow speed from the latest HERE Traffic API v7 flow capture matched to the segment. Roads without a matched HERE line take the median of matched roads of the same city and road type, then road type, then all roads (flagged group_estimate).", "source": sources["here"]},
            "travel_time": {"classification": "live_provider_observation", "basis": "T (min) = length / live HERE speed for matched roads; free-flow time / (1 - group median congestion) for unmatched roads (flagged group_estimate). No weather multiplier is added: live speeds already reflect current weather.", "source": sources["here"]},
            "time_profile": {"classification": "live_derived", "basis": "L x h, h = length-weighted mean live HERE congestion of the matched roads in the same city at the capture time: how busy the city is right now.", "source": sources["here"]},
            "weather": {"classification": "live_provider_observation_with_assumed_effect", "basis": "L x a, a = assumed severity of the current Weatherstack observation category for the road's city. Severities are assumptions, not calibrated.", "source": sources["weather"]},
            "flood_exposure": {"classification": "mapped_hazard_or_live_report", "basis": "NOAH hazard class lengths x assumed class severity; a road matched to a current HERE flooding incident counts its whole length at the highest severity.", "source": sources["flood"]},
            "capacity_burden": {"classification": "provisional_study_mapping", "basis": "L / K_live, K_live = base capacity (directional lanes x assumed 1,900 veh/h/lane x g/C) x live HERE incident factor x flood factor x weather factor (configs/live_assumptions.json). A provisional study mapping; factors are assumptions.", "source": sources["live_assumptions"]},
        })
    return registry
