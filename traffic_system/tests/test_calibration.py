"""Development-only calibration: split, hourly CRITIC averaging, mappings and profiles."""
from datetime import date

import numpy as np
import pytest

from src.routing.calibration import (CalibrationBlocked, CalibrationError, ProfileFit, class_severity, hourly_critic,
                                     load_split)
from src.routing.critic import CriticAccumulator
from src.routing.criteria import CRITERIA

SPLIT = {"development": {"start": "2025-06-01", "end": "2025-12-31"},
         "tuning": {"start": "2026-01-01", "end": "2026-02-28"}, "test": {"start": "2026-03-01", "end": "2026-05-31"}}


def test_hourly_vectors_are_averaged_with_zero_entries_kept():
    # Three-criterion arithmetic of item 15: [0.5, 0.5, 0] and [0, 0.5, 0.5] average to [0.25, 0.5, 0.25].
    vectors = np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]])
    assert vectors.mean(axis=0).tolist() == pytest.approx([0.25, 0.5, 0.25])
    # The same rule in hourly_critic: a criterion constant in one hour contributes 0 there, not "skip that hour".
    rng = np.random.default_rng(5)
    first, second = rng.random((60, len(CRITERIA))), rng.random((60, len(CRITERIA)))
    second[:, CRITERIA.index("weather")] = 0.0  # Dry hour: weather constant.
    result = hourly_critic([("h1", first, None), ("h2", second, None)])
    one = [CriticAccumulator(CRITERIA) for _ in range(2)]
    one[0].add(first)
    one[1].add(second)
    w1, w2 = one[0].result()["weights"], one[1].result()["weights"]
    assert w2["weather"] == 0
    assert result["weights"]["weather"] == pytest.approx(w1["weather"] / 2)  # Divided by |H| = 2, not by 1 active hour.
    assert result["weights"] == pytest.approx({name: (w1[name] + w2[name]) / 2 for name in CRITERIA})
    assert sum(result["weights"].values()) == pytest.approx(1)
    assert result["active_masks"]["h2"][CRITERIA.index("weather")] is False


def test_undefined_and_ineligible_hours_are_excluded_whole_and_logged():
    rng = np.random.default_rng(6)
    good = rng.random((40, len(CRITERIA)))
    constant = np.full((40, len(CRITERIA)), 0.3)
    result = hourly_critic([("good", good, None), ("constant", constant, None), ("missing", None, "input_unavailable: no weather")])
    assert result["valid_hours"] == ["good"]
    assert result["excluded_hours"]["constant"].startswith("undefined")
    assert result["excluded_hours"]["missing"] == "input_unavailable: no weather"
    with pytest.raises(CalibrationError, match="every sampled"):
        hourly_critic([("constant", constant, None), ("missing", None, "input_unavailable")])


def test_split_must_be_declared_chronological_and_in_period():
    with pytest.raises(CalibrationBlocked, match="not declared"):
        load_split({"calibration_split": {"development": {"start": None, "end": None}}})
    split = load_split({"calibration_split": SPLIT})
    assert split["development"] == (date(2025, 6, 1), date(2025, 12, 31))
    overlapping = {**SPLIT, "tuning": {"start": "2025-12-01", "end": "2026-02-28"}}
    with pytest.raises(ValueError, match="overlap"):
        load_split({"calibration_split": overlapping})
    with pytest.raises(ValueError, match="within"):
        load_split({"calibration_split": {**SPLIT, "test": {"start": "2026-03-01", "end": "2026-06-30"}}})


def test_class_severity_from_median_speeds():
    severity = class_severity({"primary": [50, 60, 40], "residential": [20, 20, 30], "trunk": [60, 60]})
    assert severity == pytest.approx({"primary": 1 - 50 / 60, "residential": 1 - 20 / 60, "trunk": 0}, abs=1e-6)  # Stored to 6 d.p.


def test_time_profile_falls_back_from_group_to_road_type_to_all_roads():
    fit = ProfileFit(groups={"r1": "Manila|primary", "r2": "Pasig|primary", "r3": "Pasig|residential"},
                     road_types={"r1": "primary", "r2": "primary", "r3": "residential"})
    for hour in range(24):
        fit.add("weekday", hour, {"r1": 0.2, "r2": 0.6, "r3": 0.1})
    fit.add("weekend", 8, {"r1": 0.4})  # Only r1 observed on a weekend, only at 08:00.
    for hour in [h for h in range(24) if h != 8]:
        fit.add("weekend", hour, {"r3": 0.05})
    profiles, fallbacks = fit.profiles()
    assert profiles["r1"]["weekday"][8] == pytest.approx(0.2) and profiles["r2"]["weekday"][8] == pytest.approx(0.6)
    assert profiles["r2"]["weekend"][8] == pytest.approx(0.4)  # Pasig|primary unseen: road type 'primary' (r1).
    assert profiles["r3"]["weekend"][8] == pytest.approx(0.4)  # Residential unseen at 08:00: all roads.
    assert fallbacks["road_type"] > 0 and fallbacks["all"] > 0


def test_no_tuning_or_test_record_enters_fitted_artifacts(network, monkeypatch):
    """Profiles, CRITIC weights, maxima and references read development dates only."""
    import scripts.build_study_network as builder
    from src.routing.network import WeatherEffect
    split = load_split({"calibration_split": {**SPLIT, "development": {"start": "2025-06-01", "end": "2025-06-30"}}})
    read = []
    rng = np.random.default_rng(7)

    def traffic(day, hour):
        read.append(day)
        return {road.id: (float(rng.random()) * 0.5, 1.0 + float(rng.random())) for road in network.roads}

    def weather(day, hour, net):
        read.append(day)
        return {road.source: WeatherEffect(severity=float(rng.random()), travel_time_multiplier=1) for road in net.roads}, {}
    for road in network.roads:
        road.traffic_location, road.free_flow_time_s = road.id, 10.0
        road.weather["historical"] = road.weather["clear"]
    network.provenance["assumptions"] = {"bpr_alpha": .15, "bpr_beta": 4}
    monkeypatch.setattr(builder, "traffic_snapshot", traffic)
    monkeypatch.setattr(builder, "historical_snapshot", weather)
    builder.fit_time_profiles(network, split["development"])
    result = builder.calibrate(network, split, network.provenance["assumptions"])
    assert read and max(read) <= date(2025, 6, 30) and min(read) >= date(2025, 6, 1)
    assert all(hour_id < "2025-07" for hour_id in result["valid_hours"] + list(result["excluded_hours"]))
    assert sum(result["weights"].values()) == pytest.approx(1)
    assert result["development_max"]["distance"] == pytest.approx(1.0)  # The 1000 m road.
    assert result["cheng_references"]["flow_provenance"] == "simulation_adaptation_inverse_bpr_of_assumed_congestion"
