"""The versioned criterion contract (criteria.py): additive, unclipped, validated."""
import pytest

from src.routing.criteria import CRITERIA, CriterionInputError, SegmentInputs, check_calibration, criteria, scale

WEIGHTS = dict.fromkeys(CRITERIA, 0.1)
MAXIMA = {**dict.fromkeys(CRITERIA, 1.0), "capacity_burden": 1e-3}


def segment(**changes):
    values = dict(length_m=2000, traversal_s=240, congestion=0.4, class_severity=0.5, profile_severity=0.3,
                  weather_severity=0.2, flood_portions=((0.5, 0.25), (1.5, 0.75)), capacity_vph=1800,
                  intersection=1, signal=1)
    return SegmentInputs(**{**values, **changes})


def test_flood_exposure_uses_class_lengths():
    # 0.5 km at severity 0.25 and 1.5 km at 0.75: exposure 1.25, mean severity over 2 km 0.625.
    values = criteria(segment())
    assert values["flood_exposure"] == pytest.approx(1.25)
    assert values["flood_exposure"] / values["distance"] == pytest.approx(0.625)


def test_criteria_definitions():
    values = criteria(segment())
    assert values == pytest.approx({
        "distance": 2, "congestion": 0.8, "travel_time": 4, "intersections": 1, "signals": 1, "road_type": 1,
        "time_profile": 0.6, "weather": 0.4, "flood_exposure": 1.25, "capacity_burden": 2 / 1800})


def test_subdivision_preserves_criteria_and_cost():
    # Splitting a road at a node that is neither an intersection nor a signal leaves every additive
    # criterion and the cost unchanged when the calibration is fixed.
    whole = criteria(segment())
    first = criteria(segment(length_m=500, traversal_s=60, flood_portions=((0.5, 0.25),), intersection=0, signal=0))
    second = criteria(segment(length_m=1500, traversal_s=180, flood_portions=((1.5, 0.75),)))
    assert {name: first[name] + second[name] for name in CRITERIA} == pytest.approx(whole)
    cost = scale(first, WEIGHTS, MAXIMA).cost + scale(second, WEIGHTS, MAXIMA).cost
    assert cost == pytest.approx(scale(whole, WEIGHTS, MAXIMA).cost)


def test_reference_exceedances_are_flagged_not_capped():
    result = scale(criteria(segment(length_m=3000, traversal_s=600)), WEIGHTS, MAXIMA)
    assert result.values["distance"] == pytest.approx(3) and result.values["travel_time"] == pytest.approx(10)
    assert {"distance", "travel_time"} <= set(result.exceedances)
    assert result.cost > 1


def test_zero_weight_criterion_needs_no_maximum():
    weights = {**WEIGHTS, "signals": 0, "distance": 0.2}
    maxima = {**MAXIMA, "signals": 0}
    check_calibration(weights, maxima)
    result = scale(criteria(segment()), weights, maxima)
    assert result.values["signals"] is None and result.contributions["signals"] == 0
    with pytest.raises(CriterionInputError, match="positive weight"):
        check_calibration(WEIGHTS, maxima)


@pytest.mark.parametrize("change", [
    {"length_m": 0}, {"traversal_s": float("nan")}, {"capacity_vph": 0}, {"congestion": 1.2},
    {"weather_severity": -0.1}, {"flood_portions": ((3.0, 0.5),)}, {"flood_portions": ((0.5, 1.5),)},
    {"intersection": 2}, {"signal": True + 1}, {"class_severity": float("inf")},
])
def test_invalid_inputs_are_rejected(change):
    with pytest.raises(CriterionInputError):
        criteria(segment(**change))


def test_calibration_requires_the_ten_criteria_and_unit_sum():
    with pytest.raises(CriterionInputError):
        check_calibration({**WEIGHTS, "distance": 0.5}, MAXIMA)
    with pytest.raises(CriterionInputError):
        check_calibration({"distance": 1.0}, MAXIMA)
