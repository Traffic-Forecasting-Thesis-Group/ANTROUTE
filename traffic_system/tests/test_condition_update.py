"""
Route Quality Before and After Condition Updates: the old route and the recalculated one are
both measured under the NEW conditions, and improvement is old minus recalculated.
"""

import numpy as np
import pytest

from src.routing.condition_update import compare_update
from test_aco_routing import weighted_risky_direct

DETOUR = [100, 101, 102, 103]     # 3 x 200 m, risk 0.1 each
DIRECT = [100, 103]               # 500 m, risk 0.9


def test_both_routes_are_measured_under_the_new_conditions():
    wg_after = weighted_risky_direct()
    impact = np.zeros(wg_after.n_edges)
    impact[wg_after.edge_id(wg_after.index_of(100), wg_after.index_of(103))] = 1.0   # incident on the old route
    row = compare_update(wg_after, impact, old_path=DIRECT, new_path=DETOUR)
    assert row["route_changed"] is True
    assert row["old_congestion_exposure"] == pytest.approx(0.9)
    assert row["recalculated_congestion_exposure"] == pytest.approx(0.3)
    assert row["old_event_exposure"] == pytest.approx(1.0) and row["recalculated_event_exposure"] == 0.0
    assert row["old_avoidable_event_exposure"] == 0.0     # the incident's road joins origin and destination
    assert row["congestion_exposure_improvement"] == pytest.approx(0.6)     # positive: recalculation helped
    assert row["event_exposure_improvement"] == pytest.approx(1.0)
    assert row["distance_m_change"] == pytest.approx(100.0)                 # descriptive: 600 m vs 500 m


def test_an_unchanged_route_shows_no_improvement():
    wg_after = weighted_risky_direct()
    row = compare_update(wg_after, None, DETOUR, DETOUR)
    assert row["route_changed"] is False
    assert row["congestion_exposure_improvement"] == 0.0 and row["event_exposure_improvement"] == 0.0


def test_routes_for_different_trips_cannot_be_compared():
    with pytest.raises(ValueError, match="same origin"):
        compare_update(weighted_risky_direct(), None, [100, 101], DETOUR)
