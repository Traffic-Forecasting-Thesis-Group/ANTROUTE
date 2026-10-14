"""
The assumed tier's estimate: a ridge fit from the non-visual inputs to the decoder's risk.
"""

from datetime import datetime

import numpy as np
import pytest

from src.routing.context_risk import FEATURES, ContextRiskModel, EdgeContext, evenly, fit_from_windows


def test_the_fit_recovers_a_relationship_that_is_there():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, size=(400, len(FEATURES)))
    y = np.clip(0.2 + 0.5 * x[:, 0] + 0.2 * x[:, 1], 0, 1)          # flood and rain drive risk
    model = ContextRiskModel.fit(x, y, alpha=1e-3)
    assert model.r2_fit > 0.99
    assert model.predict(x[:5]) == pytest.approx(y[:5], abs=0.01)


def test_predictions_stay_within_the_risk_range():
    x = np.vstack([np.zeros(len(FEATURES)), np.ones(len(FEATURES))])
    model = ContextRiskModel.fit(x, np.array([0.0, 1.0]))
    far_out = np.full((1, len(FEATURES)), 50.0)
    assert 0.0 <= model.predict(far_out)[0] <= 1.0


def test_a_constant_input_does_not_break_the_fit():
    x = np.zeros((10, len(FEATURES)))
    x[:, 0] = np.linspace(0, 1, 10)
    model = ContextRiskModel.fit(x, np.linspace(0.1, 0.9, 10))
    assert np.isfinite(model.coef).all()


def test_holdout_r2_is_reported_from_windows_the_fit_never_saw():
    n = 6
    context = EdgeContext(
        flood=np.linspace(0, 1, n), speed=np.full(n, 0.4), log_length=np.full(n, 0.5),
        lanes=np.full(n, 1 / 3), road_class=np.full(n, 0.55),
        cell_index=np.zeros(n, dtype=int), cells=["c"], weather={},
    )
    edges = np.arange(n)
    risk = 0.2 + 0.6 * context.flood                                   # flood-prone roads are riskier
    train = [(datetime(2026, 5, d, 7, 30), edges, risk, None) for d in (4, 6, 8)]
    held = [(datetime(2026, 5, 25, 7, 30), edges, risk, None)]
    model = fit_from_windows(context, train, held, alpha=1e-3)
    assert model.n_windows == 3 and model.n_holdout_windows == 1
    assert model.r2_holdout == pytest.approx(1.0, abs=1e-3)
    assert set(model.describe()["standardised_coefficients"]) == set(FEATURES)


def test_evenly_spreads_a_sample_over_the_whole_range():
    assert evenly(list(range(10)), 3) == [0, 4, 9] or evenly(list(range(10)), 3) == [0, 5, 9]
    assert evenly([1, 2], 5) == [1, 2]


def test_hourly_weather_gives_each_hour_its_own_value():
    n = 3
    hot, wet = np.array([1.0, 0.0, 0.2]), np.array([0.2, 1.0, 0.9])
    day = datetime(2026, 5, 18).date()
    context = EdgeContext(
        flood=np.zeros(n), speed=np.full(n, 0.4), log_length=np.full(n, 0.5), lanes=np.full(n, 1 / 6),
        road_class=np.full(n, 0.15), cell_index=np.zeros(n, dtype=int), cells=["c"],
        weather={(day, 7, "c"): hot, (day, 17, "c"): wet},
    )
    assert context.hourly
    assert context.weather_at(datetime(2026, 5, 18, 7, 40))[0] == pytest.approx(hot)
    assert context.weather_at(datetime(2026, 5, 18, 17, 5))[0] == pytest.approx(wet)


def test_quantile_map_keeps_the_ranking_and_takes_the_reference_scale():
    from src.routing.context_risk import quantile_map

    reference_pred = np.array([0.1, 0.2, 0.3, 0.4])
    reference_risk = np.array([0.60, 0.65, 0.70, 0.75])      # the decoder's scale
    mapped = quantile_map(np.array([0.05, 0.25, 0.9]), reference_pred, reference_risk)
    assert mapped[0] < mapped[1] < mapped[2]                  # ranking kept
    assert 0.60 <= mapped.min() and mapped.max() <= 0.75      # values on the decoder's scale


def test_a_dropped_input_gets_no_weight():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, size=(300, len(FEATURES)))
    y = 0.5 * x[:, FEATURES.index("live_text_incidents")] + 0.1 * x[:, 0]
    full = ContextRiskModel.fit(x, y, alpha=1e-3)
    without = ContextRiskModel.fit(x, y, alpha=1e-3, dropped=["live_text_incidents"])
    assert abs(full.coef[FEATURES.index("live_text_incidents")]) > 0.1
    assert without.coef[FEATURES.index("live_text_incidents")] == pytest.approx(0.0, abs=1e-9)
    assert full.r2_fit > without.r2_fit


def test_live_text_decays_over_the_hour_after_a_post():
    from datetime import datetime as dt

    from src.data.text_prior import LIVE_MINUTES, LiveText

    posted = dt(2026, 5, 18, 7, 0)
    live = LiveText(created=np.array([posted.timestamp() / 60.0]), node_rows=[np.array([1])],
                    incident=np.array([True]), n_nodes=3)
    fresh, later, gone = (live.at(dt(2026, 5, 18, 7, m)) for m in (0, 30, 59))
    assert fresh[1, 0] > later[1, 0] > gone[1, 0] > 0 and fresh[0].sum() == 0
    assert fresh[1, 1] == fresh[1, 0]                          # an incident post counts in both columns
    assert live.at(dt(2026, 5, 18, 8, int(LIVE_MINUTES) % 60 + 1)).sum() == 0
    assert live.at(dt(2026, 5, 18, 6, 59)).sum() == 0           # not before it was posted
