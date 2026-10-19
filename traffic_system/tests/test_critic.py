import numpy as np
import pytest

from src.routing.critic import CriticAccumulator, critic_weights

CRITERIA = ("a", "b", "c")


def direct_critic(rows):
    """CRITIC straight from its definition, on the full matrix."""
    rows = np.asarray(rows, dtype=float)
    scaled = (rows - rows.min(axis=0)) / (rows.max(axis=0) - rows.min(axis=0))
    information = scaled.std(axis=0) * (1 - np.corrcoef(rows, rowvar=False)).sum(axis=1)
    return information / information.sum()


def test_matches_the_definition():
    rows = np.random.default_rng(0).random((500, 3)) * [1, 5, 20]
    rows[:, 1] += rows[:, 0] * 3  # Correlated criteria share information.
    weights = critic_weights(rows, CRITERIA)["weights"]
    assert [weights[name] for name in CRITERIA] == pytest.approx(direct_critic(rows))
    assert sum(weights.values()) == pytest.approx(1)


def test_streamed_chunks_equal_one_pass():
    rows = np.random.default_rng(1).random((300, 3))
    accumulator = CriticAccumulator(CRITERIA)
    for chunk in np.array_split(rows, 7):
        accumulator.add(chunk)
    assert accumulator.result()["weights"] == pytest.approx(critic_weights(rows, CRITERIA)["weights"])


def test_duplicate_criteria_lose_weight_and_constants_get_none():
    rng = np.random.default_rng(2)
    independent = rng.random((1000, 2))
    rows = np.column_stack([independent[:, 0], independent[:, 0], independent[:, 1]])
    weights = critic_weights(rows, CRITERIA)["weights"]
    assert weights["a"] == pytest.approx(weights["b"])
    assert weights["c"] > weights["a"]
    rows[:, 2] = 0.5
    assert critic_weights(rows, CRITERIA)["weights"]["c"] == 0


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        critic_weights([[1, 2, np.nan], [0, 1, 2]], CRITERIA)
    with pytest.raises(ValueError):
        critic_weights([[1, 1, 1], [1, 1, 1]], CRITERIA)


def test_within_hour_critic_gives_no_weight_to_a_feature_constant_within_each_hour():
    import random
    from scripts.build_study_network import critic_within_hours
    from src.routing.network import FEATURES
    rng = random.Random(1)
    peak, weather = FEATURES.index("time_of_day"), FEATURES.index("weather")
    hours = []
    for hour_value in (0.0, 1.0):  # Off-peak hour, then peak hour: same value on every road within an hour.
        rows = [[rng.random() for _ in FEATURES] for _ in range(50)]
        for row in rows:
            row[peak] = hour_value
        hours.append(rows)
    hours[0] = [row[:weather] + [0.0] + row[weather + 1:] for row in hours[0]]  # Dry hour: weather constant.
    hours.append([[0.5] * len(FEATURES)] * 3)  # Every feature constant: skipped.
    result = critic_within_hours(hours)
    assert result["weights"]["time_of_day"] == 0 and result["active_hours"]["time_of_day"] == 0
    # Weather's weight comes only from the hour in which it varies, not diluted by the dry hour.
    assert result["active_hours"]["weather"] == 1 and result["weights"]["weather"] > 0
    assert sum(result["weights"].values()) == pytest.approx(1)
    assert result["hours_used"] == 2
