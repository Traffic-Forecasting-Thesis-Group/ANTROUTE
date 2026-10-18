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
