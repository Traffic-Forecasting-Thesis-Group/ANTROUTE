import numpy as np
import pytest

from src.routing.critic import CriticAccumulator, UndefinedCriticError, critic_weights

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


def test_duplicate_criteria_lose_weight():
    rng = np.random.default_rng(2)
    independent = rng.random((1000, 2))
    rows = np.column_stack([independent[:, 0], independent[:, 0], independent[:, 1]])
    weights = critic_weights(rows, CRITERIA)["weights"]
    assert weights["a"] == pytest.approx(weights["b"])
    assert weights["c"] > weights["a"]


def test_constant_column_is_invariant():
    # A constant criterion gets weight 0 and leaves the other weights exactly as if it were absent.
    rows = np.random.default_rng(3).random((400, 2)) * [1, 7]
    without = critic_weights(rows, ("a", "b"))
    with_constant = critic_weights(np.column_stack([rows, np.full(400, 0.4)]), CRITERIA)
    assert with_constant["weights"]["c"] == 0 and not with_constant["active"]["c"]
    assert [with_constant["weights"][name] for name in ("a", "b")] == pytest.approx([without["weights"][name] for name in ("a", "b")])
    # Unavailable correlations are None, not an artificial zero.
    assert with_constant["correlation"]["a"]["c"] is None and with_constant["conflict"]["c"] is None


def test_perfectly_concordant_varying_columns_are_undefined():
    x = np.linspace(0, 1, 50)
    with pytest.raises(UndefinedCriticError, match="concordant"):
        critic_weights(np.column_stack([x, 2 * x + 1, np.full(50, 3.0)]), CRITERIA)


def test_centred_statistics_survive_large_offsets():
    rows = np.random.default_rng(4).random((500, 3))
    shifted = rows + [1e9, -1e9, 5e8]  # Raw-moment formulas lose these variances to cancellation.
    assert critic_weights(shifted, CRITERIA)["weights"] == pytest.approx(critic_weights(rows, CRITERIA)["weights"], rel=1e-5)
    streamed = CriticAccumulator(CRITERIA)
    for chunk in np.array_split(shifted, 9):
        streamed.add(chunk)
    assert streamed.result()["weights"] == pytest.approx(critic_weights(rows, CRITERIA)["weights"], rel=1e-5)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        critic_weights([[1, 2, np.nan], [0, 1, 2]], CRITERIA)
    with pytest.raises(ValueError):
        critic_weights([[1, 2], [0, 1]], CRITERIA)
    with pytest.raises(UndefinedCriticError, match="constant"):
        critic_weights([[1, 1, 1], [1, 1, 1]], CRITERIA)
    with pytest.raises(UndefinedCriticError, match="two alternatives"):
        critic_weights([[1, 2, 3]], CRITERIA)
