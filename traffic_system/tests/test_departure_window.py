import pandas as pd
import pytest
from src.routing.departure_window import available_windows, resolve_departure_window


def make_risk_csv(tmp_path):
    path = tmp_path / "risk_edges.csv"
    pd.DataFrame(
        {
            "window_start": ["2026-05-25T07:00:00", "2026-05-25T17:00:00", "2026-05-25T17:30:00"],
            "window_end": ["2026-05-25T07:30:00", "2026-05-25T17:30:00", "2026-05-25T18:00:00"],
            "source_node_id": [1, 1, 1],
            "target_node_id": [2, 2, 2],
            "risk": [0.1, 0.5, 0.9],
        }
    ).to_csv(path, index=False)
    return path


def test_no_depart_at_returns_none(tmp_path):
    assert resolve_departure_window(make_risk_csv(tmp_path), None) is None


def test_exact_containment_picks_the_right_window(tmp_path):
    path = make_risk_csv(tmp_path)
    assert resolve_departure_window(path, "2026-05-25T17:15:00") == "2026-05-25T17:00:00"


def test_boundary_time_belongs_to_the_window_that_starts_there(tmp_path):
    path = make_risk_csv(tmp_path)
    assert resolve_departure_window(path, "2026-05-25T17:30:00") == "2026-05-25T17:30:00"


def test_uncovered_time_falls_back_to_the_nearest_window(tmp_path):
    path = make_risk_csv(tmp_path)
    assert resolve_departure_window(path, "2026-05-25T23:00:00") == "2026-05-25T17:30:00"


def test_uncovered_early_time_falls_back_to_the_nearest_window(tmp_path):
    path = make_risk_csv(tmp_path)
    assert resolve_departure_window(path, "2026-05-25T00:00:00") == "2026-05-25T07:00:00"


def test_a_file_with_only_window_end_still_resolves(tmp_path):
    path = tmp_path / "risk_edges.csv"
    pd.DataFrame(
        {
            "window_end": ["2026-05-25T07:30:00", "2026-05-25T17:30:00"],
            "source_node_id": [1, 1],
            "target_node_id": [2, 2],
            "risk": [0.1, 0.9],
        }
    ).to_csv(path, index=False)
    windows = available_windows(path)
    assert list(windows["window_end"]) == ["2026-05-25T07:30:00", "2026-05-25T17:30:00"]


def test_unparseable_windows_raise_clearly(tmp_path):
    path = tmp_path / "risk_edges.csv"
    pd.DataFrame({"source_node_id": [1], "target_node_id": [2], "risk": [0.5]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="no parseable"):
        available_windows(path)