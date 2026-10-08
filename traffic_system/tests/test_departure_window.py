from datetime import datetime

import pandas as pd
import pytest
from src.routing.departure_window import available_windows, match_recorded_window, resolve_departure_window


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


def session_starts(day: str, sessions=("AM", "PM")):
    """Window starts the decoder writes: every 5 min from 7:00-8:30 and 17:00-18:30."""
    hours = {"AM": 7, "PM": 17}
    return [
        pd.Timestamp(f"{day}T{hours[s]:02d}:00:00") + pd.Timedelta(minutes=m)
        for s in sessions
        for m in range(0, 95, 5)
    ]


def recorded(*days, sessions=("AM", "PM")):
    return [t.isoformat() for d in days for t in session_starts(d, sessions)]


# 2026-05-13 and 2026-05-20 are Wednesdays, 2026-05-22 a Friday, 2026-05-25 a Monday.
WINDOWS = recorded("2026-05-13", "2026-05-20", "2026-05-22", "2026-05-25")


def test_a_departure_months_later_matches_on_weekday_and_time_of_day():
    # Monday 5 Oct, 5:32 PM -> the most recent recorded Monday's 5:30 PM window.
    assert match_recorded_window(WINDOWS, datetime(2026, 10, 5, 17, 32)) == ("2026-05-25T17:30:00", True)


def test_the_window_starting_nearest_the_departure_wins_within_a_day():
    assert match_recorded_window(WINDOWS, datetime(2026, 10, 7, 7, 13)) == ("2026-05-20T07:15:00", True)


def test_a_weekday_never_recorded_falls_back_to_the_most_recent_day():
    # Tuesday: no recorded Tuesday, so the newest day (Mon 25 May).
    assert match_recorded_window(WINDOWS, datetime(2026, 10, 6, 17, 30)) == ("2026-05-25T17:30:00", True)


def test_a_recorded_date_uses_that_exact_day_over_a_newer_same_weekday():
    assert match_recorded_window(WINDOWS, datetime(2026, 5, 13, 8, 0)) == ("2026-05-13T08:00:00", True)


def test_time_of_day_beats_weekday_when_that_weekday_lacks_the_session():
    # Monday recorded only in the morning: a Monday-evening trip should use another
    # day's evening, not that Monday's 8:30 AM.
    windows = recorded("2026-05-22") + recorded("2026-05-25", sessions=("AM",))
    assert match_recorded_window(windows, datetime(2026, 10, 5, 17, 30)) == ("2026-05-22T17:30:00", True)


def test_the_tail_of_a_session_is_still_covered_by_its_last_window():
    # The 6:30 PM window runs to 7:00 PM.
    assert match_recorded_window(WINDOWS, datetime(2026, 10, 5, 18, 50)) == ("2026-05-25T18:30:00", True)


def test_a_departure_outside_the_recorded_peaks_uses_the_closest_and_says_so():
    # Noon: 8:30 AM is 3.5 h away, 5:00 PM is 5 h away.
    assert match_recorded_window(WINDOWS, datetime(2026, 10, 5, 12, 0)) == ("2026-05-25T08:30:00", False)


def test_matching_with_no_parseable_windows_raises():
    with pytest.raises(ValueError, match="no parseable"):
        match_recorded_window(["not a date"], datetime(2026, 10, 5, 8, 0))