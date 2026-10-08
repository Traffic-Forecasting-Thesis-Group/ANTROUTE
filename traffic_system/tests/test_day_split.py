"""The whole-date 70/15/15 split every modality follows (training_data.assign_day_splits)."""

from datetime import date

import pytest

from src.data.split_report import check_split, split_shares
from src.data.training_data import (
    assign_day_splits,
    assign_session_splits,
    load_split,
    save_split,
    split_days_overlap,
)

DAYS = [date(2026, 5, d) for d in (4, 6, 8, 11, 13, 15, 18, 20, 22, 25)]
ALL = [(d, s) for d in DAYS for s in ("AM", "PM")]


def test_the_session_split_can_put_one_date_on_both_sides():
    # Why the day split exists: 7 sessions -> 5 / 1 / 1 cuts a date in half.
    split = assign_session_splits(ALL[:7])
    assert split_days_overlap(split)


def test_no_date_is_in_two_splits_and_the_order_is_chronological():
    split = assign_day_splits(ALL)
    assert split_days_overlap(split) == {}
    order = [split[s] for s in sorted(split)]
    assert order == sorted(order, key=["train", "val", "test"].index)     # train, then val, then test
    for d in DAYS:
        assert split[(d, "AM")] == split[(d, "PM")]


def test_shares_come_as_close_to_70_15_15_as_whole_dates_allow():
    split = assign_day_splits(ALL)
    shares = split_shares(split)
    assert shares["train"] == pytest.approx(0.70) and min(shares["val"], shares["test"]) == pytest.approx(0.10)
    assert check_split(split) == []


def test_label_weights_steer_the_cut_toward_the_labelled_data():
    # Most labels sit on the last days, so a session count would starve train of targets.
    weights = {s: (50 if s[0] >= DAYS[6] else 5) for s in ALL}
    by_labels, by_sessions = assign_day_splits(ALL, weights), assign_day_splits(ALL)

    def miss(split):  # distance from 70/15/15 in labelled frames
        shares = split_shares(split, weights)
        return sum(abs(shares[s] - t) for s, t in (("train", .70), ("val", .15), ("test", .15)))

    assert by_labels != by_sessions and miss(by_labels) < miss(by_sessions)
    assert split_days_overlap(by_labels) == {}


def test_too_few_dates_fill_what_they_can():
    one, two = assign_day_splits(ALL[:2]), assign_day_splits(ALL[:4])
    assert set(one.values()) == {"train"}
    assert sorted(set(two.values())) == ["train", "val"]
    assert "test is empty" in check_split(two)


def test_a_split_round_trips_through_its_file(tmp_path):
    split = assign_day_splits(ALL)
    save_split(split, tmp_path / "split.json")
    assert load_split(tmp_path / "split.json") == split
    (tmp_path / "bad.json").write_text('{"2026-05-04|AM": "holdout"}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_split(tmp_path / "bad.json")


def test_check_split_names_a_crossing_date():
    split = assign_session_splits(ALL[:7])
    assert any("is in" in p for p in check_split(split))
