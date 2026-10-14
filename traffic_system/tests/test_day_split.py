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


def test_the_road_text_history_may_not_use_posts_from_a_split_date():
    from datetime import date

    from src.data.split_report import source_problems

    split = {(date(2026, 5, 4), "AM"): "train", (date(2026, 5, 4), "PM"): "train",
             (date(2026, 5, 20), "AM"): "val", (date(2026, 5, 25), "PM"): "test"}
    assert source_problems(split, date(2026, 5, 1)) == []                      # before the first session
    assert source_problems(split, date(2026, 5, 4)) == []                      # up to the first session's date
    leak = source_problems(split, date(2026, 5, 21))                           # would include May 4 and May 20's posts
    assert len(leak) == 1 and "2026-05-04" in leak[0] and "leak" in leak[0]
    assert source_problems(split, None) == []                                  # no text history, nothing to leak


def test_the_split_table_reports_every_source_including_the_text_history(tmp_path):
    from datetime import date

    import pandas as pd

    from src.data.split_report import split_table

    frames = tmp_path / "frames"
    frames.mkdir()
    pd.DataFrame({"camera_id": ["A", "A"], "timestamp": ["2026-05-04T17:05:00", "2026-05-20T07:05:00"],
                  "frame_path": ["a.jpg", "b.jpg"]}).to_csv(frames / "manifest.csv", index=False)
    weather = tmp_path / "w.csv"
    pd.DataFrame({"date": ["2026-05-04", "2026-05-04", "2026-05-20"], "hour": [7, 8, 7], "grid_cell": ["1_1"] * 3,
                  "ws_temp_c": [30, 31, 29]}).to_csv(weather, index=False)
    twitter = tmp_path / "twitter" / "2026-05-04"
    twitter.mkdir(parents=True)
    (twitter / "tweets_1700_1900.json").write_text(
        '{"data": [{"createdAt": "Mon May 04 09:20:00 +0000 2026", "text": "x"}, '
        '{"createdAt": "Wed Apr 29 09:20:00 +0000 2026", "text": "y"}]}', encoding="utf-8")
    split = {(date(2026, 5, 4), "AM"): "train", (date(2026, 5, 4), "PM"): "train",
             (date(2026, 5, 20), "AM"): "val", (date(2026, 5, 25), "PM"): "test"}
    table = split_table(split, [frames], {"a.jpg": 2}, weather_csv=weather, raw_twitter_root=twitter,
                        text_prior_before=date(2026, 5, 1))
    mods = set(table["modality"])
    assert {"hourly weather", "road text history", "live road text", "event / text"} <= mods
    history = table[table["modality"] == "road text history"].iloc[0]
    assert (history["train"], history["val"], history["test"]) == (0, 0, 0)    # the Apr 29 post is no split date's
    hours = table[(table["modality"] == "hourly weather") & table["measure"].str.contains("rows")].iloc[0]
    assert hours["train"] == 2 and hours["val"] == 1
