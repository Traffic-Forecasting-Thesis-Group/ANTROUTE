from collections import Counter
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.data.alignment import (
    BUFFER_START,
    NONVISUAL_START,
    VISUAL_SPLIT,
    build_sessions,
    build_window_index,
    nonvisual_split,
    window_starts,
)

WEATHER_COLUMNS = ["precip", "humidity"]


def make_data(extra_weather=None):
    frames = pd.DataFrame({
        "camera_id": ["cam1", "cam1", "cam1"],
        "timestamp": ["2026-05-04 07:00:10", "2026-05-04 07:00:50", "2026-05-04 07:05:30"],
        "frame_path": ["a.jpg", "b.jpg", "c.jpg"],
    })
    tweets = pd.DataFrame({
        "camera_id": ["cam1", "cam1", "cam1", "cam2"],
        "created_at": [
            "2026-05-04 07:05:30",
            "2026-05-04 07:05:59",
            "2026-05-04 09:00:00",     # outside AM session
            "2026-05-04 07:05:30",     # other camera
        ],
        "embedding": [np.ones(4), 3 * np.ones(4), np.ones(4), np.ones(4)],
    })
    rows = [
        {"date": "2026-05-04", "precip": 0.0, "humidity": 60.0},
        {"date": "2026-05-06", "precip": 10.0, "humidity": 80.0},
    ]
    rows += extra_weather or []
    return frames, tweets, pd.DataFrame(rows)


def get(records, camera_id, day, session):
    return next(
        r for r in records
        if r.camera_id == camera_id and r.day == day and r.session == session
    )


# Splits and windows

def test_visual_split_is_14_3_3():
    assert Counter(VISUAL_SPLIT.values()) == {"train": 14, "val": 3, "test": 3}


def test_visual_split_is_chronological():
    order = {"AM": 0, "PM": 1}
    keys = {s: [(d, order[p]) for (d, p), v in VISUAL_SPLIT.items() if v == s]
            for s in ("train", "val", "test")}
    assert max(keys["train"]) < min(keys["val"])
    assert max(keys["val"]) < min(keys["test"])


def test_window_starts():
    starts = window_starts()
    assert len(starts) == 19
    assert starts[0] == 0 and starts[-1] == 90


def test_nonvisual_split_chronological_with_buffer():
    days = [NONVISUAL_START + timedelta(days=i)
            for i in range((date(2026, 5, 26) - NONVISUAL_START).days)]
    split = {d: nonvisual_split(d) for d in days}

    train = [d for d in days if split[d] == "train"]
    val = [d for d in days if split[d] == "val"]
    test = [d for d in days if split[d] == "test"]

    assert max(train) < min(val) and max(val) < min(test)
    assert max(test) < BUFFER_START
    assert all(split[d] is None for d in days if d >= BUFFER_START)
    assert abs(len(train) / (len(train) + len(val) + len(test)) - 0.70) < 0.01


# Alignment

def test_frames_and_posts_land_in_the_right_minute():
    frames, tweets, weather = make_data()
    records, _, _ = build_sessions(frames, tweets, weather, ["cam1", "cam2"], WEATHER_COLUMNS)
    r = get(records, "cam1", date(2026, 5, 4), "AM")

    assert r.regime == "visual" and r.split == "train"
    assert r.frame_paths[0] == "b.jpg"          # latest frame in the minute wins
    assert r.frame_paths[5] == "c.jpg"
    assert sum(p is not None for p in r.frame_paths) == 2

    assert set(r.text_steps) == {5}             # 09:00 post excluded
    assert np.allclose(r.text_steps[5], 2.0)    # mean of the two posts


def test_posts_are_per_camera():
    frames, tweets, weather = make_data()
    records, _, _ = build_sessions(frames, tweets, weather, ["cam1", "cam2"], WEATHER_COLUMNS)
    r = get(records, "cam2", date(2026, 5, 4), "AM")

    assert set(r.text_steps) == {5}
    assert all(p is None for p in r.frame_paths)    # cam2 has no frames


def test_utc_posts_are_converted_to_manila_time():
    frames, tweets, weather = make_data()
    tweets = pd.DataFrame({
        "camera_id": ["cam1"],
        "created_at": pd.to_datetime(["2026-05-03 23:05:30"]).tz_localize("UTC"),
        "embedding": [np.ones(4)],
    })
    records, _, _ = build_sessions(frames, tweets, weather, ["cam1"], WEATHER_COLUMNS)
    r = get(records, "cam1", date(2026, 5, 4), "AM")

    assert set(r.text_steps) == {5}             # 23:05 UTC = 07:05 Manila


def test_days_without_weather_are_skipped():
    frames, tweets, weather = make_data()
    records, _, skipped = build_sessions(frames, tweets, weather, ["cam1"], WEATHER_COLUMNS)

    assert {r.day for r in records} == {date(2026, 5, 4), date(2026, 5, 6)}
    assert skipped > 0


def test_weather_scaler_uses_training_days_only():
    frames, tweets, weather = make_data(
        extra_weather=[{"date": "2026-05-25", "precip": 20.0, "humidity": 70.0}]
    )
    records, scaler, _ = build_sessions(frames, tweets, weather, ["cam1"], WEATHER_COLUMNS)

    assert np.allclose(scaler.max_, [10.0, 80.0])           # May 25 (test) not used
    r = get(records, "cam1", date(2026, 5, 25), "PM")
    assert r.split == "test"
    assert np.isclose(r.weather[0], 2.0)                    # above training range


def test_window_index():
    frames, tweets, weather = make_data()
    records, _, _ = build_sessions(frames, tweets, weather, ["cam1"], WEATHER_COLUMNS)
    index = build_window_index(records, "train", regime="visual")

    # May 4 and May 6, AM + PM, 19 windows each
    assert len(index) == 4 * 19