"""
How text reaches the cameras: a tweet attaches to every camera within the placement radius of
the place it names, and stays in the text input for the active window after it is posted. The
defaults stay the original rules (own intersection, one minute) so older checkpoints are scored
exactly as they were trained.
"""

from datetime import date

import numpy as np
import pandas as pd

from src.data.alignment import build_sessions, session_start
from src.data.training_data import TEXT_DEFAULTS
from src.data.tweet_geocode import cameras_within, intersection_coordinates, place

BY_INTERSECTION = {"EDSA-Ortigas": ["cam_ortigas"], "EDSA-Aurora": ["cam_aurora"]}
COORDS = {
    "EDSA-Ortigas": (14.5876, 121.0567),
    "EDSA-Aurora": (14.6206, 121.0521),       # ~3.7 km north of Ortigas
    "EDSA-Guadalupe": (14.5621, 121.0423),    # ~3.1 km south of Ortigas
    "EDSA-Reliance": (14.5767, 121.0505),     # ~1.4 km from Ortigas
}


def test_radius_zero_keeps_the_original_rule():
    assert cameras_within(BY_INTERSECTION, COORDS, 0.0) == BY_INTERSECTION


def test_a_nearby_place_reaches_the_cameras_within_the_radius():
    near = cameras_within(BY_INTERSECTION, COORDS, 2.0)
    assert near["EDSA-Reliance"] == ["cam_ortigas"]          # 1.4 km: in reach
    assert "EDSA-Guadalupe" not in near                      # 3.1 km: out of reach, still dropped
    assert near["EDSA-Ortigas"] == ["cam_ortigas"]           # a camera keeps its own tweets
    landmarks = {"edsa reliance": "EDSA-Reliance", "edsa guadalupe": "EDSA-Guadalupe"}
    assert place(["Crash at EDSA Reliance NB", "Stalled bus EDSA Guadalupe"], landmarks, near) == [["cam_ortigas"], []]


def test_every_camera_intersection_has_coordinates():
    coords = intersection_coordinates()
    for label in ("EDSA-Ortigas", "EDSA-Shaw", "EDSA-Aurora", "Roxas Blvd-Kalaw", "8337-Ayala NB 1-PTZ"):
        assert label in coords


def _sessions(tweet_minutes, active):
    day = date(2026, 5, 18)
    start = session_start(day, "AM")
    weather = pd.DataFrame([{"date": day, "ws_temp_c": 30.0, "ws_precip_mm": 0.0, "ws_humidity_pct": 60.0}])
    frames = pd.DataFrame({"camera_id": ["cam"], "timestamp": [start], "frame_path": ["f.jpg"]})
    tweets = pd.DataFrame({
        "camera_id": ["cam"] * len(tweet_minutes),
        "created_at": [start + pd.Timedelta(minutes=m) for m in tweet_minutes],
        "embedding": [np.full(4, float(i + 1), dtype=np.float32) for i in range(len(tweet_minutes))],
    })
    records, _, _ = build_sessions(frames, tweets, weather, ["cam"], ["ws_temp_c", "ws_precip_mm", "ws_humidity_pct"],
                                   visual_split={(day, "AM"): "train"}, text_active_minutes=active)
    return next(r for r in records if r.session == "AM").text_steps


def test_one_minute_is_the_original_rule():
    steps = _sessions([5], active=1)
    assert sorted(steps) == [5]


def test_a_tweet_stays_active_for_the_window_and_tweets_before_the_session_carry_in():
    steps = _sessions([-20, 5], active=60)            # one 20 minutes before the session starts
    assert min(steps) == 0 and max(steps) == 64        # -20 covers 0..39, 5 covers 5..64
    assert np.allclose(steps[0], 1.0)                  # only the earlier tweet is active yet
    assert np.allclose(steps[10], 1.5)                # both active: their mean
    assert np.allclose(steps[50], 2.0)                 # only the second is still active


def test_new_training_uses_the_wider_reach():
    assert TEXT_DEFAULTS == {"text_radius_km": 2.0, "text_active_minutes": 60}
