"""
Temporal alignment, windowing, and chronological splits for
ANTROUTE's CNN + LSTM module.

Collection rates:
    Visual (MMDA CCTV):  1 frame every 60 s, peak sessions only,
                         May 4-25, 2026 (Mon/Wed/Fri)
    Unstructured (X):    retrieved every 2 h, aligned by each post's created_at
    Temporal (weather):  daily

Common timestep: 1 minute.
Step k of a session covers [session_start + k min, session_start + (k + 1) min).

Expected inputs (pandas DataFrames):
    frames:  camera_id, timestamp, frame_path
    tweets:  camera_id, created_at, embedding   (768-d DistilBERT vector per post,
                                                 camera_id = geocoded intersection)
    weather: date, <weather_columns...>         (one row per day)

Naive timestamps are assumed to already be Manila local time.
Timezone-aware timestamps (e.g. X API created_at in UTC) are converted.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# Timestep settings
STEP_MINUTES = 1
SESSION_STEPS = 120          # 2-hour peak session
WINDOW_STEPS = 30            # 30-minute window
STRIDE_STEPS = 5             # 5-minute stride
SESSION_STARTS = {"AM": time(7, 0), "PM": time(17, 0)}
LOCAL_TZ = "Asia/Manila"

# Non-visual regime (visual modality missing)
NONVISUAL_START = date(2025, 6, 1)
BUFFER_START = date(2026, 4, 27)     # buffer week: Apr 27 - May 3, 2026 (excluded)

# Visual regime: chronological split by session (14 / 3 / 3)
_TRAIN_DAYS = (4, 6, 8, 11, 13, 15, 18)
VISUAL_SPLIT: Dict[Tuple[date, str], str] = {
    **{(date(2026, 5, d), s): "train" for d in _TRAIN_DAYS for s in ("AM", "PM")},
    (date(2026, 5, 20), "AM"): "val",
    (date(2026, 5, 20), "PM"): "val",
    (date(2026, 5, 22), "AM"): "val",
    (date(2026, 5, 22), "PM"): "test",
    (date(2026, 5, 25), "AM"): "test",
    (date(2026, 5, 25), "PM"): "test",
}


# Splits

def nonvisual_days(
    start: date = NONVISUAL_START,
    buffer_start: date = BUFFER_START,
) -> List[date]:
    """All days of the non-visual regime, ending before the buffer week."""
    return [start + timedelta(days=i) for i in range((buffer_start - start).days)]


def nonvisual_split(
    day: date,
    start: date = NONVISUAL_START,
    buffer_start: date = BUFFER_START,
) -> Optional[str]:
    """Chronological 70/15/15 split by day. None = outside the regime."""
    n = (buffer_start - start).days
    if n <= 0 or day < start or day >= buffer_start:
        return None

    i = (day - start).days
    if i < round(0.70 * n):
        return "train"
    if i < round(0.85 * n):
        return "val"
    return "test"


# Windowing

def window_starts(
    n_steps: int = SESSION_STEPS,
    window: int = WINDOW_STEPS,
    stride: int = STRIDE_STEPS,
) -> List[int]:
    """Window start steps inside one session (windows never cross sessions)."""
    if n_steps < window:
        return []
    return list(range(0, n_steps - window + 1, stride))


# Time helpers

def to_local(timestamps: pd.Series) -> pd.Series:
    """Parse timestamps and convert tz-aware values to naive Manila time."""
    ts = pd.to_datetime(timestamps)
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    return ts


def session_start(day: date, session: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(day, SESSION_STARTS[session]))


def step_index(timestamps: pd.Series, start: pd.Timestamp) -> np.ndarray:
    """Minute bin of each timestamp relative to the session start."""
    return ((timestamps - start) // pd.Timedelta(minutes=STEP_MINUTES)).to_numpy()


# Weather normalization

@dataclass
class MinMaxScaler:
    """Sequence normalization to [0, 1]. Fit on TRAINING days only."""

    min_: Optional[np.ndarray] = None
    max_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "MinMaxScaler":
        self.min_ = x.min(axis=0)
        self.max_ = x.max(axis=0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        span = self.max_ - self.min_
        span = np.where(span > 0, span, 1.0)
        # Not clipped: val/test values outside the training range stay visible
        return ((x - self.min_) / span).astype(np.float32)


# Session records

@dataclass
class SessionRecord:
    camera_id: str
    day: date
    session: str                              # "AM" or "PM"
    regime: str                               # "visual" or "nonvisual"
    split: str                                # "train", "val", "test"
    frame_paths: List[Optional[str]]          # length SESSION_STEPS, None = no frame
    text_steps: Dict[int, np.ndarray] = field(default_factory=dict)  # step -> mean embedding
    weather: Optional[np.ndarray] = None      # normalized daily weather vector


def _session_plan(
    nonvisual_start: date,
    buffer_start: date,
) -> List[Tuple[date, str, str, str]]:
    plan = [
        (day, session, "visual", split)
        for (day, session), split in VISUAL_SPLIT.items()
    ]
    for day in nonvisual_days(nonvisual_start, buffer_start):
        split = nonvisual_split(day, nonvisual_start, buffer_start)
        for session in SESSION_STARTS:
            plan.append((day, session, "nonvisual", split))
    return plan


def build_sessions(
    frames: pd.DataFrame,
    tweets: pd.DataFrame,
    weather: pd.DataFrame,
    camera_ids: Sequence[str],
    weather_columns: Sequence[str],
    nonvisual_start: date = NONVISUAL_START,
    buffer_start: date = BUFFER_START,
) -> Tuple[List[SessionRecord], MinMaxScaler, int]:
    """
    Align all three streams to 1-minute timesteps per camera and session.

    Returns:
        records, fitted weather scaler, number of sessions skipped (no weather row)
    """
    weather_columns = list(weather_columns)

    # Normalize timestamps
    frames = frames.assign(timestamp=to_local(frames["timestamp"]))
    tweets = tweets.assign(created_at=to_local(tweets["created_at"]))
    weather = (
        weather.assign(date=pd.to_datetime(weather["date"]).dt.date)
        .drop_duplicates("date", keep="last")
        .set_index("date")
    )

    plan = _session_plan(nonvisual_start, buffer_start)

    # Weather scaler: training days only (prevents leakage)
    train_days = sorted(
        {day for day, _, _, split in plan if split == "train"} & set(weather.index)
    )
    if not train_days:
        raise ValueError("No weather rows found for any training day.")

    scaler = MinMaxScaler().fit(
        weather.loc[train_days, weather_columns].to_numpy(dtype=np.float32)
    )

    frames_by_camera = {c: g for c, g in frames.groupby("camera_id")}
    tweets_by_camera = {c: g for c, g in tweets.groupby("camera_id")}

    session_length = pd.Timedelta(minutes=SESSION_STEPS * STEP_MINUTES)
    records: List[SessionRecord] = []
    skipped = 0

    for camera_id in camera_ids:
        camera_frames = frames_by_camera.get(camera_id, frames.iloc[0:0])
        camera_tweets = tweets_by_camera.get(camera_id, tweets.iloc[0:0])

        for day, session, regime, split in plan:

            # Weather check
            if day not in weather.index:
                skipped += 1
                continue

            start = session_start(day, session)
            end = start + session_length

            # Visual: latest frame inside each minute
            frame_paths: List[Optional[str]] = [None] * SESSION_STEPS
            if regime == "visual" and len(camera_frames):
                sel = camera_frames[
                    (camera_frames["timestamp"] >= start)
                    & (camera_frames["timestamp"] < end)
                ].sort_values("timestamp")
                for k, path in zip(step_index(sel["timestamp"], start), sel["frame_path"]):
                    frame_paths[int(k)] = path

            # Text: mean embedding of posts created inside each minute
            text_steps: Dict[int, np.ndarray] = {}
            sel = camera_tweets[
                (camera_tweets["created_at"] >= start)
                & (camera_tweets["created_at"] < end)
            ]
            if len(sel):
                steps = step_index(sel["created_at"], start)
                embeddings = np.stack(
                    [np.asarray(e, dtype=np.float32) for e in sel["embedding"]]
                )
                for k in np.unique(steps):
                    text_steps[int(k)] = embeddings[steps == k].mean(axis=0)

            # Temporal: normalized daily weather
            weather_vector = scaler.transform(
                weather.loc[[day], weather_columns].to_numpy(dtype=np.float32)
            )[0]

            records.append(
                SessionRecord(
                    camera_id=camera_id,
                    day=day,
                    session=session,
                    regime=regime,
                    split=split,
                    frame_paths=frame_paths,
                    text_steps=text_steps,
                    weather=weather_vector,
                )
            )

    return records, scaler, skipped


def build_window_index(
    records: Sequence[SessionRecord],
    split: str,
    regime: Optional[str] = None,
) -> List[Tuple[int, int]]:
    """(record index, window start step) pairs for one split."""
    return [
        (i, start)
        for i, record in enumerate(records)
        if record.split == split and (regime is None or record.regime == regime)
        for start in window_starts()
    ]