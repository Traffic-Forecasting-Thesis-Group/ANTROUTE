"""
What each data source contributes to the train / val / test split.

The split is decided once, per session (and, with training_data.assign_day_splits, per whole
date). Every modality then follows it, because every model input is looked up *by the window's
own time*: a validation window gets that validation session's frames, tweets, daily weather,
live incidents and clock, and nothing from a training session. The road graph and the flood
hazard map are properties of the roads, not samples -- the same 1,400-odd nodes appear in every
split, the way the same city does -- so they are reported, not split.

split_table() makes that checkable: one row per modality, one column per split.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.data.alignment import SESSION_STEPS, STEP_MINUTES, session_start, window_starts
from src.data.node_context import EVENT_TTL_MINUTES, load_events
from src.data.training_data import load_frames_table, split_days_overlap

SPLITS = ("train", "val", "test")
TARGET = {"train": 0.70, "val": 0.15, "test": 0.15}


def _session_bounds(split: Dict[tuple, str]):
    """(split name, start, end) per session, naive Manila time."""
    for (day, session), name in sorted(split.items()):
        start = session_start(day, session)
        yield name, start, start + pd.Timedelta(minutes=SESSION_STEPS * STEP_MINUTES)


def _in_sessions(times: pd.Series, split: Dict[tuple, str], lead: timedelta = timedelta(0)) -> pd.Series:
    """Which split each time falls in ('' if none); `lead` also counts the time just before a
    session (an incident reported 30 minutes earlier is still in force at its start)."""
    out = pd.Series("", index=times.index, dtype=object)
    for name, start, end in _session_bounds(split):
        out[(times >= start - lead) & (times < end)] = name
    return out


def _tweet_times(raw_twitter_root: Optional[Path]) -> pd.Series:
    if raw_twitter_root is None or not Path(raw_twitter_root).exists():
        return pd.Series([], dtype="datetime64[ns]")
    created = []
    for path in sorted(Path(raw_twitter_root).rglob("tweets_*.json")):
        content = json.loads(path.read_text(encoding="utf-8"))
        tweets = content.get("data", content) if isinstance(content, dict) else content
        created.extend(t.get("createdAt") for t in tweets if isinstance(t, dict))
    stamps = pd.to_datetime(pd.Series(created), format="%a %b %d %H:%M:%S %z %Y", errors="coerce", utc=True)
    return stamps.dropna().dt.tz_convert("Asia/Manila").dt.tz_localize(None)


def split_table(
    split: Dict[tuple, str],
    frames_roots,
    lookup: Dict[str, int],
    weather_csv: Optional[Path] = None,
    raw_twitter_root: Optional[Path] = None,
    landmarks_csv: Optional[Path] = None,
    graph=None,
    spatial_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Rows: what each modality puts in each split. Columns: train / val / test / note."""
    rows = []

    def add(modality, measure, counts, note=""):
        rows.append({"modality": modality, "measure": measure,
                     **{s: counts.get(s, 0) for s in SPLITS}, "note": note})

    sessions = pd.Series(split)
    add("sessions", "sessions (date x AM/PM)", sessions.value_counts().to_dict())
    add("sessions", "dates", pd.Series({d: n for (d, _), n in split.items()}).groupby(level=0).first()
        .value_counts().to_dict(), "a date is never in two splits" if not split_days_overlap(split) else "OVERLAP")

    frames = load_frames_table(frames_roots)
    times = pd.to_datetime(frames["timestamp"], format="ISO8601")
    where = _in_sessions(times, split)
    add("CCTV / image", "frames", where[where != ""].value_counts().to_dict())
    labelled = frames["frame_path"].isin(lookup)
    add("CCTV / image", "labelled frames (training targets)", where[labelled & (where != "")].value_counts().to_dict())
    add("CCTV / image", "cameras with frames", {s: frames.loc[where == s, "camera_id"].nunique() for s in SPLITS})

    per_session = len(window_starts())
    add("temporal", "30-min windows (5-min stride)", {s: n * per_session for s, n in sessions.value_counts().items()},
        "clock + weekday computed from each window's own time")
    add("temporal", "AM / PM sessions", {s: f"{sum(1 for (d, x), v in split.items() if v == s and x == 'AM')} / "
                                           f"{sum(1 for (d, x), v in split.items() if v == s and x == 'PM')}"
                                        for s in SPLITS})

    if weather_csv is not None and Path(weather_csv).exists():
        weather = pd.read_csv(weather_csv, usecols=["date", "grid_cell"])
        weather["date"] = pd.to_datetime(weather["date"]).dt.date
        day_split = {d: v for (d, _), v in split.items()}
        weather["split"] = weather["date"].map(day_split)
        add("daily weather", "dates with weather", weather.dropna(subset=["split"]).groupby("split")["date"].nunique().to_dict(),
            "scaling fitted on train dates only")
        add("daily weather", "grid-cell x date rows", weather["split"].value_counts().to_dict())

    tweet_times = _tweet_times(raw_twitter_root)
    tweet_where = _in_sessions(tweet_times, split)
    add("event / text", "tweets posted in the split's sessions", tweet_where[tweet_where != ""].value_counts().to_dict())
    events = load_events(raw_twitter_root, landmarks_csv)
    if events:
        at = pd.Series([e.at.replace(tzinfo=None) for e in events])
        event_where = _in_sessions(at, split, lead=timedelta(minutes=EVENT_TTL_MINUTES))
        add("event / text", "MMDA incidents in force (placeable)", event_where[event_where != ""].value_counts().to_dict(),
            f"incl. reports up to {EVENT_TTL_MINUTES:.0f} min before a session")

    if graph is not None:
        nodes, edges = int(graph.n_nodes), int(graph.edge_index.shape[1])
        add("spatial / road network", "graph nodes / edges", {s: f"{nodes} / {edges}" for s in SPLITS},
            "static: the same roads in every split")
        if spatial_dir is not None:
            static = pd.read_csv(Path(spatial_dir) / "full_network_static_features.csv",
                                 usecols=["node_id", "flood_hazard_level"]).set_index("node_id")
            levels = static.reindex(np.asarray(graph.node_ids))["flood_hazard_level"]
            add("flood hazard", "nodes with hazard level 1-3", {s: int((levels > 0).sum()) for s in SPLITS},
                "static map; enters each window x that date's rain")
    return pd.DataFrame(rows)


def split_shares(split: Dict[tuple, str], weights: Optional[Dict[tuple, float]] = None) -> Dict[str, float]:
    """Share of sessions (or of `weights`, e.g. labelled frames) in each split."""
    total = {s: 0.0 for s in SPLITS}
    for key, name in split.items():
        total[name] += float((weights or {}).get(key, 1.0)) if weights else 1.0
    whole = sum(total.values()) or 1.0
    return {s: total[s] / whole for s in SPLITS}


def check_split(split: Dict[tuple, str], weights: Optional[Dict[tuple, float]] = None,
                tolerance: float = 0.10) -> list:
    """Problems with a split, [] when clean: a date in two splits, an empty split, or a share
    more than `tolerance` away from 70/15/15."""
    problems = [f"date {d} is in {', '.join(v)}" for d, v in split_days_overlap(split).items()]
    shares = split_shares(split, weights)
    for s in SPLITS:
        if not any(v == s for v in split.values()):
            problems.append(f"{s} is empty")
        elif abs(shares[s] - TARGET[s]) > tolerance:
            problems.append(f"{s} holds {shares[s]:.0%}, target {TARGET[s]:.0%}")
    return problems
