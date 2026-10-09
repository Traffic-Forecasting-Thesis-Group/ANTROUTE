"""
Builds the CNN + LSTM training data from the three real sources:

  frames   <frames_root>/manifest.csv        (extract_frames.py)
  labels   <frames_root>/auto_labels.csv + labels.csv   (autolabel + label_viewer)
  text     data/processed/embeddings.pt + data/raw/twitter (createdAt per tweet)
  weather  data/processed/temporal/weatherstack_historical.csv

and hands them to alignment.build_sessions. Only the visual regime is built
(sessions that have CCTV frames), since the congestion labels come from frames.
"""

import collections
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from src.data.tweet_geocode import (
    cameras_by_intersection,
    load_camera_map,
    load_label_locations,
    place,
)
from src.routing.event_layer import load_landmarks

from src.data.alignment import (
    BUFFER_START,
    LEGACY_TEXT_ACTIVE_MINUTES,
    SESSION_STEPS,
    window_starts,
    WINDOW_STEPS,
    LOCAL_TZ,
    SessionRecord,
    build_sessions,
)
from src.data.dataset import ANTROUTEWindowDataset
from src.vision.label_store import load_frames
from src.vision.timeline import corrected_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
LANDMARKS_CSV = REPO_ROOT / "configs" / "event_landmarks.csv"

WEATHER_COLUMNS = ["ws_temp_c", "ws_precip_mm", "ws_humidity_pct"]
CLASS_TO_IDX = {"Light": 0, "Medium": 1, "Heavy": 2}
IGNORE_INDEX = -100


def frame_key(frames_root: Path, relative_path: str) -> str:
    return str(Path(frames_root) / relative_path)


def as_roots(frames_roots) -> List[Path]:
    """One frames folder or several (e.g. notebook A's, notebook B's and the pilot's)."""
    if isinstance(frames_roots, (str, Path)):
        return [Path(frames_roots)]
    return [Path(r) for r in frames_roots]


def load_frames_table(frames_roots, allow_estimated: bool = False) -> pd.DataFrame:
    """manifest.csv of each folder -> camera_id, timestamp, frame_path (absolute).

    Frames whose segment start is only a lower bound (an earlier segment failed) are
    dropped by default because their timestamps can be wrong. The same frame in two folders
    (same camera and time) is kept once, from the folder listed first.
    """
    tables = []
    for root in as_roots(frames_roots):
        df = corrected_manifest(root)
        if not allow_estimated and "start_estimated" in df:
            df = df[~df["start_estimated"].astype(str).eq("True")]
        df = df[["camera_id", "timestamp", "frame_path"]].copy()
        df["frame_path"] = [frame_key(root, p) for p in df["frame_path"]]
        tables.append(df)
    merged = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=["camera_id", "timestamp", "frame_path"])
    return merged.drop_duplicates(["camera_id", "timestamp"], keep="first").reset_index(drop=True)


def load_weather_daily(csv_path: Path) -> pd.DataFrame:
    """One row per date: mean over that date's grid cells (no per-camera geocoding exists)."""
    df = pd.read_csv(csv_path, usecols=["date"] + WEATHER_COLUMNS)
    return df.groupby("date", as_index=False)[WEATHER_COLUMNS].mean()


def load_tweets_table(
    raw_twitter_root: Path,
    embeddings_path: Path,
    camera_ids: Sequence[str],
    days: Optional[Sequence] = None,
    camera_map: Optional[Dict[str, str]] = None,
    landmarks: Optional[Dict[str, str]] = None,
    locations: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    created_at + embedding per tweet, at the camera the tweet is about.

    Row i of embeddings.pt is the i-th tweet in sorted(rglob('tweets_*.json')) order, exactly
    as TwitterTrafficDataset enumerated them, so createdAt is recovered by replaying that walk.
    That order is load-bearing: the walk must not be deduplicated or filtered, or every
    embedding after the first dropped tweet belongs to the wrong post.

    Placement is src/data/tweet_geocode.py. A tweet that names no camera's intersection is
    dropped: the cameras are the only places with visual ground truth, so text elsewhere has
    nothing to inform. This previously assigned every tweet to every camera, which made the
    text feature identical across cameras and so useless to the model.

    `locations` maps tweet id -> a hand-read location phrase (from
    data/labels/tweet_training_window.csv) for the posts no rule can place.
    """
    created, texts, ids = [], [], []
    for path in sorted(Path(raw_twitter_root).rglob("tweets_*.json")):
        with path.open(encoding="utf-8") as f:
            content = json.load(f)
        tweets = content.get("data", content) if isinstance(content, dict) else content
        for tweet in tweets:
            is_dict = isinstance(tweet, dict)
            created.append(tweet.get("createdAt") if is_dict else None)
            texts.append((tweet.get("text") or tweet.get("full_text") or "") if is_dict else "")
            ids.append(str(tweet.get("id") or "") if is_dict else "")

    payload = torch.load(embeddings_path, map_location="cpu")
    embeddings = (payload["embeddings"] if isinstance(payload, dict) else payload).float().numpy()
    if len(created) != len(embeddings):
        raise ValueError(
            f"{embeddings_path} has {len(embeddings)} rows but {raw_twitter_root} now contains "
            f"{len(created)} tweets. embeddings.pt is stale; regenerate it with main.py."
        )

    stamps = pd.to_datetime(pd.Series(created), format="%a %b %d %H:%M:%S %z %Y",
                            errors="coerce", utc=True)
    keep = stamps.notna().to_numpy().copy()
    if days is not None:
        local_dates = stamps.dt.tz_convert(LOCAL_TZ).dt.date
        keep &= local_dates.isin(set(days)).to_numpy()
    if not keep.any() or not len(camera_ids):
        return empty_tweets()

    kept = np.flatnonzero(keep)
    kept_texts = [texts[i] for i in kept]
    # Default to the hand-labelled sheet; pass locations={} to train without it.
    if locations is None:
        locations = load_label_locations()
    overrides = [locations.get(ids[i]) for i in kept]

    by_intersection = cameras_by_intersection(
        camera_ids, load_camera_map() if camera_map is None else camera_map
    )
    cameras_per_tweet = place(
        kept_texts,
        load_landmarks(LANDMARKS_CSV) if landmarks is None else landmarks,
        by_intersection,
        locations=overrides,
    )

    rows_created, rows_embedding, rows_camera = [], [], []
    for position, cameras in zip(kept, cameras_per_tweet):
        for camera_id in cameras:
            rows_created.append(stamps.iloc[position])
            rows_embedding.append(embeddings[position])
            rows_camera.append(camera_id)
    if not rows_camera:
        return empty_tweets()
    return pd.DataFrame({
        "camera_id": rows_camera,
        "created_at": pd.to_datetime(pd.Series(rows_created), utc=True),
        "embedding": rows_embedding,
    })


def empty_tweets() -> pd.DataFrame:
    return pd.DataFrame({"camera_id": pd.Series(dtype=str),
                         "created_at": pd.to_datetime(pd.Series(dtype="datetime64[ns]")),
                         "embedding": pd.Series(dtype=object)})


def session_of(timestamp) -> tuple:
    """(date, 'AM' | 'PM') of a frame time: sessions start at 07:00 and 17:00."""
    ts = pd.Timestamp(timestamp)
    return ts.date(), ("AM" if ts.hour < 12 else "PM")


def labelled_sessions(frames_roots, lookup: Dict[str, int], min_labels: int = 1) -> List[tuple]:
    """Sessions (date, AM/PM) with at least `min_labels` labelled frames, in chronological order.
    A session with only a handful of labels would make a meaningless validation or test set."""
    frames = load_frames_table(frames_roots)
    times = pd.to_datetime(frames.loc[frames["frame_path"].isin(lookup), "timestamp"], format="ISO8601")
    counts = collections.Counter(session_of(t) for t in times)
    return sorted(s for s, n in counts.items() if n >= min_labels)


def assign_session_splits(sessions: Sequence[tuple], ratios=(0.70, 0.15, 0.15)) -> Dict[tuple, str]:
    """Chronological train / val / test split by WHOLE sessions (never inside one, so a test frame is
    never a minute away from a training frame). Fewer than 3 sessions cannot fill all three sets:
    1 session -> train, 2 -> train + val."""
    ordered = sorted(set(sessions))
    n = len(ordered)
    if n < 3:
        n_val, n_test = (1 if n == 2 else 0), 0
    else:
        n_val, n_test = max(1, round(n * ratios[1])), max(1, round(n * ratios[2]))
        while n - n_val - n_test < 1:
            n_val, n_test = max(1, n_val - 1), max(1, n_test - 1)
    n_train = n - n_val - n_test
    return {s: ("train" if i < n_train else "val" if i < n_train + n_val else "test")
            for i, s in enumerate(ordered)}


def labelled_session_counts(frames_roots, lookup: Dict[str, int], min_labels: int = 1) -> Dict[tuple, int]:
    """(date, AM/PM) -> labelled frames, for sessions with at least `min_labels` of them."""
    frames = load_frames_table(frames_roots)
    times = pd.to_datetime(frames.loc[frames["frame_path"].isin(lookup), "timestamp"], format="ISO8601")
    counts = collections.Counter(session_of(t) for t in times)
    return {s: n for s, n in sorted(counts.items()) if n >= min_labels}


def assign_day_splits(sessions: Sequence[tuple], weights: Optional[Dict[tuple, float]] = None,
                      ratios=(0.70, 0.15, 0.15)) -> Dict[tuple, str]:
    """
    Chronological train / val / test split by WHOLE DAYS: both sessions of a date always land
    in the same split.

    assign_session_splits can put a date's AM session in train and its PM session in val. Every
    modality keyed by the date then crosses the boundary -- the daily weather is one value per
    date, and a morning incident is still in force in the afternoon's context -- so a held-out
    session would share inputs with a training one. Splitting by date closes that.

    Days are indivisible, so the shares cannot always hit 70/15/15 exactly; the two cut points
    are the ones whose shares of `weights` (labelled frames per session, or one per session when
    None) come closest to `ratios`, with at least one day in each split. Fewer than 3 days cannot
    fill all three: 1 day -> train, 2 -> train + val.
    """
    weights = weights or {}
    by_day: Dict = collections.OrderedDict()
    for s in sorted(set(sessions)):
        by_day.setdefault(s[0], []).append(s)
    days = list(by_day)
    load = [sum(float(weights.get(s, 1.0)) for s in by_day[d]) for d in days]
    total = sum(load) or 1.0
    n = len(days)
    if n < 3:
        name_of = {d: ("train" if i == 0 else "val") for i, d in enumerate(days)}
    else:
        best = None
        for i in range(1, n - 1):              # train = days[:i]
            for j in range(i + 1, n):          # val = days[i:j], test = days[j:]
                shares = (sum(load[:i]) / total, sum(load[i:j]) / total, sum(load[j:]) / total)
                miss = sum(abs(a - b) for a, b in zip(shares, ratios))
                if best is None or miss < best[0] - 1e-12:
                    best = (miss, i, j)
        _, i, j = best
        name_of = {d: ("train" if k < i else "val" if k < j else "test") for k, d in enumerate(days)}
    return {s: name_of[s[0]] for d in days for s in by_day[d]}


def split_days_overlap(split: Dict[tuple, str]) -> Dict:
    """Dates that appear in more than one split ({} when the split is clean)."""
    seen: Dict = collections.defaultdict(set)
    for (day, _), name in split.items():
        seen[day].add(name)
    return {day: sorted(names) for day, names in seen.items() if len(names) > 1}


def save_split(split: Dict[tuple, str], path: Path) -> None:
    """The split as {"YYYY-MM-DD|AM": "train", ...}, the format checkpoints already store."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({f"{d.isoformat()}|{s}": v for (d, s), v in sorted(split.items())}, indent=2),
                    encoding="utf-8")


def load_split(path: Path) -> Dict[tuple, str]:
    from datetime import date

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for key, value in raw.items():
        day, session = key.split("|")
        if value not in ("train", "val", "test"):
            raise ValueError(f"{path}: {key} has split {value!r}; expected train, val or test")
        out[(date.fromisoformat(day), session)] = value
    return out


def describe_split(split: Dict[tuple, str]) -> str:
    lines = []
    for name in ("train", "val", "test"):
        members = ", ".join(f"{d:%b %d} {ses}" for (d, ses), s in sorted(split.items()) if s == name)
        lines.append(f"  {name:5} ({sum(s == name for s in split.values())}): {members or '-'}")
    return "\n".join(lines)


def build_training_records(
    frames_root,
    weather_csv: Path,
    raw_twitter_root: Optional[Path] = None,
    embeddings_path: Optional[Path] = None,
    visual_split: Optional[Dict[tuple, str]] = None,
    camera_map: Optional[Dict[str, str]] = None,
    locations: Optional[Dict[str, str]] = None,
    text_active_minutes: int = LEGACY_TEXT_ACTIVE_MINUTES,
):
    """
    Sessions for every camera on the visual-regime days. Text is skipped if no paths are given.
    `text_active_minutes`: how long a post stays current (alignment.ALERT_ACTIVE_MINUTES).

    `camera_map` (camera id -> intersection) and `locations` (tweet id -> hand-read location)
    steer where a tweet lands; both default to the project's own configs. See
    src/data/tweet_geocode.py.
    """
    frames = load_frames_table(frames_root)
    if frames.empty:
        raise ValueError("No usable frames in manifest.csv (all missing or start_estimated).")
    camera_ids = sorted(frames["camera_id"].unique())
    days = sorted(pd.to_datetime(frames["timestamp"], format="ISO8601").dt.date.unique())

    if raw_twitter_root is not None and embeddings_path is not None:
        tweets = load_tweets_table(raw_twitter_root, embeddings_path, camera_ids, days,
                                   camera_map=camera_map, locations=locations)
    else:
        tweets = empty_tweets()

    # nonvisual_start == buffer_start makes the non-visual regime empty
    return build_sessions(
        frames, tweets, load_weather_daily(weather_csv), camera_ids, WEATHER_COLUMNS,
        nonvisual_start=BUFFER_START, buffer_start=BUFFER_START, visual_split=visual_split,
        text_active_minutes=text_active_minutes,
    )


def load_label_lookup(frames_roots, human_only: bool = False) -> Dict[str, int]:
    """absolute frame path -> class index (human label wins over auto label).

    human_only keeps just the frames a person reviewed: the automatic labels are relative to each
    camera and agree with human judgement far less often than they look like they should."""
    lookup: Dict[str, int] = {}
    for root in as_roots(frames_roots):
        df = load_frames(root)
        if human_only:
            df = df[df["source"] == "human"]
        for p, label in zip(df["frame_path"], df["label"]):
            lookup.setdefault(frame_key(root, p), CLASS_TO_IDX[label])
    return lookup


class TargetFn:
    """Per-step class index for a window; IGNORE_INDEX where the step has no labelled frame.
    A class rather than a closure so datasets can be sent to DataLoader worker processes (Windows)."""

    def __init__(self, lookup: Dict[str, int]):
        self.lookup = lookup

    def __call__(self, record: SessionRecord, start: int) -> torch.Tensor:
        target = torch.full((WINDOW_STEPS,), IGNORE_INDEX, dtype=torch.long)
        for j in range(WINDOW_STEPS):
            path = record.frame_paths[start + j]
            if path is not None and path in self.lookup:
                target[j] = self.lookup[path]
        return target


def make_target_fn(lookup: Dict[str, int]) -> Callable[[SessionRecord, int], torch.Tensor]:
    return TargetFn(lookup)


TIME_FEATURES = 2   # position within the 2-hour session (0..1) and PM-session flag


def append_time_features(item: dict, record: SessionRecord, start: int) -> dict:
    """Add how far into the session each step is, and the PM flag, as two columns of `temporal`."""
    steps = (start + torch.arange(WINDOW_STEPS, dtype=torch.float32)) / (SESSION_STEPS - 1)
    pm = torch.full_like(steps, 1.0 if record.session == "PM" else 0.0)
    item["temporal"] = torch.cat([item["temporal"], torch.stack([steps, pm], dim=1)], dim=1)
    return item


class LabeledWindowDataset(ANTROUTEWindowDataset):
    """ANTROUTEWindowDataset restricted to visual windows containing at least one labelled frame.

    time_features=True appends two columns to `temporal`: how far into the session each step is
    (congestion builds through the peak) and whether it is the PM session (AM peaks are lighter)."""

    def __init__(self, records: Sequence[SessionRecord], split: str, lookup: Dict[str, int],
                 image_size: int = 224, time_features: bool = False):
        super().__init__(records, split, regime="visual", image_size=image_size,
                         target_fn=make_target_fn(lookup))
        self.time_features = time_features
        self.temporal_dim = (len(records[0].weather) if len(records) else 0) + (TIME_FEATURES if time_features else 0)
        self.index = [
            (i, s) for i, s in self.index
            if any(records[i].frame_paths[s + j] in lookup for j in range(WINDOW_STEPS)
                   if records[i].frame_paths[s + j] is not None)
        ]

    def __getitem__(self, i: int) -> dict:
        item = super().__getitem__(i)
        if self.time_features:
            record_index, start = self.index[i]
            append_time_features(item, self.records[record_index], start)
        return item


class InferenceWindowDataset(ANTROUTEWindowDataset):
    """Every 30-step window that has at least one frame, from every session (labels are not needed).
    With a `lookup`, each item also carries the human target so predictions can be scored."""

    def __init__(self, records: Sequence[SessionRecord], image_size: int = 224, time_features: bool = False,
                 lookup: Optional[Dict[str, int]] = None):
        super().__init__(records, "__none__", regime="visual", image_size=image_size,
                         target_fn=make_target_fn(lookup) if lookup else None)
        self.index = [
            (i, s) for i, r in enumerate(records) if r.regime == "visual" for s in window_starts()
            if any(p is not None for p in r.frame_paths[s:s + WINDOW_STEPS])
        ]
        self.time_features = time_features
        self.temporal_dim = (len(records[0].weather) if len(records) else 0) + (TIME_FEATURES if time_features else 0)

    def __getitem__(self, i: int) -> dict:
        item = super().__getitem__(i)
        if self.time_features:
            record_index, start = self.index[i]
            append_time_features(item, self.records[record_index], start)
        return item
