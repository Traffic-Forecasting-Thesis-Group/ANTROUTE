from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Tuple
import pandas as pd

# Two departures whose nearest recorded window starts differ by no more than this count
# as the same time slot (the decoder's window stride), so the day preference below can
# choose between them instead of a few minutes' difference deciding it.
SLOT_MINUTES = 5

# A departure further than this from every recorded window start is outside the recorded
# peak sessions: the nearest window is still used, but flagged as not covering it.
COVER_MINUTES = 30


def available_windows(risk_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(risk_csv)
    for column in ("window_start", "window_end"):
        if column not in frame.columns:
            frame[column] = pd.NA
    windows = frame[["window_start", "window_end"]].drop_duplicates().reset_index(drop=True)
    windows["window_start_dt"] = pd.to_datetime(windows["window_start"], errors="coerce")
    windows["window_end_dt"] = pd.to_datetime(windows["window_end"], errors="coerce")
    if windows["window_start_dt"].isna().all() and windows["window_end_dt"].isna().all():
        raise ValueError(f"{risk_csv} has no parseable window_start or window_end values")
    windows["_sort_key"] = windows["window_start_dt"].fillna(windows["window_end_dt"])
    windows = windows.sort_values("_sort_key").drop(columns="_sort_key").reset_index(drop=True)
    return windows


def resolve_departure_window(risk_csv: Path, depart_at: Optional[str]) -> Optional[str]:
    if depart_at is None:
        return None
    windows = available_windows(risk_csv)
    depart_dt = pd.to_datetime(depart_at)
    has_start = windows["window_start_dt"].notna()
    has_end = windows["window_end_dt"].notna()
    containing = windows[
        has_start
        & has_end
        & (windows["window_start_dt"] <= depart_dt)
        & (depart_dt < windows["window_end_dt"])
    ]
    if len(containing):
        row = containing.iloc[0]
    else:
        reference = windows["window_start_dt"].fillna(windows["window_end_dt"])
        distance = (reference - depart_dt).abs()
        row = windows.loc[distance.idxmin()]
        print(
            f"WARNING: no window covers {depart_at}; using the closest one instead ({row['window_start']} to {row['window_end']})."
        )
    for column in ("window_start", "window_end"):
        if pd.notna(row[column]):
            return str(row[column])
    raise ValueError(f"{risk_csv} has no usable window_start/window_end for the matched row")


def match_recorded_window(window_starts: Iterable[str], depart_at: datetime) -> Tuple[str, bool]:
    """
    The recorded window that stands in for a trip leaving at `depart_at` (naive local time).

    resolve_departure_window() matches on the full date, which only works for a date that
    was actually recorded -- any real departure today is months past the last session and
    would always land on the newest window, whatever the hour. Congestion repeats by time
    of day and weekday, so this matches on those instead: the window starting closest to
    the departure's time of day, preferring the same date, then the same weekday, then
    the most recent recorded day.

    Returns (window_start, covered). covered is False when the departure is outside every
    recorded session (e.g. midday -- only the AM and PM peaks were recorded); the closest
    window is still returned so a route can be planned, but the caller should say so.
    """
    starts = pd.Series(list(window_starts), dtype=object)
    parsed = pd.to_datetime(starts, errors="coerce")
    keep = parsed.notna()
    if not keep.any():
        raise ValueError("no parseable window starts to match a departure against")
    starts, parsed = starts[keep], parsed[keep]

    minute_of_day = parsed.dt.hour * 60 + parsed.dt.minute
    depart_minute = depart_at.hour * 60 + depart_at.minute
    distance = (minute_of_day - depart_minute).abs()
    best = int(distance.min())

    slot = distance <= best + SLOT_MINUTES
    day_rank = pd.Series(2, index=parsed.index)
    day_rank[parsed.dt.weekday == depart_at.weekday()] = 1
    day_rank[parsed.dt.date == depart_at.date()] = 0
    candidates = pd.DataFrame(
        {"start": starts, "day_rank": day_rank, "date": parsed.dt.normalize(), "distance": distance}
    )[slot]
    chosen = candidates.sort_values(["day_rank", "date", "distance"], ascending=[True, False, True]).iloc[0]
    return str(chosen["start"]), best <= COVER_MINUTES