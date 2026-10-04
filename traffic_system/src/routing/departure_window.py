from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd


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