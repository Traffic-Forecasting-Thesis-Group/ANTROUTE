from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("temporal_weather")

DATE_START = date(2025, 6, 1)
DATE_END = date(2026, 5, 31)
WEATHER_GRID_SIZE_DEG = 0.05
WEATHER_FEATURES = ["ws_temp_c", "ws_precip_mm", "ws_humidity_pct"]
WINDOW = 6


def get_api_key() -> str:
    key = os.environ.get("WEATHERSTACK_API_KEY")
    if not key:
        raise RuntimeError("WEATHERSTACK_API_KEY is not set.")
    return key


def build_grid_points(full_nodes_csv: Path) -> pd.DataFrame:
    full_nodes_df = pd.read_csv(full_nodes_csv)
    log.info(f"Loaded {len(full_nodes_df):,} nodes")

    full_nodes_df["grid_cell"] = (
        (full_nodes_df["lat"] / WEATHER_GRID_SIZE_DEG).round().astype(int).astype(str) + "_" +
        (full_nodes_df["lon"] / WEATHER_GRID_SIZE_DEG).round().astype(int).astype(str)
    )
    grid_points = full_nodes_df.groupby("grid_cell")[["lat", "lon"]].mean().reset_index()
    log.info(f"{len(grid_points)} weather grid points")
    return grid_points


def fetch_ws_historical_chunk(lat: float, lon: float, start: date, end: date, api_key: str) -> list[dict]:
    url = "https://api.weatherstack.com/historical"
    params = {
        "access_key": api_key, "query": f"{lat},{lon}",
        "historical_date_start": start.isoformat(), "historical_date_end": end.isoformat(),
        "hourly": 1, "interval": 24,
    }
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise ValueError(data["error"].get("info", "unknown WeatherStack error"))
    rows = []
    for day_str, day_data in data.get("historical", {}).items():
        hourly = day_data.get("hourly", [{}])
        h = hourly[0] if hourly else {}
        rows.append({
            "date": day_str,
            "ws_temp_c": h.get("temperature"),
            "ws_precip_mm": h.get("precip"),
            "ws_humidity_pct": h.get("humidity"),
        })
    return rows


def date_chunks(start: date, end: date, max_days: int = 60):
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=max_days - 1), end)
        yield chunk_start, chunk_end
        chunk_start = chunk_end + timedelta(days=1)


def fetch_or_load_historical(grid_points: pd.DataFrame, save_path: Path, api_key: str,
                              date_start: date, date_end: date, sleep_s: float = 0.5) -> pd.DataFrame:
    if save_path.exists():
        log.info(f"Found existing data at {save_path}. Skipping API calls.")
        df = pd.read_csv(save_path)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    total_days = (date_end - date_start).days + 1
    n_chunks = len(list(date_chunks(date_start, date_end)))
    log.info(f"{len(grid_points)} grid points x {total_days} days = "
             f"{len(grid_points) * total_days:,} quota units, "
             f"{n_chunks * len(grid_points)} total API calls")

    ws_rows = []
    for _, r in grid_points.iterrows():
        for chunk_start, chunk_end in date_chunks(date_start, date_end):
            try:
                chunk_rows = fetch_ws_historical_chunk(r["lat"], r["lon"], chunk_start, chunk_end, api_key)
                for row in chunk_rows:
                    row["grid_cell"] = r["grid_cell"]
                    ws_rows.append(row)
            except Exception as e:
                log.warning(f"WeatherStack failed for grid {r['grid_cell']} ({chunk_start} to {chunk_end}): {e}")
            time.sleep(sleep_s)

    df = pd.DataFrame(ws_rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    log.info(f"Fetched {len(df):,} (grid_cell, date) rows")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    log.info(f"Saved fetched data to {save_path}")
    return df


def zscore_normalize(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    norm_df = df.copy()
    for col in columns:
        mean, std = df[col].mean(), df[col].std()
        std = std if std > 0 else 1.0
        norm_df[col] = (df[col] - mean) / std
    return norm_df


def build_sequences(df: pd.DataFrame, group_col: str, time_col: str,
                     feature_columns: list[str], window_size: int):
    sequences, labels = [], []
    for name, group in df.groupby(group_col):
        group = group.sort_values(time_col)
        values = group[feature_columns].to_numpy()
        if len(values) < window_size:
            continue
        for i in range(len(values) - window_size + 1):
            sequences.append(values[i:i + window_size])
            labels.append(name)
    if not sequences:
        return np.empty((0, window_size, len(feature_columns))), []
    return np.stack(sequences), labels


def run(data_dir: Path, date_start: date = DATE_START, date_end: date = DATE_END,
        window: int = WINDOW) -> None:
    spatial_proc_dir = data_dir / "processed" / "spatial"
    temporal_proc_dir = data_dir / "processed" / "temporal"
    temporal_proc_dir.mkdir(parents=True, exist_ok=True)

    full_nodes_csv = spatial_proc_dir / "full_network_static_features.csv"
    if not full_nodes_csv.exists():
        raise FileNotFoundError(f"{full_nodes_csv} not found — run spatial_topology.py first.")

    api_key = get_api_key()
    grid_points = build_grid_points(full_nodes_csv)

    save_path = temporal_proc_dir / "weatherstack_historical.csv"
    ws_historical_df = fetch_or_load_historical(grid_points, save_path, api_key, date_start, date_end)

    ws_norm_df = zscore_normalize(ws_historical_df.dropna(subset=WEATHER_FEATURES), WEATHER_FEATURES)
    sequences, labels = build_sequences(ws_norm_df, "grid_cell", "date", WEATHER_FEATURES, window)

    log.info(f"Sequence Norm output shape: {sequences.shape}")
    np.save(temporal_proc_dir / "sequence_norm_weather.npy", sequences)
    pd.Series(labels).to_csv(temporal_proc_dir / "sequence_norm_labels.csv", index=False, header=["grid_cell"])
    log.info(f"Saved sequence_norm_weather.npy and sequence_norm_labels.csv to {temporal_proc_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch WeatherStack history and build normalized sequences.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--date-start", type=lambda s: date.fromisoformat(s), default=DATE_START)
    parser.add_argument("--date-end", type=lambda s: date.fromisoformat(s), default=DATE_END)
    parser.add_argument("--window", type=int, default=WINDOW)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(data_dir=args.data_dir, date_start=args.date_start, date_end=args.date_end, window=args.window)