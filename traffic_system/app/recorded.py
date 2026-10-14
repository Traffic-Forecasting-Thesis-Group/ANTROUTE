"""
The recorded inputs behind a route, for the app's "Inputs used" panel, the per-route fields and
the simulation mode.

There is no live feed: every input is read from what was recorded in one window of risk_edges.csv
(30 minutes of one AM or PM session), which is why the panel is labelled "recorded":

    cameras         each camera on the map: the model's risk on its edges, the human congestion
                    label there (Light / Medium / Heavy, from weak_target) and the mean YOLO
                    vehicle count over the window
    incidents       the MMDA alerts in force at the window's moment (event_layer.active_at)
    weather         WeatherStack hourly weather of that day at that hour (city mean; along a
                    route, the grid cells the route crosses)
    flood           Project NOAH flood hazard level (0-3) of the roads
    matched window  which recorded window, day, session and split these all belong to

route_inputs() gives the same fields for one route: the cameras on it, the incidents on it, the
weather and flood along it, and the vehicle counts at its cameras; risk sources are reported by
risk_routing.risk_sources beside them.

Simulation mode picks a recorded day and session instead of matching a departure time
(recorded_windows(), risk_routing.plan_real_routes(window=...)). May 25 is held out: the model
never saw it in training or validation, so it is flagged "held out" (split "test").
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from app import risk_routing as rr
from src.data.alignment import WINDOW_STEPS
from src.data.node_context import grid_cell
from src.routing.baseline_router import camera_flow, camera_observations, load_vehicle_counts

LABELS = ("Light", "Medium", "Heavy")        # weak_target 0 / 0.5 / 1, the human congestion label
MAX_FLOOD_LEVEL = 3
ON_ROUTE_M = 200.0      # a camera or incident within this distance of the route counts as on it
SPLIT_NOTE = {
    "train": "the model trained on this session",
    "val": "used to validate the model during training",
    "test": "held out: the model never saw this session in training or validation",
    "infer": "not labelled; scored by the model but not used to train or test it",
}


def _label(score: Optional[float]) -> Optional[str]:
    if score is None or not math.isfinite(score):
        return None
    return LABELS[min(2, int(round(score * 2)))]


def _number(value, digits: int = 2) -> Optional[float]:
    return None if value is None or not math.isfinite(float(value)) else round(float(value), digits)


# ---------------------------------------------------------------- the recorded sessions
def _split_of(frame: pd.DataFrame) -> str:
    return str(frame["split"].iloc[0]) if "split" in frame.columns and len(frame) else "infer"


def _session_label(day: str, session: str) -> str:
    return f"{datetime.fromisoformat(day):%a %b} {datetime.fromisoformat(day).day}, {session}"


def recorded_windows() -> List[dict]:
    """
    Every recorded session (day + AM/PM) with its windows, for the simulation picker. Sessions are
    in date order; `held_out` is True for the test split (May 25).
    """
    net = rr._network()
    sessions: Dict[tuple, dict] = {}
    for window in sorted(net.risk_by_window):
        frame = net.risk_by_window[window]
        start = datetime.fromisoformat(window)
        day = f"{start:%Y-%m-%d}"
        session = "AM" if start.hour < 12 else "PM"
        entry = sessions.setdefault((day, session), {
            "day": day, "session": session, "label": _session_label(day, session),
            "split": _split_of(frame), "windows": [],
            "labelled_edges": 0,
        })
        entry["windows"].append({"start": window, "end": (start + timedelta(minutes=WINDOW_STEPS)).isoformat()})
        if "weak_target" in frame.columns:
            entry["labelled_edges"] += int(frame["weak_target"].notna().sum())
    out = []
    for (day, session), entry in sorted(sessions.items()):
        entry["held_out"] = entry["split"] == "test"
        entry["note"] = SPLIT_NOTE.get(entry["split"], "")
        out.append(entry)
    return out


def window_description(window: str) -> dict:
    """Which recorded window this is: day, session, clock span, split."""
    net = rr._network()
    frame = net.risk_by_window[window]
    start = datetime.fromisoformat(window)
    end = start + timedelta(minutes=WINDOW_STEPS)
    split = _split_of(frame)
    return {
        "window": window,
        "window_end": end.isoformat() if end else None,
        "day": f"{start:%Y-%m-%d}",
        "session": "AM" if start.hour < 12 else "PM",
        "label": f"{start:%a %b} {start.day}, {rr._clock(start)}" + (f" - {rr._clock(end)}" if end else ""),
        "split": split,
        "held_out": split == "test",
        "split_note": SPLIT_NOTE.get(split, ""),
    }


# ---------------------------------------------------------------- weather, flood, vehicle counts
@lru_cache(maxsize=1)
def _weather_table() -> Optional[pd.DataFrame]:
    """WeatherStack hourly rows indexed by (date, hour, grid_cell); None when the file is absent."""
    path = Path(rr.WEATHER_CSV) if rr.WEATHER_CSV else None
    if path is None or not path.exists():
        return None
    frame = pd.read_csv(path, usecols=lambda c: c in {"date", "hour", "grid_cell", "ws_temp_c", "ws_precip_mm", "ws_humidity_pct"})
    if "hour" not in frame.columns:     # the daily file: one row per day and cell
        frame["hour"] = -1
    return frame.set_index(["date", "hour", "grid_cell"]).sort_index()


def _weather_rows(when: datetime, cells: Optional[Sequence[str]]) -> Optional[pd.DataFrame]:
    table = _weather_table()
    if table is None:
        return None
    day = f"{when:%Y-%m-%d}"
    for hour in (when.hour, -1):             # hourly rows, else the daily file's
        try:
            rows = table.xs((day, hour), level=("date", "hour"))
        except KeyError:
            continue
        if cells is not None:
            rows = rows[rows.index.isin(list(cells))]
        return rows if len(rows) else None
    return None


def _weather_summary(when: Optional[datetime], cells: Optional[Sequence[str]], where: str) -> dict:
    rows = _weather_rows(when, cells) if when else None
    if rows is None:
        return {"available": False, "note": "no weather recorded for this day"}
    return {
        "available": True,
        "where": where,
        "hour": f"{when:%Y-%m-%d} {when.hour:02d}:00",
        "temp_c": _number(rows["ws_temp_c"].mean(), 1),
        "precip_mm": _number(rows["ws_precip_mm"].mean(), 2),
        "precip_mm_max": _number(rows["ws_precip_mm"].max(), 2),
        "humidity_pct": _number(rows["ws_humidity_pct"].mean(), 0),
        "cells": int(len(rows)),
        "source": "WeatherStack, hourly, that day",
    }


@lru_cache(maxsize=1)
def _node_flood_and_cell() -> Dict[int, tuple]:
    """node id -> (flood hazard level 0-3, WeatherStack grid cell)."""
    net = rr._network()
    nodes = pd.read_csv(rr.SPATIAL_DIR / "full_network_static_features.csv",
                        usecols=["node_id", "lat", "lon", "flood_hazard_level"])
    nodes = nodes[nodes["node_id"].isin(set(int(n) for n in net.graph.node_ids))]
    cells = grid_cell(nodes["lat"].to_numpy(), nodes["lon"].to_numpy())
    flood = nodes["flood_hazard_level"].fillna(0).clip(0, MAX_FLOOD_LEVEL).to_numpy()
    return {int(n): (int(f), str(c)) for n, f, c in zip(nodes["node_id"], flood, cells)}


@lru_cache(maxsize=1)
def _vehicle_counts() -> Optional[pd.DataFrame]:
    net = rr._network()
    folder = Path(rr.settings.vehicle_counts_dir)
    if not folder.is_absolute():
        folder = rr.REPO_ROOT / folder
    paths = sorted(folder.glob("*.csv")) if folder.exists() else []
    if not paths:
        return None
    return load_vehicle_counts(paths, net.graph.camera_nodes, rr.REPO_ROOT / "configs/camera_nodes.csv")


def _camera_table(window: str) -> List[dict]:
    """Each camera: position, model risk, human label and vehicle count in this window."""
    net = rr._network()
    frame = net.risk_by_window[window]
    node_ids = np.asarray(net.graph.node_ids)
    indices = list(net.graph.camera_nodes.values())
    risk = camera_observations(frame, node_ids, indices, column="risk")    # edges at exactly one camera
    observed = camera_observations(frame[frame["weak_target"].notna()], node_ids, indices) \
        if "weak_target" in frame.columns else {}
    counts = _vehicle_counts()
    start = pd.Timestamp(window)
    flow = camera_flow(counts, start, start + pd.Timedelta(minutes=WINDOW_STEPS)) if counts is not None else {}
    cameras = []
    for label, index in sorted(net.graph.camera_nodes.items()):
        node = int(node_ids[index])
        lat, lon = net.node_coords[node]
        score = observed.get(index)
        cameras.append({
            "camera": label, "node_id": node, "lat": lat, "lon": lon,
            "model_risk": _number(risk.get(index), 3),
            "human_score": _number(score, 2), "human_label": _label(score),
            "vehicle_count": _number(flow.get(index), 1),
        })
    return cameras


def _incident_table(window: str) -> List[dict]:
    net = rr._network()
    when = rr._as_of(window)
    node_ids = np.asarray(net.graph.node_ids)
    out = []
    for event, strength in rr.live_events(net, window):
        index = net.event_intersections.get(event.intersection)
        coords = net.node_coords.get(int(node_ids[index])) if index is not None else None
        out.append({
            "location": event.landmark or event.location_text, "intersection": event.intersection,
            "kind": event.kind, "lanes": event.lanes_occupied,
            "strength": round(float(strength), 2),
            "reported_at": event.at.isoformat(timespec="minutes"),
            "minutes_ago": int(round((when - event.at).total_seconds() / 60)) if when else None,
            "node_id": int(node_ids[index]) if index is not None else None,
            "lat": coords[0] if coords else None, "lon": coords[1] if coords else None,
        })
    return out


def inputs_for_window(window: str) -> dict:
    """The panel: every recorded input of one window, over the whole map."""
    net = rr._network()
    if window not in net.risk_by_window:
        raise KeyError(window)
    flood = _node_flood_and_cell()
    levels = np.array([level for level, _ in flood.values()]) if flood else np.zeros(0)
    when = rr._as_of(window)
    return {
        "recorded": True,
        "matched_window": window_description(window),
        "cameras": _camera_table(window),
        "incidents": _incident_table(window),
        "weather": _weather_summary(when, None, "city mean over WeatherStack grid cells"),
        "flood": {
            "source": "Project NOAH flood hazard, 0 (none) to 3 (high)",
            "mean_level": _number(levels.mean(), 2) if levels.size else None,
            "share_of_roads_flood_prone": _number((levels > 0).mean(), 3) if levels.size else None,
        },
        "vehicle_counts_available": _vehicle_counts() is not None,
    }


# ---------------------------------------------------------------- one route
def _near_route(net, nodes: Sequence[int], things: List[dict]) -> List[dict]:
    """The cameras or incidents within ON_ROUTE_M of any node of the route, each with its distance."""
    if not things:
        return []
    coords = np.array([net.node_coords[int(n)] for n in nodes if int(n) in net.node_coords])
    if not coords.size:
        return []
    out = []
    for thing in things:
        if thing.get("lat") is None:
            continue
        dlat = np.radians(coords[:, 0] - thing["lat"])
        dlon = np.radians(coords[:, 1] - thing["lon"]) * math.cos(math.radians(thing["lat"]))
        metres = float(6371000.0 * np.hypot(dlat, dlon).min())
        if metres <= ON_ROUTE_M:
            out.append({**thing, "distance_m": round(metres)})
    return sorted(out, key=lambda t: t["distance_m"])


def route_inputs(net, wg, nodes: Sequence[int]) -> Optional[dict]:
    """
    The recorded inputs along one route: cameras on it (with the model's risk, the human label and
    the vehicle count at each), incidents on it, the weather and flood along it. None without a
    recorded window.
    """
    window = wg.window
    if window is None or window not in net.risk_by_window:
        return None
    cameras = _near_route(net, nodes, _camera_table(window))
    incidents = _near_route(net, nodes, _incident_table(window))
    flood = _node_flood_and_cell()
    along = [flood[int(n)] for n in nodes if int(n) in flood]
    cells = sorted({cell for _, cell in along})
    levels = np.array([level for level, _ in along]) if along else np.zeros(0)
    counts = [c["vehicle_count"] for c in cameras if c["vehicle_count"] is not None]
    return {
        "recorded": True,
        "window": window,
        "cameras": cameras,
        "incidents": incidents,
        "weather": _weather_summary(rr._as_of(window), cells, f"{len(cells)} grid cell(s) the route crosses"),
        "flood": {
            "max_level": int(levels.max()) if levels.size else 0,
            "share_of_route_flood_prone": _number((levels > 0).mean(), 3) if levels.size else 0.0,
        },
        "vehicle_count_mean": _number(np.mean(counts), 1) if counts else None,
    }
