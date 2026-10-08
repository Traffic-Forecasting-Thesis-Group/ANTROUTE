"""
Per-node context for the Congestion Risk Score: what every road node -- with or without a
camera -- is known to face at each minute of a window.

Before this, the only inputs to the RADR STGNN were the CNN+LSTM features of the camera
nodes; every other node got one shared learned placeholder. Weather, time and text reached
the graph only through the cameras, and a 2-layer GCN carries that two hops, so ~89% of the
scored edges had the same risk in every window. This module gives each node its own
non-visual inputs, at the resolution the data actually has:

  weather   daily, per 0.05-degree WeatherStack grid cell (the node's own cell): temperature,
            precipitation, humidity, min-max scaled on the training days. Daily, because that
            is what weatherstack_historical.csv holds -- nothing here pretends to be hourly.
  flood     Project NOAH 5-year hazard level as an ordinal (level / 3), and its interaction
            with that day's rainfall in the node's cell (hazard only matters when it rains).
            Ordinal rather than one-hot: the camera nodes, the only supervised ones, sit on
            levels 0 and 1, so a per-level embedding would leave levels 2-3 untrained.
  events    MMDA incident alerts (event_layer.parse_alerts) placed at their intersection,
            decaying linearly over EVENT_TTL_MINUTES: strongest live impact at the node and
            the number of live incidents there, per minute.
  temporal  the real clock: minute of day and day of week, each as sin/cos.
  spatial   from the road graph itself: in/out degree, mean and max free-flow speed of the
            node's roads (speed limits, so a proxy for road class), whether it has a camera,
            and its hop distance to the nearest camera (how far observed traffic has to travel).

GROUPS names the column blocks so a whole source can be dropped for ablation (its columns are
zeroed), and so verify_crs_sources.py can perturb one source at a time.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import shortest_path

from src.data.alignment import STEP_MINUTES, WINDOW_STEPS, session_start
from src.data.graph_data import GraphData

WEATHER_COLUMNS = ["ws_temp_c", "ws_precip_mm", "ws_humidity_pct"]
WEATHER_GRID_SIZE_DEG = 0.05  # src/ingestion/temporal_weather.py
PRECIP_COLUMN = 1             # index of ws_precip_mm in WEATHER_COLUMNS
MAX_FLOOD_LEVEL = 3
MAX_CAMERA_HOPS = 8
EVENT_TTL_MINUTES = 60.0      # src/routing/event_layer.py
MAX_EVENT_SNAP_KM = 0.5       # an event intersection farther than this from every node is dropped

FEATURES: List[str] = [
    # weather (daily, node's grid cell)
    "temp", "precip", "humidity",
    # flood
    "flood_level", "flood_x_precip",
    # events (per minute)
    "event_impact", "event_count",
    # temporal (per minute)
    "minute_sin", "minute_cos", "weekday_sin", "weekday_cos",
    # spatial
    "in_degree", "out_degree", "mean_speed", "max_speed", "is_camera", "camera_hops",
]
GROUPS: Dict[str, List[str]] = {
    "weather": ["temp", "precip", "humidity"],
    "flood": ["flood_level", "flood_x_precip"],
    "events": ["event_impact", "event_count"],
    "temporal": ["minute_sin", "minute_cos", "weekday_sin", "weekday_cos"],
    "spatial": ["in_degree", "out_degree", "mean_speed", "max_speed", "is_camera", "camera_hops"],
}
N_FEATURES = len(FEATURES)
COLUMN = {name: i for i, name in enumerate(FEATURES)}


def group_columns(groups: Iterable[str]) -> List[int]:
    unknown = set(groups) - set(GROUPS)
    if unknown:
        raise ValueError(f"unknown context group(s) {sorted(unknown)}; choose from {sorted(GROUPS)}")
    return [COLUMN[name] for g in groups for name in GROUPS[g]]


def grid_cell(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """WeatherStack grid cell id per point, as temporal_weather.build_grid_points writes it."""
    return np.char.add(
        np.char.add(np.round(np.asarray(lat) / WEATHER_GRID_SIZE_DEG).astype(int).astype(str), "_"),
        np.round(np.asarray(lon) / WEATHER_GRID_SIZE_DEG).astype(int).astype(str),
    )


def clock_features(when: Sequence[datetime]) -> np.ndarray:
    """[T, 4]: minute of day and day of week, each as sin/cos."""
    minute = np.array([t.hour * 60 + t.minute for t in when], dtype=np.float64)
    weekday = np.array([t.weekday() for t in when], dtype=np.float64)
    return np.stack(
        [
            np.sin(2 * np.pi * minute / 1440), np.cos(2 * np.pi * minute / 1440),
            np.sin(2 * np.pi * weekday / 7), np.cos(2 * np.pi * weekday / 7),
        ],
        axis=1,
    ).astype(np.float32)


@dataclass
class NodeContext:
    """Everything needed to build a [T, N, N_FEATURES] context tensor for any window."""

    static: np.ndarray                                  # [N, len(FEATURES)], spatial + flood level filled
    flood: np.ndarray                                   # [N] ordinal in [0, 1]
    cells: np.ndarray                                   # [N] grid cell id per node
    weather: Dict[Tuple[date, str], np.ndarray]         # (day, cell) -> scaled [3]
    weather_min: np.ndarray                             # [3] from the training days
    weather_max: np.ndarray
    events: Dict[int, List[Tuple[datetime, float]]] = field(default_factory=dict)  # node -> (time, impact)
    drop: Tuple[str, ...] = ()                          # groups zeroed for ablation
    missing_weather: set = field(default_factory=set)   # days asked for that have no weather row

    # ------------------------------------------------------------------ build
    @classmethod
    def build(
        cls,
        graph: GraphData,
        spatial_dir: Path,
        weather_csv: Optional[Path],
        events: Sequence = (),
        event_nodes: Optional[Dict[str, int]] = None,
        camera_index: Optional[Sequence[int]] = None,
        train_days: Optional[Iterable[date]] = None,
        weather_min: Optional[np.ndarray] = None,
        weather_max: Optional[np.ndarray] = None,
        drop: Sequence[str] = (),
    ) -> "NodeContext":
        """
        `events` are event_layer.TrafficEvent (already placed at an intersection label);
        `event_nodes` maps those labels to graph indices (see event_node_index). The weather
        scaling comes from `train_days` unless `weather_min`/`weather_max` are given, which is
        how prediction reuses the scaling a checkpoint was trained with.
        """
        group_columns(drop)  # validates
        spatial_dir = Path(spatial_dir)
        n = graph.n_nodes
        nodes = pd.read_csv(spatial_dir / "full_network_static_features.csv").set_index("node_id")
        nodes = nodes.loc[graph.node_ids]
        lat, lon = nodes["lat"].to_numpy(), nodes["lon"].to_numpy()
        flood = np.clip(nodes["flood_hazard_level"].to_numpy(dtype=np.float64), 0, MAX_FLOOD_LEVEL) / MAX_FLOOD_LEVEL

        cameras = sorted(graph.camera_nodes.values()) if camera_index is None else [int(c) for c in camera_index]
        static = np.zeros((n, N_FEATURES), dtype=np.float32)
        static[:, COLUMN["flood_level"]] = flood
        for name, values in road_features(graph, spatial_dir, cameras).items():
            static[:, COLUMN[name]] = values

        weather, w_min, w_max = load_cell_weather(weather_csv, train_days, weather_min, weather_max)
        by_node: Dict[int, List[Tuple[datetime, float]]] = {}
        for event in events:
            node = (event_nodes or {}).get(event.intersection)
            if node is None:
                continue
            at = event.at.replace(tzinfo=None) if event.at.tzinfo else event.at
            by_node.setdefault(int(node), []).append((at, float(event.base_impact)))
        return cls(static, flood.astype(np.float32), grid_cell(lat, lon), weather, w_min, w_max, by_node, tuple(drop))

    # ------------------------------------------------------------------ query
    def at_times(self, times: Sequence[datetime]) -> torch.Tensor:
        """[T, N, N_FEATURES] for consecutive minutes `times` (naive Manila wall clock)."""
        t_count, n = len(times), self.static.shape[0]
        out = np.broadcast_to(self.static, (t_count, n, N_FEATURES)).copy()

        weather = self.node_weather(times[0].date())
        out[:, :, [COLUMN[c] for c in GROUPS["weather"]]] = weather[None]
        out[:, :, COLUMN["flood_x_precip"]] = (self.flood * weather[:, PRECIP_COLUMN])[None]
        out[:, :, [COLUMN[c] for c in GROUPS["temporal"]]] = clock_features(times)[:, None, :]

        ttl = timedelta(minutes=EVENT_TTL_MINUTES)
        for node, reports in self.events.items():
            for at, impact in reports:
                for step, when in enumerate(times):
                    age = when - at
                    if timedelta(0) <= age < ttl:
                        strength = impact * (1.0 - age / ttl)
                        out[step, node, COLUMN["event_impact"]] = max(out[step, node, COLUMN["event_impact"]], strength)
                        out[step, node, COLUMN["event_count"]] += 1.0
        out[:, :, COLUMN["event_count"]] = np.log1p(out[:, :, COLUMN["event_count"]])

        if self.drop:
            out[:, :, group_columns(self.drop)] = 0.0
        return torch.from_numpy(out)

    def window(self, day: date, session: str, start: int, steps: int = WINDOW_STEPS) -> torch.Tensor:
        """[steps, N, N_FEATURES] for the window starting `start` minutes into a session."""
        begin = session_start(day, session).to_pydatetime() + timedelta(minutes=int(start) * STEP_MINUTES)
        return self.at_times([begin + timedelta(minutes=i * STEP_MINUTES) for i in range(steps)])

    def node_weather(self, day: date) -> np.ndarray:
        """[N, 3] scaled daily weather of each node's cell; the day's citywide mean for a cell
        without a row, and mid-range (0.5) when the day has no weather at all."""
        rows = [self.weather.get((day, c)) for c in self.cells]
        known = [r for r in rows if r is not None]
        if not known:
            self.missing_weather.add(day)
            return np.full((len(rows), len(WEATHER_COLUMNS)), 0.5, dtype=np.float32)
        mean = np.mean(known, axis=0)
        return np.stack([mean if r is None else r for r in rows]).astype(np.float32)

    def config(self) -> dict:
        """What a checkpoint must keep to rebuild the same context at prediction time."""
        return {
            "context_features": FEATURES,
            "context_weather_min": self.weather_min.tolist(),
            "context_weather_max": self.weather_max.tolist(),
            "context_drop": list(self.drop),
        }


# ---------------------------------------------------------------------- helpers
def _endpoints(graph: GraphData) -> Tuple[np.ndarray, np.ndarray]:
    """(src, dst) as writable int64 arrays; scipy's fancy indexing rejects read-only views."""
    return (np.array(graph.edge_index[0].cpu().numpy(), dtype=np.int64),
            np.array(graph.edge_index[1].cpu().numpy(), dtype=np.int64))


def road_features(graph: GraphData, spatial_dir: Path, cameras: Sequence[int]) -> Dict[str, np.ndarray]:
    """Spatial columns from the road graph: degree, free-flow speed of incident roads, camera reach."""
    n = graph.n_nodes
    src, dst = _endpoints(graph)
    speed = edge_speed_kmh(graph, spatial_dir)
    in_deg = np.bincount(dst, minlength=n).astype(np.float64)
    out_deg = np.bincount(src, minlength=n).astype(np.float64)
    total = np.bincount(src, weights=speed, minlength=n) + np.bincount(dst, weights=speed, minlength=n)
    count = in_deg + out_deg
    mean_speed = np.where(count > 0, total / np.maximum(count, 1), 0.0)
    max_speed = np.zeros(n)
    np.maximum.at(max_speed, src, speed)
    np.maximum.at(max_speed, dst, speed)

    is_camera = np.zeros(n)
    hops = np.full(n, float(MAX_CAMERA_HOPS))
    if len(cameras):
        is_camera[list(cameras)] = 1.0
        undirected = sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
        undirected = undirected + undirected.T
        dist = shortest_path(undirected, unweighted=True, directed=False, indices=list(cameras)).min(axis=0)
        hops = np.minimum(np.where(np.isfinite(dist), dist, MAX_CAMERA_HOPS), MAX_CAMERA_HOPS)
    return {
        "in_degree": np.log1p(in_deg) / math.log(5),
        "out_degree": np.log1p(out_deg) / math.log(5),
        "mean_speed": mean_speed / 100.0,
        "max_speed": max_speed / 100.0,
        "is_camera": is_camera,
        "camera_hops": hops / MAX_CAMERA_HOPS,
    }


def edge_speed_kmh(graph: GraphData, spatial_dir: Path) -> np.ndarray:
    """Free-flow speed per edge (graph.edge_index order) from length / travel time."""
    src, dst = _endpoints(graph)
    length = np.asarray(graph.adjacency[src, dst]).ravel()
    path = Path(spatial_dir) / "metro_manila_travel_time.npz"
    if not path.exists():
        return np.full(len(src), 25.0)
    order = np.load(Path(spatial_dir) / "metro_manila_node_order.npy")
    position = {int(nid): i for i, nid in enumerate(order)}
    full = np.array([position[int(nid)] for nid in graph.node_ids])
    seconds = np.asarray(sp.load_npz(path).tocsr()[full[src], full[dst]]).ravel()
    return np.where(seconds > 0, length / np.maximum(seconds, 1e-9) * 3.6, 25.0)


def edge_features(graph: GraphData, spatial_dir: Path) -> torch.Tensor:
    """[E, 2] per-edge road attributes for the decoder: log length and free-flow speed."""
    src, dst = _endpoints(graph)
    length = np.asarray(graph.adjacency[src, dst]).ravel()
    speed = edge_speed_kmh(graph, spatial_dir)
    return torch.tensor(np.stack([np.log1p(length) / math.log(1000), speed / 100.0], axis=1), dtype=torch.float32)


def load_cell_weather(
    weather_csv: Optional[Path],
    train_days: Optional[Iterable[date]] = None,
    weather_min: Optional[np.ndarray] = None,
    weather_max: Optional[np.ndarray] = None,
) -> Tuple[Dict[Tuple[date, str], np.ndarray], np.ndarray, np.ndarray]:
    """(day, cell) -> min-max scaled weather, scaled on the training days unless min/max given."""
    if weather_csv is None or not Path(weather_csv).exists():
        zeros = np.zeros(len(WEATHER_COLUMNS))
        return {}, zeros if weather_min is None else np.asarray(weather_min), np.ones(len(WEATHER_COLUMNS)) if weather_max is None else np.asarray(weather_max)
    df = pd.read_csv(weather_csv)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.dropna(subset=WEATHER_COLUMNS)
    if weather_min is None or weather_max is None:
        fit = df[df["date"].isin(set(train_days))] if train_days is not None else df
        if fit.empty:
            fit = df
        weather_min = fit[WEATHER_COLUMNS].min().to_numpy(dtype=np.float64)
        weather_max = fit[WEATHER_COLUMNS].max().to_numpy(dtype=np.float64)
    weather_min, weather_max = np.asarray(weather_min, dtype=np.float64), np.asarray(weather_max, dtype=np.float64)
    span = np.where(weather_max > weather_min, weather_max - weather_min, 1.0)
    # Not clipped, like alignment.MinMaxScaler: a wetter day than any in training stays visible.
    scaled = ((df[WEATHER_COLUMNS].to_numpy(dtype=np.float64) - weather_min) / span).astype(np.float32)
    table = {(d, str(c)): row for d, c, row in zip(df["date"], df["grid_cell"], scaled)}
    return table, weather_min, weather_max


def event_node_index(graph: GraphData, spatial_dir: Path, intersections_csv: Optional[Path]) -> Dict[str, int]:
    """
    Incident intersection label -> graph index: the camera intersections, plus the geocoded
    landmarks in configs/event_intersections.csv snapped to the nearest node within
    MAX_EVENT_SNAP_KM (the same placement app/risk_routing.py uses for routing).
    """
    index = dict(graph.camera_nodes)
    if intersections_csv is None or not Path(intersections_csv).exists():
        return index
    nodes = pd.read_csv(Path(spatial_dir) / "full_network_static_features.csv").set_index("node_id").loc[graph.node_ids]
    lat, lon = np.radians(nodes["lat"].to_numpy()), np.radians(nodes["lon"].to_numpy())
    with Path(intersections_csv).open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            label = row["intersection"].strip()
            if label in index:
                continue
            plat, plon = math.radians(float(row["lat"])), math.radians(float(row["lon"]))
            a = np.sin((lat - plat) / 2) ** 2 + np.cos(lat) * math.cos(plat) * np.sin((lon - plon) / 2) ** 2
            km = 2 * 6371.0 * np.arcsin(np.sqrt(a))
            nearest = int(np.argmin(km))
            if km[nearest] <= MAX_EVENT_SNAP_KM:
                index[label] = nearest
    return index


def load_events(raw_twitter_root: Optional[Path], landmarks_csv: Optional[Path]) -> list:
    """Parsed, placeable MMDA alerts (event_layer.parse_alerts); [] when the inputs are missing."""
    from src.routing.event_layer import load_landmarks, parse_alerts

    if raw_twitter_root is None or landmarks_csv is None or not Path(raw_twitter_root).exists():
        return []
    landmarks = load_landmarks(landmarks_csv)
    return parse_alerts(raw_twitter_root, landmarks) if landmarks else []
