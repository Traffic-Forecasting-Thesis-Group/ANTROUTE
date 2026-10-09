"""
Non-visual Congestion Risk for roads outside the camera subgraph.

Why this exists
---------------
The RADR STGNN (traffic_risk_model_edge.py) scores only the k-hop camera subgraph: its GCN
needs a dense normalised adjacency (~14 GB over the city's 59,521 nodes), and it was trained
with CNN+LSTM features at the 7 camera nodes plus a learned *placeholder* elsewhere. It was
never trained or evaluated with the visual modality absent everywhere, so feeding it zeros
(or the placeholder) for a road far from any camera is not a valid prediction. Until now the
app gave the ~86% of roads that are neither scored nor within 1.5 km of a scored road the
window's median risk -- a constant, which made routing there distance-only.

What this does instead
----------------------
A separate non-visual model predicts risk from inputs every road has, aligned to that road
and that window, and never from camera features:

    weather    the day's temperature, rainfall and humidity in the road's 0.05-degree cell
    flood      Project NOAH 5-year hazard level of the road's endpoints, and x rainfall
    incidents  MMDA alerts live during the window within INCIDENT_RADIUS_KM of the road
    clock      time of day and day of week of the window
    road       OSMnx network: length, free-flow speed and travel time, degree and speeds at
               both endpoints, and 2-hop neighbourhood means (sparse adjacency products) --
               built for every road, but NOT used by the deployed model (see MODEL_FEATURES)

Missing inputs are explicit, never silently zero:
    * incidents are UNAVAILABLE (NaN, flag 0) when the alert feed has no data that day, or
      when the road is farther than INCIDENT_COVERAGE_KM from every place the gazetteer can
      locate an alert -- "no incident reported" (0, flag 1) is only claimed where an incident
      would have been placed had one been reported;
    * weather is UNAVAILABLE (NaN, flag 0) when the road's cell has no row for that day.
The model (gradient-boosted trees) splits on NaN natively, and training adds modality-
dropout copies of every row with incidents and weather withheld, so the "unavailable" branch
is learned rather than left to chance.

What it is trained on, and what that limits
-------------------------------------------
Labels exist only on the 42 road segments touching the 7 EDSA cameras (weak_target:
Light/Medium/Heavy as 0/0.5/1). The model is fitted on those, so it can only learn how the
non-visual inputs relate to congestion *where cameras are*, and then applies that to roads it
has never seen labelled. Leave-one-camera-out validation (scripts/train_nonvisual_risk.py)
is the honest estimate of how that transfers to an unseen road; there is no ground truth off
the cameras to check it against. Deliberately excluded: latitude/longitude, camera flags and
hops to the nearest camera -- with 7 labelled places they would let the model memorise the
cameras, and every non-camera road would be out of their training range.

No dense city-wide matrix is built anywhere here: the 2-hop neighbourhood features are sparse
matrix-vector products over the 147,197-edge network.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.data.graph_data import GraphData

# Where each edge's risk came from, in the risk-source arrays the app and reports carry.
SOURCE_PREDICTED = 0     # the RADR STGNN scored this edge (camera subgraph, visual + context)
SOURCE_BORROWED = 1      # copied from a scored edge within MAX_FILL_M by road
SOURCE_NONVISUAL = 2     # predicted by this module from non-visual inputs only
SOURCE_UNAVAILABLE = 3   # no prediction of any kind (model or inputs missing)
SOURCE_NAMES = {
    SOURCE_PREDICTED: "predicted",
    SOURCE_BORROWED: "borrowed",
    SOURCE_NONVISUAL: "nonvisual",
    SOURCE_UNAVAILABLE: "unavailable",
}

WEATHER_COLUMNS = ["ws_temp_c", "ws_precip_mm", "ws_humidity_pct"]
WEATHER_GRID_SIZE_DEG = 0.05
MAX_FLOOD_LEVEL = 3
EVENT_TTL_MINUTES = 60.0           # an alert's impact fades to 0 over this (event_layer)
INCIDENT_RADIUS_KM = 0.5           # an alert affects roads this close to where it was placed
INCIDENT_COVERAGE_KM = 1.0         # farther than this from every placeable point: unknown, not 0
WINDOW_MINUTES = 30                # alignment.WINDOW_STEPS x STEP_MINUTES

STATIC_FEATURES = [
    "length_log", "edge_speed", "edge_travel_time_log",
    "src_in_degree", "src_out_degree", "dst_in_degree", "dst_out_degree",
    "src_mean_speed", "dst_mean_speed", "endpoint_max_speed",
    "flood_level", "nbr_mean_speed", "nbr_degree", "nbr_flood",
]
DYNAMIC_FEATURES = [
    "temp", "precip", "humidity", "weather_available", "flood_x_precip",
    "minute_sin", "minute_cos", "weekday_sin", "weekday_cos",
    "incident_impact", "incident_count", "incident_feed_available", "incident_covered",
]
FEATURES = STATIC_FEATURES + DYNAMIC_FEATURES
COLUMN = {name: i for i, name in enumerate(FEATURES)}

# What the deployed model actually uses: the time-varying inputs aligned to each road (its own
# weather cell, flood x that day's rain, incidents near it, the clock), not the road-network
# and speed columns. Chosen by leave-one-camera-out validation (scripts/train_nonvisual_risk.py,
# report["feature_selection"]): with all 14 road columns added, accuracy on a camera the model
# had not seen fell below a constant (MAE 0.336-0.343 vs 0.324), because with 9 labelled places
# -- all on EDSA and Roxas Blvd -- those columns identify the camera rather than describe
# congestion. They are still built (RoadFeatures.static) for the report and for retraining once
# labels exist at more varied roads.
MODEL_FEATURES = list(DYNAMIC_FEATURES)
INCIDENT_VALUE_COLUMNS = ["incident_impact", "incident_count"]
WEATHER_VALUE_COLUMNS = ["temp", "precip", "humidity", "flood_x_precip"]


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=np.float64)) for v in (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def grid_cell(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """WeatherStack grid cell id, as src/ingestion/temporal_weather.py writes it."""
    return np.char.add(
        np.char.add(np.round(np.asarray(lat) / WEATHER_GRID_SIZE_DEG).astype(int).astype(str), "_"),
        np.round(np.asarray(lon) / WEATHER_GRID_SIZE_DEG).astype(int).astype(str),
    )


# --------------------------------------------------------------------------- incidents
@dataclass
class PlacedIncidents:
    """
    MMDA alerts placed on the road network, and where and when the feed could have reported one.

    `points` are the places the gazetteer can locate an alert (lat, lon); a road farther than
    INCIDENT_COVERAGE_KM from all of them has *unknown* incident status. `feed_days` are the days
    the raw alert feed has data at all; on any other day the status is unknown everywhere.
    """

    events: pd.DataFrame          # columns: at (naive Manila time), lat, lon, impact
    points: np.ndarray            # [P, 2] lat, lon
    feed_days: frozenset

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.events.assign(at=self.events["at"].astype(str)).to_csv(path, index=False)
        meta = {"points": self.points.tolist(), "feed_days": sorted(str(d) for d in self.feed_days)}
        path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "PlacedIncidents":
        path = Path(path)
        events = pd.read_csv(path)
        events["at"] = pd.to_datetime(events["at"])
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        return cls(events, np.asarray(meta["points"], dtype=np.float64).reshape(-1, 2),
                   frozenset(date.fromisoformat(d) for d in meta["feed_days"]))

    @classmethod
    def build(cls, raw_twitter_root: Path, landmarks_csv: Path, intersections_csv: Path,
              camera_points: Dict[str, Tuple[float, float]]) -> "PlacedIncidents":
        """Parse the raw feed once (slow: minutes) into the small table the app loads."""
        from src.routing.event_layer import load_landmarks, parse_alerts

        located = dict(camera_points)
        table = pd.read_csv(intersections_csv)
        for row in table.itertuples():
            located.setdefault(str(row.intersection).strip(), (float(row.lat), float(row.lon)))
        landmarks = load_landmarks(landmarks_csv)
        rows = []
        for event in parse_alerts(Path(raw_twitter_root), landmarks):
            where = located.get(event.intersection)
            if where is None:
                continue
            at = event.at.replace(tzinfo=None) if event.at.tzinfo else event.at
            rows.append({"at": at, "lat": where[0], "lon": where[1], "impact": float(event.base_impact)})
        feed_days = frozenset(
            date.fromisoformat(p.name) for p in Path(raw_twitter_root).iterdir()
            if p.is_dir() and any(p.glob("*.json"))
        )
        points = np.array(sorted(set(located.values())), dtype=np.float64)
        return cls(pd.DataFrame(rows, columns=["at", "lat", "lon", "impact"]), points, feed_days)


# --------------------------------------------------------------------------- features
@dataclass
class RoadFeatures:
    """
    Per-edge non-visual features for one road graph (the whole city, or any subgraph), built
    with sparse operations only. `static` is time-invariant; `at_window` adds the window's own
    weather, clock and incidents.
    """

    node_ids: np.ndarray            # [N]
    src: np.ndarray                 # [E] graph index of each edge's tail
    dst: np.ndarray                 # [E]
    static: np.ndarray              # [E, len(STATIC_FEATURES)]
    flood: np.ndarray               # [E] endpoint max flood level / 3
    cells: np.ndarray               # [E] weather grid cell of the edge's tail
    lat: np.ndarray                 # [N]
    lon: np.ndarray                 # [N]
    weather: Dict[Tuple[date, str], np.ndarray] = field(default_factory=dict)
    incidents: Optional[PlacedIncidents] = None
    covered: Optional[np.ndarray] = None   # [E] within INCIDENT_COVERAGE_KM of a placeable point

    @classmethod
    def build(cls, graph: GraphData, spatial_dir: Path, weather_csv: Optional[Path],
              incidents: Optional[PlacedIncidents]) -> "RoadFeatures":
        spatial_dir = Path(spatial_dir)
        n = graph.n_nodes
        src = np.array(graph.edge_index[0].cpu().numpy(), dtype=np.int64)
        dst = np.array(graph.edge_index[1].cpu().numpy(), dtype=np.int64)
        length = np.asarray(graph.adjacency[src, dst]).ravel().astype(np.float64)
        seconds = _free_flow_seconds(graph, spatial_dir, src, dst, length)
        speed = np.where(seconds > 0, length / np.maximum(seconds, 1e-9) * 3.6, np.nan)

        in_deg = np.bincount(dst, minlength=n).astype(np.float64)
        out_deg = np.bincount(src, minlength=n).astype(np.float64)
        spd = np.nan_to_num(speed, nan=25.0)
        total = np.bincount(src, weights=spd, minlength=n) + np.bincount(dst, weights=spd, minlength=n)
        count = in_deg + out_deg
        node_speed = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        node_max = np.zeros(n)
        np.maximum.at(node_max, src, spd)
        np.maximum.at(node_max, dst, spd)

        nodes = pd.read_csv(spatial_dir / "full_network_static_features.csv").set_index("node_id").loc[graph.node_ids]
        lat, lon = nodes["lat"].to_numpy(dtype=np.float64), nodes["lon"].to_numpy(dtype=np.float64)
        node_flood = np.clip(nodes["flood_hazard_level"].to_numpy(dtype=np.float64), 0, MAX_FLOOD_LEVEL) / MAX_FLOOD_LEVEL

        # 2-hop neighbourhood means through the (undirected) road graph: sparse products only.
        adj = sp.csr_matrix((np.ones(src.size), (src, dst)), shape=(n, n))
        adj = ((adj + adj.T) > 0).astype(np.float64)
        deg = np.asarray(adj.sum(axis=1)).ravel()
        walk = sp.diags(1.0 / np.maximum(deg, 1.0)) @ adj
        two_hop = lambda x: walk @ (walk @ x)  # noqa: E731
        nbr_speed = two_hop(np.nan_to_num(node_speed, nan=25.0))
        nbr_degree = two_hop(np.log1p(deg))
        nbr_flood = two_hop(node_flood)

        static = np.column_stack([
            np.log1p(length),
            speed,
            np.log1p(np.where(np.isfinite(seconds), seconds, np.nan)),
            np.log1p(in_deg[src]), np.log1p(out_deg[src]), np.log1p(in_deg[dst]), np.log1p(out_deg[dst]),
            node_speed[src], node_speed[dst], np.maximum(node_max[src], node_max[dst]),
            np.maximum(node_flood[src], node_flood[dst]),
            0.5 * (nbr_speed[src] + nbr_speed[dst]),
            0.5 * (nbr_degree[src] + nbr_degree[dst]),
            0.5 * (nbr_flood[src] + nbr_flood[dst]),
        ]).astype(np.float32)

        covered = None
        if incidents is not None and len(incidents.points):
            near = np.full(n, np.inf)
            for plat, plon in incidents.points:
                near = np.minimum(near, _haversine_km(lat, lon, plat, plon))
            node_covered = near <= INCIDENT_COVERAGE_KM
            covered = node_covered[src] | node_covered[dst]

        return cls(
            node_ids=np.asarray(graph.node_ids), src=src, dst=dst, static=static,
            flood=np.maximum(node_flood[src], node_flood[dst]).astype(np.float32),
            cells=grid_cell(lat[src], lon[src]), lat=lat, lon=lon,
            weather=load_weather(weather_csv), incidents=incidents, covered=covered,
        )

    def at_window(self, start: datetime, edges: Optional[np.ndarray] = None) -> np.ndarray:
        """[E, len(FEATURES)] for the 30-minute window starting `start` (naive Manila time)."""
        idx = np.arange(self.src.size) if edges is None else np.asarray(edges)
        out = np.full((idx.size, len(FEATURES)), np.nan, dtype=np.float32)
        out[:, : len(STATIC_FEATURES)] = self.static[idx]

        # weather of the edge's cell that day; NaN + flag 0 where the cell has no row
        day = start.date()
        rows = [self.weather.get((day, c)) for c in self.cells[idx]]
        have = np.array([r is not None for r in rows])
        if have.any():
            values = np.stack([r for r in rows if r is not None])
            out[have, COLUMN["temp"]] = values[:, 0]
            out[have, COLUMN["precip"]] = values[:, 1]
            out[have, COLUMN["humidity"]] = values[:, 2]
            out[have, COLUMN["flood_x_precip"]] = self.flood[idx][have] * values[:, 1]
        out[:, COLUMN["weather_available"]] = have.astype(np.float32)

        middle = start + timedelta(minutes=WINDOW_MINUTES / 2)
        minute = middle.hour * 60 + middle.minute
        out[:, COLUMN["minute_sin"]] = math.sin(2 * math.pi * minute / 1440)
        out[:, COLUMN["minute_cos"]] = math.cos(2 * math.pi * minute / 1440)
        out[:, COLUMN["weekday_sin"]] = math.sin(2 * math.pi * middle.weekday() / 7)
        out[:, COLUMN["weekday_cos"]] = math.cos(2 * math.pi * middle.weekday() / 7)

        self._incidents_into(out, idx, start)
        return out

    def _incidents_into(self, out: np.ndarray, idx: np.ndarray, start: datetime) -> None:
        feed = self.incidents is not None and start.date() in self.incidents.feed_days
        covered = self.covered[idx] if (self.covered is not None) else np.zeros(idx.size, dtype=bool)
        known = covered & feed
        out[:, COLUMN["incident_feed_available"]] = float(feed)
        out[:, COLUMN["incident_covered"]] = covered.astype(np.float32)
        # Known status: 0 unless an alert is live nearby. Unknown status stays NaN.
        out[known, COLUMN["incident_impact"]] = 0.0
        out[known, COLUMN["incident_count"]] = 0.0
        if not feed or not known.any():
            return
        ev = self.incidents.events
        middle = start + timedelta(minutes=WINDOW_MINUTES / 2)
        ttl = timedelta(minutes=EVENT_TTL_MINUTES)
        live = ev[(ev["at"] <= middle) & (ev["at"] > middle - ttl)]
        if live.empty:
            return
        src_lat, src_lon = self.lat[self.src[idx]], self.lon[self.src[idx]]
        dst_lat, dst_lon = self.lat[self.dst[idx]], self.lon[self.dst[idx]]
        for row in live.itertuples():
            age = (middle - row.at.to_pydatetime()) / ttl
            strength = float(row.impact) * (1.0 - age)
            near = (np.minimum(_haversine_km(src_lat, src_lon, row.lat, row.lon),
                               _haversine_km(dst_lat, dst_lon, row.lat, row.lon)) <= INCIDENT_RADIUS_KM) & known
            if near.any():
                col = COLUMN["incident_impact"]
                out[near, col] = np.maximum(out[near, col], strength)
                out[near, COLUMN["incident_count"]] += 1.0


def _free_flow_seconds(graph: GraphData, spatial_dir: Path, src, dst, length) -> np.ndarray:
    """Free-flow travel time per edge from metro_manila_travel_time.npz; NaN where unknown."""
    path = spatial_dir / "metro_manila_travel_time.npz"
    if not path.exists():
        return np.full(src.size, np.nan)
    order = np.load(spatial_dir / "metro_manila_node_order.npy")
    position = {int(nid): i for i, nid in enumerate(order)}
    full = np.array([position[int(nid)] for nid in graph.node_ids])
    seconds = np.asarray(sp.load_npz(path).tocsr()[full[src], full[dst]]).ravel().astype(np.float64)
    return np.where(seconds > 0, seconds, np.nan)


def load_weather(weather_csv: Optional[Path]) -> Dict[Tuple[date, str], np.ndarray]:
    """(day, cell) -> raw [temp C, precip mm, humidity %]; {} when the file is missing."""
    if weather_csv is None or not Path(weather_csv).exists():
        return {}
    df = pd.read_csv(weather_csv).dropna(subset=WEATHER_COLUMNS)
    days = pd.to_datetime(df["date"]).dt.date
    values = df[WEATHER_COLUMNS].to_numpy(dtype=np.float32)
    return {(d, str(c)): v for d, c, v in zip(days, df["grid_cell"], values)}


# --------------------------------------------------------------------------- model
def with_modality_dropout(x: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None):
    """
    The rows, plus a copy with incidents withheld and a copy with weather withheld, flags set
    to match: the camera rows always have both, so without these copies the model would never
    see an "unavailable" input and would handle one arbitrarily at prediction time.
    """
    no_incidents = x.copy()
    no_incidents[:, [COLUMN[c] for c in INCIDENT_VALUE_COLUMNS]] = np.nan
    no_incidents[:, COLUMN["incident_covered"]] = 0.0
    no_weather = x.copy()
    no_weather[:, [COLUMN[c] for c in WEATHER_VALUE_COLUMNS]] = np.nan
    no_weather[:, COLUMN["weather_available"]] = 0.0
    xs = np.vstack([x, no_incidents, no_weather])
    ys = np.concatenate([y, y, y])
    if groups is None:
        return xs, ys
    return xs, ys, np.concatenate([groups, groups, groups])


@dataclass
class NonVisualRiskModel:
    """
    Gradient-boosted trees on `features` (MODEL_FEATURES unless told otherwise) -> risk in
    [0, 1], the weak_target scale. `fit` and `predict` both take full FEATURES rows and pick
    the model's columns themselves, so callers never have to.
    """

    estimator: object
    trained_on: dict
    features: List[str] = field(default_factory=lambda: list(MODEL_FEATURES))
    # Optional out-of-fold affine calibration onto the label scale (scripts/train_nonvisual_risk.py):
    # {"slope", "intercept", "n_samples", "fitted_on"}. Applied only when predict(calibrated=True).
    calibration: Optional[dict] = None

    # Shallow and strongly regularised: ~9,400 labelled rows, but only 9 places.
    PARAMS = dict(max_iter=150, learning_rate=0.05, max_depth=2, min_samples_leaf=80,
                  l2_regularization=2.0, random_state=0)

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, trained_on: dict, dropout: bool = True,
            features: Optional[Sequence[str]] = None) -> "NonVisualRiskModel":
        from sklearn.ensemble import HistGradientBoostingRegressor

        features = list(MODEL_FEATURES if features is None else features)
        if dropout:
            x, y = with_modality_dropout(x, y)
        estimator = HistGradientBoostingRegressor(**cls.PARAMS)
        estimator.fit(x[:, [COLUMN[f] for f in features]], y)
        return cls(estimator, dict(trained_on, features=features, params=cls.PARAMS, modality_dropout=dropout),
                   features)

    def predict(self, x: np.ndarray, calibrated: bool = False) -> np.ndarray:
        """Risk for full FEATURES rows ([n, len(FEATURES)]); `calibrated` applies the out-of-fold
        calibration (and raises if the model has none, rather than silently returning raw)."""
        if x.shape[1] != len(FEATURES):
            raise ValueError(f"expected {len(FEATURES)} feature columns, received {x.shape[1]}")
        raw = np.clip(self.estimator.predict(x[:, [COLUMN[f] for f in self.features]]), 0.0, 1.0)
        if not calibrated:
            return raw
        if not self.calibration:
            raise ValueError("this non-visual model has no calibration; retrain with scripts/train_nonvisual_risk.py")
        return np.clip(self.calibration["slope"] * raw + self.calibration["intercept"], 0.0, 1.0)

    def save(self, path: Path) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"estimator": self.estimator, "trained_on": self.trained_on,
                     "calibration": self.calibration}, path)

    @classmethod
    def load(cls, path: Path) -> "NonVisualRiskModel":
        import joblib

        blob = joblib.load(path)
        features = list(blob["trained_on"].get("features") or [])
        unknown = set(features) - set(FEATURES)
        if not features or unknown:
            raise ValueError(f"{path} uses features src/models/nonvisual_risk.py does not build: {sorted(unknown)}")
        return cls(blob["estimator"], blob["trained_on"], features, blob.get("calibration"))
