"""
Congestion risk for the ASSUMED tier: roads too far from the camera subgraph for any camera
signal to reach them (see dynamic_weight.TIERS).

Risk is predicted on the whole-city graph, in three tiers. On the camera subgraph the RADR
STGNN -> MLP decoder predicts it (OBSERVED); within MAX_FILL_M of that, the nearest predicted
road's risk is carried along the corridor (PROPAGATED). Beyond that the only information is
what every road has whether or not a camera sees it -- the non-visual inputs the STGNN itself
takes per node (src/data/node_context.py):

    weather   temperature, precipitation and humidity in the road's 0.05-degree WeatherStack
              cell, at that hour (weatherstack_hourly.csv) or for that day (the daily file)
    flood     Project NOAH 5-year hazard level of the road's endpoints, and its interaction
              with that day's rainfall
    events    the MMDA incident impact on the road at that moment (event_layer.event_impact)
    clock     minute of day and day of week, as sin/cos
    road      free-flow speed (the speed limit, posted or assumed), length, lane count and road
              class (full_network_edge_features.csv, when built)
    text      how often a year of traffic posts (before May 2026) concerned the road at that hour,
              and how often they reported an incident (src/data/text_prior.py)

ContextRiskModel maps those to risk with a ridge regression fitted on the decoder's own
outputs on the camera subgraph, across recorded windows: it learns how the decoder's risk
moves with these inputs, and applies that to roads the decoder never scored. It is fitted on
the risk file's train-split windows and its R^2 on the other windows is reported, so how much
of the decoder's behaviour the assumed tier reproduces is a number, not a claim.

Limitations, stated rather than hidden: the fit sees only the corridor around the cameras, so
road types and flood levels rare there are extrapolated (predictions are clipped to [0, 1]),
and nothing here observes traffic on the roads it scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data.graph_data import GraphData
from src.data.node_context import (
    MAX_FLOOD_LEVEL,
    MINOR_ROAD_CLASS,
    road_attributes,
    WEATHER_COLUMNS,
    _endpoints,
    clock_features,
    edge_speed_kmh,
    grid_cell,
    load_cell_weather,
)

FEATURES: List[str] = [
    "flood_level", "precip", "flood_x_precip", "temp", "humidity",
    "minute_sin", "minute_cos", "weekday_sin", "weekday_cos",
    "free_flow_speed", "log_length", "lanes", "road_class", "event_impact",
    "text_reports", "text_incidents", "live_text_reports", "live_text_incidents",
]
TEXT_FEATURES = ["text_reports", "text_incidents", "live_text_reports", "live_text_incidents"]
DEFAULT_ALPHA = 1.0     # ridge penalty, on standardised features


@dataclass
class EdgeContext:
    """The non-visual inputs of every edge of one graph, ready to be read at any moment."""

    flood: np.ndarray                 # [E] mean endpoint hazard level in [0, 1]
    speed: np.ndarray                 # [E] free-flow speed / 100 km/h
    log_length: np.ndarray            # [E] log1p(metres) / log(1000)
    lanes: np.ndarray                 # [E] lane count / MAX_LANES
    road_class: np.ndarray            # [E] node_context.ROAD_CLASS ordinal
    cell_index: np.ndarray            # [E] index into `cells` of the edge's tail node
    cells: List[str]
    weather: Dict[tuple, np.ndarray]  # (day, cell), or (day, hour, cell) when hourly
    src: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64), repr=False)
    dst: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64), repr=False)
    node_text: Optional[np.ndarray] = None   # [N, 48, 2] text_prior report / incident rates
    live: Optional[object] = None            # text_prior.LiveText over the same nodes
    _live_cache: Dict[datetime, np.ndarray] = field(default_factory=dict, repr=False)
    _by_key: Dict[tuple, np.ndarray] = field(default_factory=dict, repr=False)

    @classmethod
    def build(cls, graph: GraphData, spatial_dir: Path, weather_csv: Optional[Path]) -> "EdgeContext":
        spatial_dir = Path(spatial_dir)
        nodes = (
            pd.read_csv(spatial_dir / "full_network_static_features.csv", usecols=["node_id", "lat", "lon", "flood_hazard_level"])
            .set_index("node_id")
            .reindex(graph.node_ids)
        )
        if nodes[["lat", "lon"]].isna().any().any():
            raise ValueError("some graph nodes have no coordinates in full_network_static_features.csv")
        src, dst = _endpoints(graph)   # writable copies: scipy's fancy indexing rejects read-only views
        flood_node = np.clip(nodes["flood_hazard_level"].fillna(0).to_numpy(dtype=np.float64), 0, MAX_FLOOD_LEVEL) / MAX_FLOOD_LEVEL
        length = np.asarray(graph.adjacency[src, dst]).ravel().astype(np.float64)
        node_cells = grid_cell(nodes["lat"].to_numpy(), nodes["lon"].to_numpy())
        cells, inverse = np.unique(node_cells, return_inverse=True)
        weather, _, _ = load_cell_weather(weather_csv)
        try:
            road = road_attributes(graph, spatial_dir)
        except FileNotFoundError:
            road = {"lanes": np.full(src.size, 1 / 6), "road_class": np.full(src.size, MINOR_ROAD_CLASS)}
        try:
            from src.data.text_prior import LiveText, build_text_prior
            node_text = build_text_prior(graph, components=0).values[:, :, :2]
            live = LiveText.build(graph)
        except (OSError, ValueError, KeyError):
            node_text, live = None, None
        return cls(
            src=src, dst=dst, node_text=node_text, live=live,
            flood=(flood_node[src] + flood_node[dst]) / 2,
            speed=edge_speed_kmh(graph, spatial_dir) / 100.0,
            log_length=np.log1p(length) / np.log(1000),
            lanes=road["lanes"],
            road_class=road["road_class"],
            cell_index=inverse[src],
            cells=[str(c) for c in cells],
            weather=weather,
        )

    @property
    def n_edges(self) -> int:
        return self.flood.size

    @property
    def hourly(self) -> bool:
        return bool(self.weather) and len(next(iter(self.weather))) == 3

    def weather_at(self, when: datetime) -> np.ndarray:
        """[E, 3] scaled weather per edge at `when`: its cell's row for that hour (hourly
        weather) or day, else the mean over cells, else mid-range (0.5) with no weather at all."""
        key = (when.date(), when.hour) if self.hourly else (when.date(),)
        if key not in self._by_key:
            rows = [self.weather.get((*key, c)) for c in self.cells]
            known = [r for r in rows if r is not None]
            fill = np.mean(known, axis=0) if known else np.full(len(WEATHER_COLUMNS), 0.5)
            per_cell = np.stack([fill if r is None else r for r in rows]).astype(np.float64)
            self._by_key[key] = per_cell[self.cell_index]
        return self._by_key[key]

    def _text(self, when: datetime, idx: np.ndarray) -> np.ndarray:
        """[n, 2] text report / incident rate of each edge (mean of its endpoints) at when's hour."""
        if self.node_text is None or self.src.size == 0:
            return np.zeros((idx.size, 2))
        b = when.hour + (24 if when.weekday() >= 5 else 0)
        rates = self.node_text[:, b, :]
        return (rates[self.src[idx]] + rates[self.dst[idx]]) / 2

    def _live_text(self, when: datetime, idx: np.ndarray) -> np.ndarray:
        """[n, 2] live posts / live incident posts near each edge (mean of its endpoints)."""
        if self.live is None or self.src.size == 0:
            return np.zeros((idx.size, 2))
        if when not in self._live_cache:
            self._live_cache[when] = self.live.at(when)
        rates = self._live_cache[when]
        return (rates[self.src[idx]] + rates[self.dst[idx]]) / 2

    def features(self, when: datetime, impact: Optional[np.ndarray] = None,
                 edges: Optional[np.ndarray] = None) -> np.ndarray:
        """[n, len(FEATURES)] for `edges` (all when None) at the moment `when` (naive Manila time)."""
        idx = np.arange(self.n_edges) if edges is None else np.asarray(edges, dtype=np.int64)
        weather = self.weather_at(when)[idx]
        temp, precip, humidity = weather[:, 0], weather[:, 1], weather[:, 2]
        clock = np.broadcast_to(clock_features([when])[0].astype(np.float64), (idx.size, 4))
        flood = self.flood[idx]
        event = np.zeros(idx.size) if impact is None else np.asarray(impact, dtype=np.float64)[idx]
        return np.column_stack([
            flood, precip, flood * precip, temp, humidity,
            clock, self.speed[idx], self.log_length[idx], self.lanes[idx], self.road_class[idx], event,
            self._text(when, idx), self._live_text(when, idx),
        ])


@dataclass
class ContextRiskModel:
    """Ridge regression from FEATURES to risk, fitted on standardised features."""

    coef: np.ndarray
    intercept: float
    mean: np.ndarray
    scale: np.ndarray
    alpha: float
    n_rows: int
    n_windows: int
    r2_fit: float
    r2_holdout: Optional[float] = None
    n_holdout_windows: int = 0

    dropped: Tuple[str, ...] = ()          # inputs left out (their columns zeroed), for ablation
    target: str = "decoder risk"
    holdout: Optional[dict] = None         # held-out MAE / Spearman, see fit_from_windows
    text_ablation: Optional[dict] = None   # held-out scores without / with text, see text_ablation

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, alpha: float = DEFAULT_ALPHA, n_windows: int = 0,
            dropped: Sequence[str] = ()) -> "ContextRiskModel":
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        x = _zero(x, dropped)
        if x.ndim != 2 or x.shape[1] != len(FEATURES) or x.shape[0] != y.size or y.size < 2:
            raise ValueError(f"need [n, {len(FEATURES)}] features and n >= 2 targets, received {x.shape} and {y.shape}")
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)   # a constant column (e.g. one weekday) contributes nothing
        z = (x - mean) / scale
        intercept = float(y.mean())
        coef = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - intercept))
        model = cls(coef, intercept, mean, scale, float(alpha), int(y.size), int(n_windows), float("nan"),
                    dropped=tuple(dropped))
        model.r2_fit = model.r_squared(x, y)
        return model

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = (_zero(np.asarray(x, dtype=np.float64), self.dropped) - self.mean) / self.scale
        return np.clip(self.intercept + z @ self.coef, 0.0, 1.0)

    def r_squared(self, x: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=np.float64)
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - float(np.sum((self.predict(x) - y) ** 2)) / ss_tot if ss_tot > 0 else float("nan")

    def describe(self) -> dict:
        """For provenance: what the assumed tier is, how well it reproduces the decoder."""
        return {
            "method": f"ridge regression on the {self.target} over the camera subgraph",
            "target": self.target,
            "dropped_inputs": list(self.dropped),
            "holdout": self.holdout,
            "text_ablation": self.text_ablation,
            "inputs": FEATURES,
            "standardised_coefficients": {f: round(float(c), 5) for f, c in zip(FEATURES, self.coef)},
            "intercept": round(self.intercept, 5),
            "alpha": self.alpha,
            "fit_rows": self.n_rows,
            "fit_windows": self.n_windows,
            "r2_fit": None if not np.isfinite(self.r2_fit) else round(self.r2_fit, 4),
            "r2_holdout": None if self.r2_holdout is None or not np.isfinite(self.r2_holdout) else round(self.r2_holdout, 4),
            "holdout_windows": self.n_holdout_windows,
        }


def evenly(items: Sequence, limit: int) -> List:
    """At most `limit` items spread evenly over `items`, in order."""
    items = list(items)
    if len(items) <= limit:
        return items
    return [items[int(i)] for i in np.linspace(0, len(items) - 1, limit).round().astype(int)]


def _zero(x: np.ndarray, dropped: Sequence[str]) -> np.ndarray:
    """`x` with the columns of the `dropped` inputs set to 0 (a constant: the fit gives them no weight)."""
    if not dropped:
        return x
    x = np.array(x, dtype=np.float64, copy=True)
    x[:, [FEATURES.index(f) for f in dropped]] = 0.0
    return x


def _holdout_scores(model: "ContextRiskModel", x: np.ndarray, y: np.ndarray, baseline: float) -> dict:
    from scipy.stats import spearmanr

    pred = model.predict(x)
    rho = spearmanr(pred, y).statistic if np.unique(y).size > 1 and np.unique(pred).size > 1 else float("nan")
    return {
        "rows": int(y.size),
        "mae": round(float(np.mean(np.abs(pred - y))), 4),
        "mae_constant": round(float(np.mean(np.abs(baseline - y))), 4),
        "spearman": None if not np.isfinite(rho) else round(float(rho), 4),
        "r2": None if not np.isfinite(model.r_squared(x, y)) else round(model.r_squared(x, y), 4),
    }


def fit_from_windows(
    context: EdgeContext,
    samples: Iterable[Tuple[datetime, np.ndarray, np.ndarray, Optional[np.ndarray]]],
    holdout: Iterable[Tuple[datetime, np.ndarray, np.ndarray, Optional[np.ndarray]]] = (),
    alpha: float = DEFAULT_ALPHA,
    dropped: Sequence[str] = (),
    target: str = "decoder risk",
) -> ContextRiskModel:
    """
    Fit on (when, edge ids, target on them, [E] event impact or None) per window -- the target
    being the decoder's risk or the human congestion labels -- and score the `holdout` windows:
    R^2, and MAE / Spearman against a constant at the training mean.
    """
    x, y, n = _stack(context, samples)
    if x is None:
        raise ValueError("no windows to fit the assumed-tier risk on")
    model = ContextRiskModel.fit(x, y, alpha=alpha, n_windows=n, dropped=dropped)
    model.target = target
    hx, hy, hn = _stack(context, holdout)
    if hx is not None:
        model.r2_holdout, model.n_holdout_windows = model.r_squared(hx, hy), hn
        model.holdout = _holdout_scores(model, hx, hy, float(y.mean()))
    return model


def _stack(context: EdgeContext, rows):
    xs, ys, n = [], [], 0
    for when, edges, values, impact in rows:
        xs.append(context.features(when, impact, edges))
        ys.append(np.asarray(values, dtype=np.float64))
        n += 1
    return (np.vstack(xs), np.concatenate(ys), n) if n else (None, None, 0)


def text_ablation(context: EdgeContext, samples: list, holdout: list, alpha: float = DEFAULT_ALPHA,
                  target: str = "human congestion labels") -> Dict[str, dict]:
    """
    Held-out scores of the same fit without text, with the text prior only, and with prior + live
    text: whether text actually carries the missing camera signal, as a number.
    """
    variants = {
        "no_text": TEXT_FEATURES,
        "text_prior": ["live_text_reports", "live_text_incidents"],
        "text_prior_and_live": [],
    }
    return {name: fit_from_windows(context, samples, holdout, alpha, dropped, target).holdout
            for name, dropped in variants.items()}


def quantile_map(pred: np.ndarray, reference_pred: np.ndarray, reference_values: np.ndarray) -> np.ndarray:
    """
    Put `pred` on the scale of `reference_values` by rank: each prediction takes the value at its
    percentile among `reference_pred`. Used so assumed-tier risk, ranked by a model fitted on the
    human labels, sits on the same scale as the decoder's risk on the observed tier of the same
    window -- otherwise far roads and camera roads would be priced on different scales.
    """
    order = np.sort(np.asarray(reference_pred, dtype=np.float64))
    if order.size == 0:
        return np.asarray(pred, dtype=np.float64)
    pct = np.searchsorted(order, pred, side="right") / order.size
    return np.quantile(np.asarray(reference_values, dtype=np.float64), np.clip(pct, 0.0, 1.0))
