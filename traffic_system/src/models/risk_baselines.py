"""
The Congestion Risk Score against simple predictors, on exactly the same labelled rows.

A model's edge-risk MAE means little on its own: on this data the labels are ~70% Heavy, so
a constant can score well, and the scored risk_edges.csv is the only artefact that holds the
model's real held-out predictions. Every reference here is fitted on the TRAIN split alone and
then applied unchanged to val and test, the way the model itself is:

  constant 0.5           the "always Medium" forecaster used during training
  train mean             the best constant under squared error (BCE's optimum too)
  train median           the best constant under absolute error (MAE's optimum)
  edge x AM/PM history   each camera edge's mean training label for that session type --
                         a "historical average" forecaster using only the road and the
                         time of day, both permitted inputs
  calibrated model       the model's own scores through an affine map fitted on train
                         (src/models/risk_calibration.py), monotone, so rankings unchanged
  nearest camera         propagation: the human label observed at the NEAREST OTHER camera in
                         the same window (the next nearest when that one has no label then,
                         else the train mean) -- what borrowing from a neighbouring camera,
                         as the router's propagated tier does, would predict

Besides MAE / RMSE / R2 every table reports Spearman's rank correlation (does a higher score
mean more congestion), class_means() the mean score per Light / Medium / Heavy label (it
should rise with the label), and per_camera() the same comparison camera by camera.

These are comparisons of edge-risk regression only. They say nothing about routes, which
scripts/evaluate_routing.py compares against the Improved ACO baseline with Google Maps as the
reference.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from src.models import risk_calibration

SPLITS = ("train", "val", "test")
CLASSES = {0.0: "Light", 0.5: "Medium", 1.0: "Heavy"}
NEAREST = "nearest camera (same window)"


class Cameras:
    """
    Which camera each camera edge belongs to, and every camera's other cameras by distance.
    An edge belongs to a camera when exactly one of its endpoints is that camera's node.
    """

    def __init__(self, node_camera: Mapping[int, str], coords: Mapping[str, tuple]):
        self.node_camera = {int(n): c for n, c in node_camera.items()}
        names = sorted(coords)
        self.order: Dict[str, List[str]] = {}
        for c in names:
            lat, lon = coords[c]
            others = [o for o in names if o != c]
            self.order[c] = sorted(others, key=lambda o: (coords[o][0] - lat) ** 2
                                   + ((coords[o][1] - lon) * np.cos(np.radians(lat))) ** 2)

    def of(self, source: int, target: int) -> Optional[str]:
        a, b = self.node_camera.get(int(source)), self.node_camera.get(int(target))
        if (a is None) == (b is None):
            return None
        return a or b


def with_cameras(rows: pd.DataFrame, cameras: Cameras) -> pd.DataFrame:
    out = rows.copy()
    out["camera"] = [cameras.of(u, v) for u, v in zip(out["source_node_id"], out["target_node_id"])]
    return out


def labelled_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Camera-edge rows with a human target, from risk_edges.csv."""
    camera = frame["camera_edge"].astype(str).str.lower().eq("true")
    out = frame[camera & frame["weak_target"].notna()].copy()
    out["edge"] = out["source_node_id"].astype(str) + ">" + out["target_node_id"].astype(str)
    return out


def scores(target: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    target, prediction = np.asarray(target, float), np.asarray(prediction, float)
    error = prediction - target
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    return {
        "n": int(target.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "r2": 1 - float(np.sum(error ** 2)) / ss_tot if ss_tot > 0 else float("nan"),
        "spearman": spearman(target, prediction),
    }


def spearman(target, prediction) -> float:
    """Rank correlation; NaN when either side is constant (a constant forecaster ranks nothing)."""
    target, prediction = np.asarray(target, float), np.asarray(prediction, float)
    if target.size < 3 or np.ptp(target) == 0 or np.ptp(prediction) == 0:
        return float("nan")
    return float(stats.spearmanr(target, prediction).statistic)


def fit_predictors(train: pd.DataFrame) -> Dict[str, object]:
    """Every reference's parameters, from the training rows only."""
    if train.empty:
        raise ValueError("no labelled training rows to fit the baselines on")
    history = train.groupby(["edge", "session"])["weak_target"].mean().to_dict()
    try:
        calibration = risk_calibration.fit(train["risk"].to_numpy(float), train["weak_target"].to_numpy(float))
    except ValueError:
        calibration = None
    return {
        "mean": float(train["weak_target"].mean()),
        "median": float(train["weak_target"].median()),
        "history": history,
        "calibration": calibration,
    }


def nearest_camera(rows: pd.DataFrame, labelled: pd.DataFrame, cameras: Cameras, fallback: float) -> np.ndarray:
    """
    For each row, the mean human label at the nearest OTHER camera in the same window, walking
    outwards until a camera has a label then; `fallback` (the train mean) when none has.
    """
    observed = labelled.dropna(subset=["camera"]).groupby(["camera", "window_start"])["weak_target"].mean().to_dict()
    out = np.full(len(rows), fallback, dtype=float)
    for i, (camera, window) in enumerate(zip(rows["camera"], rows["window_start"])):
        for other in cameras.order.get(camera, []) if camera else []:
            value = observed.get((other, window))
            if value is not None:
                out[i] = value
                break
    return out


def predictions(
    rows: pd.DataFrame,
    fitted: Dict[str, object],
    labelled: Optional[pd.DataFrame] = None,
    cameras: Optional[Cameras] = None,
) -> Dict[str, np.ndarray]:
    n = len(rows)
    keys = list(zip(rows["edge"], rows["session"]))
    history = np.array([fitted["history"].get(k, fitted["mean"]) for k in keys], dtype=float)
    out = {
        "model": rows["risk"].to_numpy(float),
        "constant 0.5": np.full(n, 0.5),
        "train mean": np.full(n, fitted["mean"]),
        "train median": np.full(n, fitted["median"]),
        "edge x AM/PM history": history,
    }
    if fitted["calibration"] is not None:
        out["model, calibrated on train"] = fitted["calibration"].apply(rows["risk"].to_numpy(float))
    if cameras is not None and labelled is not None and "camera" in rows.columns:
        out[NEAREST] = nearest_camera(rows, labelled, cameras, fitted["mean"])
    return out


def _prepared(frame: pd.DataFrame, cameras: Optional[Cameras]):
    rows = labelled_rows(frame)
    if cameras is not None:
        rows = with_cameras(rows, cameras)
    return rows, fit_predictors(rows[rows["split"] == "train"])


def baseline_table(frame: pd.DataFrame, cameras: Optional[Cameras] = None) -> pd.DataFrame:
    """One row per (split, predictor): n, MAE, RMSE, R2, Spearman."""
    rows, fitted = _prepared(frame, cameras)
    table: List[dict] = []
    for split in SPLITS:
        part = rows[rows["split"] == split]
        if part.empty:
            continue
        y = part["weak_target"].to_numpy(float)
        for name, p in predictions(part, fitted, rows, cameras).items():
            table.append({"split": split, "predictor": name, **scores(y, p)})
    return pd.DataFrame(table)


def breakdown(frame: pd.DataFrame, by: str, cameras: Optional[Cameras] = None) -> pd.DataFrame:
    """
    MAE per split and `by` ('session' = date + AM/PM, 'target', or 'camera') for every
    predictor; per camera also the model's Spearman correlation there.
    """
    rows, fitted = _prepared(frame, cameras)
    rows["session_key"] = rows["day"].astype(str) + " " + rows["session"]
    if by == "camera" and "camera" not in rows.columns:
        raise ValueError("a per-camera breakdown needs the cameras")
    key = {"session": "session_key", "target": "weak_target", "camera": "camera"}[by]
    out = []
    for (split, value), part in rows[rows["split"].isin(SPLITS)].groupby(["split", key]):
        y = part["weak_target"].to_numpy(float)
        entry = {"split": split, by: value, "n": len(part), "target_mean": float(y.mean())}
        for name, p in predictions(part, fitted, rows, cameras).items():
            entry[name] = float(np.mean(np.abs(p - y)))
        if by == "camera":
            entry["model_spearman"] = spearman(y, part["risk"].to_numpy(float))
        out.append(entry)
    return pd.DataFrame(out)


def class_means(frame: pd.DataFrame) -> pd.DataFrame:
    """
    The model's mean score per human label, per split: a score that means something rises
    from Light to Medium to Heavy. Also the share of Light/Heavy pairs ranked the right way
    round (the probability a Heavy row scores above a Light one; 0.5 is chance).
    """
    rows = labelled_rows(frame)
    out = []
    for split in SPLITS:
        part = rows[rows["split"] == split]
        if part.empty:
            continue
        entry = {"split": split}
        for value, name in CLASSES.items():
            scores_ = part.loc[part["weak_target"] == value, "risk"].to_numpy(float)
            entry[f"{name}_n"] = int(scores_.size)
            entry[f"{name}_mean_risk"] = float(scores_.mean()) if scores_.size else float("nan")
        heavy = part.loc[part["weak_target"] == 1.0, "risk"].to_numpy(float)
        light = part.loc[part["weak_target"] == 0.0, "risk"].to_numpy(float)
        if heavy.size and light.size:
            u = stats.mannwhitneyu(heavy, light, alternative="two-sided")
            entry["heavy_above_light"] = float(u.statistic / (heavy.size * light.size))
            entry["heavy_vs_light_p"] = float(u.pvalue)
        out.append(entry)
    return pd.DataFrame(out)


def variance_split(frame: pd.DataFrame) -> Dict[str, float]:
    """
    How much of the predicted risk varies between time windows versus between roads.

    A model that mostly moves every road together (a network-wide shift per window) is not
    making road-specific predictions, whatever its MAE.
    """
    edge = frame["source_node_id"].astype(str) + ">" + frame["target_node_id"].astype(str)
    grid = frame.assign(edge=edge).pivot_table(index="window_start", columns="edge", values="risk")
    total = float(np.nanvar(grid.values))
    return {
        "total": total,
        "between_windows": float(grid.mean(axis=1).var(ddof=0)) / total if total else float("nan"),
        "between_roads": float(grid.mean(axis=0).var(ddof=0)) / total if total else float("nan"),
    }
