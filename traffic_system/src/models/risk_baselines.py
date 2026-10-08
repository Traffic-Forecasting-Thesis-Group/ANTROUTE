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

These are comparisons of edge-risk regression only. They say nothing about routes or ETAs,
which scripts/evaluate_routing.py measures against the Improved ACO baseline and Apple Maps.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from src.models import risk_calibration

SPLITS = ("train", "val", "test")


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
    }


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


def predictions(rows: pd.DataFrame, fitted: Dict[str, object]) -> Dict[str, np.ndarray]:
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
    return out


def baseline_table(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (split, predictor): n, MAE, RMSE, R2."""
    rows = labelled_rows(frame)
    fitted = fit_predictors(rows[rows["split"] == "train"])
    table: List[dict] = []
    for split in SPLITS:
        part = rows[rows["split"] == split]
        if part.empty:
            continue
        y = part["weak_target"].to_numpy(float)
        for name, p in predictions(part, fitted).items():
            table.append({"split": split, "predictor": name, **scores(y, p)})
    return pd.DataFrame(table)


def breakdown(frame: pd.DataFrame, by: str) -> pd.DataFrame:
    """MAE per split and `by` ('session' = date + AM/PM, or 'target') for every predictor."""
    rows = labelled_rows(frame)
    fitted = fit_predictors(rows[rows["split"] == "train"])
    rows["session_key"] = rows["day"].astype(str) + " " + rows["session"]
    key = {"session": "session_key", "target": "weak_target"}[by]
    out = []
    for (split, value), part in rows[rows["split"].isin(SPLITS)].groupby(["split", key]):
        y = part["weak_target"].to_numpy(float)
        entry = {"split": split, by: value, "n": len(part), "target_mean": float(y.mean())}
        for name, p in predictions(part, fitted).items():
            entry[name] = float(np.mean(np.abs(p - y)))
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
