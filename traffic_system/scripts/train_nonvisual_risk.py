"""
Train and validate the non-visual Congestion Risk model (src/models/nonvisual_risk.py) that
scores roads outside the camera subgraph.

    python scripts/train_nonvisual_risk.py

Labels: the camera road segments of risk_edges.csv (weak_target, 42 segments at 7 cameras)
with that file's whole-date train / val / test split. Features: only non-visual inputs, built
on the whole-city graph with the same code the app uses at prediction time.

Validation, all against the same held-out rows:
  temporal   fit on train, report val; fit on train+val, report test (the deployed model)
  spatial    leave one camera out: fit on the other six cameras' train+val days, predict the
             held-out camera's test days -- the closest available stand-in for a road with no
             camera, which is the setting the model is used in
Baselines: the training-label mean; the training mean per session (AM / PM peak); the
window's median predicted risk (what roads outside the subgraph got before this change); and,
for reference only, the camera model's own prediction at that segment (not available off the
subgraph) and each segment's own training mean per session (not available for an unseen road).

Writes the model to data/processed/risk_scores/nonvisual_risk.joblib, the placed-incident
table it needs to data/processed/events/placed_incidents.csv, and the validation report to
outputs/nonvisual_risk/report.json. No city-wide risk file is written: the app predicts per
window on demand.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.graph_data import build_full_graph  # noqa: E402
from src.models.nonvisual_risk import (  # noqa: E402
    COLUMN,
    FEATURES,
    MODEL_FEATURES,
    NonVisualRiskModel,
    PlacedIncidents,
    RoadFeatures,
    with_modality_dropout,
)
from src.models import risk_calibration  # noqa: E402


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    from scipy.stats import spearmanr

    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    rho = spearmanr(y, p).correlation if np.std(p) > 0 and np.std(y) > 0 else float("nan")
    return {
        "n": int(y.size),
        "mae": float(np.mean(np.abs(y - p))),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "spearman": None if not np.isfinite(rho) else float(rho),
        "prediction_mean": float(np.mean(p)),
        "prediction_sd": float(np.std(p)),
    }


def camera_points(spatial_dir: Path) -> dict:
    nodes = pd.read_csv(spatial_dir / "full_network_static_features.csv")
    cams = nodes[nodes["is_cctv_node"].astype(str).str.lower().isin(["true", "1"])]
    return {str(r.cctv_label): (float(r.lat), float(r.lon)) for r in cams.itertuples() if isinstance(r.cctv_label, str)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--risk-edges", type=Path, default=REPO_ROOT / "data/processed/risk_scores/risk_edges.csv")
    p.add_argument("--spatial-dir", type=Path, default=REPO_ROOT / "data/processed/spatial")
    p.add_argument("--weather-csv", type=Path, default=REPO_ROOT / "data/processed/temporal/weatherstack_historical.csv")
    p.add_argument("--twitter-root", type=Path, default=REPO_ROOT / "data/raw/twitter")
    p.add_argument("--incidents", type=Path, default=REPO_ROOT / "data/processed/events/placed_incidents.csv")
    p.add_argument("--model-out", type=Path, default=REPO_ROOT / "data/processed/risk_scores/nonvisual_risk.joblib")
    p.add_argument("--report", type=Path, default=REPO_ROOT / "outputs/nonvisual_risk/report.json")
    a = p.parse_args()

    graph = build_full_graph(a.spatial_dir)
    if a.incidents.exists():
        incidents = PlacedIncidents.load(a.incidents)
    else:
        print("placing the MMDA alert feed on the map (once; takes a few minutes)...")
        incidents = PlacedIncidents.build(
            a.twitter_root, REPO_ROOT / "configs/event_landmarks.csv",
            REPO_ROOT / "configs/event_intersections.csv", camera_points(a.spatial_dir),
        )
        incidents.save(a.incidents)
    print(f"incidents: {len(incidents.events)} placed alerts, {len(incidents.points)} placeable points, "
          f"{len(incidents.feed_days)} feed days")
    roads = RoadFeatures.build(graph, a.spatial_dir, a.weather_csv, incidents)

    # ---- labelled camera segments, mapped onto the whole-city edge list
    header = pd.read_csv(a.risk_edges, nrows=0).columns
    frame = pd.read_csv(a.risk_edges, usecols=["window_start", "session", "source_node_id", "target_node_id",
                                               "risk", "weak_target", "split"]
                        + (["risk_calibrated"] if "risk_calibrated" in header else []))
    median_by_window = frame.groupby("window_start")["risk"].median()  # the fallback being replaced
    labelled = frame[frame["weak_target"].notna() & frame["split"].isin(["train", "val", "test"])].copy()
    edge_of = {(int(u), int(v)): i for i, (u, v) in enumerate(zip(roads.node_ids[roads.src], roads.node_ids[roads.dst]))}
    labelled["edge"] = [edge_of.get((int(u), int(v)), -1) for u, v in zip(labelled.source_node_id, labelled.target_node_id)]
    labelled = labelled[labelled["edge"] >= 0].reset_index(drop=True)
    node_label = {}
    static = pd.read_csv(a.spatial_dir / "full_network_static_features.csv", usecols=["node_id", "cctv_label"])
    for nid, label in zip(static["node_id"], static["cctv_label"]):
        if isinstance(label, str) and label:
            node_label[int(nid)] = label
    labelled["camera"] = [node_label.get(int(u)) or node_label.get(int(v)) or "?" for u, v in
                          zip(labelled.source_node_id, labelled.target_node_id)]
    labelled["is_am"] = pd.to_datetime(labelled["window_start"]).dt.hour < 12

    # ---- features for every labelled row, window by window
    x = np.zeros((len(labelled), len(FEATURES)), dtype=np.float32)
    for window, rows in labelled.groupby("window_start").groups.items():
        start = datetime.fromisoformat(str(window))
        x[np.asarray(rows)] = roads.at_window(start, labelled.loc[rows, "edge"].to_numpy())
    y = labelled["weak_target"].to_numpy(dtype=float)
    split = labelled["split"].to_numpy()
    labelled["median_fallback"] = labelled["window_start"].map(median_by_window).to_numpy(dtype=float)
    print(f"{len(labelled)} labelled rows, {labelled['edge'].nunique()} segments, {labelled['camera'].nunique()} cameras")

    def baselines(fit_mask: np.ndarray, eval_mask: np.ndarray) -> dict:
        fit, ev = labelled[fit_mask], labelled[eval_mask]
        mean = float(fit["weak_target"].mean())
        session_mean = fit.groupby("is_am")["weak_target"].mean()
        per_edge = fit.groupby(["edge", "is_am"])["weak_target"].mean()
        edge_hist = np.array([per_edge.get((e, s), session_mean.get(s, mean)) for e, s in zip(ev["edge"], ev["is_am"])])
        yy = ev["weak_target"].to_numpy(dtype=float)
        return {
            "train_mean": metrics(yy, np.full(len(ev), mean)),
            "session_mean": metrics(yy, ev["is_am"].map(session_mean).fillna(mean).to_numpy(dtype=float)),
            "window_median_fallback": metrics(yy, ev["median_fallback"].to_numpy(dtype=float)),
            "camera_model_reference": metrics(yy, ev["risk"].to_numpy(dtype=float)),
            "segment_history_reference": metrics(yy, edge_hist),
        }

    report = {"features_built": FEATURES, "model_features": MODEL_FEATURES,
              "params": NonVisualRiskModel.PARAMS, "rows": int(len(labelled)),
              "segments": int(labelled["edge"].nunique()), "cameras": sorted(labelled["camera"].unique())}

    # ---- temporal hold-out
    train, val, test = split == "train", split == "val", split == "test"
    m_val = NonVisualRiskModel.fit(x[train], y[train], {"split": "train"})
    report["temporal_val"] = {"nonvisual": metrics(y[val], m_val.predict(x[val])), **baselines(train, val)}
    deployed = NonVisualRiskModel.fit(x[train | val], y[train | val], {
        "split": "train+val", "rows": int((train | val).sum()), "trained_at": datetime.now().isoformat(timespec="seconds"),
        "risk_edges": str(a.risk_edges),
    })
    report["temporal_test"] = {"nonvisual": metrics(y[test], deployed.predict(x[test])), **baselines(train | val, test)}

    # ---- spatial hold-out: leave one camera out, test days only
    per_camera, pooled_y, pooled = {}, [], {"nonvisual": [], "train_mean": [], "session_mean": [], "window_median_fallback": []}
    for cam in sorted(labelled["camera"].unique()):
        held = labelled["camera"].to_numpy() == cam
        fit_mask, eval_mask = (~held) & (train | val), held & test
        if eval_mask.sum() == 0 or fit_mask.sum() == 0:
            continue
        model = NonVisualRiskModel.fit(x[fit_mask], y[fit_mask], {"split": f"loco-{cam}"})
        pred = model.predict(x[eval_mask])
        fit = labelled[fit_mask]
        mean = float(fit["weak_target"].mean())
        session_mean = fit.groupby("is_am")["weak_target"].mean()
        ev = labelled[eval_mask]
        preds = {
            "nonvisual": pred,
            "train_mean": np.full(len(ev), mean),
            "session_mean": ev["is_am"].map(session_mean).fillna(mean).to_numpy(dtype=float),
            "window_median_fallback": ev["median_fallback"].to_numpy(dtype=float),
        }
        per_camera[cam] = {k: metrics(y[eval_mask], v) for k, v in preds.items()}
        pooled_y.append(y[eval_mask])
        for k, v in preds.items():
            pooled[k].append(v)
    yy = np.concatenate(pooled_y)
    report["leave_one_camera_out"] = {
        "pooled": {k: metrics(yy, np.concatenate(v)) for k, v in pooled.items()},
        "per_camera": per_camera,
    }

    # ---- why MODEL_FEATURES: the same leave-one-camera-out run with every built column
    def loco_mae(features):
        ys, ps = [], []
        for cam in sorted(labelled["camera"].unique()):
            held = labelled["camera"].to_numpy() == cam
            fit_mask, eval_mask = (~held) & (train | val), held & test
            if eval_mask.sum() == 0 or fit_mask.sum() == 0:
                continue
            model = NonVisualRiskModel.fit(x[fit_mask], y[fit_mask], {"split": f"loco-{cam}"}, features=features)
            ys.append(y[eval_mask])
            ps.append(model.predict(x[eval_mask]))
        return metrics(np.concatenate(ys), np.concatenate(ps))
    report["feature_selection"] = {
        "criterion": "leave-one-camera-out MAE on test days (lower is better)",
        "inputs_only (deployed)": loco_mae(MODEL_FEATURES),
        "all_built_features": loco_mae(FEATURES),
        "constant_train_mean": report["leave_one_camera_out"]["pooled"]["train_mean"],
    }

    # ---- out-of-fold calibration onto the label scale (training data only)
    # Each train+val row is predicted by a model fitted without its camera, and with incidents
    # withheld as well as known (most roads off the cameras have incident status unavailable),
    # so the calibration sees the model as it is used: on places it was not trained on.
    fold_x, fold_y, fold_p = [], [], []
    trainval = train | val
    for cam in sorted(labelled["camera"].unique()):
        held = labelled["camera"].to_numpy() == cam
        if not (held & trainval).any():
            continue
        model = NonVisualRiskModel.fit(x[(~held) & trainval], y[(~held) & trainval], {"split": f"oof-{cam}"})
        xs, ys = with_modality_dropout(x[held & trainval], y[held & trainval])
        fold_p.append(model.predict(xs))
        fold_y.append(ys)
    oof_p, oof_y = np.concatenate(fold_p), np.concatenate(fold_y)
    calibration = risk_calibration.fit(oof_p, oof_y, fitted_on="out-of-fold train+val (leave one camera out, with modality dropout)")
    deployed.calibration = calibration.as_dict()
    print(f"\nnon-visual calibration: {calibration}")
    test_rows = labelled.loc[test]
    report["calibration"] = {
        "nonvisual": deployed.calibration,
        "nonvisual_oof_mae": {"raw": metrics(oof_y, oof_p), "calibrated": metrics(oof_y, calibration.apply(oof_p))},
        # Comparability on the held-out test day, where both models exist: camera segments.
        "test_day_camera_segments": {
            "labels_mean": float(y[test].mean()),
            "nonvisual_raw": metrics(y[test], deployed.predict(x[test])),
            "nonvisual_calibrated": metrics(y[test], deployed.predict(x[test], calibrated=True)),
            "camera_model_raw": metrics(y[test], test_rows["risk"].to_numpy(dtype=float)),
            "camera_model_calibrated": (metrics(y[test], test_rows["risk_calibrated"].to_numpy(dtype=float))
                                        if "risk_calibrated" in test_rows else "run scripts/calibrate_risk_edges.py --apply"),
        },
    }
    for k, v in report["calibration"]["test_day_camera_segments"].items():
        if isinstance(v, dict):
            print(f"  test-day camera segments, {k:24s} MAE {v['mae']:.3f}  sd(pred) {v['prediction_sd']:.3f}")

    # ---- does the prediction respond to each road-specific input? (response, not accuracy)
    rng = np.random.default_rng(0)
    off_camera = rng.choice(roads.src.size, size=3000, replace=False)
    base_x = roads.at_window(datetime(2026, 5, 25, 17, 30), off_camera)
    base = deployed.predict(base_x)
    def shifted(col, value):
        z = base_x.copy()
        z[:, COLUMN[col]] = value
        return float(np.mean(deployed.predict(z) - base))
    report["response_checks"] = {
        "baseline_mean": float(base.mean()), "baseline_sd": float(base.std()),
        "precip_plus_20mm": shifted("precip", np.nan_to_num(base_x[:, COLUMN["precip"]]) + 20.0),
        "incident_impact_1": shifted("incident_impact", 1.0),
        "flood_level_max": shifted("flood_level", 1.0),
        "edge_speed_halved": shifted("edge_speed", base_x[:, COLUMN["edge_speed"]] * 0.5),
        "am_peak_clock": float(np.mean(deployed.predict(roads.at_window(datetime(2026, 5, 25, 7, 30), off_camera)) - base)),
        "share_with_known_incident_status": float(np.mean(np.isfinite(base_x[:, COLUMN["incident_impact"]]))),
        "share_with_weather": float(np.mean(base_x[:, COLUMN["weather_available"]])),
    }
    importances = {}
    try:
        from sklearn.inspection import permutation_importance
        sample = rng.choice(np.flatnonzero(test), size=min(800, int(test.sum())), replace=False)
        cols = [COLUMN[f] for f in deployed.features]
        pi = permutation_importance(deployed.estimator, x[sample][:, cols], y[sample], n_repeats=5, random_state=0,
                                    scoring="neg_mean_absolute_error")
        importances = {f: float(v) for f, v in sorted(zip(deployed.features, pi.importances_mean), key=lambda t: -t[1])}
    except Exception as err:  # importance is diagnostic only
        importances = {"error": str(err)}
    report["permutation_importance_test"] = importances

    deployed.save(a.model_out)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def line(name, block):
        return f"  {name:28s} MAE {block['mae']:.3f}  RMSE {block['rmse']:.3f}  sd(pred) {block['prediction_sd']:.3f}"
    for section in ("temporal_val", "temporal_test"):
        print(f"\n{section}:")
        for k, v in report[section].items():
            print(line(k, v))
    print("\nleave one camera out (test days), pooled:")
    for k, v in report["leave_one_camera_out"]["pooled"].items():
        print(line(k, v))
    print("\nfeature selection (leave one camera out):")
    for k, v in report["feature_selection"].items():
        if isinstance(v, dict):
            print(line(k, v))
    print("\nresponse checks (mean change in predicted risk on 3,000 random roads):")
    for k, v in report["response_checks"].items():
        print(f"  {k:34s} {v:+.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"\nwrote {a.model_out} and {a.report}")


if __name__ == "__main__":
    main()
