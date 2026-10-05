"""
Compute the real ANTROUTE-vs-baseline numbers for app/evaluation.py from a scored
risk_edges.csv, instead of hand-typing them.

    python scripts/compute_comparison_metrics.py --risk-edges data/processed/risk_scores/risk_edges.csv

Evaluates on the held-out TEST split only, against the real human-labelled
Light/Medium/Heavy targets (weak_target) on camera edges -- the same ground truth
train_stgnn_edge.py trains against. "Baseline" here is the naive "always predict
Medium (0.5)" constant, the same reference point used throughout training (a
roughly balanced Light/Medium/Heavy label mix gives it MAE ~0.33-0.38 depending on
the exact split).

This is a genuinely harder bar than train_stgnn_edge.py's own reported val_mae:
that number comes from its own internal windowing/target alignment during
training, while this reads the actual deployed artifact (risk_edges.csv) that
predict_congestion_risk.py produced and that the live app serves routes from.
The two numbers are not expected to match, and if the model you're evaluating
barely beats the naive baseline here, that is real and should be reported as
such, not smoothed over.

"Always 0.5" is a weak bar: 57% of the labels are Heavy, so 0.5 is not the mean
label. The output also reports the stronger "always predict the TRAIN-split mean
label" constant, and a collapse check -- the mean predicted risk on Light, Medium
and Heavy edges, and how much the risk varies at all. A model that has collapsed to
(roughly) one value everywhere routes exactly like a shortest-distance router, since
W = dist * (1 + lambda * c) is then the same scaling on every edge.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--risk-edges", type=Path, default=REPO_ROOT / "data/processed/risk_scores/risk_edges.csv")
    p.add_argument("--split", default="test")
    a = p.parse_args()

    df = pd.read_csv(a.risk_edges)
    labelled = df[(df["camera_edge"] == True) & df["weak_target"].notna() & (df["split"] == a.split)]  # noqa: E712
    if labelled.empty:
        raise SystemExit(f"no labelled camera edges in split {a.split!r} of {a.risk_edges}")

    y_true = labelled["weak_target"].to_numpy(dtype=float)
    y_pred = labelled["risk"].to_numpy(dtype=float)
    y_naive = np.full_like(y_true, 0.5)
    train = df[(df["camera_edge"] == True) & df["weak_target"].notna() & (df["split"] == "train")]  # noqa: E712
    train_mean = float(train["weak_target"].mean()) if len(train) else float("nan")

    antroute = regression_metrics(y_true, y_pred)
    baseline = regression_metrics(y_true, y_naive)
    naive_mean = regression_metrics(y_true, np.full_like(y_true, train_mean)) if len(train) else None
    by_label = {
        str(k): round(float(v), 4) for k, v in labelled.groupby("weak_target")["risk"].mean().items()
    }
    spread = (by_label.get("1.0", np.nan) - by_label.get("0.0", np.nan))

    result = {
        "risk_edges": str(a.risk_edges),
        "split": a.split,
        "n_labelled_edges": int(len(labelled)),
        "label_distribution": {str(k): int(v) for k, v in y_true_counts(y_true).items()},
        "antroute": antroute,
        "baseline": baseline,
        "naive_train_mean": {"value": train_mean, **naive_mean} if naive_mean else None,
        "collapse_check": {
            "pearson_r": float(np.corrcoef(y_true, y_pred)[0, 1]) if y_pred.std() > 0 else 0.0,
            "mean_risk_by_label": by_label,
            "heavy_minus_light": round(float(spread), 4),
            "risk_std_all_edges": round(float(df["risk"].std()), 4),
            "risk_std_within_window_median": round(
                float(df.groupby(window_column(df))["risk"].std().median()), 4),
        },
    }
    print(json.dumps(result, indent=2))
    print()
    if naive_mean and antroute["mae"] >= naive_mean["mae"]:
        print(f"WARNING: MAE {antroute['mae']:.3f} does not beat always predicting the training mean "
              f"({naive_mean['mae']:.3f}); the model has not learned usable per-edge risk.")
    if not spread > 0.2:
        print(f"WARNING: Heavy edges score only {spread:.3f} above Light ones; the risk barely separates "
              "congestion levels, so routing will behave like shortest distance.")
    print("Paste the antroute/baseline mae, rmse and r2 values above into")
    print("app/evaluation.py's RAW_RESULTS, with this command in the comment above it.")


def window_column(df: pd.DataFrame) -> str:
    return "window_start" if "window_start" in df.columns else "window_end"


def y_true_counts(y_true: np.ndarray) -> dict:
    values, counts = np.unique(y_true, return_counts=True)
    return dict(zip(values.tolist(), counts.tolist()))


if __name__ == "__main__":
    main()
