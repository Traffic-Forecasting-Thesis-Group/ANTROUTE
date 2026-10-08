"""
Compute the ANTROUTE-vs-baseline congestion forecast numbers from a scored
risk_edges.csv -- the same calculation app/evaluation.py serves live.

    python scripts/compute_comparison_metrics.py --risk-edges data/processed/risk_scores/risk_edges.csv

Evaluates on the held-out TEST split only, against the real human-labelled
Light/Medium/Heavy targets (weak_target) on camera edges -- the same ground truth
train_stgnn_edge.py trains against. "Baseline" here is the naive "always predict
Medium (0.5)" constant, the same reference point used throughout training.

Two constants are reported, not one, because 0.5 is a weak bar on this data: the
labels are not balanced -- the test split is 67% Heavy, mean 0.784 -- so a constant
sitting at the *training* mean is a much stronger reference, and a reader will ask
for it. Both are printed so neither can be chosen after the fact.

Likewise both the raw `risk` column and `risk_calibrated` are scored when the latter
is present (scripts/calibrate_risk_edges.py adds it). The raw scores lose to the
train-mean constant purely through being miscentred rather than misordered; see
src/models/risk_calibration.py. Whichever figures the write-up quotes, quote the
pair, and say which column produced them.

This is a genuinely harder bar than train_stgnn_edge.py's own reported val_mae:
that number comes from its own internal windowing/target alignment during
training, while this reads the actual deployed artifact (risk_edges.csv) that
predict_congestion_risk.py produced and that the live app serves routes from.
The two numbers are not expected to match, and if the model you're evaluating
barely beats the naive baseline here, that is real and should be reported as
such, not smoothed over.
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

    # Fitted on the train split, so it is a held-out reference for every other split.
    train = df[(df["camera_edge"] == True) & df["weak_target"].notna() & (df["split"] == "train")]  # noqa: E712
    train_mean = float(train["weak_target"].mean()) if not train.empty else 0.5

    result = {
        "risk_edges": str(a.risk_edges),
        "split": a.split,
        "n_labelled_edges": int(len(labelled)),
        "label_distribution": {str(k): int(v) for k, v in y_true_counts(y_true).items()},
        "label_mean": float(y_true.mean()),
        "antroute": regression_metrics(y_true, y_pred),
        "baseline": regression_metrics(y_true, np.full_like(y_true, 0.5)),
        "baseline_train_mean": {
            "constant": train_mean,
            **regression_metrics(y_true, np.full_like(y_true, train_mean)),
        },
    }
    if "risk_calibrated" in labelled.columns:
        result["antroute_calibrated"] = regression_metrics(
            y_true, labelled["risk_calibrated"].to_numpy(dtype=float)
        )
    print(json.dumps(result, indent=2))
    print()
    print("GET /routes/comparison-metrics computes these same numbers live from")
    print("risk_edges.csv (when no routing_eval/metrics.json exists); use this to check them offline.")


def y_true_counts(y_true: np.ndarray) -> dict:
    values, counts = np.unique(y_true, return_counts=True)
    return dict(zip(values.tolist(), counts.tolist()))


if __name__ == "__main__":
    main()
