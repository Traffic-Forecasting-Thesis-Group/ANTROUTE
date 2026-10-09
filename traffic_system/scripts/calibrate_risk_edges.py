"""
Add a calibrated risk column to a scored risk_edges.csv.

    python scripts/calibrate_risk_edges.py                      # preview the fit, write nothing
    python scripts/calibrate_risk_edges.py --apply              # add risk_calibrated in place
    python scripts/calibrate_risk_edges.py --apply --out other.csv

The STGNN's sigmoid output sits well below the scale of the Light/Medium/Heavy targets it is
scored against, so most of its reported MAE is a constant offset rather than a wrong ranking
of roads. src/models/risk_calibration.py explains why that happens and why a single affine
map fitted on the training split is the right correction.

The new column is added alongside `risk`, never over it, so the raw and calibrated figures
can be reported side by side. The map is monotone, so the ordering of edges by risk is
unchanged -- but routes are not guaranteed to be: W = d * (1 + lambda * Risk) weighs risk
against distance, and rescaling Risk changes that balance. The app routes on `risk` unless
RISK_SCORES=calibrated (see src/models/risk_calibration.py).

The fit uses the TRAIN split only; the val and test figures printed here are held out from it.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from src.models.risk_calibration import class_separation, fit, metrics  # noqa: E402

RISK_EDGES = REPO_ROOT / "data" / "processed" / "risk_scores" / "risk_edges.csv"
CONSTANT_BASELINES = {"always 0.5": 0.5}


def labelled(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    return frame[(frame["camera_edge"] == True) & frame["weak_target"].notna()  # noqa: E712
                 & (frame["split"] == split)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--risk-edges", type=Path, default=RISK_EDGES)
    parser.add_argument("--out", type=Path, help="write here instead of in place")
    parser.add_argument("--apply", action="store_true", help="write the file (default: preview only)")
    parser.add_argument("--json", type=Path, help="also save the fit and the metrics here")
    args = parser.parse_args()

    frame = pd.read_csv(args.risk_edges)
    for column in ("camera_edge", "weak_target", "split", "risk"):
        if column not in frame.columns:
            raise SystemExit(f"{args.risk_edges} has no {column!r} column")

    train = labelled(frame, "train")
    calibration = fit(train["risk"].to_numpy(), train["weak_target"].to_numpy(), "train")
    print(calibration)

    train_mean = float(train["weak_target"].mean())
    baselines = dict(CONSTANT_BASELINES, **{f"always {train_mean:.3f} (train mean)": train_mean})

    report = {"calibration": calibration.as_dict(), "splits": {}}
    for split in ("train", "val", "test"):
        rows = labelled(frame, split)
        if rows.empty:
            continue
        target = rows["weak_target"].to_numpy()
        raw = rows["risk"].to_numpy()
        per_class, r = class_separation(target, raw)

        print(f"\n--- {split}  (n={len(rows)}, label mean {target.mean():.4f}) ---")
        print(f"  raw prediction mean {raw.mean():.4f}   pearson r {r:.4f}")
        print("  mean raw prediction per true class: "
              + "  ".join(f"{k:.1f}->{v:.4f}" for k, v in sorted(per_class.items())))
        print(f"\n  {'predictor':34s} {'MAE':>8s} {'RMSE':>8s} {'R2':>8s}")
        entry = {"n": len(rows), "label_mean": target.mean(), "pearson_r": r,
                 "mean_prediction_per_class": per_class, "predictors": {}}
        candidates = {"ANTROUTE raw": raw, "ANTROUTE calibrated": calibration.apply(raw)}
        candidates.update({name: pd.Series(value, index=rows.index).to_numpy()
                           for name, value in ((n, [v] * len(rows)) for n, v in baselines.items())})
        for name, prediction in candidates.items():
            m = metrics(target, prediction)
            entry["predictors"][name] = m
            print(f"  {name:34s} {m['mae']:8.4f} {m['rmse']:8.4f} {m['r2']:8.4f}")
        report["splits"][split] = entry

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")

    if not args.apply:
        print("\npreview only; pass --apply to add the risk_calibrated column")
        return

    frame["risk_calibrated"] = calibration.apply(frame["risk"].to_numpy())
    destination = args.out or args.risk_edges
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    print(f"\nwrote {len(frame)} rows with a risk_calibrated column -> {destination}")


if __name__ == "__main__":
    main()
