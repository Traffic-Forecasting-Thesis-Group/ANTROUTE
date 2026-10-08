"""
Score a risk_edges.csv against simple predictors on the same labelled rows
(src/models/risk_baselines.py), every reference fitted on the train split only.

    python scripts/evaluate_risk_baselines.py --risk-edges data/processed/risk_scores/risk_edges.csv \
        --out outputs/risk_eval/v7_baselines.json

Prints the overall table, MAE per session and per target class, and how much of the model's
risk varies between windows versus between roads. This is edge-risk regression only; route
quality and ETA are scripts/evaluate_routing.py's job.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.models.risk_baselines import baseline_table, breakdown, variance_split  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--risk-edges", type=Path, default=REPO_ROOT / "data/processed/risk_scores/risk_edges.csv")
    p.add_argument("--out", type=Path, default=None, help="also write every table as JSON here")
    a = p.parse_args()

    frame = pd.read_csv(a.risk_edges)
    pd.set_option("display.width", 200)
    overall = baseline_table(frame)
    sessions, targets = breakdown(frame, "session"), breakdown(frame, "target")
    spread = variance_split(frame)
    print("Edge-risk regression, every reference fitted on the TRAIN split only:\n")
    print(overall.round(4).to_string(index=False))
    print("\nMAE per session:\n" + sessions.round(3).to_string(index=False))
    print("\nMAE per target class:\n" + targets.round(3).to_string(index=False))
    print(f"\nshare of predicted-risk variance between windows {spread['between_windows']:.0%}, "
          f"between roads {spread['between_roads']:.0%}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({
            "risk_edges": str(a.risk_edges),
            "overall": overall.to_dict(orient="records"),
            "per_session": sessions.to_dict(orient="records"),
            "per_target": targets.to_dict(orient="records"),
            "variance": spread,
        }, indent=2, default=float), encoding="utf-8")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
