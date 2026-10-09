"""
Are the camera model's and the non-visual model's risk scores comparable, raw and calibrated?

    python scripts/verify_risk_calibration.py
    python scripts/verify_risk_calibration.py --windows 2026-05-25T07:30:00 2026-05-25T17:30:00

Routing mixes three layers on one city graph (app/risk_routing._risk_layers): the camera model's
predictions, those borrowed within 1.5 km, and the non-visual model everywhere else. If the
layers sit at different levels for reasons unrelated to congestion, W = d * (1 + lambda * Risk)
favours whichever layer reads lower. This reports, per window and per layer, the mean and spread
of the risk under RISK_SCORES=raw and =calibrated, and the gap between the non-visual layer and
the camera-model layers (predicted + borrowed).

Read with care: a smaller gap means the layers are on a more similar scale, not that either is
more accurate -- they cover different roads, so their true congestion levels may differ too.
Accuracy is in outputs/nonvisual_risk/report.json (calibration section) and
scripts/calibrate_risk_edges.py.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from app import risk_routing as rr  # noqa: E402
from app.config import settings  # noqa: E402
from src.models.nonvisual_risk import SOURCE_BORROWED, SOURCE_NAMES, SOURCE_NONVISUAL, SOURCE_PREDICTED  # noqa: E402

DEFAULT_WINDOWS = ["2026-05-25T07:00:00", "2026-05-25T07:30:00", "2026-05-25T17:30:00", "2026-05-25T18:20:00"]


def layers(window: str, mode: str) -> dict:
    settings.risk_scores = mode
    for cached in (rr._risk_layers, rr._nonvisual_risk, rr._weighted_graph, rr._informed_edges):
        cached.cache_clear()
    risk, sources, _ = rr._risk_layers(window)
    out = {}
    for code, name in SOURCE_NAMES.items():
        r = risk[sources == code]
        if r.size:
            out[name] = {"n": int(r.size), "mean": float(r.mean()), "sd": float(r.std()),
                         "p10": float(np.percentile(r, 10)), "p90": float(np.percentile(r, 90))}
    camera = risk[(sources == SOURCE_PREDICTED) | (sources == SOURCE_BORROWED)]
    nonvisual = risk[sources == SOURCE_NONVISUAL]
    out["gap_nonvisual_minus_camera_layers"] = float(nonvisual.mean() - camera.mean()) if camera.size and nonvisual.size else None
    # How much more an average non-visual road costs than an average camera-layer road, per metre.
    out["cost_ratio_nonvisual_vs_camera"] = (
        float((1 + rr.DEFAULT_LAMBDA * nonvisual.mean()) / (1 + rr.DEFAULT_LAMBDA * camera.mean()))
        if camera.size and nonvisual.size else None
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--windows", nargs="+", default=DEFAULT_WINDOWS)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/nonvisual_risk/calibration_check.json")
    a = p.parse_args()

    original = settings.risk_scores
    report = {}
    try:
        for window in a.windows:
            report[window] = {mode: layers(window, mode) for mode in ("raw", "calibrated")}
    finally:
        settings.risk_scores = original

    for window, modes in report.items():
        print(f"\n== {window}")
        for mode, res in modes.items():
            parts = [f"{n} {res[n]['mean']:.3f}+-{res[n]['sd']:.3f}" for n in ("predicted", "borrowed", "nonvisual") if n in res]
            print(f"  {mode:10s} " + "   ".join(parts)
                  + f"   | gap {res['gap_nonvisual_minus_camera_layers']:+.3f}   cost ratio {res['cost_ratio_nonvisual_vs_camera']:.3f}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
