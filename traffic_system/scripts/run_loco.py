"""
Run the leave-one-camera-out retrains (on Colab: it is one full training run per camera).

    python scripts/run_loco.py --frames-root /content/drive/MyDrive/MMDA_FRAMES /content/drive/MyDrive/MMDA_FRAMES_B \
        --out-dir /content/drive/MyDrive/MMDA_LOCO

For each of the nine intersections, and once with none held out (the control: same settings and
seed, so "full model" vs "LOCO" differs only by the camera), it trains with v8's own settings --
daily weather, length + speed road attributes, the text rules v8 used (0 km, 1 minute, no text
prior), 8 epochs, the v8 split (configs/visual_split_v8.json) -- and scores the subgraph with
the result. Then:

    python scripts/evaluate_loco.py --reference <out-dir>/control/risk_edges.csv \
        --runs <out-dir>/loco_* --out outputs/risk_eval/loco.json

A finished camera (its risk_edges.csv exists) is skipped, so the loop resumes after a Colab
disconnect. --cameras limits it to some intersections; --dry-run prints the commands.
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CAMERAS = (
    "8337-Ayala NB 1-PTZ", "EDSA-Aurora", "EDSA-Kamuning", "EDSA-Ortigas", "EDSA-Quezon Ave",
    "EDSA-Regalia (P. Tuazon)", "EDSA-Shaw", "Roxas Blvd-Kalaw", "Roxas-Padre Burgos",
)
V8_SETTINGS = [
    "--weather-csv", str(REPO_ROOT / "data/processed/temporal/weatherstack_historical.csv"),
    "--edge-features", "v1", "--text-radius-km", "0", "--text-active-minutes", "1", "--no-text-prior",
    "--epochs", "8", "--split-file", str(REPO_ROOT / "configs/visual_split_v8.json"), "--seed", "0",
]


def slug(camera: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in camera).strip("_")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames-root", type=Path, nargs="+", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--cameras", nargs="*", default=list(CAMERAS))
    p.add_argument("--no-control", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--extra", default="", help="extra train_stgnn_edge.py arguments, e.g. '--batch-size 2'")
    a = p.parse_args()
    runs = ([] if a.no_control else [("control", None)]) + [(f"loco_{slug(c)}", c) for c in a.cameras]
    frames = [str(f) for f in a.frames_root]
    for name, camera in runs:
        run_dir = a.out_dir / name
        if (run_dir / "risk_edges.csv").exists():
            print(f"{name}: done, skipped")
            continue
        checkpoint = run_dir / f"{name}.pt"
        train = [sys.executable, str(REPO_ROOT / "scripts/train_stgnn_edge.py"), "--frames-root", *frames,
                 "--out", str(checkpoint), *V8_SETTINGS, *shlex.split(a.extra)]
        if camera:
            train += ["--hold-out-camera", camera]
        score = [sys.executable, str(REPO_ROOT / "scripts/predict_congestion_risk.py"), "--checkpoint", str(checkpoint),
                 "--frames-root", *frames, "--out-dir", str(run_dir),
                 "--weather-csv", str(REPO_ROOT / "data/processed/temporal/weatherstack_historical.csv")]
        for command in (train, score):
            print("$ " + " ".join(shlex.quote(c) for c in command), flush=True)
            if not a.dry_run:
                run_dir.mkdir(parents=True, exist_ok=True)
                subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
