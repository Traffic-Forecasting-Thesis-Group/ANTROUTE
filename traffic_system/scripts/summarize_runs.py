"""
Average and spread of training runs that differ only by seed.

    python scripts/summarize_runs.py checkpoints/v9_*.result.json
    python scripts/summarize_runs.py "/content/drive/MyDrive/MMDA_CHECKPOINTS/v9_*.result.json" --out runs.json

Each `.result.json` is written by scripts/train_stgnn_edge.py next to its checkpoint. Runs with
the same training options (visual init, augmentation, post duration, incident weight, patience,
class weights, context, text) form one configuration; for each, this reports mean +- standard
deviation over its seeds of the best-epoch validation loss and MAE and the test loss and MAE,
plus the epoch it stopped at. Validation decides between configurations; test is reported, never
used to choose. One seed is not a result: a configuration with fewer than three is flagged.
"""

import argparse
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path

MIN_SEEDS = 3


def mean_sd(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else None)


def fmt(m, s):
    if m is None:
        return "    -        "
    return f"{m:.4f} +- {s:.4f}" if s is not None else f"{m:.4f} (1 run)"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", nargs="+", help=".result.json files or glob patterns")
    p.add_argument("--out", type=Path, default=None, help="also write the summary as JSON")
    a = p.parse_args()

    paths = sorted({f for pattern in a.results for f in glob.glob(pattern)} | {r for r in a.results if Path(r).is_file()})
    if not paths:
        raise SystemExit("no .result.json files matched")
    groups = defaultdict(list)
    for path in paths:
        run = json.loads(Path(path).read_text(encoding="utf-8"))
        key = json.dumps(run.get("options", {}), sort_keys=True)
        groups[key].append(run)

    summary = []
    for key, runs in groups.items():
        val = [r.get("val") or {} for r in runs]
        test = [r.get("test") or {} for r in runs]
        row = {
            "options": json.loads(key),
            "seeds": sorted(r.get("seed") for r in runs),
            "val_loss": mean_sd([v.get("loss") for v in val]),
            "val_mae": mean_sd([v.get("mae") for v in val]),
            "test_loss": mean_sd([t.get("loss") for t in test]),
            "test_mae": mean_sd([t.get("mae") for t in test]),
            "best_epoch": mean_sd([r.get("best_epoch") for r in runs]),
            "stopped_early": sum(bool(r.get("stopped_early")) for r in runs),
            "enough_seeds": len(runs) >= MIN_SEEDS,
        }
        summary.append(row)
    summary.sort(key=lambda r: (r["val_loss"][0] is None, r["val_loss"][0] or 0.0))

    for row in summary:
        o = row["options"]
        print(f"\n{o}")
        print(f"  seeds {row['seeds']}" + ("" if row["enough_seeds"] else f"   <-- fewer than {MIN_SEEDS} seeds"))
        print(f"  val  loss {fmt(*row['val_loss'])}   MAE {fmt(*row['val_mae'])}")
        print(f"  test loss {fmt(*row['test_loss'])}   MAE {fmt(*row['test_mae'])}")
        print(f"  best epoch {fmt(*row['best_epoch'])}   stopped early in {row['stopped_early']} of {len(row['seeds'])}")
    print("\nOrdered by mean validation loss. Choose on validation; report test as it falls.")
    if a.out:
        a.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
