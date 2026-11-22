"""Run the frozen evaluator with per-run durability, outside all search timers."""
import inspect
import json
import sys
import hashlib
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from scripts import evaluate_sop as evaluator

OUT = Path(__file__).resolve().parents[1] / "data/live/closeout_2026-11-22"
CHECKPOINT = OUT / "evaluation_checkpoint.jsonl"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protected = ("configs/live_experiment_manifest.json", "configs/live_study.json", "configs/live_assumptions.json",
                 "configs/study_assumptions.json", "data/processed/live_calibration.json",
                 "data/processed/live_network.json", "data/processed/live_release.json",
                 "outputs/live_search_tuning.json", "src/routing/aco_kernel.py", "scripts/evaluate_sop.py")
    original_hashes = json.loads((OUT / "backup_sha256.json").read_text())
    def verify_frozen():
        for name in protected:
            actual = hashlib.sha256((evaluator.ROOT / name).read_bytes()).hexdigest()
            if actual != original_hashes[name]:
                raise RuntimeError(f"Frozen input changed since backup: {name}")
    verify_frozen()
    if CHECKPOINT.exists():
        raise SystemExit("Checkpoint already exists; refusing to overwrite or rerun measured searches.")
    count = 0
    with CHECKPOINT.open("x", encoding="utf-8", buffering=1) as stream:
        def save(row):
            nonlocal count
            stream.write(json.dumps(row) + "\n")
            count += 1
            if count % 10 == 0:
                print(f"Checkpoint: {count}/1500 completed runs", flush=True)

        original_static = evaluator.static_run

        def static_run(*args, **kwargs):
            row = original_static(*args, **kwargs)
            save(row)
            return row

        class CheckpointRows(list):
            def append(self, row):
                save(row)
                super().append(row)

            def __iadd__(self, rows):
                for row in rows:
                    self.append(row)
                return self

        # Only the result container changes. The frozen algorithms, inputs, order,
        # timing boundaries, and evaluation implementation remain unchanged.
        evaluator.static_run = static_run
        namespace = dict(evaluator.__dict__, CheckpointRows=CheckpointRows)
        source = inspect.getsource(evaluator.dynamic_runs)
        assert source.count("rows = []") == 1
        exec(compile(source.replace("rows = []", "rows = CheckpointRows()"), "<checkpointed dynamic_runs>", "exec"), namespace)
        evaluator.dynamic_runs = namespace["dynamic_runs"]
        evaluator.LIVE["enabled"] = True
        (OUT / "evaluation_started.json").write_text(json.dumps({"started_at": datetime.now(timezone.utc).isoformat(), "expected_static": 300, "expected_dynamic": 1200, "argv": "scripts.evaluate_sop --live run --dynamic"}, indent=2))
        rows = evaluator.run(dynamic=True)
        evaluator.write(rows)
        verify_frozen()
        manifest = json.loads(evaluator.manifest_path().read_text(encoding="utf-8"))
        expected = {(experiment, request["id"], algorithm, seed)
                    for experiment in ("static", *(f"dynamic:{s}" for s in evaluator.LIVE_DYNAMIC_SCENARIOS))
                    for request in manifest["requests"] for algorithm in evaluator.ALGORITHMS for seed in manifest["seeds"]}
        keys = [(r["experiment"], r["request"], r["algorithm"], r["seed"]) for r in rows]
        if len(keys) != len(set(keys)) or set(keys) != expected:
            raise RuntimeError("Evaluation has duplicate or missing run identities; inspect checkpoint.")
        results_path = evaluator.ROOT / "outputs/live_sop_results.json"
        results = json.loads(results_path.read_text(encoding="utf-8"))
        audit = json.loads((OUT / "collection_audit.json").read_text())
        analysis = results["analysis"]
        lines = ["# Live study closeout review", "",
                 f"Completed {datetime.now(timezone.utc).isoformat()}", "",
                 "Collection task disabled and stopped; in-app collection disabled with ANTROUTE_COLLECT=0.",
                 "Backup is a verified local archive in this closeout folder; copy it to another drive or storage account for an independent backup.", "",
                 f"Archive payloads that failed integrity checks: {len(audit['corrupt_files'])}.",
                 f"Frozen departure availability checks: {dict(Counter(r['status'] for r in audit['frozen_input_checks']))}.",
                 "Availability at departure does not guarantee an entire replay trip is usable; dynamic failures below are retained.", "",
                 "## Collection audit", "",
                 "| Period | Source | Expected scheduled | Saved in period | Missing scheduled |",
                 "|---|---|---:|---:|---:|"]
        for period, sources in audit["periods"].items():
            for source, summary in sources.items():
                lines.append(f"| {period} | {source} | {summary['expected']} | {summary['saved_in_period']} | {summary['missing_scheduled']} |")
        lines += ["", "Missing scheduled weather observations may have later retries; see collection_audit.json for all exact gaps.",
                  "", "## Evaluation completeness", "",
                  f"Verified {len(rows)} unique expected runs: 300 static and 1,200 dynamic.",
                  f"Paired static requests: {analysis['complete_requests']}/{analysis['requests']}; excluded requests: {analysis['incomplete_requests']}.",
                  "", "## Four primary outcomes", "",
                  "| Outcome | Matched requests | ANTROUTE mean | Cheng mean | Mean Cheng minus ANTROUTE | Holm p | Significant |",
                  "|---|---:|---:|---:|---:|---:|---|"]
        for measure, value in analysis["primary"].items():
            def fmt(v):
                return "unavailable" if v is None else f"{v:.8g}"
            lines.append(f"| {measure} | {value['n']} | {fmt(value['proposed_mean'])} | {fmt(value['baseline_mean'])} | {fmt(value['mean_difference'])} | {fmt(value['p_holm'])} | {value['significant']} |")
        lines += ["", "Positive mean differences favor ANTROUTE under the frozen lower-is-better definitions. A nonsignificant result does not establish equivalence.",
                  "Results describe this frozen sample, adapted Cheng baseline and recorded conditions. They do not establish global optimality or universal superiority.",
                  "", "## Run outcomes", "", "| Experiment | Algorithm | Status | Runs |", "|---|---|---|---:|"]
        outcomes = Counter((r["experiment"], r["algorithm"], r["status"]) for r in rows)
        for (experiment, algorithm, status), n in sorted(outcomes.items()):
            lines.append(f"| {experiment} | {algorithm} | {status} | {n} |")
        lines += ["", "Failures remain in the run outputs and are never scored as zero. Dynamic results are descriptive and are not part of the four primary tests.",
                  "Search times were measured on this computer during this run; machine load and hardware limit their generalization.",
                  "", "## Preservation", "", "The frozen manifest, calibration, network, tuning results and search defaults were preserved. No retuning or request replacement was performed."]
        (OUT / "closeout_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        final_hashes = {}
        for name in ("live_sop_runs.csv", "live_sop_results.json", "live_sop_report.md"):
            source = evaluator.ROOT / "outputs" / name
            shutil.copy2(source, OUT / name)
            final_hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        (OUT / "evaluation_sha256.json").write_text(json.dumps(final_hashes, indent=2))
        (OUT / "evaluation_complete.json").write_text(json.dumps({"completed_at": datetime.now(timezone.utc).isoformat(), "rows": len(rows), "checkpoints": count}, indent=2))


if __name__ == "__main__":
    main()
