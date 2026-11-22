"""Back up and audit the unchanged live study archive before evaluation."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import zipfile
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/live/closeout_2026-11-22"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = ROOT / "data/live/raw"
    files = list(raw.rglob("*.json.gz"))
    for folder in ("configs", "src", "app", "scripts", "data/processed/features"):
        files.extend(p for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    files.extend(ROOT / p for p in (
        "data/live/collection_log.csv", "data/processed/live_calibration.json",
        "data/processed/live_network.json", "data/processed/live_release.json",
        "outputs/live_search_tuning.json", "requirements.txt"))
    archive = OUT / "study_backup.zip"
    if archive.exists():
        raise SystemExit("Backup already exists; refusing to overwrite it.")
    digests = {}
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for p in sorted(set(files)):
            name = p.relative_to(ROOT).as_posix()
            digest = hashlib.sha256()
            with p.open("rb") as source, z.open(name, "w", force_zip64=True) as dest:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                    dest.write(chunk)
            digests[name] = digest.hexdigest()
    (OUT / "backup_sha256.json").write_text(json.dumps(digests, indent=2))
    with zipfile.ZipFile(archive) as z:
        for name, expected in digests.items():
            h = hashlib.sha256()
            with z.open(name) as source:
                while chunk := source.read(1024 * 1024):
                    h.update(chunk)
            if h.hexdigest() != expected:
                raise RuntimeError(f"Backup checksum mismatch: {name}")
    print(f"Backup verified: {len(digests)} files, {archive.stat().st_size:,} bytes", flush=True)
    observed, corrupt = {}, []
    for p in sorted(raw.rglob("*.json.gz")):
        try:
            with gzip.open(p, "rt", encoding="utf-8") as handle:
                record = json.load(handle)
            moment = datetime.fromisoformat(record["captured_at_manila"])
            kind = record["kind"]
            if p.name != f"{moment:%H%M}_{kind}.json.gz" or p.parent.name != f"{moment:%Y-%m-%d}":
                raise ValueError("Filename and capture timestamp differ")
            if not isinstance(record["payload"], dict):
                raise ValueError("Payload is not an object")
            observed.setdefault(kind, {})[moment] = p
        except Exception as error:
            corrupt.append({"file": str(p.relative_to(ROOT)), "error": str(error)})
    config = json.loads((ROOT / "configs/live_study.json").read_text())
    tz = ZoneInfo(config["timezone"])
    missing, summaries = [], {}
    for period, bounds in {"collection": config["collection"], **config["calibration_split"]}.items():
        if not isinstance(bounds, dict) or "start" not in bounds:
            continue
        start = datetime.fromisoformat(bounds["start"]).replace(tzinfo=tz)
        end = datetime.fromisoformat(bounds["end"]).replace(tzinfo=tz) + timedelta(days=1)
        summary = {}
        for kind, interval in (("flow", 15), ("incidents", 15), ("weather", 60)):
            expected = int((end - start).total_seconds() / (interval * 60))
            times = observed.get(kind, {})
            gaps = []
            tick = start
            while tick < end:
                if tick not in times:
                    gaps.append(tick.isoformat())
                tick += timedelta(minutes=interval)
            summary[kind] = {"expected": expected, "saved_in_period": sum(start <= t < end for t in times), "missing_scheduled": len(gaps)}
            missing.extend({"period": period, "kind": kind, "at": at} for at in gaps)
        summaries[period] = summary
    from src.live.snapshot import nearest_capture, weather_at
    from scripts.evaluate_sop import LIVE_DYNAMIC_SCENARIOS, peak_departure
    manifest = json.loads((ROOT / "configs/live_experiment_manifest.json").read_text())
    checks = []
    for request in manifest["requests"]:
        requested = datetime.fromisoformat(request["live_at"])
        for scenario in ("static", *LIVE_DYNAMIC_SCENARIOS):
            departure = peak_departure(manifest, requested.date(), scenario)
            moment = requested.replace(hour=departure[0], minute=departure[1]) if departure else requested
            row = {"request": request["id"], "scenario": scenario, "requested": moment.isoformat()}
            try:
                capture = nearest_capture(moment)
                weather, payload = weather_at(capture)
                row.update(status="available", capture=capture.isoformat(), weather=weather.isoformat(), cities=len(payload.get("cities", {})))
            except Exception as error:
                row.update(status="unavailable", error=str(error))
            checks.append(row)
    release = json.loads((ROOT / "data/processed/live_release.json").read_text())
    tuning = json.loads((ROOT / "outputs/live_search_tuning.json").read_text())
    from src.routing.aco_kernel import SearchConfig
    integrity = {
        "manifest_calibration_matches": manifest["release"]["calibration_id"] == release["calibration_id"] == tuning["calibration_id"],
        "release_artifact_hashes": {name: hashlib.sha256((ROOT / "data/processed" / name).read_bytes()).hexdigest() == digest for name, digest in release["artifacts"].items()},
        "search_defaults": SearchConfig().model_dump(),
        "tuning_selection": {k: v for k, v in tuning.items() if k not in ("requests", "grid", "rule")},
    }
    with (ROOT / "data/live/collection_log.csv").open(newline="", encoding="utf-8") as handle:
        errors = [r for r in csv.DictReader(handle) if r["status"] != "ok"]
    result = {"generated_at": datetime.now(tz).isoformat(), "backup": str(archive.relative_to(ROOT)), "backup_verified_files": len(digests),
              "periods": summaries, "missing_slots": missing, "corrupt_files": corrupt,
              "frozen_input_checks": checks, "integrity": integrity, "logged_errors": errors,
              "note": "Availability checks validate capture files and weather lookup; evaluation validates road matching and full-trip usability. Missing scheduled weather can be retried later in its hour."}
    (OUT / "collection_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"periods": summaries, "corrupt_files": corrupt, "frozen_checks": dict(Counter(r["status"] for r in checks)), "integrity": integrity}, indent=2), flush=True)


if __name__ == "__main__":
    main()
