"""
The app's what-if table: precomputed cases showing the checkpoint accepts weather, flood, incidents
and the camera and text inputs.

    python scripts/verify_crs_sources.py --checkpoint <stgnn_edge_v8.pt> --frames-root <frames...> \
        --out outputs/google_eval/stgnn_edge_v8.sources_with_text.json
    python scripts/build_what_if.py outputs/google_eval/stgnn_edge_v8.sources_with_text.json

verify_crs_sources.py loads the real checkpoint and changes ONE input at a time in real batches
(src/models/source_sensitivity.py), measuring how far the Congestion Risk Score moves on camera
edges and on all other edges. This script puts friendly case names on that result and, if a
condition-update run with an injected incident exists, adds what the incident does to a route
(outputs/condition_updates/event/summary.json). The result is outputs/what_if/<checkpoint>.json,
which GET /routes/what-if serves. It is re-inference by the checkpoint done offline, not live:
serving the checkpoint needs PyTorch and the frames on the server.
"""

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

CASES = {
    "weather": ("Heavy rain", "Every node gets the wettest, most humid day the model trained on."),
    "flood": ("Flood level 3", "Every node becomes Project NOAH flood hazard level 3, with the batch's own rainfall."),
    "events": ("New incidents", "A fresh two-lane incident at every fifth node, for the whole window."),
    "cctv": ("Cameras off", "Every camera goes offline: the visual input is masked."),
    "text": ("No traffic posts", "The tweet text goes silent, as for a minute with no posts."),
    "temporal": ("Time of day moves", "The clock moves ten hours (AM peak to PM peak) and three weekdays on."),
    "spatial": ("Road attributes shuffled", "A control: speed, lanes and road class are shuffled between nodes."),
    "text_prior": ("Text history removed", "The year of traffic posts per road is removed (checkpoints trained with it)."),
}


def route_effect(update_dir: Path) -> dict | None:
    path = update_dir / "summary.json"
    if not path.exists():
        return None
    s = json.loads(path.read_text(encoding="utf-8"))
    out = {"update": s.get("update"), "n_trips": None, "lanes": s.get("lanes"),
           "active_minutes": s.get("event_active_minutes"), "systems": {}}
    for system, v in s["by_system"].items():
        changes, quality = v["route_changes"], v["route_quality_all_trips"]
        out["n_trips"] = v["n_trips"]
        out["systems"][system] = {
            "routes_changed_share": changes["share_changed"],
            "event_exposure_old": quality["event_exposure"]["baseline_mean"],
            "event_exposure_recalculated": quality["event_exposure"]["proposed_mean"],
            "congestion_exposure_old": quality["congestion_exposure"]["baseline_mean"],
            "congestion_exposure_recalculated": quality["congestion_exposure"]["proposed_mean"],
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sources", type=Path, help="verify_crs_sources.py --out result")
    p.add_argument("--condition-updates", type=Path, default=REPO_ROOT / "outputs/condition_updates/event")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs/what_if")
    a = p.parse_args()
    src = json.loads(a.sources.read_text(encoding="utf-8"))
    name = Path(src["checkpoint"].replace("\\", "/")).stem
    cases = []
    for key, (label, description) in CASES.items():
        s = (src.get("sources") or {}).get(key)
        if not s:
            continue
        cases.append({"key": key, "label": label, "description": description,
                      "camera_mean": s["camera_mean"], "camera_max": s["camera_max"],
                      "other_mean": s["other_mean"], "other_max": s["other_max"],
                      "other_moved": s["other_moved"]})
    out = {
        "model": name,
        "windows": src["windows"],
        "note": ("Precomputed from the real checkpoint: each case changes one input in real batches and "
                 "measures how far the Congestion Risk Score moves (mean absolute change, 0-1 scale). "
                 "Offline re-inference, not a live run."),
        "text_present_share": src.get("text_present_share"),
        "cases": cases,
        "route_effect": route_effect(a.condition_updates),
    }
    a.out_dir.mkdir(parents=True, exist_ok=True)
    path = a.out_dir / f"{name}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"{len(cases)} cases -> {path}")


if __name__ == "__main__":
    main()
