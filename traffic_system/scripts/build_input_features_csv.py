"""
The input features of one recorded day, as a CSV the app can upload: one row per 30-minute window and
camera intersection, with every input the system used then.

    python scripts/build_input_features_csv.py --day 2026-05-25 \
        --checkpoint "G:/My Drive/MMDA_CHECKPOINTS/stgnn_edge_v10.pt" \
        --out outputs/test_inputs/may25_input_features.csv

Columns, in groups:

    window          day, session, window_start, window_end, split (the day's split in the risk file)
    camera          camera (intersection label), node_id, lat, lon
    observed        human_label / human_score (Light, Medium, Heavy as 0, 0.5, 1) and vehicle_count_mean
                    (mean YOLO count over the window) -- what the cameras recorded
    model           model_risk: the checkpoint's recorded risk on the camera's edges (an output, not an input)
    weather         temp_c, precip_mm, humidity_pct: WeatherStack hourly, at the camera's grid cell
    flood           flood_level: Project NOAH hazard, 0-3
    incidents       incident_count, incident_strength (strongest, 0-1), incident_lanes, incident_places
    text            live_text_reports / live_text_incidents (posts placed on the camera's roads in the last
                    60 minutes, log-scaled 0-1), history_reports / history_incidents (the road text history
                    for that hour of the week)
    clock           hour, weekday (0 = Monday)
    f_*             the model's own input vector at the window start: every context feature of the checkpoint,
                    scaled exactly as in training (src/data/node_context.py)

Inputs are read "as of" the window start. The same file is what the app's Upload File box loads.
"""

import argparse
import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import predict_congestion_risk as predict  # noqa: E402
from app import recorded, risk_routing as rr  # noqa: E402
from src.data.graph_data import build_subgraph  # noqa: E402
from src.data.node_context import COLUMN  # noqa: E402
from src.data.text_prior import LiveText  # noqa: E402
from src.data.training_data import default_weather_csv  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--day", default="2026-05-25")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/test_inputs/may25_input_features.csv")
    p.add_argument("--embeddings", type=Path, default=REPO_ROOT / "data/processed/embeddings.pt")
    a = p.parse_args()

    cfg = torch.load(a.checkpoint, map_location="cpu", weights_only=False)["config"]
    spatial = REPO_ROOT / "data/processed/spatial"
    graph = build_subgraph(spatial, cfg["k"])
    labels = list(cfg["node_labels"])
    camera_index = torch.tensor([graph.camera_nodes[label] for label in labels], dtype=torch.long)
    context = predict.build_context(
        cfg, graph, spatial, default_weather_csv(cfg.get("weather_resolution", "daily")), camera_index,
        REPO_ROOT / "data/raw/twitter", REPO_ROOT / "configs/event_landmarks.csv",
        REPO_ROOT / "configs/event_intersections.csv", REPO_ROOT / "data/raw/twitter", a.embeddings)
    names = list(cfg["context_features"])

    net = rr._network()
    windows = [w for w in sorted(net.risk_by_window) if w.startswith(a.day)]
    if not windows:
        raise SystemExit(f"no recorded window on {a.day}")
    times = [datetime.fromisoformat(w) for w in windows]
    features = context.at_times(times).numpy()                       # [T, N, F], names order = cfg context_features
    live = LiveText.build(net.graph)
    flood_cell = recorded._node_flood_and_cell()
    node_of = {label: int(np.asarray(graph.node_ids)[i]) for label, i in zip(labels, camera_index.tolist())}

    rows = []
    for t_index, window in enumerate(windows):
        info = recorded.window_description(window)
        cameras = {c["camera"]: c for c in recorded._camera_table(window)}
        incidents = recorded._incident_table(window)
        live_now = live.at(times[t_index])                          # [N_full, 2]
        for slot, label in enumerate(labels):
            node = node_of[label]
            cam = cameras.get(label, {})
            level, cell = flood_cell.get(node, (0, None))
            wx = recorded._weather_rows(times[t_index], [cell]) if cell else None
            here = [i for i in incidents if i["node_id"] == node]
            full_index = net.graph.camera_nodes.get(label)
            f = features[t_index, camera_index[slot].item()]
            rows.append({
                "day": info["day"], "session": info["session"], "window_start": window,
                "window_end": info["window_end"], "split": info["split"],
                "camera": label, "node_id": node, "lat": cam.get("lat"), "lon": cam.get("lon"),
                "human_label": cam.get("human_label"), "human_score": cam.get("human_score"),
                "vehicle_count_mean": cam.get("vehicle_count"), "model_risk": cam.get("model_risk"),
                "temp_c": None if wx is None else round(float(wx["ws_temp_c"].mean()), 1),
                "precip_mm": None if wx is None else round(float(wx["ws_precip_mm"].mean()), 2),
                "humidity_pct": None if wx is None else round(float(wx["ws_humidity_pct"].mean()), 0),
                "flood_level": level,
                "incident_count": len(here),
                "incident_strength": max((i["strength"] for i in here), default=0.0),
                "incident_lanes": max((i["lanes"] for i in here), default=0),
                "incident_places": " | ".join(i["location"] for i in here),
                "live_text_reports": None if full_index is None else round(float(live_now[full_index, 0]), 3),
                "live_text_incidents": None if full_index is None else round(float(live_now[full_index, 1]), 3),
                "history_reports": round(float(f[COLUMN["text_reports"]]), 4) if "text_reports" in names else None,
                "history_incidents": round(float(f[COLUMN["text_incidents"]]), 4) if "text_incidents" in names else None,
                "hour": times[t_index].hour, "weekday": times[t_index].weekday(),
                **{f"f_{name}": round(float(f[COLUMN[name]]), 5) for name in names},
            })
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows ({len(windows)} windows x {len(labels)} cameras), {len(rows[0])} columns -> {a.out}")


if __name__ == "__main__":
    main()
