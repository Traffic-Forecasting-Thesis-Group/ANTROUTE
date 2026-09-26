"""
Congestion viewer / labeling tool.

    pip install streamlit
    streamlit run app/label_viewer.py -- "G:/My Drive/MMDA_FRAMES"

Shows extracted CCTV frames with detected vehicles, the congestion timeline of a
camera over its session, and lets you correct the Light / Medium / Heavy label.
Corrections are saved to <frames root>/labels.csv and override the auto label.
"""

import collections
import sys
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.graph_data import (  # noqa: E402
    cctv_labels, derive_camera_map, expected_camera_counts, load_camera_map, save_camera_map,
)
from src.vision.label_store import LABELS, load_frames, parse_boxes, save_human_label  # noqa: E402

COLORS = {"Light": "#2e8b57", "Medium": "#d98e04", "Heavy": "#c0392b"}
DEFAULT_ROOT = "G:/My Drive/MMDA_FRAMES"

st.set_page_config(page_title="Congestion labeling", layout="wide")

root_arg = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT
root = Path(st.sidebar.text_input("Frames folder", root_arg))
if not (root / "auto_labels.csv").exists():
    st.error(f"auto_labels.csv not found in {root}. Run extract_frames.py and autolabel_frames.py first.")
    st.stop()

df = load_frames(root)

mode = st.sidebar.radio("Mode", ["Label frames", "Map cameras"])
if mode == "Map cameras":
    camera_csv = Path(st.sidebar.text_input("Camera map CSV", str(REPO_ROOT / "configs" / "camera_nodes.csv")))
    intersections = cctv_labels(REPO_ROOT / "data" / "processed" / "spatial")
    cameras = sorted(df["camera_id"].unique())
    saved = load_camera_map(camera_csv)
    guessed = derive_camera_map(cameras, intersections)
    unset = "(not mapped)"

    st.subheader("Which intersection does each camera view?")
    st.caption("One frame per camera. Guesses come from the footage folder name; your saved choices win. "
               "Several cameras may share an intersection. Cameras left unmapped are skipped in STGNN training.")
    choices = {}
    columns = st.columns(3)
    for i, cam in enumerate(cameras):
        frames = df[df["camera_id"] == cam]
        current = saved.get(cam) or guessed.get(cam) or unset
        with columns[i % 3]:
            st.image(str(root / frames.iloc[len(frames) // 2]["frame_path"]), caption=f"{cam[-14:]}  ({len(frames)} frames)")
            choices[cam] = st.selectbox("Intersection", [unset] + intersections, key=f"map_{cam}",
                                        index=([unset] + intersections).index(current) if current in intersections else 0)
    mapped = {c: v for c, v in choices.items() if v != unset}
    st.write(f"{len(mapped)} of {len(cameras)} cameras mapped to {len(set(mapped.values()))} of {len(intersections)} intersections")
    expected = expected_camera_counts(REPO_ROOT / "configs" / "cctv_locations.csv")
    if expected:
        assigned = collections.Counter(mapped.values())
        st.table({"Intersection": intersections,
                  "Cameras expected": [expected.get(i, 0) for i in intersections],
                  "Cameras assigned": [assigned.get(i, 0) for i in intersections]})
    if st.button("Save mapping"):
        save_camera_map(camera_csv, mapped)
        st.success(f"Saved {len(mapped)} cameras to {camera_csv}")
    st.stop()

date = st.sidebar.selectbox("Date", sorted(df["date"].unique()))
day = df[df["date"] == date]
camera = st.sidebar.selectbox("Camera", sorted(day["camera_id"].unique()))
view = day[day["camera_id"] == camera].reset_index(drop=True)

only_unreviewed = st.sidebar.checkbox("Only frames without a human label", value=False)
label_filter = st.sidebar.multiselect("Show labels", LABELS, default=list(LABELS))
shown = view[view["label"].isin(label_filter)]
if only_unreviewed:
    shown = shown[shown["source"] == "auto"]
if shown.empty:
    st.info("No frames match the current filters.")
    st.stop()

counts = view["label"].value_counts()
cols = st.columns(4)
cols[0].metric("Frames", len(view))
for col, name in zip(cols[1:], LABELS):
    col.metric(name, int(counts.get(name, 0)))
st.caption(f"{int((view['source'] == 'human').sum())} of {len(view)} frames human-labeled")

reviewed = df[df["source"] == "human"]
if len(reviewed):
    agree_all = float((reviewed["label"] == reviewed["auto_label"]).mean())
    this_cam = reviewed[reviewed["camera_id"] == camera]
    agree_cam = f"{(this_cam['label'] == this_cam['auto_label']).mean():.0%} on this camera ({len(this_cam)})" if len(this_cam) else "no frames from this camera yet"
    st.caption(f"Your labels match the automatic ones {agree_all:.0%} of the time across {len(reviewed)} reviewed frames; {agree_cam}. "
               "Above ~85% the automatic labels are good enough to train on.")
else:
    st.caption("Review some frames to see how often your labels agree with the automatic ones.")

st.subheader("Congestion over the session")
chart = view.assign(label_color=view["label"])
st.scatter_chart(chart, x="timestamp", y="occupancy", color="label_color", size=40)

st.subheader("Frame")
key = f"pos_{camera}_{date}"
pos = min(st.session_state.get(key, 0), len(shown) - 1)
nav = st.columns([1, 1, 6])
if nav[0].button("Previous", disabled=pos == 0):
    pos -= 1
if nav[1].button("Next", disabled=pos >= len(shown) - 1):
    pos += 1
pos = st.slider("Frame position", 0, len(shown) - 1, pos) if len(shown) > 1 else 0
st.session_state[key] = pos
row = shown.iloc[pos]

image = Image.open(root / row["frame_path"]).convert("RGB")
draw = ImageDraw.Draw(image)
w, h = image.size
for x1, y1, x2, y2, *_ in parse_boxes(row["boxes"]):
    draw.rectangle([x1 * w, y1 * h, x2 * w, y2 * h], outline=COLORS[row["label"]], width=2)

left, right = st.columns([3, 1])
left.image(image, caption=f"{row['timestamp']}  |  {row['frame_path']}", width="stretch")
right.markdown(f"**Current label:** {row['label']} ({row['source']})")
right.markdown(f"Auto label: {row['auto_label']}")
right.markdown(f"Vehicles: {int(row['n_vehicles'])}  \nOccupancy: {row['occupancy']:.3f}")
for name in LABELS:
    if right.button(name, key=f"set_{name}", width="stretch"):
        save_human_label(root, row["frame_path"], name)
        st.rerun()
if right.button("Clear human label", key="clear", width="stretch"):
    save_human_label(root, row["frame_path"], None)
    st.rerun()
