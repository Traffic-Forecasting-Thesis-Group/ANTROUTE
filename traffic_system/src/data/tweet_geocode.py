"""
Placing a tweet at the camera it is about.

src/data/alignment.py has always documented the text stream as "camera_id = geocoded
intersection", but nothing geocoded: load_tweets_table assigned every tweet to every camera, so
a crash reported at EDSA-Ortigas arrived at the Roxas Blvd camera too. The text feature was then
identical at all eight cameras at any instant, and a model cannot learn "an incident here raises
congestion here" from a signal that is the same everywhere. That is what this module fixes.

Placement reuses the Event-Aware layer's gazetteer (configs/event_landmarks.csv), so routing and
training agree on where a landmark is, and widening the gazetteer improves both at once. A tweet
is attached to the cameras whose own intersection it names -- and, with a placement radius
(cameras_within), to every camera within that distance of the intersection it names, because a
crash a kilometre up the road affects the traffic the camera sees. The cameras are the only
places with visual ground truth to learn against, so text with no camera in reach is dropped.

Deliberately conservative, for the same reason event_layer.py is: a tweet that cannot be placed
is dropped rather than guessed at. Spreading it over every camera is what produced the bug.
"""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from src.data import graph_data
from src.routing.event_layer import _LOCATION, match_landmark

REPO_ROOT = Path(__file__).resolve().parents[2]
CAMERA_MAP_CSV = REPO_ROOT / "configs" / "camera_nodes.csv"


LABELS_CSV = REPO_ROOT / "data" / "labels" / "tweet_training_window.csv"


def load_label_locations(path: Path = LABELS_CSV) -> Dict[str, str]:
    """
    tweet id -> the location a human read out of it, from the hand-labelled sheet.

    Only rows marked relevant with a location are returned: a tweet a labeller judged not to
    be about a road condition should not reach a camera just because it names a place.

    These are the posts no rule can place -- news items, radio bulletins, ordinary prose --
    which is why a person had to read them. The structured feeds (MMDA alerts, @MakatiTraffic
    updates) are placed from their own text and need no sheet.
    """
    path = Path(path)
    if not path.exists():
        return {}
    out: Dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            relevant = str(row.get("label_relevant", "")).strip()
            location = str(row.get("label_location", "")).strip()
            tweet_id = str(row.get("tweet_id", "")).strip()
            if tweet_id and location and relevant in ("1", "1.0"):
                out[tweet_id] = location
    return out


def load_camera_map(path: Path = CAMERA_MAP_CSV) -> Dict[str, str]:
    """camera id -> the intersection label it watches, from configs/camera_nodes.csv."""
    return graph_data.load_camera_map(path)


def cameras_by_intersection(camera_ids: Sequence[str], camera_map: Dict[str, str]) -> Dict[str, List[str]]:
    """
    intersection label -> the camera ids watching it.

    A list, not one id, because two cameras can watch the same intersection from different
    angles; a tweet about it belongs to both.
    """
    out: Dict[str, List[str]] = defaultdict(list)
    for camera_id in camera_ids:
        label = camera_map.get(camera_id)
        if label:
            out[label].append(camera_id)
    return dict(out)


INTERSECTIONS_CSV = REPO_ROOT / "configs" / "event_intersections.csv"
SPATIAL_DIR = REPO_ROOT / "data" / "processed" / "spatial"


def intersection_coordinates(spatial_dir: Path = SPATIAL_DIR,
                             intersections_csv: Path = INTERSECTIONS_CSV) -> Dict[str, tuple]:
    """
    (lat, lon) of every intersection label a landmark can map to: the camera intersections from
    the road graph's own nodes (full_network_static_features.csv, cctv_label), and the other
    intersections the gazetteer names from configs/event_intersections.csv.
    """
    coords: Dict[str, tuple] = {}
    path = Path(intersections_csv)
    if path.exists():
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                coords[row["intersection"].strip()] = (float(row["lat"]), float(row["lon"]))
    static = Path(spatial_dir) / "full_network_static_features.csv"
    if static.exists():
        sums: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0, 0])
        with static.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("is_cctv_node") == "True" and row.get("cctv_label"):
                    s = sums[row["cctv_label"].strip()]
                    s[0] += float(row["lat"]); s[1] += float(row["lon"]); s[2] += 1
        for label, (lat, lon, n) in sums.items():
            coords[label] = (lat / n, lon / n)
    return coords


def _km(a: tuple, b: tuple) -> float:
    (la1, lo1), (la2, lo2) = a, b
    p1, p2 = math.radians(la1), math.radians(la2)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def cameras_within(
    by_intersection: Dict[str, List[str]], coords: Dict[str, tuple], radius_km: float
) -> Dict[str, List[str]]:
    """
    intersection label -> the cameras within `radius_km` (straight line) of it, for every label
    with coordinates; with radius 0 it is `by_intersection` itself (a camera's own intersection).

    This is what lets a tweet about a nearby intersection -- a crash at EDSA-Guadalupe, a few
    hundred metres from the Ortigas cameras' corridor -- inform the cameras whose traffic it
    affects, instead of being dropped for not naming their exact intersection. A label without
    coordinates still reaches its own cameras.
    """
    if radius_km <= 0:
        return dict(by_intersection)
    out: Dict[str, List[str]] = {label: list(cams) for label, cams in by_intersection.items()}
    camera_points = [(label, coords[label]) for label in by_intersection if label in coords]
    for label, point in coords.items():
        near = [cam for cam_label, cam_point in camera_points if _km(point, cam_point) <= radius_km
                for cam in by_intersection[cam_label]]
        if near:
            out[label] = sorted(set(out.get(label, [])) | set(near))
    return out


def locate(text: str, landmarks: Dict[str, str]) -> Optional[str]:
    """
    The intersection a tweet is about, or None.

    Two passes, narrowest first. An MMDA alert states its location in a fixed slot ("... at
    <location> involving/as of ..."), and reading only that slot avoids being misled by a
    second place named later in the sentence. Anything else is matched over the whole text,
    which is how a news item or a traveller's post names a road at all.
    """
    if not text:
        return None
    slot = _LOCATION.search(text)
    if slot:
        phrase = match_landmark(slot.group(1), landmarks)
        if phrase:
            return landmarks[phrase]
    phrase = match_landmark(text, landmarks)
    return landmarks[phrase] if phrase else None


def place(
    texts: Iterable[str],
    landmarks: Dict[str, str],
    by_intersection: Dict[str, List[str]],
    locations: Optional[Sequence[Optional[str]]] = None,
) -> List[List[str]]:
    """
    Per tweet, the camera ids it belongs to (empty when it names no camera's intersection).

    `locations` optionally supplies a hand-read location phrase per tweet, from
    data/labels/tweet_training_window.csv, which takes precedence over reading the raw text.
    That is how the unstructured posts -- news items and the like, where no rule finds a
    location -- reach a camera.
    """
    texts = list(texts)
    if locations is not None and len(locations) != len(texts):
        raise ValueError(f"{len(texts)} tweets but {len(locations)} location override(s)")

    out: List[List[str]] = []
    for i, text in enumerate(texts):
        label = None
        if locations is not None and locations[i]:
            phrase = match_landmark(str(locations[i]), landmarks)
            label = landmarks[phrase] if phrase else None
        if label is None:
            label = locate(text, landmarks)
        out.append(list(by_intersection.get(label, ())) if label else [])
    return out
