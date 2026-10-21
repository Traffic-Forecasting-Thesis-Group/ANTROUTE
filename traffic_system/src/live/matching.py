"""Match live HERE stretches to the study's directed road segments.

A segment is matched to a live line when most of the segment lies within a distance tolerance of the
line and the two run in the same direction. Lengths are measured in UTM zone 51N (EPSG:32651), which
covers Metro Manila. Unmatched segments stay unmatched: they are reported, never filled with zero.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass

from pyproj import Transformer
from shapely import wkt
from shapely.geometry import LineString
from shapely.ops import transform
from shapely.strtree import STRtree

_TO_METRES = Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True).transform
SAMPLE_FRACTIONS = (0.1, 0.3, 0.5, 0.7, 0.9)


@dataclass(frozen=True)
class Segment:
    road_id: str
    road_type: str
    length_m: float
    line: LineString  # metres


def load_segments(path) -> list[Segment]:
    segments = []
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            line = transform(_TO_METRES, wkt.loads(row["geometry_wkt"]))
            segments.append(Segment(row["road_id"], row["road_type"], float(row["distance_m"]), line))
    return segments


def project(points: list[tuple[float, float]]) -> LineString:
    return transform(_TO_METRES, LineString(points))


def _direction(line: LineString, distance: float) -> tuple[float, float]:
    a = line.interpolate(max(distance - 1.0, 0.0))
    b = line.interpolate(min(distance + 1.0, line.length))
    dx, dy = b.x - a.x, b.y - a.y
    norm = math.hypot(dx, dy) or 1.0
    return dx / norm, dy / norm


def match(segments: list[Segment], items: list[dict], tolerance_m: float = 20.0, min_share: float = 0.6,
          max_angle_degrees: float = 45.0, directional: bool = True) -> dict[str, int]:
    """road_id -> index into items of the best matching live line."""
    lines, owner = [], []
    for index, item in enumerate(items):
        for points in item["lines"]:
            lines.append(project(points))
            owner.append(index)
    if not lines:
        return {}
    tree, cos_limit, result = STRtree(lines), math.cos(math.radians(max_angle_degrees)), {}
    for segment in segments:
        if segment.line.length <= 0:
            continue
        samples = [segment.line.interpolate(f, normalized=True) for f in SAMPLE_FRACTIONS]
        directions = [_direction(segment.line, f * segment.line.length) for f in SAMPLE_FRACTIONS]
        best = None
        for candidate in tree.query(segment.line.buffer(tolerance_m)):
            line = lines[candidate]
            hits, distance_sum = 0, 0.0
            for point, (sx, sy) in zip(samples, directions):
                distance = line.distance(point)
                if distance > tolerance_m:
                    continue
                if directional:
                    lx, ly = _direction(line, line.project(point))
                    if sx * lx + sy * ly < cos_limit:
                        continue
                hits += 1
                distance_sum += distance
            share = hits / len(samples)
            if share >= min_share:
                key = (share, -distance_sum / hits)
                if best is None or key > best[0]:
                    best = (key, owner[int(candidate)])
        if best is not None:
            result[segment.road_id] = best[1]
    return result


def coverage(segments: list[Segment], matched: dict[str, int]) -> dict:
    """Share of segments and of length that received a live value, overall and by road type."""
    by_type: dict[str, dict] = {}
    for segment in segments:
        row = by_type.setdefault(segment.road_type, {"segments": 0, "matched": 0, "km": 0.0, "matched_km": 0.0})
        row["segments"] += 1
        row["km"] += segment.length_m / 1000
        if segment.road_id in matched:
            row["matched"] += 1
            row["matched_km"] += segment.length_m / 1000
    total = {key: sum(row[key] for row in by_type.values()) for key in ("segments", "matched", "km", "matched_km")}
    for row in (*by_type.values(), total):
        row["segment_share"] = row["matched"] / row["segments"] if row["segments"] else 0.0
        row["length_share"] = row["matched_km"] / row["km"] if row["km"] else 0.0
    return {"total": total, "by_road_type": dict(sorted(by_type.items(), key=lambda kv: -kv[1]["km"]))}
