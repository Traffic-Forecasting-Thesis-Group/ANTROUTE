"""
Event-Aware layer: turns MMDA incident reports into a routing penalty.

The RADR STGNN predicts *recurring* congestion from what the cameras see. It cannot
predict a crash that happened ten minutes ago, because nothing in its training signal
represents one. This module covers that second, complementary signal: the unstructured
MMDA feed reports incidents ("Road crash incident at EDSA Ortigas NB ... Two lanes
occupied"), which are parsed here, matched to intersections of the road graph, and
turned into a per-edge impact that the Dynamic Weight Engine multiplies into its cost:

    W(u, v) = dist(u, v) * (1 + lambda * Risk(u, v)) * (1 + mu * Event(u, v))

Event(u, v) is in [0, 1] and decays to zero over EVENT_TTL_MINUTES, so an incident
stops diverting traffic once it has been cleared long enough. mu scales how strongly
the router avoids incidents, exactly as lambda scales how strongly it avoids predicted
congestion (mu = 0 reproduces the risk-only behaviour).

Matching is deliberately conservative: an alert only counts when its location text
names a landmark in configs/event_landmarks.csv, which maps landmark phrases to graph
intersections whose coordinates are already known. An unmatched alert is dropped
rather than guessed at, so a route is never diverted by an incident the system could
not actually place.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from src.data.graph_data import GraphData
from src.routing.dynamic_weight import WeightedGraph, dynamic_weight

LOCAL_TZ = "Asia/Manila"

# How long an incident keeps affecting routing, and how its impact falls off. MMDA
# alerts report the incident, not its clearance, so this is a decay assumption rather
# than an observed duration -- 60 minutes matches the usual clear-up time for the
# single-lane incidents that dominate the feed (2,215 of 2,322 report one lane).
EVENT_TTL_MINUTES = 60.0

# Event sensitivity, the mu in the formula above. 1.0 makes a fresh two-lane incident
# roughly double the cost of the edges at that intersection.
DEFAULT_MU = 1.0

# Impact of a single incident at full strength, by how much of the road it takes out.
LANES_IMPACT = {1: 0.5, 2: 1.0, 3: 1.0}

_ALERT_MARKER = "MMDA ALERT"
_LOCATION = re.compile(r"\bat (.+?) (?:involving|as of)", re.I)
_LANES = re.compile(r"\b(one|two|three|1|2|3) lanes? occupied", re.I)
_DIRECTION = re.compile(r"\b(NB|SB|EB|WB)\b")
_WORD_TO_INT = {"one": 1, "two": 2, "three": 3, "1": 1, "2": 2, "3": 3}


@dataclass(frozen=True)
class TrafficEvent:
    """One parsed MMDA incident report, placed on the graph."""

    at: datetime                  # when it was reported
    kind: str                     # "crash", "stalled" or "other"
    location_text: str            # the raw location phrase, for traceability
    landmark: str                 # the gazetteer phrase it matched
    intersection: str             # graph intersection label it was placed at
    lanes_occupied: int
    direction: Optional[str]      # NB/SB/EB/WB when stated

    @property
    def base_impact(self) -> float:
        return LANES_IMPACT.get(self.lanes_occupied, 1.0)


def classify(text: str) -> str:
    lowered = text.lower()
    if "road crash" in lowered or "collision" in lowered:
        return "crash"
    if "stalled" in lowered:
        return "stalled"
    return "other"


def load_landmarks(path: Path) -> Dict[str, str]:
    """landmark phrase -> graph intersection label, from configs/event_landmarks.csv."""
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as f:
        return {
            row["landmark"].strip().lower(): row["intersection"].strip()
            for row in csv.DictReader(f)
            if row.get("landmark") and row.get("intersection")
        }


def match_landmark(location_text: str, landmarks: Dict[str, str]) -> Optional[str]:
    """
    Longest matching landmark phrase in the location text, or None.

    Longest-first matters: "edsa quezon ave" must not be matched by a shorter, more
    generic entry that happens to also be a substring of it.
    """
    lowered = location_text.lower()
    for phrase in sorted(landmarks, key=len, reverse=True):
        if phrase in lowered:
            return phrase
    return None


def _tweet_texts(raw_twitter_root: Path) -> Iterable[tuple]:
    for path in sorted(Path(raw_twitter_root).rglob("tweets_*.json")):
        with path.open(encoding="utf-8") as f:
            content = json.load(f)
        tweets = content.get("data", content) if isinstance(content, dict) else content
        for tweet in tweets:
            if not isinstance(tweet, dict):
                continue
            text = tweet.get("text") or tweet.get("full_text") or ""
            if text:
                yield re.sub(r"\s+", " ", text), tweet.get("createdAt")


def parse_alerts(raw_twitter_root: Path, landmarks: Dict[str, str]) -> List[TrafficEvent]:
    """
    Every MMDA alert that could be parsed *and* placed on the graph, in time order.

    The tweet's own timestamp is used as the incident time rather than the "as of
    H:MM AM/PM" inside the text: the text time carries no date, and the feed posts
    essentially in real time, so the two agree to within minutes.
    """
    import pandas as pd

    events: List[TrafficEvent] = []
    for text, created in _tweet_texts(raw_twitter_root):
        if _ALERT_MARKER not in text.upper():
            continue
        location = _LOCATION.search(text)
        if not location:
            continue
        location_text = location.group(1).strip()
        landmark = match_landmark(location_text, landmarks)
        if landmark is None:
            continue
        stamp = pd.to_datetime(created, format="%a %b %d %H:%M:%S %z %Y", errors="coerce", utc=True)
        if pd.isna(stamp):
            continue
        lanes_match = _LANES.search(text)
        direction_match = _DIRECTION.search(location_text)
        events.append(
            TrafficEvent(
                at=stamp.tz_convert(LOCAL_TZ).to_pydatetime(),
                kind=classify(text),
                location_text=location_text,
                landmark=landmark,
                intersection=landmarks[landmark],
                lanes_occupied=_WORD_TO_INT.get(lanes_match.group(1).lower(), 1) if lanes_match else 1,
                direction=direction_match.group(1).upper() if direction_match else None,
            )
        )
    events.sort(key=lambda e: e.at)
    return events


def active_at(
    events: Sequence[TrafficEvent], when: datetime, ttl_minutes: float = EVENT_TTL_MINUTES
) -> List[tuple]:
    """(event, strength) for every incident still in effect at `when`, strength in (0, 1]."""
    live = []
    window = timedelta(minutes=ttl_minutes)
    for event in events:
        age = when - event.at
        if timedelta(0) <= age < window:
            decay = 1.0 - age / window
            live.append((event, event.base_impact * decay))
    return live


def event_impact(
    graph: GraphData, live: Sequence[tuple], intersections: Optional[Dict[str, int]] = None
) -> np.ndarray:
    """
    Per-edge impact in [0, 1] from the live incidents, aligned to graph.edge_index.

    An incident affects every edge touching its intersection. Overlapping incidents at
    one place take the strongest rather than the sum, so impact stays bounded and three
    simultaneous fender-benders cannot make an edge infinitely expensive.
    """
    intersections = graph.camera_nodes if intersections is None else intersections
    src, dst = graph.edge_index[0].numpy(), graph.edge_index[1].numpy()
    impact = np.zeros(src.size, dtype=np.float64)

    for event, strength in live:
        node = intersections.get(event.intersection)
        if node is None:
            continue
        touching = (src == node) | (dst == node)
        impact[touching] = np.maximum(impact[touching], strength)

    return np.clip(impact, 0.0, 1.0)


def apply_events(wg: WeightedGraph, impact: np.ndarray, mu: float = DEFAULT_MU) -> WeightedGraph:
    """
    The same weighted graph with the event penalty folded into its weights.

    W = dist * (1 + lambda * Risk) * (1 + mu * Event); mu = 0 (or no live incident)
    returns the risk-only weights unchanged.
    """
    impact = np.asarray(impact, dtype=np.float64)
    if impact.shape != wg.weight.shape:
        raise ValueError(f"impact must have shape {wg.weight.shape}, received {impact.shape}")
    if not np.isfinite(mu) or mu < 0:
        raise ValueError(f"mu must be finite and >= 0, received {mu}")
    if not np.all(np.isfinite(impact)) or np.any((impact < 0) | (impact > 1)):
        raise ValueError("every event impact must be finite and within [0, 1]")

    return WeightedGraph(
        node_ids=wg.node_ids,
        src=wg.src,
        dst=wg.dst,
        distance=wg.distance,
        risk=wg.risk,
        weight=dynamic_weight(wg.distance, wg.risk, wg.lam) * (1.0 + mu * impact),
        lam=wg.lam,
        coverage=wg.coverage,
        scored_edges=wg.scored_edges,
        window=wg.window,
    )
