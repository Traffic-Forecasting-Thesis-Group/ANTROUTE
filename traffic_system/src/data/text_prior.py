"""
The text prior: what a year of traffic posts says about each road, at each hour.

Live text (the tweets posted during a recorded camera session) can only ever be the few hundred
posts that happen to fall inside the 20 two-hour sessions of May 2026. The other ~45,000 posts
describe the same roads over the twelve months before -- which roads get reported, at what
hours, and what about. This module turns them into per-node inputs for the RADR STGNN, the way
node_context gives every node its weather and flood level:

  text_reports     how often posts concern this node at this hour (log-scaled, 0..1)
  text_incidents   the same, counting only incident posts (crash, stalled, flood, ...)
  text_pc1..8      the mean DistilBERT embedding of those posts, reduced by PCA to 8 numbers
                   (scaled to unit variance), so what was said -- not just how often -- reaches
                   the model

per node and per hour bucket: hour of day (0-23), weekday or weekend -- 48 buckets.

Placement on the road graph, most specific first:

  1. a landmark from configs/event_landmarks.csv (the same gazetteer as the Event-Aware layer):
     the graph nodes within LANDMARK_RADIUS_M of that intersection;
  2. otherwise every major road the post names (OSM edge names, e.g. "Kalayaan Ave", plus
     aliases such as EDSA): every node on that road. #MakatiTraffic posts ("along Kalayaan Ave.
     from Makati Ave. to Rockwell Drive") are placed this way.

A post naming no known place is not used. Road names are matched only for roads that are not
residential and carry the name on at least MIN_NAME_EDGES edges, so a common word does not
place a post on a side street; a name shared by two same-named roads reaches both.

No leakage: the prior is built only from posts created before `before` (default 1 May 2026),
and every camera session -- train, validation and test -- is in May 2026. A checkpoint records
the cutoff ("text_prior_before") and is rebuilt with the same one at prediction.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.graph_data import GraphData

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_TWITTER = REPO_ROOT / "data/raw/twitter"
EMBEDDINGS = REPO_ROOT / "data/processed/embeddings.pt"
SPATIAL_DIR = REPO_ROOT / "data/processed/spatial"
LANDMARKS_CSV = REPO_ROOT / "configs/event_landmarks.csv"

DEFAULT_BEFORE = date(2026, 5, 1)
N_COMPONENTS = 8
N_BUCKETS = 48                       # 24 hours x {weekday, weekend}
LANDMARK_RADIUS_M = 300.0
MIN_NAME_EDGES = 5
MIN_NAME_LENGTH = 5
MINOR_ROADS = {"residential", "unclassified", "service", "living_street", "track", "road"}

FEATURES: List[str] = ["text_reports", "text_incidents"] + [f"text_pc{i + 1}" for i in range(N_COMPONENTS)]

_SUFFIX = re.compile(r"\s+(avenue|ave\.?|street|st\.?|road|rd\.?|boulevard|blvd\.?|drive|dr\.?|"
                     r"highway|hwy\.?|extension|ext\.?|parkway)$", re.I)
# Short names people use for roads whose OSM name is long.
ALIASES = {
    "edsa": "epifanio de los santos",
    "c5": "carlos p. garcia",
    "c-5": "carlos p. garcia",
    "slex": "south luzon expressway",
    "nlex": "north luzon expressway",
}
INCIDENT = re.compile(r"\b(crash|collision|accident|aksidente|banggaan|stalled|overturned|"
                      r"flood\w*|baha|fire|sunog|road closure|closed|disabled|hit[- ]and[- ]run|"
                      r"vehicular|incident)\b", re.I)


def bucket(when: datetime) -> int:
    """Hour of day, +24 on Saturday and Sunday."""
    return when.hour + (24 if when.weekday() >= 5 else 0)


def _bare(name: str) -> str:
    return _SUFFIX.sub("", str(name).strip().lower())


# ------------------------------------------------------------------ the corpus
def read_corpus(raw_twitter_root: Path = RAW_TWITTER, embeddings_path: Path = EMBEDDINGS):
    """
    (created [naive Manila time or None], texts, embeddings [n, 768]) for every post, in the
    order embeddings.pt was written (training_data.read_tweet_walk).
    """
    import torch

    from src.data.training_data import read_tweet_walk

    created_raw, texts, _ = read_tweet_walk(raw_twitter_root)
    payload = torch.load(embeddings_path, map_location="cpu", weights_only=False)
    embeddings = (payload["embeddings"] if isinstance(payload, dict) else payload).float().numpy()
    if len(embeddings) != len(texts):
        raise ValueError(f"{embeddings_path} has {len(embeddings)} rows for {len(texts)} posts; it is stale "
                         "(copy MMDA_TEXT/embeddings.pt from Drive, or regenerate it with main.py)")
    stamps = pd.to_datetime(pd.Series(created_raw), format="%a %b %d %H:%M:%S %z %Y", errors="coerce", utc=True)
    local = stamps.dt.tz_convert("Asia/Manila").dt.tz_localize(None)
    created = [None if pd.isna(t) else t.to_pydatetime() for t in local]
    return created, texts, embeddings


# ------------------------------------------------------------------ placement on the road graph
@dataclass
class RoadPlacer:
    """Which road-graph node ids a post concerns."""

    road_nodes: Dict[str, np.ndarray]          # bare road name -> node ids on it
    road_pattern: Optional[re.Pattern]
    landmark_nodes: Dict[str, np.ndarray]      # intersection label -> node ids around it
    landmarks: Dict[str, str]                  # landmark phrase -> intersection label

    @classmethod
    def build(cls, spatial_dir: Path = SPATIAL_DIR, landmarks_csv: Path = LANDMARKS_CSV) -> "RoadPlacer":
        from src.data.tweet_geocode import intersection_coordinates
        from src.routing.event_layer import load_landmarks

        edges = pd.read_csv(Path(spatial_dir) / "full_network_edge_features.csv",
                            usecols=["source_node_id", "target_node_id", "road_type", "name"]).dropna(subset=["name"])
        edges = edges[~edges["road_type"].isin(MINOR_ROADS)]
        edges = edges.assign(bare=edges["name"].map(_bare))
        counts = edges["bare"].value_counts()
        keep = counts[(counts >= MIN_NAME_EDGES) & (counts.index.str.len() >= MIN_NAME_LENGTH)].index
        road_nodes: Dict[str, np.ndarray] = {}
        for name, group in edges[edges["bare"].isin(keep)].groupby("bare"):
            road_nodes[name] = np.unique(np.concatenate([group["source_node_id"], group["target_node_id"]]))
        for alias, target in ALIASES.items():
            if target in road_nodes:
                road_nodes[alias] = road_nodes[target]
        names = sorted(road_nodes, key=len, reverse=True)
        pattern = re.compile(r"(?<![a-z0-9])(" + "|".join(re.escape(n) for n in names) + r")(?![a-z0-9])", re.I) \
            if names else None

        nodes = pd.read_csv(Path(spatial_dir) / "full_network_static_features.csv", usecols=["node_id", "lat", "lon"])
        lat, lon = np.radians(nodes["lat"].to_numpy()), np.radians(nodes["lon"].to_numpy())
        ids = nodes["node_id"].to_numpy()
        landmark_nodes: Dict[str, np.ndarray] = {}
        for label, (plat, plon) in intersection_coordinates(spatial_dir).items():
            pl, po = math.radians(plat), math.radians(plon)
            h = np.sin((lat - pl) / 2) ** 2 + np.cos(lat) * math.cos(pl) * np.sin((lon - po) / 2) ** 2
            metres = 2 * 6371000.0 * np.arcsin(np.sqrt(h))
            near = ids[metres <= LANDMARK_RADIUS_M]
            if near.size:
                landmark_nodes[label] = near
        return cls(road_nodes, pattern, landmark_nodes, load_landmarks(landmarks_csv))

    def place(self, text: str) -> np.ndarray:
        """Node ids the post concerns: around its landmark, else along every road it names."""
        from src.data.tweet_geocode import locate

        label = locate(text, self.landmarks)
        if label is not None and label in self.landmark_nodes:
            return self.landmark_nodes[label]
        if self.road_pattern is None:
            return np.empty(0, dtype=np.int64)
        found = {m.group(1).lower() for m in self.road_pattern.finditer(text)}
        found = {f for f in found if f in self.road_nodes}
        if not found:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate([self.road_nodes[f] for f in found]))


@dataclass
class PlacedCorpus:
    """Usable posts: when, where (node ids), whether an incident, embedding."""

    created: List[datetime]
    nodes: List[np.ndarray]
    incident: np.ndarray
    embeddings: np.ndarray
    n_posts: int                          # all posts in the corpus, for reporting
    before: Optional[date]                # None: every placeable post, whenever posted

    def subset(self, keep: np.ndarray, before: Optional[date]) -> "PlacedCorpus":
        return PlacedCorpus([self.created[i] for i in keep], [self.nodes[i] for i in keep], self.incident[keep],
                            self.embeddings[keep], self.n_posts, before)


CACHE = REPO_ROOT / "data/processed/text_prior_placement.npz"


def _cache_key(raw_twitter_root: Path, embeddings_path: Path, spatial_dir: Path) -> str:
    """Changes whenever an input of the placement does: the posts, embeddings, gazetteer or road names."""
    parts = ["all-posts-v2", str(len(list(Path(raw_twitter_root).rglob("tweets_*.json"))))]
    for path in (embeddings_path, LANDMARKS_CSV, REPO_ROOT / "configs/event_intersections.csv",
                 Path(spatial_dir) / "full_network_edge_features.csv"):
        path = Path(path)
        parts.append(f"{path.name}:{path.stat().st_size}:{int(path.stat().st_mtime)}" if path.exists() else path.name)
    return "|".join(parts)


@lru_cache(maxsize=1)
def all_placed(raw_twitter_root: Path = RAW_TWITTER, embeddings_path: Path = EMBEDDINGS,
               spatial_dir: Path = SPATIAL_DIR) -> PlacedCorpus:
    """
    Every placeable post, whenever posted. Placing 46,000 posts takes about a minute, so the result
    is cached in data/processed/text_prior_placement.npz and reused while its inputs are unchanged.
    """
    key = _cache_key(raw_twitter_root, embeddings_path, spatial_dir)
    if CACHE.exists():
        cached = np.load(CACHE, allow_pickle=False)
        if str(cached["key"]) == key:
            import torch
            payload = torch.load(embeddings_path, map_location="cpu", weights_only=False)
            embeddings = (payload["embeddings"] if isinstance(payload, dict) else payload).float().numpy()
            offsets, all_nodes = cached["offsets"], cached["nodes"]     # an NpzFile re-reads on every access
            nodes = [all_nodes[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]
            created = [datetime.fromisoformat(str(t)) for t in cached["created"]]
            return PlacedCorpus(created, nodes, cached["incident"].astype(bool), embeddings[cached["rows"]],
                                int(cached["n_posts"]), None)
    created, texts, embeddings = read_corpus(raw_twitter_root, embeddings_path)
    placer = RoadPlacer.build(spatial_dir)
    keep_created, keep_nodes, keep_incident, rows = [], [], [], []
    for i, (when, text) in enumerate(zip(created, texts)):
        if when is None:
            continue
        nodes = placer.place(text)
        if nodes.size == 0:
            continue
        keep_created.append(when)
        keep_nodes.append(nodes)
        keep_incident.append(bool(INCIDENT.search(text)))
        rows.append(i)
    corpus = PlacedCorpus(keep_created, keep_nodes, np.array(keep_incident, dtype=bool),
                          embeddings[rows] if rows else np.zeros((0, embeddings.shape[1]), np.float32),
                          len(texts), None)
    offsets = np.concatenate([[0], np.cumsum([len(n) for n in corpus.nodes])]).astype(np.int64)
    np.savez_compressed(
        CACHE, key=np.array(key), created=np.array([t.isoformat() for t in corpus.created]),
        nodes=np.concatenate(corpus.nodes).astype(np.int64) if corpus.nodes else np.zeros(0, np.int64),
        offsets=offsets, incident=corpus.incident, rows=np.array(rows, dtype=np.int64), n_posts=corpus.n_posts,
    )
    return corpus


def placed_corpus(before: date = DEFAULT_BEFORE, **corpus_kwargs) -> PlacedCorpus:
    """The placeable posts created before `before` -- what the text prior is built from."""
    corpus = all_placed(**corpus_kwargs)
    keep = np.array([i for i, t in enumerate(corpus.created) if t.date() < before], dtype=np.int64)
    return corpus.subset(keep, before)


# ------------------------------------------------------------------ live text
LIVE_MINUTES = 60.0      # how long a post keeps counting, as event_layer.EVENT_TTL_MINUTES
LIVE_CAP = 5.0           # decay-weighted posts at a node that count as the maximum (1.0)


@dataclass
class LiveText:
    """
    The posts placed near each node in the last LIVE_MINUTES before a moment, decaying linearly
    from 1 at posting to 0 -- the text counterpart of the Event-Aware layer's incident impact,
    from every placeable post, not only MMDA alerts. Two columns: all posts, incident posts.
    """

    created: np.ndarray                    # [P] posting time, minutes since the epoch, sorted
    node_rows: List[np.ndarray]            # [P] graph indices of each post's nodes
    incident: np.ndarray                   # [P]
    n_nodes: int

    @classmethod
    def build(cls, graph: GraphData, **corpus_kwargs) -> "LiveText":
        corpus = all_placed(**corpus_kwargs)
        index = {int(v): i for i, v in enumerate(graph.node_ids)}
        minutes, rows, incident = [], [], []
        for when, nodes, inc in zip(corpus.created, corpus.nodes, corpus.incident):
            r = np.array([index[int(v)] for v in nodes if int(v) in index], dtype=np.int64)
            if r.size:
                minutes.append(when.timestamp() / 60.0)
                rows.append(r)
                incident.append(bool(inc))
        order = np.argsort(minutes)
        return cls(np.asarray(minutes)[order], [rows[i] for i in order], np.asarray(incident, dtype=bool)[order],
                   graph.n_nodes)

    def at(self, when: datetime) -> np.ndarray:
        """[N, 2] in [0, 1]: decay-weighted live posts and live incident posts at each node."""
        now = when.timestamp() / 60.0
        lo, hi = np.searchsorted(self.created, [now - LIVE_MINUTES, now], side="right")
        out = np.zeros((self.n_nodes, 2))
        for k in range(lo, hi):
            weight = 1.0 - (now - self.created[k]) / LIVE_MINUTES
            out[self.node_rows[k], 0] += weight
            if self.incident[k]:
                out[self.node_rows[k], 1] += weight
        return np.minimum(np.log1p(out) / np.log1p(LIVE_CAP), 1.0)


# ------------------------------------------------------------------ the prior for one graph
@dataclass
class TextPrior:
    """[N, N_BUCKETS, len(FEATURES)] for one graph, plus what built it."""

    values: np.ndarray
    before: date
    n_posts_used: int
    n_posts_total: int
    explained_variance: float              # share of embedding variance the PCA components keep

    def at(self, when: datetime) -> np.ndarray:
        """[N, len(FEATURES)] at the hour bucket of `when`."""
        return self.values[:, bucket(when), :]

    def describe(self) -> dict:
        return {"before": self.before.isoformat(), "posts_used": self.n_posts_used, "posts_total": self.n_posts_total,
                "share_used": round(self.n_posts_used / max(self.n_posts_total, 1), 4),
                "pca_components": N_COMPONENTS, "pca_explained_variance": round(self.explained_variance, 4),
                "nodes_with_reports": int((self.values[:, :, 0] > 0).any(axis=1).sum())}


def build_text_prior(graph: GraphData, before: date = DEFAULT_BEFORE, components: int = N_COMPONENTS,
                     **corpus_kwargs) -> TextPrior:
    """
    The text prior on `graph`'s nodes. `components=0` builds the two report rates only (cheap
    enough for the whole-city graph); the embedding columns are then zero.
    """
    corpus = placed_corpus(before, **corpus_kwargs)
    n = graph.n_nodes
    index = {int(v): i for i, v in enumerate(graph.node_ids)}
    counts = np.zeros((n, N_BUCKETS))
    incidents = np.zeros((n, N_BUCKETS))
    pcs = np.zeros((n, N_BUCKETS, N_COMPONENTS))
    explained = 0.0
    projected = None
    if components and len(corpus.embeddings) > N_COMPONENTS:
        x = corpus.embeddings.astype(np.float64)
        x = x - x.mean(axis=0)
        _, s, vt = np.linalg.svd(x, full_matrices=False)
        projected = x @ vt[:N_COMPONENTS].T
        projected /= projected.std(axis=0).clip(min=1e-9)
        explained = float((s[:N_COMPONENTS] ** 2).sum() / (s ** 2).sum())
    used = 0
    for k, (when, nodes) in enumerate(zip(corpus.created, corpus.nodes)):
        rows = [index[int(v)] for v in nodes if int(v) in index]
        if not rows:
            continue
        used += 1
        b = bucket(when)
        counts[rows, b] += 1
        if corpus.incident[k]:
            incidents[rows, b] += 1
        if projected is not None:
            pcs[rows, b, :] += projected[k]
    with np.errstate(invalid="ignore", divide="ignore"):
        pcs = np.where(counts[:, :, None] > 0, pcs / counts[:, :, None], 0.0)
    values = np.zeros((n, N_BUCKETS, len(FEATURES)), dtype=np.float32)
    values[:, :, 0] = np.log1p(counts) / max(np.log1p(counts.max()), 1e-9)
    values[:, :, 1] = np.log1p(incidents) / max(np.log1p(incidents.max()), 1e-9)
    values[:, :, 2:] = pcs
    return TextPrior(values, before, used, corpus.n_posts, explained)


def coverage_report(before: date = DEFAULT_BEFORE) -> str:
    corpus = placed_corpus(before)
    everything = all_placed()
    return (f"text prior: {len(corpus.created):,} of {corpus.n_posts:,} posts "
            f"({len(corpus.created) / max(corpus.n_posts, 1):.1%}) placed on the road graph and posted "
            f"before {before.isoformat()}; {int(corpus.incident.sum()):,} of them incidents. "
            f"Placeable at any date (live text): {len(everything.created):,} "
            f"({len(everything.created) / max(everything.n_posts, 1):.1%})")


if __name__ == "__main__":
    print(coverage_report())
    print(json.dumps({"features": FEATURES}, indent=1))
