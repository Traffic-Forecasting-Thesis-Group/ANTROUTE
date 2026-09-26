"""
Road-graph inputs for RADR STGNN, cut down to the neighbourhood of the CCTV intersections.

The full network has 59,521 nodes but only 8 carry cameras, and the 8x8 key-intersection
graph has no edges. A k-hop subgraph around the camera nodes keeps real road connectivity
at a size (hundreds to a few thousand nodes) where a dense normalised adjacency is cheap.
"""

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch

from src.models.congestion_risk_score import edge_index_from_adjacency
from src.models.radr_stgnn import normalize_adjacency

DEFAULT_K = 8   # ~1.4k nodes; links most EDSA intersections (nearest pairs are 5-16 hops apart)


@dataclass
class GraphData:
    node_ids: np.ndarray                 # road-network node id per subgraph index
    adjacency: sp.csr_matrix             # subgraph adjacency
    a_hat: torch.Tensor                  # dense D^-1/2 (A+I) D^-1/2, [N, N]
    edge_index: torch.Tensor             # [2, E]
    camera_nodes: Dict[str, int]         # intersection label -> subgraph index

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)


def khop_nodes(adjacency: sp.csr_matrix, seeds: Sequence[int], k: int) -> np.ndarray:
    """Sorted indices of every node within k hops of a seed (edges treated as undirected)."""
    undirected = (adjacency + adjacency.T).tocsr()
    seen = np.zeros(adjacency.shape[0], dtype=bool)
    seen[list(seeds)] = True
    frontier = np.asarray(list(seeds))
    for _ in range(k):
        reached = np.unique(undirected[frontier].indices)
        frontier = reached[~seen[reached]]
        seen[frontier] = True
        if frontier.size == 0:
            break
    return np.flatnonzero(seen)


def subgraph_from_arrays(adjacency: sp.csr_matrix, node_order: np.ndarray,
                         camera_full_index: Dict[str, int], k: int) -> GraphData:
    keep = khop_nodes(adjacency, list(camera_full_index.values()), k)
    position = {int(full): i for i, full in enumerate(keep)}
    sub = adjacency[keep][:, keep].tocsr()
    return GraphData(
        node_ids=np.asarray(node_order)[keep],
        adjacency=sub,
        a_hat=normalize_adjacency(sub),
        edge_index=edge_index_from_adjacency(sub),
        camera_nodes={label: position[full] for label, full in camera_full_index.items()},
    )


def load_camera_full_index(spatial_dir: Path, node_order: np.ndarray) -> Dict[str, int]:
    """Intersection label -> index in the full graph, for nodes flagged is_cctv_node."""
    position = {int(n): i for i, n in enumerate(node_order)}
    with (Path(spatial_dir) / "full_network_static_features.csv").open(encoding="utf-8", newline="") as f:
        return {r["cctv_label"]: position[int(r["node_id"])]
                for r in csv.DictReader(f) if r["is_cctv_node"] == "True"}


def build_subgraph(spatial_dir: Path, k: int = DEFAULT_K) -> GraphData:
    spatial_dir = Path(spatial_dir)
    adjacency = sp.load_npz(spatial_dir / "metro_manila_adjacency.npz").tocsr()
    node_order = np.load(spatial_dir / "metro_manila_node_order.npy")
    return subgraph_from_arrays(adjacency, node_order, load_camera_full_index(spatial_dir, node_order), k)


def graph_sizes(spatial_dir: Path, ks: Sequence[int]) -> List[Tuple[int, int, int]]:
    """(k, nodes, directed edges) for each k, to choose a subgraph size."""
    spatial_dir = Path(spatial_dir)
    adjacency = sp.load_npz(spatial_dir / "metro_manila_adjacency.npz").tocsr()
    node_order = np.load(spatial_dir / "metro_manila_node_order.npy")
    seeds = list(load_camera_full_index(spatial_dir, node_order).values())
    out = []
    for k in ks:
        keep = khop_nodes(adjacency, seeds, k)
        out.append((k, len(keep), int(adjacency[keep][:, keep].nnz)))
    return out


# ---------------------------------------------------------------------------
# camera id -> intersection
# ---------------------------------------------------------------------------
def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def load_camera_map(path: Path) -> Dict[str, str]:
    """camera_id -> intersection label from a CSV with columns camera_id, intersection."""
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as f:
        return {r["camera_id"].strip(): r["intersection"].strip()
                for r in csv.DictReader(f) if r.get("camera_id") and r.get("intersection")}


def cctv_labels(spatial_dir: Path) -> List[str]:
    """Intersection labels of the camera nodes, in file order."""
    with (Path(spatial_dir) / "full_network_static_features.csv").open(encoding="utf-8", newline="") as f:
        return [r["cctv_label"] for r in csv.DictReader(f) if r["is_cctv_node"] == "True"]


def expected_camera_counts(path: Path) -> Dict[str, int]:
    """Cameras expected per graph intersection, from configs/cctv_locations.csv (location, intersection)."""
    path = Path(path)
    if not path.exists():
        return {}
    counts: Dict[str, int] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            counts[row["intersection"]] = counts.get(row["intersection"], 0) + 1
    return counts


def save_camera_map(path: Path, mapping: Dict[str, str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["camera_id", "intersection"])
        writer.writerows(sorted(mapping.items()))


def derive_camera_map(camera_ids: Sequence[str], intersections: Sequence[str]) -> Dict[str, str]:
    """Best-effort mapping from footage folder names.

    Matches on the numeric camera code (folder '..._8337 - Ayala NB 1' <-> label '8337-Ayala
    NB 1-PTZ') or on the descriptive suffix after ' - '. Cameras whose folder name carries
    neither are left out; list them with `unmapped` and fill them in the CSV by hand.
    """
    mapping = {}
    for camera in camera_ids:
        code = re.search(r"_(\d+)(?:\s|$)", camera)
        suffix = camera.split(" - ", 1)[1] if " - " in camera else ""
        for label in intersections:
            by_code = code and label.startswith(f"{code.group(1)}-")
            tokens, label_tokens = _tokens(suffix), _tokens(label)
            by_name = bool(tokens) and (tokens <= label_tokens or label_tokens <= tokens)
            if by_code or by_name:
                mapping[camera] = label
                break
    return mapping


def resolve_camera_map(camera_ids: Sequence[str], intersections: Sequence[str],
                       csv_path: Optional[Path] = None) -> Tuple[Dict[str, str], List[str]]:
    """CSV entries win over names derived from folders. Returns (camera_id -> label, unmapped ids)."""
    mapping = derive_camera_map(camera_ids, intersections)
    if csv_path:
        mapping.update({c: lab for c, lab in load_camera_map(csv_path).items() if lab in set(intersections)})
    unmapped = [c for c in camera_ids if c not in mapping]
    return {c: mapping[c] for c in camera_ids if c in mapping}, unmapped
