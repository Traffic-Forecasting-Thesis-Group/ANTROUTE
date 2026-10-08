"""
Does each data source actually move the Congestion Risk Score -- and on roads without a camera?

source_sensitivity() changes one source at a time in a real batch and measures how far the
predicted risk moves, separately on camera edges (where the model is supervised) and on all
other edges (where, before the node context, nothing but the nearby cameras could move it):

  cctv      every camera goes offline (visual_mask all False)
  weather   every node's day becomes the wettest, most humid training day (scaled 1.0)
  flood     every node becomes Project NOAH level 3, with the batch's own rainfall
  events    a fresh two-lane incident at every fifth node, for the whole window
  temporal  the clock moves ten hours (AM peak -> PM peak) and three weekdays on
  spatial   the road attributes are shuffled between nodes

constant_edges() is the dataset-level check: the share of non-camera edges whose risk is the
same in every window, which was ~89% for the CCTV-only checkpoint.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import torch

from src.data.node_context import COLUMN, GROUPS

SOURCES = ("cctv", "weather", "flood", "events", "temporal", "spatial")
MOVED = 1e-3  # an edge counts as responding when its risk moves by more than this


def _rotate(sin: torch.Tensor, cos: torch.Tensor, fraction: float):
    angle = 2 * math.pi * fraction
    return sin * math.cos(angle) + cos * math.sin(angle), cos * math.cos(angle) - sin * math.sin(angle)


def perturb(batch: Dict[str, torch.Tensor], source: str, seed: int = 0) -> Dict[str, torch.Tensor]:
    """A copy of `batch` with one source changed (see the module docstring)."""
    out = {k: v.clone() for k, v in batch.items()}
    if source == "cctv":
        out["visual_mask"] = torch.zeros_like(out["visual_mask"])
        return out
    if "context" not in out:
        raise ValueError(f"source {source!r} lives in the node context, which this batch does not have")
    ctx = out["context"]                                   # [B, T, N, C]
    if source == "weather":
        ctx[..., COLUMN["precip"]] = 1.0
        ctx[..., COLUMN["humidity"]] = 1.0
        ctx[..., COLUMN["flood_x_precip"]] = ctx[..., COLUMN["flood_level"]] * 1.0
    elif source == "flood":
        ctx[..., COLUMN["flood_level"]] = 1.0
        ctx[..., COLUMN["flood_x_precip"]] = ctx[..., COLUMN["precip"]]
    elif source == "events":
        ctx[:, :, ::5, COLUMN["event_impact"]] = 1.0
        ctx[:, :, ::5, COLUMN["event_count"]] = math.log1p(1.0)
    elif source == "temporal":
        s, c = _rotate(ctx[..., COLUMN["minute_sin"]], ctx[..., COLUMN["minute_cos"]], 10 / 24)
        ctx[..., COLUMN["minute_sin"]], ctx[..., COLUMN["minute_cos"]] = s, c
        s, c = _rotate(ctx[..., COLUMN["weekday_sin"]], ctx[..., COLUMN["weekday_cos"]], 3 / 7)
        ctx[..., COLUMN["weekday_sin"]], ctx[..., COLUMN["weekday_cos"]] = s, c
    elif source == "spatial":
        cols = [COLUMN[name] for name in GROUPS["spatial"]]
        order = torch.randperm(ctx.shape[2], generator=torch.Generator().manual_seed(seed))
        ctx[..., cols] = ctx[:, :, order][..., cols]
    else:
        raise ValueError(f"unknown source {source!r}; choose from {SOURCES}")
    return out


@torch.no_grad()
def source_sensitivity(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    a_hat: torch.Tensor,
    camera_index: torch.Tensor,
    edge_index: torch.Tensor,
    camera_edges: torch.Tensor,
    sources: Sequence[str] = SOURCES,
) -> Dict[str, Dict[str, float]]:
    """
    Per source: mean and max |change in risk| on camera edges and on the other edges, and the
    share of the other edges whose risk moved by more than MOVED.
    """
    model.eval()
    base = torch.sigmoid(model(batch, a_hat, camera_index, edge_index).float())
    other = torch.ones(base.shape[-1], dtype=torch.bool, device=base.device)
    other[camera_edges] = False
    result = {}
    for source in sources:
        if source != "cctv" and "context" not in batch:
            continue
        changed = torch.sigmoid(model(perturb(batch, source), a_hat, camera_index, edge_index).float())
        delta = (changed - base).abs()
        result[source] = {
            "camera_mean": float(delta[:, camera_edges].mean()),
            "camera_max": float(delta[:, camera_edges].max()),
            "other_mean": float(delta[:, other].mean()) if other.any() else 0.0,
            "other_max": float(delta[:, other].max()) if other.any() else 0.0,
            "other_moved": float((delta[:, other] > MOVED).float().mean()) if other.any() else 0.0,
        }
    return result


def constant_edges(risk_by_window: torch.Tensor, camera_edges: Optional[torch.Tensor] = None,
                   tolerance: float = 1e-4) -> float:
    """Share of (non-camera) edges whose risk never moves by more than `tolerance` across
    windows. risk_by_window: [W, E]."""
    spread = risk_by_window.max(dim=0).values - risk_by_window.min(dim=0).values
    if camera_edges is not None:
        keep = torch.ones(spread.shape[0], dtype=torch.bool)
        keep[camera_edges.cpu()] = False
        spread = spread[keep]
    return float((spread <= tolerance).float().mean())
