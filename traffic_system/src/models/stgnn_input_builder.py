"""
Shape helpers shared by both risk models (traffic_risk_model.py and traffic_risk_model_edge.py)
for turning per-camera CNNLSTMFusion output into the [B, T, N, F] node tensor RADR STGNN reads.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def scatter_camera_features(
    camera_features: torch.Tensor, camera_index: torch.Tensor, n_nodes: int, fill: torch.Tensor
) -> torch.Tensor:
    """[B, T, K, F] -> [B, T, N, F]: camera nodes get their features, all others `fill` [F]."""
    b, t, _, f = camera_features.shape
    out = fill.view(1, 1, 1, f).expand(b, t, n_nodes, f).clone()
    # Under autocast camera_features comes back float16 while `out` (built from the plain fp32
    # placeholder parameter) is float32; fancy-index assignment needs an exact dtype match.
    out[:, :, camera_index, :] = camera_features.to(out.dtype)
    return out


def build_stgnn_input(
    fusion: nn.Module,
    images: torch.Tensor,
    text_embeddings: torch.Tensor,
    temporal_features: torch.Tensor,
    lengths: torch.Tensor = None,
    visual_mask: torch.Tensor = None,
    text_mask: torch.Tensor = None,
) -> torch.Tensor:
    """Run `fusion` on every node's sequence at once and return [B, T, N, F] for the STGNN.

    Inputs are node-major ([B, N, T, ...]); they are flattened to B*N sequences for the fusion
    model, then reshaped and permuted to the time-major layout RADR STGNN expects.
    """
    B, N, T = images.shape[:3]

    images_flat = images.reshape(B * N, T, *images.shape[3:])
    text_flat = text_embeddings.reshape(B * N, T, -1)
    temporal_flat = temporal_features.reshape(B * N, T, -1)

    lengths_flat = None
    if lengths is not None:
        lengths_flat = lengths.unsqueeze(1).expand(B, N).reshape(B * N)

    visual_mask_flat = None if visual_mask is None else visual_mask.reshape(B * N, T)
    text_mask_flat = None if text_mask is None else text_mask.reshape(B * N, T)

    fused_flat = fusion(
        images_flat,
        text_flat,
        temporal_flat,
        lengths=lengths_flat,
        visual_mask=visual_mask_flat,
        text_mask=text_mask_flat,
    )

    feature_dim = fused_flat.shape[-1]
    fused = fused_flat.reshape(B, N, T, feature_dim)
    stgnn_input = fused.permute(0, 2, 1, 3)
    return stgnn_input