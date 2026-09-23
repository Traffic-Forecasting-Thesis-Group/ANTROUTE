from __future__ import annotations

import torch
import torch.nn as nn


def build_stgnn_input(
    fusion: nn.Module,
    images: torch.Tensor,
    text_embeddings: torch.Tensor,
    temporal_features: torch.Tensor,
    lengths: torch.Tensor = None,
    visual_mask: torch.Tensor = None,
    text_mask: torch.Tensor = None,
) -> torch.Tensor:
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