from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.mlp_decoder import MLPDecoder
from src.models.radr_stgnn import RADRSTGNN
from src.models.stgnn_input_builder import build_stgnn_input

IGNORE_INDEX = -100
CLASS_RISK = torch.tensor([0.0, 0.5, 1.0])
N_CLASSES = 3


def scatter_camera_features(
    camera_features: torch.Tensor, camera_index: torch.Tensor, n_nodes: int, fill: torch.Tensor
) -> torch.Tensor:
    b, t, _, f = camera_features.shape
    out = fill.view(1, 1, 1, f).expand(b, t, n_nodes, f).clone()
    # Under autocast camera_features comes back float16 while `out` (built from the plain fp32
    # placeholder parameter) is float32; fancy-index assignment needs an exact dtype match.
    out[:, :, camera_index, :] = camera_features.to(out.dtype)
    return out


class TrafficRiskModel(nn.Module):

    def __init__(
        self,
        fusion: nn.Module,
        stgnn: RADRSTGNN,
        decoder: Optional[MLPDecoder] = None,
        n_camera_nodes: int = 0,
    ):
        super().__init__()
        self.fusion = fusion
        self.stgnn = stgnn
        self.decoder = decoder or MLPDecoder(node_embedding_dim=stgnn.output_dim)
        self.placeholder = nn.Parameter(torch.zeros(fusion.lstm.hidden_size))
        # One learned vector per camera intersection, added to its fused features before the graph.
        # The Light/Medium/Heavy labels are relative to each camera's view, so without an identity
        # signal the model cannot tell "Heavy for this camera" from "Heavy for that one". Zero-init:
        # it starts as a no-op. Kept optional so checkpoints trained without it still load.
        self.camera_embedding = (
            nn.Embedding(n_camera_nodes, fusion.lstm.hidden_size) if n_camera_nodes else None
        )
        if self.camera_embedding is not None:
            nn.init.zeros_(self.camera_embedding.weight)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        a_hat: torch.Tensor,
        camera_index: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        features = build_stgnn_input(
            self.fusion,
            batch["images"],
            batch["text"],
            batch["temporal"],
            visual_mask=batch["visual_mask"],
            text_mask=batch["text_mask"],
        )
        if self.camera_embedding is not None:
            features = features + self.camera_embedding.weight.to(features.dtype)
        x = scatter_camera_features(features, camera_index, a_hat.shape[0], self.placeholder)
        nodes = self.stgnn(x, a_hat)
        return self.decoder(nodes, edge_index)


def camera_edge_ids(edge_index: torch.Tensor, camera_index: torch.Tensor, n_nodes: int) -> torch.Tensor:
    is_camera = torch.zeros(n_nodes, dtype=torch.bool)
    is_camera[camera_index.cpu()] = True
    src, dst = (edge_index[0].cpu(), edge_index[1].cpu())
    touching = is_camera[src] | is_camera[dst]
    ids = touching.nonzero(as_tuple=True)[0]
    if len(ids) == 0:
        raise ValueError("No graph edge touches a camera node.")
    return ids


def camera_edge_targets(
    target: torch.Tensor,
    camera_index: torch.Tensor,
    n_nodes: int,
    edge_index: torch.Tensor,
    edge_ids: torch.Tensor,
) -> torch.Tensor:
    known = target != IGNORE_INDEX
    values = CLASS_RISK.to(target.device)[target.clamp(min=0)]
    values = torch.where(known, values, torch.full_like(values, float("nan")))
    node_risk = torch.full((target.shape[0], n_nodes), float("nan"), device=target.device)
    node_risk[:, camera_index.to(target.device)] = values
    src, dst = (edge_index[0, edge_ids], edge_index[1, edge_ids])
    pair = torch.stack([node_risk[:, src], node_risk[:, dst]], dim=-1)
    return torch.nanmean(pair, dim=-1)


def sample_weights_from_targets(
    targets: torch.Tensor, class_weights: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    """Per-edge BCE weight from its Light/Medium/Heavy bucket (targets are only ever
    0.0, 0.5, 1.0 or nan, since they come from CLASS_RISK). None if class_weights is None."""
    if class_weights is None:
        return None
    # round(target * 2) maps 0.0/0.5/1.0 -> 0/1/2; nan rows are excluded by the loss's own
    # mask before this is indexed, so the clamp here only has to avoid an out-of-range read.
    class_idx = torch.nan_to_num(targets * 2, nan=0.0).round().long().clamp(0, N_CLASSES - 1)
    return class_weights.to(targets.device)[class_idx]


def edge_risk_loss(
    logits: torch.Tensor,
    edge_ids: torch.Tensor,
    targets: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, int]:
    at_camera_edges = logits[:, edge_ids]
    mask = ~torch.isnan(targets)
    count = int(mask.sum())
    if count == 0:
        return (at_camera_edges.sum() * 0.0, 0)
    weight = sample_weights_from_targets(targets, class_weights)
    loss = F.binary_cross_entropy_with_logits(
        at_camera_edges[mask], targets[mask],
        weight=weight[mask] if weight is not None else None,
        reduction="sum",
    )
    return (loss / count, count)