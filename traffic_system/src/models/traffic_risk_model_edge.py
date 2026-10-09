"""
===============================================================================
THE WHOLE MODEL, ASSEMBLED  --  start here if you are reading the code
                                against the architecture diagram
===============================================================================
This file is where blocks 9, 15, 16 and 17 are wired into one network, and
where its output is compared against block 14 in the training loss. If you
read only one model file, read this one: TrafficRiskModel.forward() is the
diagram's entire bottom half, top to bottom, in about twenty lines.

    [BLOCK 9 ] CNN + LSTM          build_stgnn_input(self.fusion, ...)
                                       |
                                       v  scatter_camera_features()
                                   put each camera's features on ITS node;
                                   every other node gets a learned
                                   placeholder meaning "no camera here"
                                       |
                                       +  ContextEncoder(context)
                                          weather / flood / events / clock
                                          for EVERY node (blocks 11 + 12)
                                       |
                                       v
    [BLOCK 15] RADR STGNN          self.stgnn(x, a_hat)
                                       |  -> one embedding per junction
                                       v
    [BLOCK 16] MLP Decoder         self.decoder(nodes, edge_index, ...)
                                       |  -> one logit per road segment
                                       v
    [BLOCK 17] Congestion Risk Score   sigmoid(logit) -> risk in [0, 1]
                                       ^
    [BLOCK 14] YOLOv8 Auto-Labeling ---+  edge_risk_loss() compares the two

WHY THE CONTEXT ENCODER EXISTS (the "multimodal" fix)
    Originally only camera nodes had real inputs and every other node shared
    one placeholder, so weather, floods and tweets could only reach a road
    through at most two GCN hops from a camera. With ~9 camera junctions in a
    59k-node city that meant most roads could not move at all. ContextEncoder
    gives all nodes their own 17-feature context vector, which is what makes
    the Congestion Risk Score genuinely multimodal across the whole graph.

HOW LABELS BECOME EDGE TARGETS
    CLASS_RISK = [0.0, 0.5, 1.0] maps Light/Medium/Heavy (block 14) onto the
    same [0, 1] scale the model predicts. camera_edge_targets() then averages
    the two endpoint junctions' labels to get a target for the EDGE between
    them, because labels are observed at cameras (nodes) but predicted per
    road (edge). Unlabelled entries are nan and masked out of the loss.

A NOTE ON CLASS WEIGHTS (open decision, see SESSION_HANDOFF.md)
    class_weights re-weights Light/Medium/Heavy to counter the ~58% Heavy
    skew. But balanced weights make the loss's best constant prediction
    exactly 0.5 while the labels average ~0.72, which builds a calibration
    offset in by construction. src/models/risk_calibration.py exists to undo
    it. Training unweighted is the current recommendation for calibrated risk.

KEY NAMES
    scatter_camera_features()  camera features -> their graph nodes
    ContextEncoder             non-visual sources -> every node
    TrafficRiskModel.forward() blocks 9 -> 15 -> 16 in one pass
    camera_edge_targets()      block 14 labels -> per-edge targets
    edge_risk_loss()           where prediction meets training target
===============================================================================
"""

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


class ContextEncoder(nn.Module):
    """
    Per-node, per-minute context (src/data/node_context.py: daily weather of the node's cell,
    flood hazard and its interaction with rain, live incidents, clock, road attributes) ->
    a vector of the CNN+LSTM output size, added to every node's STGNN input.

    This is what lets the non-visual sources reach every road: a node without a camera used
    to get one shared placeholder, so its risk could only change through camera features
    carried at most two GCN hops.
    """

    def __init__(self, context_dim: int, hidden: int, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(context_dim, width), nn.ReLU(), nn.Linear(width, hidden))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context)


class TrafficRiskModel(nn.Module):
    """
    CNN+LSTM -> RADR STGNN -> MLP decoder, one Congestion Risk logit per edge.

    STGNN input per node and timestep:
        camera nodes:  CNN+LSTM fused features (CCTV + its text + its temporal input)
        other nodes:   a learned placeholder ("no camera here")
        every node:    + ContextEncoder(context) when context_dim > 0
                       + the legacy flood_level embedding, for checkpoints trained with it

    `flood_level` ([N] long, Project NOAH hazard level per graph node) is the older way flood
    entered: a learned vector per level, zero-initialised. New models carry flood inside the
    context instead (ordinal and x rainfall), so it is kept only to load those checkpoints.

    `edge_features` ([E, D], node_context.edge_features) are per-edge road attributes passed to
    the decoder, aligned to the edge_index given at forward time.
    """

    def __init__(
        self,
        fusion: nn.Module,
        stgnn: RADRSTGNN,
        decoder: Optional[MLPDecoder] = None,
        flood_level: Optional[torch.Tensor] = None,
        n_flood_levels: int = 4,
        context_dim: int = 0,
        edge_features: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.fusion = fusion
        self.stgnn = stgnn
        edge_feature_dim = 0 if edge_features is None else int(edge_features.shape[1])
        self.decoder = decoder or MLPDecoder(node_embedding_dim=stgnn.output_dim, edge_feature_dim=edge_feature_dim)
        if self.decoder.edge_feature_dim != edge_feature_dim:
            raise ValueError("the decoder's edge_feature_dim must match edge_features")
        hidden = fusion.lstm.hidden_size
        self.placeholder = nn.Parameter(torch.zeros(hidden))
        self.context_encoder = ContextEncoder(context_dim, hidden) if context_dim else None
        if edge_features is not None:
            self.register_buffer("edge_features", torch.as_tensor(edge_features, dtype=torch.float32))
        else:
            self.edge_features = None
        self.flood_embedding = None
        if flood_level is not None:
            flood_level = torch.as_tensor(flood_level, dtype=torch.long)
            if flood_level.ndim != 1 or flood_level.min() < 0 or flood_level.max() >= n_flood_levels:
                raise ValueError(f"flood_level must be [N] with values in [0, {n_flood_levels - 1}]")
            self.register_buffer("flood_level", flood_level)
            self.flood_embedding = nn.Embedding(n_flood_levels, hidden)
            nn.init.zeros_(self.flood_embedding.weight)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        a_hat: torch.Tensor,
        camera_index: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        # [BLOCK 9] CNN + LSTM: fuse CCTV frames, tweet embeddings and temporal
        # features into one sequence vector per camera, per timestep.
        features = build_stgnn_input(
            self.fusion,
            batch["images"],
            batch["text"],
            batch["temporal"],
            visual_mask=batch["visual_mask"],
            text_mask=batch["text_mask"],
        )
        # [BLOCK 9 -> 15] Place each camera's features on the graph node it
        # watches. Nodes without a camera get the learned placeholder.
        x = scatter_camera_features(features, camera_index, a_hat.shape[0], self.placeholder)
        if self.flood_embedding is not None:
            if self.flood_level.shape[0] != a_hat.shape[0]:
                raise ValueError(
                    f"flood_level covers {self.flood_level.shape[0]} nodes but the graph has {a_hat.shape[0]}"
                )
            x = x + self.flood_embedding(self.flood_level).to(x.dtype)
        if self.context_encoder is not None:
            context = batch.get("context")
            if context is None or tuple(context.shape[:3]) != tuple(x.shape[:3]):
                raise ValueError(
                    f"this model needs batch['context'] of shape [B, T, N, C] matching {tuple(x.shape[:3])}"
                )
            # [BLOCKS 11 + 12 -> 15] Weather, flood, incidents and clock for
            # EVERY node, not just camera nodes. This is what makes the risk
            # score multimodal across the whole graph.
            x = x + self.context_encoder(context.to(x.dtype)).to(x.dtype)

        # [BLOCK 15] RADR STGNN: spatial GCN then temporal GRU -> one
        # embedding per junction.
        nodes = self.stgnn(x, a_hat)

        # [BLOCK 16] MLP Decoder: junction embeddings -> one risk logit per
        # road segment. sigmoid() turns this into block 17's score.
        return self.decoder(nodes, edge_index, self.edge_features)


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
    # [BLOCK 14 -> 17] Labels are observed AT CAMERAS (nodes) but predicted per
    # ROAD (edge), so an edge's target is the mean of its two endpoints'
    # labels. Unlabelled endpoints are nan and get masked out of the loss.
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
    # [BLOCK 17 vs BLOCK 14] THE TRAINING LOSS -- the dashed arrow on the
    # diagram. Only edges at a camera can be scored, because only they have a
    # ground-truth label; the rest of the city is learned by propagation.
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