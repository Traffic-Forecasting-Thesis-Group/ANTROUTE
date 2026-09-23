import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.vision.patch_embedder import PatchEmbedder


class VisualCNNEncoder(nn.Module):
    """
    CNN branch of ANTROUTE's CNN + LSTM module.

    Input:
        [B, T, 3, H, W]   (pixels already normalized to [0, 1])

    Output:
        [B, T, visual_feature_dim]

    Assumes PatchEmbedder returns [N, num_patches, patch_embed_dim]
    with NO class token. This is checked at runtime.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        patch_embed_dim: int = 768,
        cnn_hidden_dim: int = 256,
        visual_feature_dim: int = 128,
    ):
        super().__init__()

        # Image settings
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size ({image_size}) must be divisible "
                f"by patch_size ({patch_size})"
            )

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.patch_embed_dim = patch_embed_dim
        self.visual_feature_dim = visual_feature_dim

        # Patch embedding
        self.patch_embedder = PatchEmbedder(
            img_size=image_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=patch_embed_dim,
        )

        # CNN encoder
        self.cnn = nn.Sequential(
            nn.Conv2d(patch_embed_dim, cnn_hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(cnn_hidden_dim, visual_feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(visual_feature_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """[N, 3, H, W] -> [N, visual_feature_dim]"""

        # Input validation
        if frames.ndim != 4:
            raise ValueError("frames must have shape [N, C, H, W]")

        N, C, H, W = frames.shape

        if C != 3:
            raise ValueError(f"Expected RGB images with 3 channels, received {C}")

        if H != self.image_size or W != self.image_size:
            raise ValueError(
                f"Expected {self.image_size}x{self.image_size} images, "
                f"received {H}x{W}"
            )

        # Create patches
        patches = self.patch_embedder(frames)

        # Contract check with PatchEmbedder
        expected = (N, self.grid_size ** 2, self.patch_embed_dim)
        if tuple(patches.shape) != expected:
            raise RuntimeError(
                f"PatchEmbedder returned {tuple(patches.shape)}, expected "
                f"{expected}. Remove any CLS token before the CNN."
            )

        # Restore grid: [N, P, D] -> [N, D, g, g]
        grid = patches.transpose(1, 2).reshape(
            N, self.patch_embed_dim, self.grid_size, self.grid_size
        )

        # Extract features
        return self.cnn(grid).flatten(start_dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:

        # Input validation
        if images.ndim != 5:
            raise ValueError("images must have shape [B, T, C, H, W]")

        B, T = images.shape[:2]

        # Flatten sequence, encode, restore sequence
        features = self.encode_frames(images.reshape(B * T, *images.shape[2:]))
        return features.reshape(B, T, -1)


class CNNLSTMFusion(nn.Module):
    """
    ANTROUTE CNN + LSTM feature fusion module.

    Inputs:
        images:            [B, T, 3, 224, 224]
        text_embeddings:   [B, T, text_dim]      (one pooled DistilBERT vector per timestep)
        temporal_features: [B, T, temporal_dim]  (already normalized with TRAIN-split stats)

    Optional:
        lengths:     [B]     true sequence length of each sample (for padded batches)
        visual_mask: [B, T]  True where a CCTV frame exists for that timestep
        text_mask:   [B, T]  True where at least one tweet exists for that timestep

    Output:
        [B, T, lstm_hidden_dim]   (zeros at padded timesteps)
    """

    def __init__(
        self,
        text_dim: int = 768,
        temporal_dim: int = 10,
        visual_feature_dim: int = 128,
        text_projection_dim: int = 128,
        temporal_projection_dim: int = 32,
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.2,
        image_size: int = 224,
        patch_size: int = 16,
        patch_embed_dim: int = 768,
    ):
        super().__init__()

        self.text_dim = text_dim
        self.temporal_dim = temporal_dim
        self.visual_feature_dim = visual_feature_dim

        # Visual branch
        self.visual_encoder = VisualCNNEncoder(
            image_size=image_size,
            patch_size=patch_size,
            patch_embed_dim=patch_embed_dim,
            visual_feature_dim=visual_feature_dim,
        )
        self.visual_dropout = nn.Dropout(dropout)

        # Text projection
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, text_projection_dim),
            nn.LayerNorm(text_projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Temporal projection
        self.temporal_projection = nn.Sequential(
            nn.Linear(temporal_dim, temporal_projection_dim),
            nn.LayerNorm(temporal_projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Learned placeholders for timesteps with no frame / no tweet
        self.missing_visual = nn.Parameter(torch.zeros(visual_feature_dim))
        self.missing_text = nn.Parameter(torch.zeros(text_projection_dim))

        # Fusion size + normalization (puts all modalities on one scale)
        self.fused_input_dim = (
            visual_feature_dim + text_projection_dim + temporal_projection_dim
        )
        self.fusion_norm = nn.LayerNorm(self.fused_input_dim)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.fused_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.output_dropout = nn.Dropout(dropout)

    @staticmethod
    def _as_mask(mask, B, T, name, device):
        if mask is None:
            return torch.ones(B, T, dtype=torch.bool, device=device)
        if tuple(mask.shape) != (B, T):
            raise ValueError(f"{name} must have shape [B, T] = {(B, T)}")
        return mask.to(device=device, dtype=torch.bool)

    def forward(
        self,
        images: torch.Tensor,
        text_embeddings: torch.Tensor,
        temporal_features: torch.Tensor,
        lengths: torch.Tensor = None,
        visual_mask: torch.Tensor = None,
        text_mask: torch.Tensor = None,
    ) -> torch.Tensor:

        # Input validation
        if images.ndim != 5:
            raise ValueError("images must have shape [B, T, C, H, W]")

        B, T = images.shape[:2]
        device = images.device

        if tuple(text_embeddings.shape) != (B, T, self.text_dim):
            raise ValueError(
                f"text_embeddings must have shape {(B, T, self.text_dim)}, "
                f"received {tuple(text_embeddings.shape)}"
            )

        if tuple(temporal_features.shape) != (B, T, self.temporal_dim):
            raise ValueError(
                f"temporal_features must have shape {(B, T, self.temporal_dim)}, "
                f"received {tuple(temporal_features.shape)}"
            )

        # Valid (non-padded) timesteps
        if lengths is None:
            valid = torch.ones(B, T, dtype=torch.bool, device=device)
        else:
            lengths = torch.as_tensor(lengths, dtype=torch.long)
            if lengths.shape != (B,) or lengths.min() < 1 or lengths.max() > T:
                raise ValueError(f"lengths must have shape [{B}] with values in [1, {T}]")
            valid = torch.arange(T, device=device)[None, :] < lengths.to(device)[:, None]

        visual_mask = self._as_mask(visual_mask, B, T, "visual_mask", device) & valid
        text_mask = self._as_mask(text_mask, B, T, "text_mask", device)

        # Visual features: encode ONLY real frames, so BatchNorm
        # statistics are never polluted by padding / blank frames
        flat_mask = visual_mask.reshape(-1)
        visual = torch.zeros(
            B * T, self.visual_feature_dim, device=device, dtype=images.dtype
        )
        if flat_mask.any():
            frames = images.reshape(B * T, *images.shape[2:])[flat_mask]
            visual[flat_mask] = self.visual_encoder.encode_frames(frames)
        if (~flat_mask).any():
            visual[~flat_mask] = self.missing_visual.to(images.dtype)
        visual = self.visual_dropout(visual.reshape(B, T, -1))

        # Text features
        text = self.text_projection(text_embeddings)
        text = torch.where(
            text_mask.unsqueeze(-1), text, self.missing_text.expand_as(text)
        )

        # Temporal features
        temporal = self.temporal_projection(temporal_features)

        # Feature fusion
        fused = self.fusion_norm(torch.cat((visual, text, temporal), dim=-1))

        # Sequence modeling
        if lengths is None:
            sequence_output, _ = self.lstm(fused)
        else:
            packed = pack_padded_sequence(
                fused, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.lstm(packed)
            sequence_output, _ = pad_packed_sequence(
                packed_output, batch_first=True, total_length=T
            )

        return self.output_dropout(sequence_output)