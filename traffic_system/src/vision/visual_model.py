import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset
import h5py
import numpy as np

try:
    from .patch_embedder import PatchEmbedder
except ImportError:
    from patch_embedder import PatchEmbedder

class CCTVSequenceDataset(Dataset):
    """
    Data Loader that groups sequential frame embeddings into sliding time windows.
    """
    def __init__(self, h5_path: str, seq_length: int = 10):
        """
        :param h5_path: Path to the extracted .h5 frames file.
        :param seq_length: The temporal window size (e.g., 10 frames = 10 seconds of context).
        """
        self.h5_path = h5_path
        self.seq_length = seq_length

        # Load indices to memory, keep actual data on disk (or load to memory if small enough)
        with h5py.File(self.h5_path, 'r') as f:
            self.total_frames = f['frames'].shape[0]

        self.valid_indices = self.total_frames - self.seq_length + 1

    def __len__(self):
        return max(0, self.valid_indices)

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            # Shape: (seq_length, 3, 224, 224)
            frames_seq = f['frames'][idx : idx + self.seq_length]

        # Convert to float32 tensor
        return torch.tensor(frames_seq, dtype=torch.float32)


class VisualTrafficModel(nn.Module):
    """
    Phase 3: CNN + LSTM Sequence Processing.
    Integrates the spatial feature extraction (CNN or ViT patches) with temporal dynamics (LSTM).
    """
    def __init__(self, use_cnn: bool = True, seq_length: int = 10, embed_dim: int = 768, hidden_dim: int = 256, num_classes: int = 3):
        super().__init__()
        self.use_cnn = use_cnn
        self.seq_length = seq_length
        self.embed_dim = embed_dim

        if self.use_cnn:
            # CNN Feature Extractor Toggle (ResNet50 Backbone)
            cnn = resnet50(weights=ResNet50_Weights.DEFAULT)
            # Remove the final classification head (fc) to output feature vectors
            self.spatial_extractor = nn.Sequential(*list(cnn.children())[:-1])
            # ResNet50 outputs 2048-dim vectors, project them to our desired embed_dim
            self.feature_proj = nn.Linear(2048, embed_dim)
        else:
            # 2D Patch Embedding approach
            self.spatial_extractor = PatchEmbedder(embed_dim=embed_dim)
            # A patch embedder outputs (Num_Patches, Embed_Dim).
            # We average pool over the spatial dimension to get one vector per frame.
            self.feature_proj = nn.Identity()

        # Temporal Sequence (LSTM)
        # Input shape to LSTM: (Batch, Seq_Length, Embed_Dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # Final prediction head (e.g., predicting Congestion Level: Light, Medium, Heavy)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Sequential frames (Batch, Seq_Length, Channels, Height, Width)
        """
        B, S, C, H, W = x.shape

        # Flatten sequence into batch dimension for spatial extraction
        # Shape becomes (B * S, C, H, W)
        x_flat = x.view(B * S, C, H, W)

        if self.use_cnn:
            # Output: (B * S, 2048, 1, 1)
            features = self.spatial_extractor(x_flat)
            # Flatten spatial dimensions: (B * S, 2048)
            features = features.view(features.size(0), -1)
            # Project to uniform embed_dim: (B * S, Embed_Dim)
            features = self.feature_proj(features)
        else:
            # Output: (B * S, Num_Patches, Embed_Dim) -> e.g., (B * S, 196, 768)
            features = self.spatial_extractor(x_flat)
            # Average pool across the patches to summarize the entire frame: (B * S, Embed_Dim)
            features = features.mean(dim=1)

        # Reshape back to sequence format for LSTM: (Batch, Seq_Length, Embed_Dim)
        seq_features = features.view(B, S, self.embed_dim)

        # Pass sequence through LSTM
        lstm_out, (h_n, c_n) = self.lstm(seq_features)

        # Take the hidden state of the final sequence step to predict traffic dynamics
        # h_n shape is (num_layers, Batch, hidden_dim). We take the last layer's output.
        final_temporal_feature = h_n[-1] # Shape: (Batch, hidden_dim)

        logits = self.classifier(final_temporal_feature) # Shape: (Batch, num_classes)

        return logits

if __name__ == "__main__":
    # Test Forward Pass
    dummy_sequence = torch.randn(2, 10, 3, 224, 224) # Batch=2, Seq=10_frames

    # 1. Test CNN pathway
    model_cnn = VisualTrafficModel(use_cnn=True)
    out_cnn = model_cnn(dummy_sequence)
    print(f"CNN+LSTM Output shape: {out_cnn.shape}")

    # 2. Test ViT Patch pathway
    model_vit = VisualTrafficModel(use_cnn=False)
    out_vit = model_vit(dummy_sequence)
    print(f"ViT+LSTM Output shape: {out_vit.shape}")
