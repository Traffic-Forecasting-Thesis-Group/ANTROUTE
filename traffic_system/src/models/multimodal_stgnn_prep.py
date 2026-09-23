import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """
    Phase 2: 2D Patch Embedding Module.
    Takes (B, C, H, W) and uses nn.Conv2d to create a grid of 2D patch tokens.
    """
    def __init__(self, in_channels: int = 3, patch_size: int = 16, embed_dim: int = 256):
        super().__init__()
        self.patch_size = patch_size

        # Generates a grid of patches. For 224x224 input, output spatial dim is 14x14.
        # Shape transition: (B, 3, 224, 224) -> (B, embed_dim, 14, 14)
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MultimodalCNNLSTM(nn.Module):
    """
    Phase 3: Multimodal CNN + LSTM Module (Feature Extractor for RADR STGNN).
    Fuses Visual Patches, DistilBERT Text Embeddings, and WeatherStack Features.
    Does NOT contain a classification head. Returns pure latent spatio-temporal nodes.
    """
    def __init__(self,
                 visual_embed_dim: int = 256,
                 text_dim: int = 768,        # e.g., DistilBERT output dim
                 weather_dim: int = 10,      # e.g., WeatherStack normalized vector dim
                 cnn_out_dim: int = 512,
                 stgnn_input_dim: int = 256):# Dimensionality required by the STGNN
        super().__init__()

        self.patch_embed = PatchEmbedding(in_channels=3, patch_size=16, embed_dim=visual_embed_dim)

        # 2D CNN Backbone to process the visual patches
        # Input: (B*S, visual_embed_dim, 14, 14) -> Output: (B*S, cnn_out_dim)
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(visual_embed_dim, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2), # 14x14 -> 7x7
            nn.Conv2d(512, cnn_out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_out_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), # Global Average Pooling -> (B*S, cnn_out_dim, 1, 1)
            nn.Flatten()                  # -> (B*S, cnn_out_dim)
        )

        # Fusion Dimensions
        self.combined_dim = cnn_out_dim + text_dim + weather_dim

        # Temporal Sequence Processing (LSTM)
        self.lstm = nn.LSTM(
            input_size=self.combined_dim,
            hidden_size=stgnn_input_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

    def forward(self,
                visual_frames: torch.Tensor,
                text_embeddings: torch.Tensor,
                weather_features: torch.Tensor,
                return_sequence: bool = True) -> torch.Tensor:
        """
        :param visual_frames: (Batch, Seq_Len, 3, 224, 224)
        :param text_embeddings: (Batch, Seq_Len, text_dim)
        :param weather_features: (Batch, Seq_Len, weather_dim)
        :param return_sequence: If True, returns (B, Seq_Len, stgnn_input_dim) for temporal STGNNs.
                                If False, returns (B, stgnn_input_dim) of the final time step.
        """
        B, S, C, H, W = visual_frames.shape

        # 1. Process Visual Modality
        # Flatten time into batch to process spatial frames efficiently
        x_visual = visual_frames.view(B * S, C, H, W)

        # (B*S, 3, 224, 224) -> (B*S, visual_embed_dim, 14, 14)
        visual_patches = self.patch_embed(x_visual)

        # (B*S, visual_embed_dim, 14, 14) -> (B*S, cnn_out_dim)
        visual_spatial_vectors = self.cnn_backbone(visual_patches)

        # Reshape back to sequence: (B, S, cnn_out_dim)
        visual_seq = visual_spatial_vectors.view(B, S, -1)

        # 2. Multimodal Fusion (Concatenation across each time step)
        # combined_seq shape: (B, S, cnn_out_dim + text_dim + weather_dim)
        combined_seq = torch.cat([visual_seq, text_embeddings, weather_features], dim=-1)

        # 3. Temporal LSTM Processing
        # lstm_out: (B, S, stgnn_input_dim), h_n: (num_layers, B, stgnn_input_dim)
        lstm_out, (h_n, c_n) = self.lstm(combined_seq)

        # Phase 4: Interface for RADR STGNN
        # Return pure latent spatio-temporal nodes to be fed into the STGNN's Graph Convolution layers.
        if return_sequence:
            return lstm_out       # Shape: [Batch, Seq_Len, stgnn_input_dim]
        else:
            return h_n[-1]        # Shape: [Batch, stgnn_input_dim]

if __name__ == "__main__":
    # Interface validation test
    print("Validating Multimodal Feature Extractor shapes for RADR STGNN...")

    BATCH_SIZE = 4
    SEQ_LEN = 10  # 10 time steps (e.g., historical window)

    # Mock Inputs
    dummy_visual = torch.randn(BATCH_SIZE, SEQ_LEN, 3, 224, 224)
    dummy_text = torch.randn(BATCH_SIZE, SEQ_LEN, 768)      # DistilBERT outputs
    dummy_weather = torch.randn(BATCH_SIZE, SEQ_LEN, 10)    # WeatherStack norms

    # Initialize the extractor (No classifier)
    stgnn_prep_model = MultimodalCNNLSTM(
        visual_embed_dim=256,
        text_dim=768,
        weather_dim=10,
        cnn_out_dim=512,
        stgnn_input_dim=128 # The final feature size needed by the STGNN nodes
    )

    # Forward Pass
    stgnn_node_features = stgnn_prep_model(dummy_visual, dummy_text, dummy_weather, return_sequence=True)

    print(f"Visual Input:  {dummy_visual.shape}")
    print(f"Text Input:    {dummy_text.shape}")
    print(f"Weather Input: {dummy_weather.shape}")
    print(f"STGNN Latent Node Features Output: {stgnn_node_features.shape}")
    print("SUCCESS: Ready to merge with OSMnx adjacency matrix!")
