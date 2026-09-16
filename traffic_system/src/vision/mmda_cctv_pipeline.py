import os
import cv2
import torch
import subprocess
import torch.nn as nn
from pathlib import Path
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ==========================================
# Phase 1: Data Extraction & Normalization
# ==========================================

class MMDACCTVDataset(Dataset):
    def __init__(self, root_dir: str, sequence_length: int = 4):
        """
        PyTorch Dataset for MMDA CCTV footage.
        :param root_dir: Root directory "D:\MMDA CCTV FOOTAGE\REQ. PUP STUDENT"
        :param sequence_length: Number of frames per temporal sequence for the LSTM
        """
        self.root_dir = Path(root_dir)
        self.sequence_length = sequence_length
        self.mp4box_path = self.root_dir / "MP4Box.exe"

        # Verify MP4Box exists
        if not self.mp4box_path.exists():
            print(f"Warning: MP4Box.exe not found at {self.mp4box_path}. Conversion may fail.")

        # Standard ImageNet normalization
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Traverse directory: Root -> Date -> Hex Folder -> dados -> raw files
        self.raw_files = []
        for date_folder in self.root_dir.iterdir():
            if date_folder.is_dir():
                for hex_folder in date_folder.iterdir():
                    if hex_folder.is_dir():
                        dados_folder = hex_folder / "dados"
                        if dados_folder.exists():
                            for raw_file in dados_folder.iterdir():
                                if raw_file.is_file() and raw_file.suffix != '.mp4':
                                    self.raw_files.append(raw_file)

        print(f"Found {len(self.raw_files)} raw CCTV chunks.")

    def _convert_to_mp4(self, raw_path: Path) -> Path:
        """Uses MP4Box.exe to convert proprietary raw files to standard .mp4"""
        temp_mp4 = raw_path.with_suffix('.temp.mp4')
        if temp_mp4.exists():
            return temp_mp4 # Skip if already converted

        try:
            # MP4Box command to mux raw video stream into MP4 container
            cmd = [str(self.mp4box_path), "-add", str(raw_path), str(temp_mp4)]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return temp_mp4
        except subprocess.CalledProcessError:
            print(f"Failed to convert {raw_path}")
            if temp_mp4.exists():
                temp_mp4.unlink()
            return None

    def _extract_frames(self, mp4_path: Path) -> List[torch.Tensor]:
        """Extracts 1 frame every 15 minutes, resizes and normalizes them."""
        cap = cv2.VideoCapture(str(mp4_path))
        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps != fps: # Handle NaN or 0
            fps = 30.0

        # 1 frame per 15 minutes (15 * 60 = 900 seconds)
        frame_interval = int(fps * 900)

        frames = []
        current_frame = 0

        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR (OpenCV) to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply ImageNet normalization & Tensor conversion
            tensor_frame = self.transform(frame_rgb)
            frames.append(tensor_frame)

            current_frame += frame_interval

        cap.release()
        return frames

    def __len__(self):
        # We return the number of raw video chunks available
        return len(self.raw_files)

    def __getitem__(self, idx):
        raw_path = self.raw_files[idx]

        try:
            # 1. Convert
            mp4_path = self._convert_to_mp4(raw_path)
            if not mp4_path:
                raise ValueError("Conversion failed")

            # 2. Extract & Normalize
            frames = self._extract_frames(mp4_path)

            # Clean up temp file to save disk space
            mp4_path.unlink()

            if not frames:
                raise ValueError("No frames extracted")

            # 3. Sequence Padding/Truncation for LSTM
            # If a video yields too few frames, pad with zeros. If too many, truncate.
            if len(frames) < self.sequence_length:
                padding = [torch.zeros(3, 224, 224) for _ in range(self.sequence_length - len(frames))]
                frames.extend(padding)
            else:
                frames = frames[:self.sequence_length]

            # Stack into shape: (Sequence_Length, Channels, Height, Width)
            sequence_tensor = torch.stack(frames)
            return sequence_tensor

        except Exception as e:
            # Skip corrupted files by returning a dummy zero tensor
            # In production, a custom collate_fn is better to filter out None objects
            return torch.zeros(self.sequence_length, 3, 224, 224)


# ==========================================
# Phase 2: 2D Patch Embedding
# ==========================================

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, embed_dim: int = 256):
        """
        Splits 224x224 frame into 16x16 patches and projects them.
        """
        super().__init__()
        self.patch_size = patch_size

        # nn.Conv2d perfectly implements patching and linear projection simultaneously
        # Input: (B, 3, 224, 224) -> Output: (B, embed_dim, 14, 14)
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Channels, H, W)
        x = self.proj(x)
        # We keep the 2D spatial grid (14x14) for the downstream CNN feature extractor
        # Shape: (Batch, embed_dim, 14, 14)
        return x


# ==========================================
# Phase 3: CNN + LSTM Sequence Module
# ==========================================

class CNNLSTM(nn.Module):
    def __init__(self, embed_dim: int = 256, lstm_hidden: int = 128, num_classes: int = 3):
        """
        Combines Patch Embedding -> CNN Spatial Extraction -> LSTM Temporal processing.
        """
        super().__init__()

        # 1. Patch Embedding
        self.patch_embed = PatchEmbedding(in_channels=3, patch_size=16, embed_dim=embed_dim)

        # 2. Lightweight CNN for Spatial Features
        # Takes the 14x14 grid of patch embeddings and extracts higher-level features
        self.cnn = nn.Sequential(
            nn.Conv2d(embed_dim, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2), # Reduces 14x14 -> 7x7
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling -> (Batch, 512, 1, 1)
        )

        # 3. LSTM Temporal Sequence processing
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # 4. Final Classifier Head (e.g., Light, Medium, Heavy Traffic)
        self.classifier = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Sequence of frames with shape (Batch, Seq_Length, C, H, W)
        """
        B, S, C, H, W = x.shape

        # Flatten time and batch to process spatial frames through CNN
        x_flat = x.view(B * S, C, H, W) # (B*S, 3, 224, 224)

        # Patch Projection -> (B*S, embed_dim, 14, 14)
        patches = self.patch_embed(x_flat)

        # CNN Feature Extraction -> (B*S, 512, 1, 1)
        spatial_features = self.cnn(patches)

        # Flatten spatial dims -> (B*S, 512)
        spatial_features = spatial_features.view(B * S, -1)

        # Reshape back to temporal sequence -> (B, S, 512)
        seq_features = spatial_features.view(B, S, -1)

        # LSTM Temporal processing
        lstm_out, (h_n, c_n) = self.lstm(seq_features)

        # Extract the last hidden state for prediction -> (B, lstm_hidden)
        last_hidden = h_n[-1]

        # Classification -> (B, num_classes)
        logits = self.classifier(last_hidden)

        return logits


# ==========================================
# Testing Block
# ==========================================
if __name__ == "__main__":
    print("Testing Pipeline Components...")

    # 1. Test Dataset Initialization
    root_dir = r"D:\MMDA CCTV FOOTAGE\REQ. PUP STUDENT"
    try:
        dataset = MMDACCTVDataset(root_dir=root_dir, sequence_length=4)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        print("DataLoader initialized successfully.")
    except Exception as e:
        print(f"DataLoader setup error: {e}")

    # 2. Test Model Forward Pass
    model = CNNLSTM()

    # Create dummy tensor: Batch=2, Seq=4, Channels=3, H=224, W=224
    dummy_input = torch.randn(2, 4, 3, 224, 224)

    output = model(dummy_input)
    print(f"Model Input shape: {dummy_input.shape}")
    print(f"Model Output shape: {output.shape} (Batch, Num_Classes)")
    print("Forward pass successful!")