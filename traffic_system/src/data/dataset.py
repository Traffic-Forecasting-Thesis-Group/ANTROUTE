from typing import Callable, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.alignment import (
    WINDOW_STEPS,
    SessionRecord,
    build_window_index,
)


class ANTROUTEWindowDataset(Dataset):
    """
    30-minute (30-step) windows for the CNN + LSTM module.

    Each item:
        images:      [T, 3, 224, 224]  pixels in [0, 1]; zeros where no frame
        text:        [T, 768]          zeros where no post
        temporal:    [T, weather_dim]  daily weather repeated per step
        visual_mask: [T]               True where a frame exists
        text_mask:   [T]               True where at least one post exists
        target:      optional, from target_fn(record, start_step)

    Windows are fixed length, so no `lengths` / padding is needed.
    """

    def __init__(
        self,
        records: Sequence[SessionRecord],
        split: str,
        regime: Optional[str] = None,
        image_size: int = 224,
        text_dim: int = 768,
        target_fn: Optional[Callable[[SessionRecord, int], torch.Tensor]] = None,
    ):
        self.records = records
        self.index = build_window_index(records, split, regime)
        self.image_size = image_size
        self.text_dim = text_dim
        self.target_fn = target_fn

    def __len__(self) -> int:
        return len(self.index)

    def _load_frame(self, path: str) -> torch.Tensor:
        # Frame normalization: RGB, resized, bounded to [0, 1]
        image = Image.open(path).convert("RGB").resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    def __getitem__(self, i: int) -> dict:
        record_index, start = self.index[i]
        record = self.records[record_index]
        T = WINDOW_STEPS
        size = self.image_size

        # Visual
        images = torch.zeros(T, 3, size, size)
        visual_mask = torch.zeros(T, dtype=torch.bool)
        for j in range(T):
            path = record.frame_paths[start + j]
            if path is not None:
                images[j] = self._load_frame(path)
                visual_mask[j] = True

        # Text
        text = torch.zeros(T, self.text_dim)
        text_mask = torch.zeros(T, dtype=torch.bool)
        for j in range(T):
            embedding = record.text_steps.get(start + j)
            if embedding is not None:
                text[j] = torch.from_numpy(embedding)
                text_mask[j] = True

        # Temporal (daily weather, repeated per step)
        temporal = torch.from_numpy(record.weather).unsqueeze(0).repeat(T, 1)

        item = {
            "images": images,
            "text": text,
            "temporal": temporal,
            "visual_mask": visual_mask,
            "text_mask": text_mask,
            "camera_id": record.camera_id,
            "day": record.day.isoformat(),
            "session": record.session,
            "regime": record.regime,
            "start_step": start,
        }

        if self.target_fn is not None:
            item["target"] = self.target_fn(record, start)

        return item