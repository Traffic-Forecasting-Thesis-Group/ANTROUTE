"""
Windows for the graph model: one 30-step window per (day, session, start) holding, for each
camera node of the road subgraph, the CNN+LSTM inputs of the camera assigned to it.

Several cameras may map to one intersection; the one with the most frames in the window is used.
Nodes with no camera in a window get zero inputs and fully masked timesteps.
"""

from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.alignment import WINDOW_STEPS, SessionRecord
from src.data.training_data import IGNORE_INDEX, LabeledWindowDataset


def last_labelled_step(target: torch.Tensor) -> int:
    """Class of the last labelled step of a window ([T] with IGNORE_INDEX gaps); ignore if none."""
    labelled = target[target != IGNORE_INDEX]
    return int(labelled[-1]) if len(labelled) else IGNORE_INDEX


class GraphWindowDataset(Dataset):
    def __init__(self, records: Sequence[SessionRecord], split: str, lookup: Dict[str, int],
                 node_labels: Sequence[str], camera_to_label: Dict[str, str], image_size: int = 224):
        self.base = LabeledWindowDataset(records, split, lookup, image_size)
        self.node_labels = list(node_labels)
        self.image_size = image_size
        self.weather_dim = len(records[0].weather) if len(records) else 0

        groups: Dict[tuple, Dict[str, tuple]] = {}
        for i, (record_index, start) in enumerate(self.base.index):
            record = records[record_index]
            label = camera_to_label.get(record.camera_id)
            if label not in self.node_labels:
                continue
            frames = sum(p is not None for p in record.frame_paths[start:start + WINDOW_STEPS])
            best = groups.setdefault((record.day, record.session, start), {})
            if label not in best or frames > best[label][1]:
                best[label] = (i, frames)
        self.index: List[Dict[str, int]] = [
            {label: i for label, (i, _) in groups[key].items()} for key in sorted(groups)
        ]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        T, S = WINDOW_STEPS, self.image_size
        images = torch.zeros(len(self.node_labels), T, 3, S, S)
        text = torch.zeros(len(self.node_labels), T, self.base.text_dim)
        temporal = torch.zeros(len(self.node_labels), T, self.weather_dim)
        visual_mask = torch.zeros(len(self.node_labels), T, dtype=torch.bool)
        text_mask = torch.zeros(len(self.node_labels), T, dtype=torch.bool)
        target = torch.full((len(self.node_labels),), IGNORE_INDEX, dtype=torch.long)

        for slot, label in enumerate(self.node_labels):
            if label not in self.index[i]:
                continue
            item = self.base[self.index[i][label]]
            images[slot], text[slot], temporal[slot] = item["images"], item["text"], item["temporal"]
            visual_mask[slot], text_mask[slot] = item["visual_mask"], item["text_mask"]
            target[slot] = last_labelled_step(item["target"])

        return {"images": images, "text": text, "temporal": temporal,
                "visual_mask": visual_mask, "text_mask": text_mask, "target": target}
