"""
Modest image augmentation for CCTV training windows.

One set of random parameters per camera window, applied to every real frame of it: the LSTM
reads the 30 frames as a sequence, so jittering each frame independently would add motion
that is not in the scene. Training split only; validation and test frames are never touched.

What changes, and by how much (all small, so a frame still shows the same traffic):
    brightness  x (1 +- BRIGHTNESS)        lighting, time of day, camera exposure
    contrast    (x - mean) x (1 +- CONTRAST) + mean
    saturation  blend with grey by (1 +- SATURATION)
    crop        keep a random CROP_MIN..1 share of each side, resized back (slight zoom / shift)

No horizontal flip: a flipped frame swaps the carriageways, and with them which direction is
congested.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

BRIGHTNESS = 0.15
CONTRAST = 0.15
SATURATION = 0.10
CROP_MIN = 0.90


def augment_window(images: torch.Tensor, visual_mask: torch.Tensor,
                   generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """
    images: [T, 3, H, W] in [0, 1]; visual_mask: [T] True where a real frame exists.
    Returns a new tensor; frames without a real image (zeros) are left as they are.
    """
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must be [T, 3, H, W]")
    out = images.clone()
    real = visual_mask.bool()
    if not real.any():
        return out

    def uniform(low: float, high: float) -> float:
        return float(torch.empty(1).uniform_(low, high, generator=generator))

    brightness = uniform(1 - BRIGHTNESS, 1 + BRIGHTNESS)
    contrast = uniform(1 - CONTRAST, 1 + CONTRAST)
    saturation = uniform(1 - SATURATION, 1 + SATURATION)
    keep = uniform(CROP_MIN, 1.0)
    _, _, h, w = images.shape
    ch, cw = max(1, int(round(h * keep))), max(1, int(round(w * keep)))
    top = int(torch.randint(0, h - ch + 1, (1,), generator=generator))
    left = int(torch.randint(0, w - cw + 1, (1,), generator=generator))

    frames = out[real]
    frames = frames * brightness
    mean = frames.mean(dim=(1, 2, 3), keepdim=True)
    frames = (frames - mean) * contrast + mean
    grey = (0.299 * frames[:, 0] + 0.587 * frames[:, 1] + 0.114 * frames[:, 2]).unsqueeze(1)
    frames = (frames - grey) * saturation + grey
    if ch != h or cw != w:
        frames = F.interpolate(frames[:, :, top:top + ch, left:left + cw], size=(h, w),
                               mode="bilinear", align_corners=False)
    out[real] = frames.clamp(0.0, 1.0)
    return out
