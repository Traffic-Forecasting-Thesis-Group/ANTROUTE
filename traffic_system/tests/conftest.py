import pytest

import src.vision.frame_extractor as frame_extractor


@pytest.fixture(autouse=True)
def small_test_videos_are_not_empty_fragments(monkeypatch):
    """Synthetic test videos are a few KB; the real 'empty fragment' cutoff is tens of MB."""
    monkeypatch.setattr(frame_extractor, "MIN_CHUNK_BYTES", 100)
