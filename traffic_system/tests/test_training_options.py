"""
The training options added for v9: post duration, window augmentation, incident loss weights,
ViT patch initialisation, non-visual calibration, and the seed summary. Behaviour only -- whether
they improve the model is decided by training runs compared on validation over seeds.
"""

import json
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.alignment import ALERT_ACTIVE_MINUTES, LEGACY_TEXT_ACTIVE_MINUTES, build_sessions
from src.data.augment import augment_window
from src.models.traffic_risk_model_edge import edge_risk_loss, incident_edge_weights
from src.vision.patch_embedder import (
    IMAGENET_MEAN,
    PatchEmbedder,
    load_vit_b16_patch_weights,
    patch_feature_stats,
)

DAY = date(2026, 5, 25)


def sessions_with_one_post(posted_at: datetime, active: int):
    frames = pd.DataFrame({"camera_id": ["cam"], "timestamp": [pd.Timestamp("2026-05-25 07:05")], "frame_path": ["x.jpg"]})
    tweets = pd.DataFrame({"camera_id": ["cam"], "created_at": [pd.Timestamp(posted_at)],
                           "embedding": [np.ones(4, dtype=np.float32)]})
    weather = pd.DataFrame({"date": [DAY], "a": [1.0]})
    records, _, _ = build_sessions(frames, tweets, weather, ["cam"], ["a"], nonvisual_start=DAY, buffer_start=DAY,
                                   visual_split={(DAY, "AM"): "train"}, text_active_minutes=active)
    return records[0].text_steps


def test_legacy_posts_count_only_in_the_minute_they_were_made():
    steps = sessions_with_one_post(datetime(2026, 5, 25, 7, 5), LEGACY_TEXT_ACTIVE_MINUTES)
    assert sorted(steps) == [5]


def test_alerts_stay_current_for_their_documented_duration():
    steps = sessions_with_one_post(datetime(2026, 5, 25, 7, 5), ALERT_ACTIVE_MINUTES)
    assert sorted(steps) == list(range(5, 5 + ALERT_ACTIVE_MINUTES))


def test_an_alert_posted_before_the_session_still_counts_at_its_start():
    steps = sessions_with_one_post(datetime(2026, 5, 25, 6, 40), ALERT_ACTIVE_MINUTES)   # 20 min before 07:00
    assert sorted(steps) == list(range(0, ALERT_ACTIVE_MINUTES - 20))
    assert sessions_with_one_post(datetime(2026, 5, 25, 5, 50), ALERT_ACTIVE_MINUTES) == {}  # expired before 07:00


def test_augmentation_changes_only_real_frames_and_the_same_way_for_the_whole_window():
    torch.manual_seed(0)
    images = torch.rand(6, 3, 32, 32) * 0.6 + 0.2
    mask = torch.tensor([True, True, False, True, True, False])
    images[~mask] = 0.0
    out = augment_window(images, mask, torch.Generator().manual_seed(1))
    assert out.shape == images.shape and (out >= 0).all() and (out <= 1).all()
    assert torch.equal(out[~mask], images[~mask])                     # missing frames untouched
    assert not torch.allclose(out[mask], images[mask])
    same = torch.stack([images[0]] * 4)                               # identical frames ...
    aug = augment_window(same, torch.ones(4, dtype=torch.bool), torch.Generator().manual_seed(2))
    assert torch.allclose(aug[0], aug[3])                             # ... stay identical: one draw per window


def test_augmentation_of_a_window_without_frames_is_a_no_op():
    images = torch.zeros(3, 3, 16, 16)
    assert torch.equal(augment_window(images, torch.zeros(3, dtype=torch.bool)), images)


def chain_a_hat(n):
    a = torch.eye(n)
    for i in range(n - 1):
        a[i, i + 1] = a[i + 1, i] = 1.0
    return a


def test_incident_weights_cover_camera_edges_within_two_hops_of_a_live_incident():
    n, impact_col = 6, 0
    edge_index = torch.tensor([[0, 4], [1, 5]])                      # camera edges 0-1 and 4-5
    edge_ids = torch.tensor([0, 1])
    context = torch.zeros(2, 3, n, 1)
    context[0, 1, 3, impact_col] = 0.5                                # window 0: incident at node 3, minute 1
    w = incident_edge_weights(context, chain_a_hat(n), edge_index, edge_ids, factor=4.0, impact_column=impact_col)
    assert w.shape == (2, 2)
    assert w[0].tolist() == [4.0, 4.0]                                # node 3 is 2 hops from node 1 and from node 5
    assert w[1].tolist() == [1.0, 1.0]                                # window 1: no incident
    far = incident_edge_weights(context, chain_a_hat(n), edge_index, edge_ids, 4.0, impact_col, hops=0)
    assert far[0].tolist() == [1.0, 1.0]                              # not touching either edge


def test_extra_weights_scale_the_loss_and_default_to_unweighted():
    logits = torch.zeros(1, 2)
    targets = torch.tensor([[1.0, 1.0]])
    plain, _ = edge_risk_loss(logits, torch.tensor([0, 1]), targets)
    weighted, _ = edge_risk_loss(logits, torch.tensor([0, 1]), targets, extra_weights=torch.tensor([[3.0, 1.0]]))
    assert weighted.item() == pytest.approx(plain.item() * 2.0)      # (3 + 1) / 2


def test_vit_patch_weights_load_with_their_input_standardisation():
    emb = PatchEmbedder()
    fake = (torch.randn(768, 3, 16, 16), torch.randn(768), torch.randn(1, 196, 768))
    info = load_vit_b16_patch_weights(emb, weights=fake)
    assert torch.equal(emb.proj.weight, fake[0]) and torch.equal(emb.pos_embed, fake[2])
    assert emb.normalize_input and torch.allclose(emb.input_mean.flatten(), torch.tensor(IMAGENET_MEAN))
    assert info["layers"] == ["proj", "pos_embed"]
    with pytest.raises(ValueError):
        load_vit_b16_patch_weights(PatchEmbedder(), weights=(torch.randn(10, 3, 16, 16), fake[1], fake[2]))


def test_old_checkpoints_still_load_because_the_statistics_are_not_saved():
    old_state = PatchEmbedder().state_dict()
    assert "input_mean" not in old_state and "input_std" not in old_state
    PatchEmbedder().load_state_dict(old_state)                        # strict load works


def test_patch_feature_stats_reports_the_compatibility_numbers():
    stats = patch_feature_stats(PatchEmbedder(img_size=32), torch.rand(4, 3, 32, 32))
    assert set(stats) == {"token_mean", "token_sd", "near_dead_channels", "between_frame_cosine"}


def test_nonvisual_calibration_is_applied_only_when_asked_and_never_silently():
    from src.models.nonvisual_risk import FEATURES, NonVisualRiskModel

    x = np.random.default_rng(0).random((300, len(FEATURES))).astype(np.float32)
    y = np.random.default_rng(1).random(300)
    model = NonVisualRiskModel.fit(x, y, {"split": "synthetic"})
    with pytest.raises(ValueError):
        model.predict(x[:5], calibrated=True)                         # no calibration yet
    model.calibration = {"slope": 0.5, "intercept": 0.2, "n_samples": 300, "fitted_on": "test"}
    assert np.allclose(model.predict(x[:5], calibrated=True), np.clip(0.5 * model.predict(x[:5]) + 0.2, 0, 1))


def test_seed_summary_groups_runs_by_options(tmp_path, capsys):
    import subprocess
    import sys

    for seed, loss in enumerate([0.50, 0.52, 0.54]):
        (tmp_path / f"v9_s{seed}.result.json").write_text(json.dumps({
            "seed": seed, "best_epoch": 3, "stopped_early": True,
            "val": {"loss": loss, "mae": 0.25}, "test": {"loss": loss + 0.05, "mae": 0.30},
            "options": {"visual_init": "random", "augment": True},
        }), encoding="utf-8")
    out = tmp_path / "summary.json"
    subprocess.run([sys.executable, "scripts/summarize_runs.py", str(tmp_path / "*.result.json"), "--out", str(out)],
                   check=True, capture_output=True)
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert len(summary) == 1 and summary[0]["seeds"] == [0, 1, 2] and summary[0]["enough_seeds"]
    assert summary[0]["val_loss"][0] == pytest.approx(0.52) and summary[0]["val_loss"][1] == pytest.approx(0.02)
