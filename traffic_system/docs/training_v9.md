# v9 training: what changed and how to run it

This covers four improvements to the CNN+LSTM → RADR-STGNN → MLP decoder model, chosen because
they need no new data collection and keep the thesis architecture and routing formula
(W = d × (1 + λ·Risk)) unchanged. Each is a **training option**: off by default, recorded in the
checkpoint, and judged by validation results over three seeds, never by assumption.

| Option | Flag | What it does | Default |
|---|---|---|---|
| Pretrained video weights | `--visual-init vit_b16` | Loads the patch projection and position embeddings from torchvision's ImageNet ViT-B/16 into the encoder's first layer, and standardises frames the way those weights expect | `random` (as v8) |
| Image augmentation | `--augment` | Small random brightness, contrast, saturation and crop, the same for all 30 frames of a camera window; training split only; no flips (they swap carriageways) | off |
| Alert duration | `--text-active-minutes 60` | A post stays "current" for 60 minutes after it is posted, instead of only in its own minute; matches the 60-minute incident decay used elsewhere | **60** for new runs (v8 used 1) |
| Incident weighting | `--incident-weight 3` | Loss weight 3 for camera segments with a live incident at, or within 2 road hops of, either end; validation and test loss stay unweighted | 1 (off) |
| Early stopping | `--patience 3` | Stop after 3 epochs without a lower validation loss; the best-validation epoch is kept | off |

Calibration of both risk models is separate (see the last section).

## Why each is needed

- **Video:** the encoder started from random weights. In v8, removing all camera frames changed
  the risk by only 0.022 on average (`stgnn_edge_v8.sources.json`). Pretrained first-layer
  filters are generic colour and edge detectors and usually transfer to new images. **But
  matching tensor shapes does not prove the weights suit this encoder.** In ViT they feed a
  transformer; here a CNN, trained from scratch, reads them as a 14 × 14 map. So training
  prints patch-token statistics on real training frames, random vs ViT (near-dead channels,
  how alike different frames look), and the decision is made on validation results.
- **Augmentation:** there was none. With 9 cameras over 7 training days, the model can memorise
  lighting and framing.
- **Alert duration:** a post reached the model only in the minute it was created, so an alert
  posted 10 minutes before a window was invisible to the text path.
- **Incident weighting:** only about 6% of labelled camera windows have an incident nearby (78
  training windows), so an unweighted loss barely notices them.
- **Early stopping and seeds:** v8's validation loss rose after epoch 2 while MAE kept falling.
  A single run cannot separate a real improvement from seed noise.

## Run it (Colab, GPU)

Same data arguments as v8: frames roots on Drive, the split file, and context on. Run every
configuration with seeds 0, 1 and 2, all writing to the same folder:

```bash
CK=/content/drive/MyDrive/MMDA_CHECKPOINTS
COMMON="--frames-root /content/drive/MyDrive/MMDA_FRAMES /content/drive/MyDrive/MMDA_FRAMES_B \
        --split-file $CK/split_70_15_15_by_date.json --epochs 15 --patience 3 --text-active-minutes 60"

for SEED in 0 1 2; do
  # A. reference: v8 settings + alert duration + early stopping
  python scripts/train_stgnn_edge.py $COMMON --seed $SEED --out $CK/v9A_s$SEED.pt
  # B. + augmentation
  python scripts/train_stgnn_edge.py $COMMON --augment --seed $SEED --out $CK/v9B_s$SEED.pt
  # C. + augmentation + ViT patch weights
  python scripts/train_stgnn_edge.py $COMMON --augment --visual-init vit_b16 --seed $SEED --out $CK/v9C_s$SEED.pt
  # D. C + incident weighting
  python scripts/train_stgnn_edge.py $COMMON --augment --visual-init vit_b16 --incident-weight 3 \
         --seed $SEED --out $CK/v9D_s$SEED.pt
done

python scripts/summarize_runs.py "$CK/v9*_s*.result.json" --out $CK/v9_summary.json
```

That is 12 runs. If time is short, run A and C first (6 runs), then D only if C helps.

## How to choose, fairly

1. `summarize_runs.py` lists each configuration's **validation** loss and MAE as mean ± SD over
   its seeds, ordered by validation loss. **Choose on validation.** Test is printed but must not
   drive the choice.
2. A configuration counts as better only if its mean validation loss is lower **by more than the
   seed SD**. Otherwise call it "no measurable difference" and keep the simpler one.
3. For the chosen configuration, run the source check on one of its seeds and compare with v8.
   Did camera video and incidents start to move the risk?

   ```bash
   python scripts/verify_crs_sources.py --checkpoint $CK/v9C_s0.pt --frames-root ...
   ```

4. Score the chosen checkpoint (`predict_congestion_risk.py`), install its `risk_edges.csv`,
   re-run `scripts/calibrate_risk_edges.py --apply` and `scripts/train_nonvisual_risk.py`, and
   only then decide whether the routing evaluation needs re-timing. Re-time only routes that
   changed.

## Calibration of both risk models

Both models are calibrated onto the Light/Medium/Heavy label scale **without using held-out
data**:

- **Camera model:** `scripts/calibrate_risk_edges.py --apply` fits one affine map on the train
  split only and adds `risk_calibrated` beside `risk`. For v8:
  `clip(1.225 × risk − 0.144)`. Test MAE 0.318 → 0.311.
- **Non-visual model:** `scripts/train_nonvisual_risk.py` fits it **out of fold**. Each
  train+val row is predicted by a model trained without its camera, including copies with
  incidents withheld, because most roads off the cameras have unknown incident status. Result:
  `clip(0.675 × risk + 0.240)`. Stored in the model file.

**Are they comparable afterwards?** `scripts/verify_risk_calibration.py`, written to
`outputs/nonvisual_risk/calibration_check.json`, compares the average risk of non-visual roads
with camera-model roads (predicted + borrowed) on the routing graph:

| Window (May 25) | Gap, raw | Gap, calibrated | Extra cost per metre on non-visual roads |
|---|---|---|---|
| 07:00 | +0.054 | +0.048 | 4.6% → 4.1% |
| 07:30 | +0.081 | +0.065 | 6.9% → 5.4% |
| 17:30 | +0.092 | +0.072 | 7.8% → 6.1% |
| 18:20 | +0.097 | +0.076 | 8.2% → 6.4% |

- **Calibration narrows the gap by about 15–25% but does not close it.** The rest comes from the
  non-visual model meeting roads unlike the cameras (other weather cells, unknown incident
  status). A scale fitted on camera data cannot correct that.
- A smaller gap means the layers are on a more similar scale. It does not show that either is
  more accurate, since the roads may truly differ.
- On the test-day camera segments, where both models exist, the calibrated models score alike:
  MAE 0.311 (camera) and 0.317 (non-visual).

**Routing:**
- Routing uses **raw** scores unless `RISK_SCORES=calibrated` is set for the server. The Apple
  Maps evaluation was run on raw scores.
- Calibration preserves each model's ranking of roads, but **not necessarily routes**, because
  W = d × (1 + λ·Risk) weighs risk against distance. Switching to calibrated scores means
  re-running the routing evaluation for any routes that change.

## For the thesis

- **Methodology, training section:** pretrained patch initialisation (ImageNet ViT-B/16 patch
  projection, with its input standardisation), window-consistent augmentation, a 60-minute post
  duration, incident-weighted loss, validation-based early stopping, and three seeds reported as
  mean ± SD. None of these changes the architecture or the routing formula.
- **Results:** report the chosen configuration against v8 on the same baselines, as mean ± SD,
  and the source-sensitivity comparison. Report "no measurable difference" where that is what
  the seeds show.
- **Calibration:** describe both fits, the data each was fitted on (train-only; out-of-fold),
  and the comparability table above, including that the gap remains.
