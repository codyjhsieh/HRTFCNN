# HRTFCNN

Convolutional Neural Network to Estimate HRTF for Spatial Audio.

> **Modernized branch (`modernize-stack`).** The original notebook + Keras/TF1 +
> Colab TPU pipeline has been replaced with a single PyTorch / PyTorch Lightning
> training script that fits on a free T4 GPU (e.g. Lightning AI). The legacy
> notebook (`FinalHRTFCNN.ipynb`) is kept for reference but is no longer the
> recommended path.

Colab notebook (legacy, TF1 era):
https://colab.research.google.com/drive/1YjlgEzn3wjde6VCa5mpQx4DTGrymTJgo

## Introduction

With the recent popularity of virtual reality, the concept of spatial audio has
generated a lot more buzz. Truly realistic spatial audio is hard: one of the
primary tools is the **Head-Related Transfer Function (HRTF)**, which describes
how a sound from a given direction propagates around a listener's head and
torso into their ear canals. HRTFs are highly individual; without an accurate
HRTF for *you*, spatial audio collapses. Today the gold-standard way to obtain
one is to sit in an anechoic chamber surrounded by a microphone/speaker array.
This project explores using a small neural network to estimate a personal HRTF
from an ear photo and a handful of anthropometric measurements.

It is a re-implementation, with modifications, of *"Personalized HRTF Modeling
Based on Deep Neural Network Using Anthropometric Measurements and Images of
the Ear"* (Lee et al., 2018, https://www.mdpi.com/2076-3417/8/11/2180/pdf).

## Repository layout

```
train.py                       PyTorch Lightning training script
predict.py                     Inference: checkpoint + (photo, anthro) -> personal SOFA file
auto_anthro.py                 CLI for estimating 17 anthropometric measurements from 2 iPhone photos
utils/hrtf.py                  CipicHRTF class + SOFA I/O via the `sofar` library
utils/dsp.py                   Log-magnitude conversion + minimum-phase IR reconstruction
utils/image_utils.py           Legacy Canny helpers used by the anthropometric notebook
AnthropomorphicFeatures.ipynb  Legacy interactive notebook (use auto_anthro.py instead)
FinalHRTFCNN.ipynb             Legacy Keras/TF1/TPU training notebook (kept for reference)
tests/test_components.py       Component tests (24 tests, all CPU, < 70 s)
data/template.sofa             SOFA template used as the schema for predicted-HRTF output
requirements.txt
```

## Data

The training pipeline expects three pieces of CIPIC data, all freely available:

| What | Where to put it |
|---|---|
| CIPIC SOFA files (one per subject) | `data/cipic_hrtf_sofa/subject_XXX.sofa` |
| Anthropometric measurements (`anthro.mat`) | `data/CIPIC_hrtf_database/anthropometry/anthro.mat` |
| Ear photos | `data/ear_photos/Subject_XXX/<photo>.jpg` |

Sources:

- **SOFA files**: `wget -r -l1 -np http://sofacoustics.org/data/database/cipic/`
  then move the `*.sofa` files into `data/cipic_hrtf_sofa/`.
- **Anthropometry + ear photos**: download `CIPIC_hrtf_database.zip` from
  https://www.ece.ucdavis.edu/cipic/spatial-sound/hrtf-data/ and arrange as above.

CIPIC has 45 subjects nominally; in practice ~32 have a complete set of
HRTFs + anthropometry + usable ear photo.

### SOFA

[SOFA](https://www.sofaconventions.org/) (Spatially Oriented Format for
Acoustics) is the AES-standardized container for HRTFs. The
`utils/hrtf.py` module wraps the
[`sofar`](https://pypi.org/project/sofar/) library and exposes the original
`CipicHRTF`, `get_hrtf_sofa`, `get_hrtf_mat`, and `create_cipic_hrtf` API used
by the rest of the codebase.

## Quickstart — training on a free T4 (Lightning AI / Colab / local CUDA)

```bash
git clone https://github.com/codyjhsieh/HRTFCNN.git
cd HRTFCNN
git checkout modernize-stack

pip install -r requirements.txt
# arrange CIPIC data under ./data/ as described above

python train.py --data-dir ./data --epochs 50 --batch-size 64
```

Outputs go to `./checkpoints/`. The script auto-selects GPU + fp16 mixed
precision when CUDA is available and falls back to CPU + fp32 otherwise.
Lightning logs both MSE and **log-spectral distance (dB)** — the latter is the
standard HRTF quality metric.

Useful flags:

- `--backbone resnet18` (default; ~11M params, fits any T4) — also try
  `resnet50`, `convnext_tiny`, `efficientnet_b0`, etc. via `timm`.
- `--batch-size`, `--epochs`, `--lr`, `--val-frac`, `--num-workers`, `--seed`.

The train/val split is **per-subject** so that no subject's ear appears on
both sides of the split.

## Model

The architecture mirrors the spirit of the original paper but uses modern
components:

- **Image branch.** The ear photo is passed through a frozen ImageNet-pretrained
  backbone (`timm`, default `resnet18`) and a small projection head. This
  replaces the original Canny-edge + small-CNN approach: with only ~32 usable
  subjects, training a vision backbone from scratch is hopeless; a frozen
  pretrained backbone gives strong pinna features for free.
- **Tabular branch.** Anthropometric measurements + (azimuth, elevation) are
  concatenated and passed through a small MLP.
- **Regressor.** The image and tabular features are concatenated and decoded
  into a **log-magnitude spectrum** (129 frequency bins) by an MLP.

Unlike the original paper, which trained one network per (azimuth, elevation)
pair, a single network is conditioned on direction and predicts the spectrum
for any direction.

**Why log-magnitude and not a raw 200-tap IR?** The legacy notebook regressed
directly on the time-domain impulse response with an MSE loss. That target is
poorly conditioned (every tap is weighted equally, dominated by the noisy
tail) and the loss does not match how HRTFs are evaluated. Predicting the
log-magnitude spectrum lets the training loss be **log-spectral distance** —
the standard HRTF metric — and reduces the regression target to 129
perceptually-meaningful dB values instead of 200 raw samples. At inference
time the predicted log-magnitude is converted back to a time-domain IR via
**minimum-phase reconstruction** (real-cepstrum folding in `utils/dsp.py`).
This is the standard HRTF personalization recipe; phase is hard to predict
and is perceptually dominated by interaural time differences anyway.

## Measurement estimation (iPhone-only)

Taking the 17 CIPIC anthropometric measurements on a real head is tedious
(tape measures, angle gauges). `auto_anthro.py` estimates them from two
iPhone photos:

```bash
python auto_anthro.py --front front.jpg --side side.jpg --output anthro.txt
```

You hold an 8.5×11 in. page in each photo as a known-scale reference. The
script auto-detects the page corners via OpenCV contour finding, applies a
4-point perspective transform to rectify the page to a known
pixels-per-inch scale, then prompts you to click anatomical landmarks. The
17 measurements are written as comma-separated cm values that drop straight
into `predict.py --anthro anthro.txt`.

The page-detection step used to require four manual clicks per photo in
`AnthropomorphicFeatures.ipynb`; that notebook is kept for reference but
`auto_anthro.py` is the recommended path — it has no notebook /
`ipywidgets` dependency and runs from any plain Python environment.

**Landmark clicks are still manual.** Fully-automatic pinna-landmark
detection is a research problem (off-the-shelf face landmark models like
MediaPipe FaceMesh cover the head but not the pinna). The measurement
accuracy is approximate (a couple of cm) — the goal is to avoid the
chamber, not to match it.

## Inference: write a personal SOFA file

```bash
python predict.py \
    --checkpoint checkpoints/hrtf-epoch49.ckpt \
    --ear-photo my_ear.jpg \
    --anthro anthro.txt \
    --template data/template.sofa \
    --output my_hrtf.sofa
```

`predict.py` loads the checkpoint, predicts a log-magnitude spectrum for
each of the 1250 CIPIC directions, reconstructs a minimum-phase impulse
response per direction, and writes a CIPIC-format `.sofa` file that drops
straight into tools like **IRCAM SPAT** (`spat5.binaural` in Max/MSP) for
real-time binaural rendering.

The model is per-ear (one image, one IR), so for a true stereo HRTF run
the script twice — once with the left-ear photo, once with the right-ear
photo — and merge. For a first pass, the current script duplicates the
single-ear prediction across both channels.

The full **iPhone-only inference pipeline** is:

```
iPhone front + side photos ─► auto_anthro.py ─► 17 anthropometric values ─┐
                                                                            ├─► predict.py ─► personal SOFA
iPhone ear photo ─────────────────────────────────────────────────────────┘
```

## Tests

```bash
python -m pytest tests/test_components.py -v
```

24 tests, all CPU, runtime ~1 minute. Coverage:

- **`utils/hrtf.py`**: coordinate-conversion round-trip; CipicHRTF loads
  `data/template.sofa`; `create_cipic_hrtf` write+reload round-trip
  preserves IR content and source positions.
- **`utils/dsp.py`**: numpy and torch log-magnitude implementations agree;
  minimum-phase reconstruction round-trips a smooth log-magnitude target
  within ~1 dB LSD.
- **`train.py`**: LSD = 0 on identical inputs; LSD > 0 on independent
  noise; model forward returns `(B, 129)` finite outputs; backbone has 0
  trainable params and stays in `eval()` after `.train()`; index builder
  filters NaN anthro + missing photos correctly; dataset items have the
  right shapes; one full Lightning training step runs end-to-end on a
  synthetic CIPIC-shaped fixture.
- **`predict.py`**: anthro parser handles inline and file forms; full
  inference produces a reloadable SOFA with shape `(1250, 2, 200)`;
  mismatched anthro dimensions raise `ValueError`.
- **`auto_anthro.py`**: corner-ordering returns TL/TR/BR/BL canonically;
  page detection works on axis-aligned synthetic pages and under random
  perspective warps; rectified page comes out at the canonical
  `(1100, 850)` size and uniformly bright; page detection returns
  `None`/garbage for pure noise; pixel-to-cm scale matches the
  page-based calibration; the CLI runs end-to-end in `--non-interactive`
  mode.

## Results (legacy)

The original Keras/TPU run reported RMLSE = −24.285 dB (vs. dataset-average
HRTF baseline at −19.23 dB, and the Lee et al. paper at −18.40 dB). Numbers
for the modernized PyTorch path will depend on the chosen backbone and
training budget; report log-spectral distance (logged automatically) for
comparison.

## Plans to improve

What's still on the old stack, in priority order:

- **Cross-dataset training.** Train jointly on CIPIC + HUTUBS + SONICOM —
  same modalities (HRTFs + anthropometry + ear photos), all freely
  available, ~6× more subjects. **Deferred** because each dataset has its
  own SOFA conventions, sample rate, IR length, anthropometric definitions,
  and ear-photo format; a clean implementation needs a per-dataset adapter
  layer and dataset on disk to test against, which is meaningfully more
  work than what's currently in this branch. This is the single biggest
  remaining quality lever.
- **Per-ear stereo prediction.** Run inference once per ear and merge into
  a true stereo SOFA file, instead of duplicating the single-ear
  prediction across both channels.
- **Automatic pinna landmark detection.** `auto_anthro.py` auto-detects
  the reference page; the anatomical landmarks are still manually clicked.
  Full automation would use MediaPipe FaceMesh for head landmarks plus a
  trained pinna-keypoint model for ear-specific landmarks.
- **Stronger image backbone.** ResNet18 is the default; `--backbone
  vit_small_patch14_dinov2.lvd142m` (DINOv2-small) gives much stronger
  pinna features and is still T4-friendly when frozen.
- **Both ears at training time.** Each CIPIC subject has two ear photos;
  the current pipeline drops one. Training on mirror-augmented pairs ~2×
  the effective data.
- **Direction encoding.** `(az, el)` is currently fed as two raw floats.
  Fourier / spherical-harmonic positional encoding reliably improves MLP
  regressors for spatial-direction tasks.
- **Multi-resolution STFT loss.** `auraloss` losses on the reconstructed
  IR can complement the log-magnitude objective at multiple FFT sizes.
- **Better evaluation.** Currently logs MSE + LSD. SOTA papers also report
  LSD per frequency band, ITD/ILD error, and perceptual localization-model
  error (Baumgartner 2014).
