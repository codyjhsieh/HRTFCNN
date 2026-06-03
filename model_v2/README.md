# model_v2 — modern, phone-first HRTF personalization

A PyTorch scaffold that modernizes the original `FinalHRTFCNN.ipynb` while keeping the
**phone-only capture** premise: the deployed model's input is a **2D ear photo** plus an
**optional** anthropometry vector. No depth, no 3D scan, no anechoic chamber.

See `../COMPARISON.md` for the full rationale and the state-of-the-art comparison.

## Design at a glance

```
            ear photo (RGB)                anthropometry (optional, dropout)
                  │                                 │
            ImageEncoder                       AnthroEncoder
         (MobileNetV3-small)                      (MLP)
                  │                                 │
                  └──────────────┬──────────────────┘
                                 ▼
                         subject embedding  z   (depth-free, phone-capturable)
                                 │
                                 ▼
                   ┌─────────────────────────────┐
   direction ───►  │  HRTFFieldDecoder (FiLM MLP) │  conditioned on z
     (az, el)      └─────────────────────────────┘
                                 │
                 ┌───────────────┼────────────────┐
                 ▼                                 ▼
       log-magnitude [2 ears, F bins]        ITD (scalar, microseconds)
```

- **Output representation:** log-magnitude spectrum per ear + a separate ITD scalar
  (recombined to an HRIR via minimum phase at inference). This replaces the old
  raw 200-tap HRIR / ReLU output.
- **Decoder is a neural field over direction** (continuous in az/el), so it is not tied to
  one dataset's measurement grid.

## Why no depth

SONICOM *contains* depth photos and 3D scans, but this pipeline deliberately ignores them
so the deployed input stays reproducible on any smartphone. See `COMPARISON.md §5`.

## Layout

| File                         | Purpose                                                      |
|------------------------------|--------------------------------------------------------------|
| `data/sonicom_dataset.py`    | SOFA + photo dataloader for SONICOM (documented assumptions) |
| `models/encoders.py`         | image encoder (MobileNetV3 w/ fallback) + anthropometry MLP  |
| `models/hrtf_net.py`         | fusion + neural-field decoder + magnitude/ITD heads          |
| `losses.py`                  | LSD / ITD / ILD losses and metrics                           |
| `train.py`                   | training loop (leave-one-subject-out friendly)               |
| `evaluate.py`                | LSD / ITD / ILD evaluation                                    |
| `dry_run.py`                 | self-contained smoke test on **synthetic** data (no dataset) |

## Quick start

```bash
pip install -r requirements.txt

# 1. Prove the model is wired correctly (no data needed, CPU only):
python -m model_v2.dry_run

# 2. Train (point --sonicom-root at an unpacked SONICOM release):
python -m model_v2.train --sonicom-root /path/to/SONICOM --epochs 100 --device cuda

# 3. Evaluate a checkpoint:
python -m model_v2.evaluate --sonicom-root /path/to/SONICOM --ckpt runs/best.pt
```

## Status / caveats

- `dry_run.py` runs on synthetic tensors and verifies shapes, a backward pass, and the
  metric code — it needs only `torch`, no dataset and no network access.
- `data/sonicom_dataset.py` encodes **documented assumptions** about SONICOM's on-disk
  layout (SOFA grid, photo filenames). Verify these against your actual SONICOM download
  and adjust the marked `TODO` spots — they could not be validated against real data in
  the environment this scaffold was written in.
- Training requires a GPU; this scaffold was authored in a CPU-only sandbox where `torch`
  could not be installed, so the training/eval paths are **unverified end-to-end**. The
  dry-run is the intended correctness gate — run it first wherever torch is available.
