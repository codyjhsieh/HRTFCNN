# HRTFCNN vs. State of the Art (2023–2025)

A grounded comparison of this project against current HRTF‑personalization research,
written to answer one practical question:

> **Can we keep the phone‑only capture approach (ear photo + light anthropometry, no
> anechoic chamber, no 3D scanner) but modernize the model and the training data?**

Short answer: **yes**, and the gap is almost entirely in the *model* and the *dataset* —
not the capture method. The original capture instinct (photo from a phone) is the same
thing Sony ships in production.

---

## 1. What this project does (baseline)

A small multi‑input network:

- **Anthropometry branch:** MLP (3×64) on 17 measurements + (elevation, azimuth) → 8‑dim.
- **Image branch:** 2‑layer / 16‑filter CNN on a 64×64 Canny‑edge ear photo → 8‑dim.
- **Fusion:** concat → 3×Dense(64) → **Dense(200) predicting the raw time‑domain HRIR**
  (ReLU output) per direction.
- **Data:** CIPIC — 45 subjects, ~32 usable after dropping missing anthropometry/images.
- **Metric:** a single "RMSLE" number; Keras + TensorFlow 1.x on a Colab TPU.
- **Source paper:** Lee & Kim 2018, *Applied Sciences* 8(11):2180.

### Two corrections worth noting up front

1. **The metric label is wrong.** The source paper reports **RMSE = −18.40 dB and
   LSD = 4.47 dB**, not "RMSLE." A negative dB figure just means small error on a dB
   scale (more negative = better). The README's "−24.285 RMLSE" is a dB‑scale RMSE under
   a non‑standard definition and is **not comparable** to published numbers.
2. **Predicting the raw 200‑tap HRIR with a ReLU output is the weakest modeling choice.**
   The field universally models the **log‑magnitude spectrum** (+ ITD separately,
   recombined via minimum phase), because that is where the perceptual cues live and what
   every standard metric scores.

---

## 2. The data gap — the single biggest lever

| Dataset            | Subjects        | Geometry modality                                   |
|--------------------|-----------------|-----------------------------------------------------|
| **CIPIC** (this)   | 45 (~32 usable) | ear photos, 37 anthropometric measurements          |
| HUTUBS             | 96              | 3D head meshes + anthropometry                      |
| 3D3A (Princeton)   | 38 (31 scanned) | 3D head/torso scans                                 |
| ARI                | 250+ (aggregate)| mostly acoustic                                     |
| **SONICOM**        | **200 → 300 (2025)** | **3D ear/head/torso scans + depth photos + anthropometry (93) + synthesized HRTFs** |

**SONICOM is ~6× more usable subjects than CIPIC and — critically — ships depth
photographs and per‑ear anthropometry**, i.e. almost exactly this project's input
modality, at scale. The README names "only 32 subjects, image‑reliant" as the fatal
weakness; SONICOM is the direct fix.

**Caveat:** you cannot naively pool datasets. Different measurement rigs leave consistent
database‑specific signatures, and direction grids don't align (CIPIC and HUTUBS can't be
jointly trained on a shared grid without harmonization). Use dataset‑aware normalization
or a neural‑field representation that absorbs different grids. Everything rides on
**SOFA / AES69‑2022** as the common file format.

---

## 3. Where the SOTA went — and which parts apply here

The dominant 2023–2025 paradigm is **neural fields / implicit representations**: the HRTF
is a continuous function of direction via a coordinate‑MLP.

- **HRTF Field** (ICASSP 2023) — foundational; unifies datasets with different grids.
- **NIIRF** (ICASSP 2024, MERL) — predicts cascaded **IIR** filter coefficients; LoRA
  personalization.
- **RANF** (ICASSP 2025, MERL) — retrieval‑augmented neural field; **won LAP'24 Task 2**.
- **HRTFformer** (2025) — transformer with a spatial‑continuity loss.

### Important framing for our goal

Most of that headline work — and the **LAP Challenge 2024** itself (an IEEE‑SPS / SONICOM
data challenge hosted at **EUSIPCO 2024**, *not* an ICASSP grand challenge) — targets
**sparse‑measurement upsampling**: reconstruct a dense HRTF from 3/5/19/100 *acoustically
measured* directions. **That assumes a measurement rig we explicitly don't want.** It is
not our task. But its *decoder and representation* (continuous neural field over
direction, magnitude + separate ITD) is exactly the backbone to adopt downstream of a
photo encoder.

The branch that **is** our task — input = photo / anthropometry, no measurements:

- **Autoencoder / latent + anthropometry regression** (Chen/Kuo ICASSP 2019 lineage;
  2023 FCN; 2025 latent representation).
- **Diffusion from anthropometry** (2025) — DDPM conditioned on measurements → HRIR; a
  score‑based variant is noted as small enough for **on‑device** use.
- **Mobile image encoders** — replace the 2‑conv CNN with MobileNetV3 / EfficientNet‑Lite
  / ViT‑tiny, quantize/distill for on‑device inference.

**Productized reality** confirms phone capture is viable: **Apple** (TrueDepth ear/face
scan — iPhone‑only), **Sony** 360RA (ear *photo* — any phone), **Genelec Aural ID** (phone
video → photogrammetry). The capture instinct here is right; the model and data are stale.

---

## 4. Evaluation — the weakest area vs. SOTA

Current standard = **three objective metrics + an auditory model + (ideally) a listening
test**:

- **LSD** (log‑spectral distortion, dB): RMS of `20·log10(|H| / |Ĥ|)` over frequency,
  averaged across directions and ears. Good systems land **~3.5–5.5 dB**.
- **ITD error** (µs) and **ILD error** (dB): the lateralization cues, scored separately.
- **Baumgartner sagittal‑plane model** (Auditory Modeling Toolbox) → **polar error (°)**
  and **quadrant‑error rate (%)** for front‑back / up‑down confusions.
- Recurring 2024–2025 warning: **low LSD ≠ better perception** — the decisive cues sit in
  a narrow spectral band, so a model can win on LSD and still localize wrong.

The single "RMSLE" number is two generations behind; on ~30 subjects with leave‑one‑out
CV, fold variance is high and overfitting near‑certain.

---

## 5. On train vs. test inputs, and depth

- The model's **input features must match at train and test** (fixed forward pass).
- Most of SONICOM's richness is the **label** (the HRTF), needed only at training — that
  is ordinary supervised learning, not a mismatch.
- **Depth:** SONICOM *contains* depth photos and 3D scans, but the `model_v2` pipeline
  here **does not consume them**. They are ignored so the deployed input stays
  phone‑capturable (2D photo). Depth's only role in the broader field is upstream:
  geometry → BEM simulation (**Mesh2HRTF**) → HRTF *labels* (how SONICOM synthesized HRTFs
  for 200 subjects). That is the dataset author's pipeline, not ours.
- Using train‑only extra modalities *correctly* (if ever desired): **privileged
  information (LUPI)**, **cross‑modal distillation (teacher→student)**, or **auxiliary
  multi‑task heads** — all keep the deployed input identical to the phone's.
- Optional anthropometry is handled with **modality dropout**: randomly zero the
  anthropometry branch during training so one model works with or without it.

---

## 6. Verdict & modernization plan

**Where this project sits:** a faithful 2018 re‑implementation — sound concept and the
right *product* instinct (phone‑only capture), but small‑data, raw‑HRIR output, toy CNN,
single non‑standard metric. Roughly two architectural generations and ~6× the data behind
2025 SOTA. The gap is in the **model and dataset, not the capture method**.

**Keep‑it‑mobile roadmap (highest leverage first):**

1. **Dataset → SONICOM** (200–300 subjects; depth photos + anthropometry). Optionally
   merge HUTUBS/3D3A with SOFA harmonization.
2. **Output representation → log‑magnitude spectrum + ITD** (recombine via minimum phase).
   Drop ReLU‑on‑raw‑HRIR.
3. **Modern edge encoder** → MobileNetV3 / EfficientNet‑Lite / ViT‑tiny on the *real* ear
   photo (not just Canny edges); keep the anthropometry MLP; fuse.
4. **Decode through a neural field** over (az, el) instead of a fixed Dense(200) — gives
   continuous, any‑direction output and dataset‑grid flexibility.
5. **Proper evaluation** → LSD + ITD + ILD (+ Baumgartner polar/quadrant error) via the
   SONICOM Spatial Audio Metrics toolbox; leave‑one‑subject‑out CV with reported variance.
6. **Port off TF1** → PyTorch (this scaffold) or current Keras 3; add a CPU dry‑run path;
   train the heavy run on Colab/cloud GPU.
7. **(Stretch, on‑device)** a small score‑based / diffusion decoder for on‑device
   personalization.

The `model_v2/` directory in this repo is a starting scaffold implementing items 2–6 with
a depth‑free, photo‑first design.

---

## Selected sources

- Lee & Kim 2018, *Applied Sciences* 8(11):2180 — https://www.mdpi.com/2076-3417/8/11/2180
- SONICOM dataset — https://www.researchgate.net/publication/370650962_The_SONICOM_HRTF_Dataset
- Extended SONICOM + SAM toolbox (2025) — https://arxiv.org/abs/2507.05053
- HRTF Field (ICASSP 2023) — https://arxiv.org/abs/2210.15196
- NIIRF (ICASSP 2024) — https://arxiv.org/abs/2402.17907
- RANF (ICASSP 2025, LAP'24 Task 2 winner) — https://arxiv.org/abs/2501.13017
- HRTFformer (2025) — https://arxiv.org/abs/2510.01891
- Diffusion HRTF personalization (2025) — https://arxiv.org/abs/2501.02871
- LAP Challenge 2024 — https://www.sonicom.eu/lap-challenge/ ·
  https://github.com/Audio-Experience-Design/LAPChallenge
- Cross‑database normalization — https://arxiv.org/pdf/2307.14547
- Baumgartner localization model (AMT) — https://amtoolbox.org/amt-0.9.8/doc/models/baumgartner2014.php
- SOFA / AES69 — https://www.sofaconventions.org/mediawiki/index.php/SOFA_(Spatially_Oriented_Format_for_Acoustics)

> Citations were gathered via web search; a few exact figures (per‑paper LSD tables,
> SONICOM's first‑release count of 200 vs 120) carry minor uncertainty and are flagged in
> the project notes. The load‑bearing facts (SONICOM scale, the neural‑field shift, the
> LAP'24 task structure, the RMSE‑vs‑RMSLE correction) were each cross‑checked across
> multiple sources.
