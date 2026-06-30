"""End-to-end component tests for HRTFCNN.

Run with: .venv/bin/python -m pytest tests/test_components.py -v
(or: .venv/bin/python tests/test_components.py to run as a script.)
"""

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import scipy.io
import sofar
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.hrtf import (  # noqa: E402
    CipicHRTF,
    create_cipic_hrtf,
    interauralPolarToVerticalPolarCoordinates,
    verticalPolarToInterauralPolarCoordinates,
)
from train import (  # noqa: E402
    CipicHRTFDataset,
    HRTFModel,
    N_FFT,
    N_MAG_BINS,
    N_TAPS,
    build_subject_index,
    make_transforms,
)
from utils.dsp import (  # noqa: E402
    ir_to_log_magnitude_np,
    ir_to_log_magnitude_torch,
    log_magnitude_to_min_phase_ir,
    log_spectral_distance_torch,
)

TEMPLATE_SOFA = ROOT / "data" / "template.sofa"


# ---------- utils/hrtf.py ----------

def test_coord_roundtrip():
    """vertical -> interaural -> vertical should be a near-identity."""
    rng = np.random.default_rng(0)
    az = rng.uniform(-45, 45, size=200)
    el = rng.uniform(-40, 80, size=200)

    el2, az2 = verticalPolarToInterauralPolarCoordinates(el.copy(), az.copy())
    el3, az3 = interauralPolarToVerticalPolarCoordinates(el2, az2)

    assert np.allclose(el3, el, atol=1e-6), "elevation round-trip failed"
    assert np.allclose(az3, az, atol=1e-6), "azimuth round-trip failed"


def test_cipic_hrtf_loads_template_sofa():
    """The sofar-backed loader should read template.sofa with correct shapes."""
    hrtf = CipicHRTF(str(TEMPLATE_SOFA), 44100.0)

    assert hrtf.impulses.shape == (1250, 2, 200)
    assert hrtf.azimuths.shape == (1250,)
    assert hrtf.elevations.shape == (1250,)
    assert hrtf.distances.shape == (1250,)
    assert hrtf.channels == ["left", "right"]
    assert hrtf.samplingRate == 44100.0
    assert np.all(np.isfinite(hrtf.impulses))


def test_create_cipic_hrtf_roundtrip(tmp_path):
    """create_cipic_hrtf should write a SOFA we can reload with matching content."""
    src = CipicHRTF(str(TEMPLATE_SOFA), 44100.0)

    rng = np.random.default_rng(1)
    new_ir = rng.standard_normal(src.impulses.shape).astype(np.float64) * 0.01

    out_path = tmp_path / "predict.sofa"
    create_cipic_hrtf(
        str(TEMPLATE_SOFA), str(out_path),
        new_ir, src.elevations, src.azimuths,
    )
    assert out_path.exists()

    reloaded = CipicHRTF(str(out_path), 44100.0)
    assert reloaded.impulses.shape == new_ir.shape
    # IR content survives the round-trip.
    assert np.allclose(reloaded.impulses, new_ir, atol=1e-9)
    # Positions survive in the interaural-polar frame the loader returns.
    assert np.allclose(reloaded.azimuths, src.azimuths, atol=1e-2)
    assert np.allclose(reloaded.elevations, src.elevations, atol=1e-2)


# ---------- train.py: pure functions ----------

def test_lsd_identity_is_zero():
    rng = np.random.default_rng(2)
    x = torch.from_numpy(rng.standard_normal((8, N_MAG_BINS)).astype(np.float32))
    lsd = log_spectral_distance_torch(x, x.clone())
    assert torch.isfinite(lsd)
    assert lsd.item() < 1e-4, f"LSD on identical inputs should be ~0, got {lsd.item()}"


def test_lsd_differs_is_positive():
    rng = np.random.default_rng(3)
    a = torch.from_numpy(rng.standard_normal((8, N_MAG_BINS)).astype(np.float32))
    b = torch.from_numpy(rng.standard_normal((8, N_MAG_BINS)).astype(np.float32))
    lsd = log_spectral_distance_torch(a, b)
    assert torch.isfinite(lsd)
    assert lsd.item() > 0.5, f"LSD on independent noise should be sizeable, got {lsd.item()}"


def test_ir_to_log_magnitude_consistency():
    """Numpy and torch implementations agree to float precision."""
    rng = np.random.default_rng(4)
    ir = rng.standard_normal((6, N_TAPS)).astype(np.float32) * 0.05
    log_mag_np = ir_to_log_magnitude_np(ir, n_fft=N_FFT)
    log_mag_torch = ir_to_log_magnitude_torch(torch.from_numpy(ir), n_fft=N_FFT).numpy()
    assert log_mag_np.shape == (6, N_MAG_BINS)
    assert np.allclose(log_mag_np, log_mag_torch, atol=1e-4)


def test_min_phase_reconstruction_matches_magnitude():
    """log-mag -> minimum-phase IR -> log-mag should be ~identity."""
    rng = np.random.default_rng(5)
    # Build a smooth magnitude response (avoid log-mag with deep nulls,
    # which behave badly under any minimum-phase reconstruction).
    base = 0.5 + 0.5 * np.cos(np.linspace(0, 3 * np.pi, N_MAG_BINS))
    log_mag = 20.0 * np.log10(base + 1e-3)
    log_mag = np.stack([log_mag, log_mag * 0.5], axis=0)  # batch of 2

    ir = log_magnitude_to_min_phase_ir(log_mag, n_fft=N_FFT, n_taps=N_TAPS)
    assert ir.shape == (2, N_TAPS)
    assert np.all(np.isfinite(ir))

    log_mag_back = ir_to_log_magnitude_np(ir, n_fft=N_FFT)
    # Truncation to N_TAPS introduces some error; ~1 dB is acceptable.
    err = np.sqrt(np.mean((log_mag_back - log_mag) ** 2, axis=-1)).max()
    assert err < 1.0, f"reconstruction LSD too high: {err:.3f} dB"


# ---------- train.py: model ----------

def _build_model(anthro_dim=17):
    return HRTFModel(anthro_dim=anthro_dim, backbone_name="resnet18", lr=1e-3)


def test_model_forward_shape():
    model = _build_model(anthro_dim=17)
    model.eval()
    image = torch.randn(4, 3, 224, 224)
    anthro = torch.randn(4, 17)
    direction = torch.randn(4, 2)
    with torch.no_grad():
        out = model(image, anthro, direction)
    assert out.shape == (4, N_MAG_BINS)
    assert torch.all(torch.isfinite(out))


def test_backbone_is_frozen():
    model = _build_model()
    trainable = [p for p in model.backbone.parameters() if p.requires_grad]
    assert trainable == [], "backbone should have no trainable params"
    head_trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    assert head_trainable > 0, "heads should still have trainable params"


def test_backbone_stays_eval_after_train():
    """train() should not flip the frozen backbone back into train mode."""
    model = _build_model()
    model.train()
    assert not model.backbone.training, "backbone must stay in eval mode"


# ---------- train.py: dataset + index ----------

def _synth_jpeg(path: Path, size=256):
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, "JPEG")


def _make_synthetic_cipic(tmp_path: Path, n_subjects: int = 3) -> Path:
    """Build a tiny CIPIC-shaped directory tree the indexer can consume."""
    data_dir = tmp_path / "data"
    sofa_dir = data_dir / "cipic_hrtf_sofa"
    photos_dir = data_dir / "ear_photos"
    anthro_dir = data_dir / "CIPIC_hrtf_database" / "anthropometry"
    sofa_dir.mkdir(parents=True)
    photos_dir.mkdir(parents=True)
    anthro_dir.mkdir(parents=True)

    # Fake anthro.mat with NaN rows for subjects we don't materialize, so that
    # the indexer's NaN filter is exercised too.
    rng = np.random.default_rng(11)
    X = np.full((50, 17), np.nan, dtype=np.float64)
    subject_ids = list(range(3, 3 + n_subjects))
    for sid in subject_ids:
        X[sid] = rng.standard_normal(17)
    scipy.io.savemat(str(anthro_dir / "anthro.mat"), {"X": X})

    # Per-subject: copy template SOFA, drop one ear photo.
    for sid in subject_ids:
        shutil.copy(TEMPLATE_SOFA, sofa_dir / f"subject_{sid:03d}.sofa")
        subj_photo_dir = photos_dir / f"Subject_{sid:03d}"
        subj_photo_dir.mkdir()
        _synth_jpeg(subj_photo_dir / f"{sid:03d}_left_side.jpg")

    return data_dir


def test_build_subject_index_filters_and_loads(tmp_path):
    data_dir = _make_synthetic_cipic(tmp_path, n_subjects=3)
    entries, az, el = build_subject_index(data_dir, max_subject=10)

    assert len(entries) == 3
    assert az.shape == (1250,)
    assert el.shape == (1250,)
    for e in entries:
        assert e["anthro"].shape == (17,)
        assert e["impulses"].shape == (1250, 200)
        assert e["ear"] in (0, 1)
        assert Path(e["image_path"]).exists()


def test_dataset_item_shapes(tmp_path):
    data_dir = _make_synthetic_cipic(tmp_path, n_subjects=2)
    entries, az, el = build_subject_index(data_dir, max_subject=10)
    ds = CipicHRTFDataset(entries, az, el, make_transforms(train=False))

    assert len(ds) == 2 * 1250
    image, anthro, direction, target = ds[0]
    assert image.shape == (3, 224, 224)
    assert anthro.shape == (17,)
    assert direction.shape == (2,)
    assert target.shape == (200,)
    assert torch.all(torch.isfinite(image))
    assert torch.all(torch.isfinite(target))


def test_one_training_step_runs(tmp_path):
    """End-to-end smoke: build a real batch from the synthetic fixture and
    take one optimizer step. Catches integration bugs between dataset, model,
    and Lightning."""
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader

    data_dir = _make_synthetic_cipic(tmp_path, n_subjects=2)
    entries, az, el = build_subject_index(data_dir, max_subject=10)
    ds = CipicHRTFDataset(entries, az, el, make_transforms(train=False))
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

    model = HRTFModel(anthro_dim=17, backbone_name="resnet18", lr=1e-3)
    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        accelerator="cpu",
        devices=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, loader, loader)
    # Lightning would have raised on shape/dtype mismatch; reaching here is the assertion.


# ---------- predict.py ----------

def _save_untrained_checkpoint(path: Path, anthro_dim: int = 17):
    """Build an untrained model, save a Lightning-style checkpoint."""
    model = HRTFModel(anthro_dim=anthro_dim, backbone_name="resnet18", lr=1e-3)
    torch.save({
        "state_dict": model.state_dict(),
        "hyper_parameters": dict(model.hparams),
    }, str(path))


def test_predict_parse_anthro_inline():
    from predict import parse_anthro
    arr = parse_anthro("1.0,2.0,3.5,4")
    assert arr.shape == (4,)
    assert np.allclose(arr, [1.0, 2.0, 3.5, 4.0])


def test_predict_parse_anthro_from_file(tmp_path):
    from predict import parse_anthro
    f = tmp_path / "anthro.txt"
    f.write_text("1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0\n")
    arr = parse_anthro(str(f))
    assert arr.shape == (17,)


def test_predict_end_to_end(tmp_path):
    """Full inference pipeline: untrained checkpoint -> SOFA file -> reload."""
    from predict import run_inference

    ckpt_path = tmp_path / "untrained.ckpt"
    _save_untrained_checkpoint(ckpt_path, anthro_dim=17)

    photo_path = tmp_path / "ear.jpg"
    _synth_jpeg(photo_path, size=256)

    anthro = np.linspace(0.0, 1.0, 17, dtype=np.float32)
    out_path = tmp_path / "predicted.sofa"

    result = run_inference(
        checkpoint=ckpt_path,
        ear_photo=photo_path,
        anthro=anthro,
        template=TEMPLATE_SOFA,
        output=out_path,
        device="cpu",
    )
    assert result == out_path
    assert out_path.exists()

    # Reload the SOFA and verify shape + finiteness.
    written = sofar.read_sofa(str(out_path))
    ir = np.asarray(written.Data_IR)
    assert ir.shape == (1250, 2, 200)
    assert np.all(np.isfinite(ir))
    # Both ear channels are identical (single-ear model duplicated).
    assert np.allclose(ir[:, 0, :], ir[:, 1, :])


def test_predict_raises_on_wrong_anthro_dim(tmp_path):
    from predict import run_inference

    ckpt_path = tmp_path / "untrained.ckpt"
    _save_untrained_checkpoint(ckpt_path, anthro_dim=17)

    photo_path = tmp_path / "ear.jpg"
    _synth_jpeg(photo_path, size=256)

    with pytest.raises(ValueError, match="anthro"):
        run_inference(
            checkpoint=ckpt_path,
            ear_photo=photo_path,
            anthro=np.zeros(10, dtype=np.float32),  # wrong size
            template=TEMPLATE_SOFA,
            output=tmp_path / "out.sofa",
            device="cpu",
        )


# ---------- auto_anthro.py ----------

def test_order_corners():
    from auto_anthro import order_corners
    # Deliberately scrambled input
    pts = np.array([
        [100, 100],   # TL
        [500, 500],   # BR
        [100, 500],   # BL
        [500, 100],   # TR
    ], dtype=np.float32)
    ordered = order_corners(pts)
    assert np.allclose(ordered[0], [100, 100])
    assert np.allclose(ordered[1], [500, 100])
    assert np.allclose(ordered[2], [500, 500])
    assert np.allclose(ordered[3], [100, 500])


def _synth_page_image(size=(900, 1200), page_tl=(150, 200),
                      page_size=(425, 550)):
    """Dark image with a single bright rectangular 'page'."""
    h, w = size
    img = np.full((h, w, 3), 30, dtype=np.uint8)  # dark gray background
    x0, y0 = page_tl
    pw, ph = page_size
    img[y0:y0 + ph, x0:x0 + pw] = 245  # bright "paper"
    return img, np.array([
        [x0, y0], [x0 + pw - 1, y0],
        [x0 + pw - 1, y0 + ph - 1], [x0, y0 + ph - 1],
    ], dtype=np.float32)


def test_detect_page_corners_on_axis_aligned():
    from auto_anthro import detect_page_corners
    img, expected = _synth_page_image()
    corners = detect_page_corners(img)
    assert corners is not None
    # Pixel-perfect contour detection isn't realistic; tolerate ~5px.
    diffs = np.linalg.norm(corners - expected, axis=1)
    assert diffs.max() < 8.0, f"corner error too large: {diffs.tolist()}"


def test_rectify_to_page_yields_canonical_size():
    from auto_anthro import (
        PAGE_HEIGHT_PX,
        PAGE_WIDTH_PX,
        auto_rectify,
    )
    img, _ = _synth_page_image()
    out = auto_rectify(img)
    assert out.image.shape[:2] == (PAGE_HEIGHT_PX, PAGE_WIDTH_PX)
    # The rectified page should be uniformly bright (it was synthetic white).
    mean_intensity = out.image[
        50:PAGE_HEIGHT_PX - 50, 50:PAGE_WIDTH_PX - 50
    ].mean()
    assert mean_intensity > 200, f"rectified page should be bright, got {mean_intensity}"


def test_detect_page_corners_under_perspective():
    """Warp the synthetic page with a known homography; detection should
    still recover ~the warped corners."""
    from auto_anthro import detect_page_corners, order_corners
    img, axis_corners = _synth_page_image()
    H, W = img.shape[:2]
    # Random-ish perspective warp.
    src = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]],
                   dtype=np.float32)
    dst = np.array([[50, 30], [W - 80, 70], [W - 40, H - 30], [20, H - 100]],
                   dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (W, H))

    expected = cv2.perspectiveTransform(
        axis_corners.reshape(-1, 1, 2), M
    ).reshape(4, 2)
    expected = order_corners(expected)

    corners = detect_page_corners(warped)
    assert corners is not None
    diffs = np.linalg.norm(corners - expected, axis=1)
    assert diffs.max() < 12.0, f"perspective corner error: {diffs.tolist()}"


def test_detect_page_returns_none_when_no_page():
    from auto_anthro import detect_page_corners
    rng = np.random.default_rng(0)
    noise = (rng.integers(0, 80, size=(400, 600, 3))).astype(np.uint8)
    corners = detect_page_corners(noise)
    # Either None, or at minimum not a sensible page contour. We accept
    # None or a contour with absurdly small area; pure-noise images rarely
    # produce convex quadrilaterals large enough to pass the filter.
    assert corners is None or cv2.contourArea(corners.astype(np.float32)) < 2000


def test_distance_cm_uses_known_scale():
    from auto_anthro import CM_PER_PIXEL, distance_cm
    d = distance_cm((0.0, 0.0), (100.0, 0.0))
    assert abs(d - 100 * CM_PER_PIXEL) < 1e-6
    assert abs(d - 2.54) < 1e-6  # 100 px / 100 px-per-inch = 1 in = 2.54 cm


def test_auto_anthro_non_interactive_cli(tmp_path):
    """End-to-end CLI smoke: pass --non-interactive, the script should
    detect+rectify both photos and exit without prompting."""
    import subprocess
    img_front, _ = _synth_page_image()
    img_side, _ = _synth_page_image(page_tl=(200, 250))
    front_path = tmp_path / "front.jpg"
    side_path = tmp_path / "side.jpg"
    cv2.imwrite(str(front_path), cv2.cvtColor(img_front, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(side_path), cv2.cvtColor(img_side, cv2.COLOR_RGB2BGR))

    proc = subprocess.run(
        [sys.executable, str(ROOT / "auto_anthro.py"),
         "--front", str(front_path),
         "--side", str(side_path),
         "--non-interactive"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}"
    assert "page corners" in proc.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
