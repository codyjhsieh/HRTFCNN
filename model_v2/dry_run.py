"""Self-contained smoke test — no dataset, no network, CPU only (needs just `torch`).

Verifies that the model, the subject-scatter training path, the loss, and the metrics are
wired correctly: shapes line up, a backward pass produces gradients, and an optimizer
step runs. This is the intended correctness gate for the scaffold — run it first wherever
torch is available:

    python -m model_v2.dry_run
"""
from __future__ import annotations

import torch

from .losses import (hrtf_loss, log_spectral_distortion, itd_error, ild_error)
from .models import HRTFNet


def _synthetic_batch(n_subjects=3, dirs_per_subject=64, anthro_dim=10,
                     img_size=64, n_freq=128):
    """Mimic one `collate_subject_batch` output with random tensors."""
    images = torch.rand(n_subjects, 3, img_size, img_size)
    anthro = torch.randn(n_subjects, anthro_dim)
    present = torch.tensor([1.0, 1.0, 0.0])[:n_subjects]  # last subject: no anthro
    az, el, log_mag, itd, subj_idx = [], [], [], [], []
    for i in range(n_subjects):
        az.append(torch.empty(dirs_per_subject).uniform_(-180, 180))
        el.append(torch.empty(dirs_per_subject).uniform_(-90, 90))
        log_mag.append(torch.randn(dirs_per_subject, 2, n_freq))
        itd.append(torch.empty(dirs_per_subject).uniform_(-30, 30))
        subj_idx.append(torch.full((dirs_per_subject,), i, dtype=torch.long))
    return {
        "image": images, "anthro": anthro, "anthro_present": present,
        "az": torch.cat(az), "el": torch.cat(el),
        "log_mag": torch.cat(log_mag), "itd": torch.cat(itd),
        "subject_index": torch.cat(subj_idx), "fs": 48000.0,
        "sids": [f"P{i:04d}" for i in range(n_subjects)],
    }


def main() -> None:
    torch.manual_seed(0)
    anthro_dim, n_freq = 10, 128
    n_subjects, dirs = 3, 64

    model = HRTFNet(anthro_dim=anthro_dim, n_freq=n_freq, image_backbone="cnn")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = _synthetic_batch(n_subjects, dirs, anthro_dim, img_size=64, n_freq=n_freq)

    print("=== model_v2 dry run ===")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}")

    # --- subject-scatter forward path (mirrors train.decode_batch) ---
    model.train()
    z = model.encode_subject(batch["image"], batch["anthro"], batch["anthro_present"])
    assert z.shape[0] == n_subjects, z.shape
    z_per_dir = z[batch["subject_index"]]
    log_mag, itd = model.decoder(z_per_dir, batch["az"], batch["el"])
    itd = itd.squeeze(-1)
    total_dirs = n_subjects * dirs
    assert log_mag.shape == (total_dirs, 2, n_freq), log_mag.shape
    assert itd.shape == (total_dirs,), itd.shape
    print(f"forward OK: log_mag {tuple(log_mag.shape)}, itd {tuple(itd.shape)}")

    # --- loss + backward + step ---
    loss, metrics = hrtf_loss(log_mag, batch["log_mag"], itd, batch["itd"])
    opt.zero_grad()
    loss.backward()
    grad_norm = sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None)
    assert torch.isfinite(loss) and grad_norm > 0, "no gradient flowed"
    opt.step()
    print(f"loss {float(loss):.4f} | LSD {metrics['lsd_db']:.3f} dB | "
          f"ITD {metrics['itd_samples']:.2f} smp | ILD {metrics['ild_db']:.3f} dB")
    print(f"backward OK: grad-norm {float(grad_norm):.2f}")

    # --- metric sanity: identical inputs -> ~0 error ---
    same = torch.randn(8, 2, n_freq)
    assert float(log_spectral_distortion(same, same)) < 1e-5
    assert float(ild_error(same, same)) < 1e-5
    z0 = torch.zeros(8)
    assert float(itd_error(z0, z0)) == 0.0
    print("metric self-consistency OK (zero error on identical inputs)")

    # --- modality dropout: photo-only inference must still run ---
    model.eval()
    with torch.no_grad():
        no_anthro_present = torch.zeros(n_subjects)
        z2 = model.encode_subject(batch["image"], batch["anthro"], no_anthro_present)
        lm2, _ = model.decoder(z2[batch["subject_index"]], batch["az"], batch["el"])
    assert lm2.shape == log_mag.shape
    print("photo-only (no anthropometry) path OK")

    print("\nALL CHECKS PASSED ✅")


if __name__ == "__main__":
    main()
