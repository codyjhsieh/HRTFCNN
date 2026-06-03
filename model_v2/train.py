"""Training loop for HRTFNet on SONICOM.

Subjects are batched whole: we encode each subject's embedding once, then scatter it to
all of that subject's directions and decode them in a single call. This matches the
neural-field design and is much cheaper than re-encoding the photo per direction.

Run:
    python -m model_v2.train --sonicom-root /path/to/SONICOM --epochs 100 --device cuda

For a leave-one-subject-out fold, pass --holdout P0042 (that subject is excluded from
training and used as the validation set).
"""
from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from .data import SonicomHRTFDataset, collate_subject_batch
from .losses import hrtf_loss
from .models import HRTFNet


def decode_batch(model: HRTFNet, batch: dict, device: str):
    """Encode per-subject embeddings, scatter to directions, decode all directions."""
    image = batch["image"].to(device)
    anthro = batch["anthro"].to(device)
    present = batch["anthro_present"].to(device)
    az = batch["az"].to(device)
    el = batch["el"].to(device)
    subj_idx = batch["subject_index"].to(device)

    z = model.encode_subject(image, anthro, present)   # (S, cond_dim)
    z_per_dir = z[subj_idx]                             # (D_total, cond_dim)
    log_mag, itd = model.decoder(z_per_dir, az, el)
    return log_mag, itd.squeeze(-1)


def run_epoch(model, loader, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    totals = {"loss": 0.0, "lsd_db": 0.0, "itd_samples": 0.0, "ild_db": 0.0, "n": 0}
    for batch in loader:
        log_mag, itd = decode_batch(model, batch, device)
        target_mag = batch["log_mag"].to(device)
        target_itd = batch["itd"].to(device)
        loss, metrics = hrtf_loss(log_mag, target_mag, itd, target_itd)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        bs = len(batch["sids"])
        totals["loss"] += float(loss.detach()) * bs
        for k in ("lsd_db", "itd_samples", "ild_db"):
            totals[k] += metrics[k] * bs
        totals["n"] += bs
    n = max(totals["n"], 1)
    return {k: totals[k] / n for k in ("loss", "lsd_db", "itd_samples", "ild_db")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sonicom-root", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--anthro-dim", type=int, default=10)
    ap.add_argument("--n-freq", type=int, default=128)
    ap.add_argument("--n-fft", type=int, default=256)
    ap.add_argument("--holdout", default=None, help="subject id for leave-one-out val")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    full = SonicomHRTFDataset(
        args.sonicom_root, img_size=128, n_fft=args.n_fft,
        n_freq=args.n_freq, anthro_dim=args.anthro_dim)
    all_ids = [s.sid for s in full.subjects]
    val_ids = [args.holdout] if args.holdout else all_ids[-max(1, len(all_ids) // 10):]
    train_ids = [s for s in all_ids if s not in val_ids]

    common = dict(root=args.sonicom_root, img_size=128, n_fft=args.n_fft,
                  n_freq=args.n_freq, anthro_dim=args.anthro_dim)
    train_ds = SonicomHRTFDataset(subject_ids=train_ids, **common)
    val_ds = SonicomHRTFDataset(subject_ids=val_ids, **common)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_subject_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            collate_fn=collate_subject_batch)

    model = HRTFNet(anthro_dim=args.anthro_dim, n_freq=args.n_freq).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    os.makedirs(args.out, exist_ok=True)

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, args.device, optimizer)
        with torch.no_grad():
            va = run_epoch(model, val_loader, args.device)
        print(f"[{epoch:3d}] train LSD {tr['lsd_db']:.3f} dB | "
              f"val LSD {va['lsd_db']:.3f} dB  ITD {va['itd_samples']:.2f} smp  "
              f"ILD {va['ild_db']:.3f} dB")
        if va["lsd_db"] < best:
            best = va["lsd_db"]
            torch.save({"model": model.state_dict(), "args": vars(args)},
                       os.path.join(args.out, "best.pt"))
    print(f"Best val LSD: {best:.3f} dB  ->  {os.path.join(args.out, 'best.pt')}")


if __name__ == "__main__":
    main()
