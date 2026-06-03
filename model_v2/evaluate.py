"""Evaluate a trained checkpoint with the standard objective metrics.

Reports LSD (dB), ITD error (microseconds), and ILD error (dB) over the held-out
subjects. These are the LAP'24 Task-2 objective metrics. Perceptual / auditory-model
metrics (Baumgartner polar & quadrant error) are intentionally left as a follow-up hook
(see ../COMPARISON.md §4) — they require the Auditory Modeling Toolbox or a Python port.

Run:
    python -m model_v2.evaluate --sonicom-root /path/to/SONICOM --ckpt runs/best.pt
"""
from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from .data import SonicomHRTFDataset, collate_subject_batch
from .losses import log_spectral_distortion, itd_error, ild_error
from .models import HRTFNet
from .train import decode_batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sonicom-root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="subject ids to evaluate (default: all in the dataset)")
    ap.add_argument("--anthro-dim", type=int, default=10)
    ap.add_argument("--n-freq", type=int, default=128)
    ap.add_argument("--n-fft", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ds = SonicomHRTFDataset(
        args.sonicom_root, subject_ids=args.subjects, img_size=128,
        n_fft=args.n_fft, n_freq=args.n_freq, anthro_dim=args.anthro_dim)
    loader = DataLoader(ds, batch_size=1, collate_fn=collate_subject_batch)

    model = HRTFNet(anthro_dim=args.anthro_dim, n_freq=args.n_freq).to(args.device)
    state = torch.load(args.ckpt, map_location=args.device)
    model.load_state_dict(state["model"])
    model.eval()

    lsd_sum = itd_sum = ild_sum = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            log_mag, itd = decode_batch(model, batch, args.device)
            tgt_mag = batch["log_mag"].to(args.device)
            tgt_itd = batch["itd"].to(args.device)
            fs = batch["fs"]
            lsd_sum += float(log_spectral_distortion(log_mag, tgt_mag))
            itd_sum += float(itd_error(itd, tgt_itd, fs=fs))
            ild_sum += float(ild_error(log_mag, tgt_mag))
            n += 1

    n = max(n, 1)
    print(f"Subjects evaluated: {n}")
    print(f"  LSD : {lsd_sum / n:.3f} dB")
    print(f"  ITD : {itd_sum / n:.2f} us")
    print(f"  ILD : {ild_sum / n:.3f} dB")


if __name__ == "__main__":
    main()
