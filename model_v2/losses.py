"""Standard HRTF objective metrics, usable both as training losses and eval metrics.

* ``log_spectral_distortion`` (LSD, dB) — the dominant objective metric.
* ``itd_error`` (samples or microseconds) — interaural time difference error.
* ``ild_error`` (dB) — interaural level difference error.

All operate on **log-magnitude** spectra of shape (..., 2, F): index 0 = left ear,
index 1 = right ear. LSD is computed on a base-10 dB scale and averaged over frequency,
ears, and any leading (batch/direction) dimensions.
"""
from __future__ import annotations

import torch

# 20 / ln(10): converts a natural-log magnitude ratio to decibels.
_DB = 20.0 / 2.302585092994046


def log_spectral_distortion(log_mag_pred: torch.Tensor,
                            log_mag_true: torch.Tensor,
                            reduce: bool = True) -> torch.Tensor:
    """LSD in dB between two log-magnitude spectra.

    Inputs are natural-log magnitudes, shape (..., 2, F). LSD per (sample, ear) is the
    RMS over frequency of the dB difference; we then average over ears and batch.
    """
    diff_db = (log_mag_pred - log_mag_true) * _DB           # dB difference
    per_ear = torch.sqrt(torch.mean(diff_db ** 2, dim=-1))  # RMS over F -> (..., 2)
    return per_ear.mean() if reduce else per_ear


def itd_error(itd_pred: torch.Tensor, itd_true: torch.Tensor,
              fs: float | None = None, reduce: bool = True) -> torch.Tensor:
    """Mean-absolute ITD error.

    ITDs are in **samples**. If ``fs`` (Hz) is given the result is converted to
    microseconds, the conventional reporting unit.
    """
    err = torch.abs(itd_pred - itd_true)
    if fs is not None:
        err = err / fs * 1e6  # samples -> microseconds
    return err.mean() if reduce else err


def ild_from_logmag(log_mag: torch.Tensor) -> torch.Tensor:
    """Broadband ILD (dB) = mean over frequency of (left - right) log-magnitude, in dB."""
    return (log_mag[..., 0, :] - log_mag[..., 1, :]).mean(dim=-1) * _DB


def ild_error(log_mag_pred: torch.Tensor, log_mag_true: torch.Tensor,
              reduce: bool = True) -> torch.Tensor:
    """Mean-absolute ILD error (dB)."""
    err = torch.abs(ild_from_logmag(log_mag_pred) - ild_from_logmag(log_mag_true))
    return err.mean() if reduce else err


def hrtf_loss(log_mag_pred: torch.Tensor, log_mag_true: torch.Tensor,
              itd_pred: torch.Tensor, itd_true: torch.Tensor,
              w_mag: float = 1.0, w_itd: float = 0.1, w_ild: float = 0.1
              ) -> tuple[torch.Tensor, dict]:
    """Combined training loss = LSD + w_itd * ITD-L1 + w_ild * ILD-L1.

    Returns (scalar_loss, metrics_dict) where the dict holds detached components for
    logging (lsd_db, itd_samples, ild_db).
    """
    lsd = log_spectral_distortion(log_mag_pred, log_mag_true)
    itd = itd_error(itd_pred, itd_true)
    ild = ild_error(log_mag_pred, log_mag_true)
    loss = w_mag * lsd + w_itd * itd + w_ild * ild
    metrics = {
        "lsd_db": float(lsd.detach()),
        "itd_samples": float(itd.detach()),
        "ild_db": float(ild.detach()),
    }
    return loss, metrics
