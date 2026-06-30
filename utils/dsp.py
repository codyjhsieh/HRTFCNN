"""DSP helpers shared by training and inference.

The HRTF model predicts a log-magnitude spectrum (in dB) for each (azimuth,
elevation) direction. To write a CIPIC-compatible SOFA file, that spectrum
has to be converted back into a time-domain HRIR. We do this via the
standard minimum-phase reconstruction (real cepstrum folding), which is the
de-facto choice in HRTF personalization work — phase is hard to predict
directly and is perceptually dominated by the interaural time difference.
"""

from typing import Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]

EPS = 1e-8


def ir_to_log_magnitude_torch(ir: torch.Tensor, n_fft: int) -> torch.Tensor:
    """(..., n_taps) real IR -> (..., n_fft//2 + 1) log-magnitude in dB."""
    spec = torch.fft.rfft(ir, n=n_fft)
    mag = torch.abs(spec)
    return 20.0 * torch.log10(mag + EPS)


def ir_to_log_magnitude_np(ir: np.ndarray, n_fft: int) -> np.ndarray:
    spec = np.fft.rfft(ir, n=n_fft, axis=-1)
    mag = np.abs(spec)
    return 20.0 * np.log10(mag + EPS)


def log_magnitude_to_min_phase_ir(
    log_mag_db: np.ndarray, n_fft: int, n_taps: int
) -> np.ndarray:
    """Reconstruct a minimum-phase time-domain IR from a log-magnitude spectrum.

    log_mag_db: (..., n_fft//2 + 1) log-magnitude in dB.
    Returns: (..., n_taps) real IR.

    Method: standard real-cepstrum folding. Given the log-magnitude
    Re{log H(w)}, the minimum-phase log-spectrum has imaginary part equal to
    the Hilbert transform of the log-magnitude. We compute it by:
      1. Symmetrizing log-mag into a full-length even spectrum.
      2. Inverse FFT to get the real cepstrum.
      3. Multiplying by [1, 2, 2, ..., 2, 1, 0, 0, ..., 0] to keep only the
         causal part (this is equivalent to applying the Hilbert transform).
      4. FFT back to get the complex log-spectrum.
      5. Exponentiate, then inverse FFT to the time domain.
    """
    log_mag_db = np.asarray(log_mag_db)
    leading_shape = log_mag_db.shape[:-1]
    n_bins = log_mag_db.shape[-1]
    assert n_bins == n_fft // 2 + 1, (
        f"log_mag length {n_bins} does not match n_fft//2+1 = {n_fft // 2 + 1}"
    )

    log_mag = log_mag_db / 20.0 * np.log(10.0)  # natural log of magnitude

    # Mirror to length n_fft (real, even). For complex IFFT we want the full
    # n_fft-length spectrum where bins [n_fft//2+1 : n_fft] mirror [1 : n_fft//2].
    full = np.empty(leading_shape + (n_fft,), dtype=np.float64)
    full[..., :n_bins] = log_mag
    full[..., n_bins:] = log_mag[..., -2:0:-1]

    cepstrum = np.fft.ifft(full, axis=-1).real

    # Causal folding: multiply by [1, 2, 2, ..., 2, 1, 0, ..., 0].
    fold = np.zeros(n_fft, dtype=np.float64)
    fold[0] = 1.0
    half = n_fft // 2
    fold[1:half] = 2.0
    fold[half] = 1.0
    cepstrum_min = cepstrum * fold

    log_spec_min = np.fft.fft(cepstrum_min, axis=-1)
    spec_min = np.exp(log_spec_min)
    ir_full = np.fft.ifft(spec_min, axis=-1).real

    return ir_full[..., :n_taps]


def log_spectral_distance_torch(
    pred_log_mag: torch.Tensor, target_log_mag: torch.Tensor
) -> torch.Tensor:
    """LSD in dB, averaged over freq then batch.

    Both inputs are already in dB (matches what the model outputs and what
    `ir_to_log_magnitude_torch` returns)."""
    return torch.sqrt(
        torch.mean((pred_log_mag - target_log_mag) ** 2, dim=-1)
    ).mean()
