"""Fusion + neural-field decoder producing log-magnitude spectra and ITD.

The subject embedding ``z`` (from photo + optional anthropometry) conditions a small
coordinate-MLP that maps a direction (azimuth, elevation) to:

* a **log-magnitude spectrum** for each ear, shape (B, 2, F);
* a scalar **ITD** in samples, shape (B, 1).

Conditioning uses FiLM (feature-wise linear modulation): ``z`` predicts per-layer
(scale, shift) applied to the direction MLP's hidden activations. This is a lightweight,
well-behaved way to make one decoder subject-specific without a giant network.

The decoder is continuous in direction, so it is not bound to any one dataset's
measurement grid — directions from CIPIC, HUTUBS, or SONICOM can all be queried.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .encoders import ImageEncoder, AnthroEncoder


def direction_features(az: torch.Tensor, el: torch.Tensor) -> torch.Tensor:
    """Map (azimuth, elevation) in degrees to a smooth, periodicity-aware encoding.

    Returns (B, 5): [sin az, cos az, sin el, cos el, el_normalized].
    Using sin/cos avoids the wrap-around discontinuity at ±180°.
    """
    az_r = az * (math.pi / 180.0)
    el_r = el * (math.pi / 180.0)
    return torch.stack(
        [torch.sin(az_r), torch.cos(az_r),
         torch.sin(el_r), torch.cos(el_r),
         el / 90.0],
        dim=-1,
    )


class _FiLMBlock(nn.Module):
    """Linear -> FiLM(scale, shift from z) -> GELU."""

    def __init__(self, dim: int, cond_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.film = nn.Linear(cond_dim, 2 * dim)
        self.act = nn.GELU()

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(cond).chunk(2, dim=-1)
        return self.act((1 + scale) * self.fc(h) + shift)


class HRTFFieldDecoder(nn.Module):
    """Direction -> (log-magnitude per ear, ITD), conditioned on subject embedding z."""

    def __init__(self, cond_dim: int, n_freq: int = 128,
                 hidden: int = 256, n_blocks: int = 4) -> None:
        super().__init__()
        self.n_freq = n_freq
        self.stem = nn.Linear(5, hidden)
        self.blocks = nn.ModuleList(
            _FiLMBlock(hidden, cond_dim) for _ in range(n_blocks)
        )
        self.mag_head = nn.Linear(hidden, 2 * n_freq)   # two ears
        self.itd_head = nn.Linear(hidden, 1)            # ITD in samples (signed)

    def forward(self, z: torch.Tensor, az: torch.Tensor,
                el: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """z: (B, cond_dim). az, el: (B,) degrees.

        Returns (log_mag, itd) with shapes (B, 2, F) and (B, 1).
        """
        h = self.stem(direction_features(az, el))
        for blk in self.blocks:
            h = blk(h, z)
        log_mag = self.mag_head(h).view(-1, 2, self.n_freq)
        itd = self.itd_head(h)
        return log_mag, itd


class HRTFNet(nn.Module):
    """End-to-end: (ear photo, optional anthropometry, direction) -> (log-mag, ITD)."""

    def __init__(self, anthro_dim: int, n_freq: int = 128,
                 img_dim: int = 64, anthro_emb: int = 32,
                 pretrained_image: bool = False,
                 image_backbone: str = "mobilenet_v3_small",
                 anthro_drop_prob: float = 0.5) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(
            out_dim=img_dim, pretrained=pretrained_image, backbone=image_backbone)
        self.anthro_encoder = AnthroEncoder(
            in_dim=anthro_dim, out_dim=anthro_emb, drop_prob=anthro_drop_prob)
        cond_dim = img_dim + anthro_emb
        self.decoder = HRTFFieldDecoder(cond_dim=cond_dim, n_freq=n_freq)

    def encode_subject(self, img: torch.Tensor, anthro: torch.Tensor,
                       anthro_present: torch.Tensor | None = None) -> torch.Tensor:
        """Compute the subject embedding once; reuse across many directions."""
        zi = self.image_encoder(img)
        za = self.anthro_encoder(anthro, anthro_present)
        return torch.cat([zi, za], dim=-1)

    def forward(self, img: torch.Tensor, anthro: torch.Tensor,
                az: torch.Tensor, el: torch.Tensor,
                anthro_present: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode_subject(img, anthro, anthro_present)
        return self.decoder(z, az, el)
