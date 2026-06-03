"""Input encoders for the phone-capturable modalities.

Two encoders, both deliberately edge-deployable:

* ``ImageEncoder``  — a 2D ear photo -> embedding. Uses torchvision MobileNetV3-small
  when available (optionally with pretrained weights), otherwise falls back to a small
  self-contained CNN so the scaffold runs with only ``torch`` installed.
* ``AnthroEncoder`` — an optional anthropometry vector -> embedding, with built-in
  *modality dropout* so a single model works with or without the measurements.

Neither encoder consumes depth or 3D geometry (see ../README.md "Why no depth").
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    """Encode a single ear photo into a fixed-size embedding.

    Args:
        out_dim: size of the produced embedding.
        pretrained: load ImageNet weights for the MobileNet backbone (needs network/
            torchvision weights). Ignored by the fallback CNN.
        backbone: ``"mobilenet_v3_small"`` (default) or ``"cnn"`` to force the fallback.
    """

    def __init__(self, out_dim: int = 64, pretrained: bool = False,
                 backbone: str = "mobilenet_v3_small") -> None:
        super().__init__()
        self.out_dim = out_dim
        self.backbone_name = backbone
        feat_dim = self._build_backbone(backbone, pretrained)
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def _build_backbone(self, backbone: str, pretrained: bool) -> int:
        if backbone == "mobilenet_v3_small":
            try:
                from torchvision.models import (mobilenet_v3_small,
                                                MobileNet_V3_Small_Weights)
                weights = (MobileNet_V3_Small_Weights.DEFAULT
                           if pretrained else None)
                net = mobilenet_v3_small(weights=weights)
                feat_dim = net.classifier[0].in_features  # 576
                net.classifier = nn.Identity()
                self.backbone = net
                return feat_dim
            except Exception:  # torchvision missing or weights unavailable offline
                self.backbone_name = "cnn"
        # Fallback: a compact CNN (no external weights, runs anywhere torch runs).
        self.backbone = _FallbackCNN()
        return self.backbone.out_features

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """img: (B, 3, H, W) in [0, 1]. Returns (B, out_dim)."""
        feats = self.backbone(img)
        if feats.dim() > 2:  # safety for backbones that return spatial maps
            feats = torch.flatten(nn.functional.adaptive_avg_pool2d(feats, 1), 1)
        return self.proj(feats)


class _FallbackCNN(nn.Module):
    """Small dependency-free CNN used when torchvision is unavailable."""

    def __init__(self) -> None:
        super().__init__()
        self.out_features = 128

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=2, padding=1),
                nn.BatchNorm2d(cout),
                nn.GELU(),
            )

        self.net = nn.Sequential(
            block(3, 16), block(16, 32), block(32, 64), block(64, 128),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AnthroEncoder(nn.Module):
    """Encode an optional anthropometry vector into an embedding.

    Implements *modality dropout*: during training the whole anthropometry vector is
    zeroed for a random subset of the batch (prob ``drop_prob``) so the fused model
    learns to work photo-only. A boolean ``present`` mask is concatenated so the network
    can tell "all zeros because measured" from "all zeros because absent".
    """

    def __init__(self, in_dim: int, out_dim: int = 32, drop_prob: float = 0.5) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.drop_prob = drop_prob
        self.net = nn.Sequential(
            nn.Linear(in_dim + 1, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 64), nn.GELU(),
            nn.Linear(64, out_dim), nn.LayerNorm(out_dim), nn.GELU(),
        )

    def forward(self, anthro: torch.Tensor,
                present: torch.Tensor | None = None) -> torch.Tensor:
        """anthro: (B, in_dim). present: (B,) bool/float, 1 if measurements exist.

        Returns (B, out_dim).
        """
        b = anthro.shape[0]
        if present is None:
            present = torch.ones(b, device=anthro.device, dtype=anthro.dtype)
        present = present.to(anthro.dtype).view(b, 1)

        if self.training and self.drop_prob > 0:
            keep = (torch.rand(b, 1, device=anthro.device) > self.drop_prob).to(anthro.dtype)
            present = present * keep

        anthro = anthro * present  # zero out absent/dropped measurements
        return self.net(torch.cat([anthro, present], dim=1))
