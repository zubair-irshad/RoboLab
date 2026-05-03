"""ISP Modification (DiffusionHarmonizer §3.2).

Realistic software ISP simulation operating in linear-light space:

  1. inverse-gamma to linear
  2. exposure (EV)
  3. per-channel white-balance gains (R/B wider than G — matches sensor IQ)
  4. near-identity 3x3 color correction matrix (the subtle hue drift different
     manufacturers' ISPs produce — captures "different camera" mismatch
     without flipping color identity the way an HSV hue-shift can)
  5. forward gamma / tone curve
  6. contrast + brightness
  7. saturation around luminance

Eq. (3) composite is then ``M ⊙ I_ISP + (1−M) ⊙ I_orig`` with ``M`` covering
manipulable foreground objects only (``runtime.object_prim_paths()``).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ISPParams:
    exposure_ev: float
    wb_gain_r: float
    wb_gain_g: float
    wb_gain_b: float
    ccm: list[list[float]]  # 3x3 list-of-lists for clean JSON serialization
    gamma: float
    contrast: float
    brightness: float
    saturation: float


def sample_isp_params(rng: random.Random, scale: float = 1.0) -> ISPParams:
    """Sample ISP params modulated uniformly by ``scale``.

    Ranges at scale=1.0 follow the realistic-ISP recipe — RGB gains span the
    same ranges real cameras vary across (R: 0.85-1.20, G: 0.95-1.05,
    B: 0.80-1.25), CCM is identity + N(0, 0.035) gaussian noise, and the
    tone curve / contrast / saturation knobs match the user-provided values.
    Lower ``scale`` shrinks every range linearly toward identity.
    """

    def around(low: float, high: float) -> float:
        center = 0.5 * (low + high)
        half = 0.5 * (high - low)
        return center + (rng.uniform(-half, half) * scale)

    ccm = [
        [1.0 + rng.gauss(0.0, 0.035) * scale if i == j else rng.gauss(0.0, 0.035) * scale for j in range(3)]
        for i in range(3)
    ]
    return ISPParams(
        exposure_ev=rng.uniform(-0.6, 0.6) * scale,
        wb_gain_r=around(0.85, 1.20),
        wb_gain_g=around(0.95, 1.05),
        wb_gain_b=around(0.80, 1.25),
        ccm=ccm,
        gamma=around(0.8, 1.3),
        contrast=around(0.85, 1.2),
        brightness=rng.uniform(-0.04, 0.04) * scale,
        saturation=around(0.8, 1.25),
    )


def apply_software_isp(image: np.ndarray, params: ISPParams, rng: random.Random | None = None) -> np.ndarray:
    """Realistic linear-light software-ISP. ``image`` is sRGB uint8 (H, W, 3)."""

    del rng  # unused — sampling happens upstream in ``sample_isp_params``
    rgb = image.astype(np.float32) / 255.0
    x = np.clip(rgb, 0.0, 1.0) ** 2.2  # inverse-gamma to linear

    # Exposure (EV stops).
    x = x * (2.0 ** float(params.exposure_ev))

    # Per-channel white balance gains.
    gains = np.array(
        [params.wb_gain_r, params.wb_gain_g, params.wb_gain_b], dtype=np.float32
    )
    x = x * gains[None, None, :]

    # Near-identity 3x3 color correction matrix.
    ccm = np.asarray(params.ccm, dtype=np.float32)
    x = x.reshape(-1, 3) @ ccm.T
    x = x.reshape(image.shape).astype(np.float32)

    # Clip before display transform.
    x = np.clip(x, 0.0, 1.0)

    # Tone / gamma.
    x = x ** (1.0 / max(float(params.gamma), 1e-3))

    # Contrast + brightness.
    x = (x - 0.5) * float(params.contrast) + 0.5 + float(params.brightness)

    # Saturation around BT.709 luminance.
    gray = (
        0.2126 * x[..., 0:1]
        + 0.7152 * x[..., 1:2]
        + 0.0722 * x[..., 2:3]
    )
    x = gray + float(params.saturation) * (x - gray)

    return (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)
