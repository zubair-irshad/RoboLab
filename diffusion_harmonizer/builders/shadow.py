"""Post-hoc shadow-simulation pair builder (DiffusionHarmonizer §3.2).

The capture phase already saved two path-traced renders per view under
identical lighting:

  * ``target.png``        - foreground present, full cast shadows.
  * ``no_shadow_fg.png``  - foreground still visible and lit, but excluded
                            from every UsdLux shadowLink collection.

The pixel-level delta between them is the shadow-only signal. This builder
just reads disk, computes the masked delta, and emits paired (input=no_shadow,
target=full_shadow) frames + auxiliary diff/mask PNGs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from diffusion_harmonizer.capture import load_mask, load_view, load_view_dirs
from diffusion_harmonizer.components.common import pair_id
from diffusion_harmonizer.image_io import save_png, write_pair


def _dilate(mask: np.ndarray, pixels: int = 7) -> np.ndarray:
    import cv2

    kernel = np.ones((pixels, pixels), dtype=np.uint8)
    return cv2.dilate((mask > 0.05).astype(np.uint8), kernel, iterations=1).astype(np.float32)


def _shadow_delta_mask(target: np.ndarray, no_shadow: np.ndarray, fg_dilated: np.ndarray) -> np.ndarray:
    diff = np.max(np.abs(target.astype(np.float32) - no_shadow.astype(np.float32)), axis=-1) / 255.0
    return np.clip(diff * (1.0 - np.clip(fg_dilated, 0.0, 1.0)), 0.0, 1.0)


def build(
    env_capture_dir: Path,
    output_dir: Path,
    count: int = 12,
    min_shadow_coverage: float = 0.001,
) -> dict[str, dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    view_dirs = load_view_dirs(env_capture_dir)
    if not view_dirs:
        raise FileNotFoundError(f"No captured views under {env_capture_dir}/sphere")

    entries: dict[str, dict[str, str]] = {}
    pair_index = 0
    for view_dir in view_dirs:
        if pair_index >= count:
            break
        try:
            target = load_view(view_dir, "target")
            no_shadow = load_view(view_dir, "no_shadow_fg")
        except FileNotFoundError:
            continue
        try:
            fg_mask = load_mask(view_dir)
        except FileNotFoundError:
            fg_mask = np.zeros(target.shape[:2], dtype=np.float32)

        diff_rgb = np.abs(target.astype(np.int16) - no_shadow.astype(np.int16)).astype(np.uint8)
        if float(np.mean(fg_mask > 0.05)) >= 0.002:
            shadow_mask = _shadow_delta_mask(target, no_shadow, _dilate(fg_mask))
        else:
            shadow_mask = np.max(diff_rgb.astype(np.float32), axis=-1) / 255.0

        if float(np.mean(shadow_mask > 0.03)) < min_shadow_coverage:
            continue

        pair_dir = output_dir / pair_id(pair_index)
        save_png(pair_dir / "shadow_diff.png", diff_rgb)
        save_png(pair_dir / "shadow_mask.png", np.repeat((shadow_mask * 255).astype(np.uint8)[..., None], 3, axis=-1))
        entries[f"shadow_{pair_id(pair_index)}"] = write_pair(
            pair_dir,
            no_shadow,  # input: full lighting but no foreground-cast shadow
            target,     # target: full lighting with foreground shadow
            {
                "component": "shadow_simulation",
                "view_dir": str(view_dir),
                "shadow_mask_coverage": float(np.mean(shadow_mask > 0.03)),
            },
        )
        pair_index += 1
    (output_dir / "pairs.json").write_text(json.dumps(entries, indent=2))
    print(f"[builder:shadow] {env_capture_dir.name}: wrote {len(entries)} pairs -> {output_dir}", flush=True)
    return entries
