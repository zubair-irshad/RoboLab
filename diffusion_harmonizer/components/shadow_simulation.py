"""Physically Based Shadow Simulation (DiffusionHarmonizer §3.2).

For every privileged sphere camera, we render the same view twice under the
same path-traced lighting:

  * **target**   - foreground present, cast shadows enabled.
  * **input**    - foreground still visible and lit, but excluded from
                   ``UsdLux.shadowLink`` so it casts no shadow on the receiver.

The pixel-level delta between the two renders is the supervision signal for
shadow inpainting. Per pair we also randomize the dome HDRI / rotation /
intensity to vary direction, softness, and color of the dominant light.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np

from diffusion_harmonizer.components.common import (
    feather_mask,
    foreground_mask_with_fallback,
    pair_id,
)
from diffusion_harmonizer.image_io import save_png, write_pair


def _gather_hdris(roots: Iterable[str | Path]) -> list[Path]:
    suffixes = {".hdr", ".exr"}
    out: list[Path] = []
    for root in roots:
        path = Path(root)
        if not path.exists():
            continue
        for candidate in path.rglob("*"):
            if candidate.is_file() and candidate.suffix.lower() in suffixes:
                out.append(candidate.resolve())
    return sorted(out)


def _shadow_delta_mask(target: np.ndarray, no_shadow: np.ndarray, foreground: np.ndarray) -> np.ndarray:
    diff = np.max(np.abs(target.astype(np.float32) - no_shadow.astype(np.float32)), axis=-1) / 255.0
    return np.clip(diff * (1.0 - np.clip(foreground, 0.0, 1.0)), 0.0, 1.0)


def _dilate_mask(mask: np.ndarray, pixels: int = 5) -> np.ndarray:
    import cv2

    kernel = np.ones((pixels, pixels), dtype=np.uint8)
    return cv2.dilate((mask > 0.05).astype(np.uint8), kernel, iterations=1).astype(np.float32)


def generate_pairs(
    runtime,
    cameras: list[str],
    output_dir: str | Path,
    count: int = 12,
    seed: int = 42,
    hdri_roots: Iterable[str | Path] = ("assets/backgrounds/indoors", "assets/backgrounds/default"),
    spp: int = 64,
) -> dict[str, dict[str, str]]:
    rng = random.Random(seed)
    output = Path(output_dir)
    foreground_paths = runtime.foreground_prim_paths()
    hdris = _gather_hdris(hdri_roots)
    entries: dict[str, dict[str, str]] = {}

    for idx in range(min(count, len(cameras))):
        camera = cameras[idx]
        if hdris:
            runtime.set_dome_hdri(rng.choice(hdris), intensity=rng.uniform(400.0, 900.0), rotation_deg=rng.uniform(0.0, 360.0))
        runtime.set_shadows_enabled(True)
        runtime.set_path_tracing(spp > 1, spp=spp)

        target_frame = runtime.capture_frame(camera, rgb=True, segmentation=True)
        target = target_frame["rgb"]
        hard_fg_mask, mask_source = foreground_mask_with_fallback(
            target_frame["segmentation"], target_frame["segmentation_mapping"] or {}, foreground_paths
        )

        # Same camera, same lights, same materials. shadowLink excludes drop the
        # foreground from each light's shadow-caster collection.
        excludes = runtime.set_shadow_link_excludes(foreground_paths, enabled=True)
        runtime.set_path_tracing(spp > 1, spp=spp)
        degraded = runtime.capture_frame(camera, rgb=True)["rgb"]
        runtime.set_shadow_link_excludes(foreground_paths, enabled=False)

        diff = np.abs(target.astype(np.int16) - degraded.astype(np.int16)).astype(np.uint8)
        if float(np.mean(hard_fg_mask > 0.05)) >= 0.002:
            shadow_mask = _shadow_delta_mask(target, degraded, _dilate_mask(hard_fg_mask, pixels=7))
        else:
            shadow_mask = np.max(diff.astype(np.float32), axis=-1) / 255.0
            mask_source = "shadow_delta_no_foreground_mask"

        pair_dir = output / pair_id(idx)
        save_png(pair_dir / "shadow_diff.png", diff)
        save_png(pair_dir / "shadow_mask.png", np.repeat((shadow_mask * 255).astype(np.uint8)[..., None], 3, axis=-1))
        key = f"shadow_{pair_id(idx)}"
        entries[key] = write_pair(
            pair_dir,
            degraded,
            target,
            {
                "component": "shadow_simulation",
                "camera": camera,
                "shadow_link_excludes": excludes,
                "mask_source": mask_source,
                "shadow_mask_coverage": float(np.mean(shadow_mask > 0.03)),
                "foreground_prim_paths": foreground_paths,
            },
        )
    return entries
