"""Asset Re-Insertion (DiffusionHarmonizer §3.2).

Reconstruct the *static background* (table / fixtures / dome) with a clean
3DGS using the privileged 100-camera capture taken with the foreground hidden,
then composite the foreground-only render (no cast shadow, no ISP harmonization)
back over the background gsplat render. This is the **degraded** image. The
matching **target** is the full path-traced render with foreground present and
shadows + harmonization on.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from diffusion_harmonizer.components.common import feather_mask, foreground_mask_with_fallback, pair_id
from diffusion_harmonizer.components.gsplat_trainer import (
    CapturedView,
    GSplatConfig,
    build_full_reference_strategy,
    run_strategy_suite,
)
from diffusion_harmonizer.image_io import save_json, save_png, write_pair


def capture_background_views(
    runtime,
    cameras: list[str],
    foreground_paths: list[str],
    spp: int = 64,
) -> list[CapturedView]:
    """Hide the foreground, path-trace each sphere camera, restore visibility."""

    runtime.set_path_tracing(spp > 1, spp=spp)
    runtime.set_prims_visibility(foreground_paths, False)
    try:
        frames = runtime.capture_multiview(cameras, rgb=True, depth=True)
    finally:
        runtime.set_prims_visibility(foreground_paths, True)
    return [
        CapturedView(
            rgb=frame["rgb"],
            depth=frame.get("depth"),
            intrinsics=np.asarray(frame["camera_intrinsics"], dtype=np.float32),
            world_T_cam=np.asarray(frame["camera_extrinsics"], dtype=np.float32),
            image_name=f"{idx:04d}.png",
        )
        for idx, frame in enumerate(frames)
    ]


def generate_pairs(
    runtime,
    cameras: list[str],
    output_dir: str | Path,
    count: int = 12,
    full_iterations: int = 30000,
    splat_kind: str = "3dgs",
    seed: int = 42,
    spp: int = 64,
    progress_cb=None,
) -> dict[str, dict[str, str]]:
    output = Path(output_dir)
    foreground_paths = runtime.foreground_prim_paths()

    bg_views = capture_background_views(runtime, cameras, foreground_paths, spp=spp)
    cfg = GSplatConfig(iterations=full_iterations, splat_kind=splat_kind, seed=seed)
    bg_reference = build_full_reference_strategy(len(bg_views), full_iters=full_iterations)
    bg_artifacts = run_strategy_suite(
        bg_views,
        cfg,
        strategies=[bg_reference],
        output_root=output / "background_renders",
        progress_cb=progress_cb,
    )
    bg_renders = bg_artifacts[bg_reference.name].renders

    entries: dict[str, dict[str, str]] = {}
    for pair_index, camera_name in enumerate(cameras[:count]):
        runtime.set_shadows_enabled(True)
        runtime.set_path_tracing(spp > 1, spp=spp)
        target_frame = runtime.capture_frame(camera_name, rgb=True, segmentation=True)
        target = target_frame["rgb"]
        mask, mask_source = foreground_mask_with_fallback(
            target_frame["segmentation"], target_frame["segmentation_mapping"] or {}, foreground_paths
        )
        mask = feather_mask(mask, sigma=2.0)
        if float(np.mean(mask > 0.05)) < 0.002:
            continue

        runtime.set_shadows_enabled(False)
        fg_frame = runtime.capture_frame(camera_name, rgb=True)
        runtime.set_shadows_enabled(True)
        fg = fg_frame["rgb"]

        bg = bg_renders.get(pair_index)
        if bg is None or bg.shape[:2] != fg.shape[:2]:
            continue
        composite = (mask[..., None] * fg.astype(np.float32) + (1.0 - mask[..., None]) * bg.astype(np.float32)).astype(np.uint8)
        pair_dir = output / pair_id(pair_index)
        save_png(pair_dir / "bg_render.png", bg)
        save_png(pair_dir / "fg_only.png", fg)
        key = f"asset_reinsertion_{pair_id(pair_index)}"
        entries[key] = write_pair(
            pair_dir,
            composite,
            target,
            {
                "component": "asset_reinsertion",
                "camera": camera_name,
                "view_index": pair_index,
                "splat_kind": splat_kind,
                "iterations": full_iterations,
                "mask_source": mask_source,
                "foreground_prim_paths": foreground_paths,
            },
            mask=mask,
        )
    save_json(output / "pairs.json", entries)
    return entries
