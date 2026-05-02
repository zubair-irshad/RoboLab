"""Asset Re-Insertion (DiffusionHarmonizer §3.2).

Reconstruct the *static background* (table + fixtures + dome) with a clean
3DGS, hide the inserted foreground, render the background-only image at every
sphere camera, then composite the foreground (no shadow / no harmonization)
back on top to create the *degraded* image. The full path-traced render with
foreground + cast shadows + matched lighting is the *target*.

Uses the in-process gsplat trainer (no CUDA sidecar).
"""

from __future__ import annotations

import argparse
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
    renderer,
    foreground_paths: list[str],
    sphere_center: tuple[float, float, float],
    sphere_radius: float,
    num_cameras: int,
    resolution: tuple[int, int],
) -> tuple[list[str], list[CapturedView]]:
    """Hide the foreground, capture sphere views of the receiver-only scene."""

    cameras = renderer.add_sphere_cameras("reinsertion_bg", center=sphere_center, radius=sphere_radius, num_cameras=num_cameras, resolution=resolution)
    renderer.set_path_tracing(True, spp=64)
    try:
        renderer.set_prims_visibility(foreground_paths, False)
        frames = renderer.capture_multiview(cameras, rgb=True, depth=True)
    finally:
        renderer.set_prims_visibility(foreground_paths, True)
    views = [
        CapturedView(
            rgb=frame["rgb"],
            depth=frame.get("depth"),
            intrinsics=np.asarray(frame["camera_intrinsics"], dtype=np.float32),
            world_T_cam=np.asarray(frame["camera_extrinsics"], dtype=np.float32),
            image_name=f"{idx:04d}.png",
        )
        for idx, frame in enumerate(frames)
    ]
    return cameras, views


def generate_pairs(
    renderer,
    foreground_paths: list[str],
    output_dir: str | Path = "data/asset_reinsertion/demo",
    count: int = 30,
    sphere_center: tuple[float, float, float] = (0.0, 0.0, 0.5),
    sphere_radius: float = 1.6,
    num_cameras: int = 100,
    resolution: tuple[int, int] = (512, 512),
    full_iterations: int = 30000,
    splat_kind: str = "3dgs",
    seed: int = 42,
) -> dict[str, dict[str, str]]:
    output = Path(output_dir)

    cameras, bg_views = capture_background_views(
        renderer,
        foreground_paths=foreground_paths,
        sphere_center=sphere_center,
        sphere_radius=sphere_radius,
        num_cameras=num_cameras,
        resolution=resolution,
    )
    cfg = GSplatConfig(iterations=full_iterations, splat_kind=splat_kind, seed=seed)
    bg_reference = build_full_reference_strategy(len(bg_views), full_iters=full_iterations)
    bg_artifacts = run_strategy_suite(bg_views, cfg, strategies=[bg_reference], output_root=output / "background_renders")
    bg_renders = bg_artifacts[bg_reference.name].renders

    entries: dict[str, dict[str, str]] = {}
    for pair_index, camera_name in enumerate(cameras[:count]):
        renderer.set_shadows_enabled(True)
        renderer.set_path_tracing(True, spp=64)
        target_frame = renderer.capture_frame(camera_name, rgb=True, segmentation=True)
        target = target_frame["rgb"]
        mask, mask_source = foreground_mask_with_fallback(
            target_frame["segmentation"], target_frame["segmentation_mapping"] or {}, foreground_paths
        )
        mask = feather_mask(mask, sigma=2.0)
        if float(np.mean(mask > 0.05)) < 0.002:
            continue

        # Capture foreground-only color (no cast shadows) at the same camera by
        # disabling shadows + hiding receivers; we keep the foreground prims lit
        # by the same dome light so albedo is consistent.
        renderer.set_shadows_enabled(False)
        fg_frame = renderer.capture_frame(camera_name, rgb=True)
        renderer.set_shadows_enabled(True)
        fg = fg_frame["rgb"]

        bg = bg_renders.get(pair_index)
        if bg is None:
            continue
        if bg.shape[:2] != fg.shape[:2]:
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
                "note": "Static background reconstructed by in-process gsplat; foreground composited without shadows or harmonization.",
            },
            mask=mask,
        )
    save_json(output / "pairs.json", entries)
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/asset_reinsertion/demo")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--full_iterations", type=int, default=30000)
    parser.add_argument("--splat_kind", choices=("3dgs", "2dgs"), default="3dgs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--foreground_paths", nargs="+", default=["/World/envs/env_0/robot", "/World/envs/env_0/scene"])
    args = parser.parse_args()
    from diffusion_harmonizer.rendering import launch_renderer

    renderer = launch_renderer(headless=True)
    try:
        entries = generate_pairs(
            renderer,
            foreground_paths=args.foreground_paths,
            output_dir=args.output_dir,
            count=args.count,
            full_iterations=args.full_iterations,
            splat_kind=args.splat_kind,
            seed=args.seed,
        )
        Path(args.output_dir).joinpath("pairs.json").write_text(json.dumps(entries, indent=2))
    finally:
        renderer.shutdown()


if __name__ == "__main__":
    main()
