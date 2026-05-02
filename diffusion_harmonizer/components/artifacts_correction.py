"""Novel-View Artifacts Correction (DiffusionHarmonizer §3.2 / DIFIX3D+ §3.2).

We capture the static scene from 100 Fibonacci-sphere cameras (privileged
multi-view), train a clean reference 3DGS once, then run four handicapped
trainers - sparse-K, deliberate-underfit, cycle-reconstruction, cross-reference -
to produce degraded renderings paired with the matching clean reference.

The entire trainer runs in-process via ``gsplat`` so no CUDA sidecar is needed.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from diffusion_harmonizer.components.common import pair_id
from diffusion_harmonizer.components.gsplat_trainer import (
    CapturedView,
    GSplatConfig,
    StrategyArtifacts,
    build_full_reference_strategy,
    default_degraded_strategies,
    run_strategy_suite,
)
from diffusion_harmonizer.image_io import save_png, save_json, write_pair


def capture_sphere_views(
    renderer,
    center: tuple[float, float, float] = (0.0, 0.0, 0.5),
    radius: float = 1.6,
    num_cameras: int = 100,
    resolution: tuple[int, int] = (512, 512),
    name_prefix: str = "harmonizer_sphere",
    spp: int = 64,
    verbose: bool = True,
) -> tuple[list[str], list[CapturedView]]:
    """Add 100 Fibonacci-sphere cameras and capture rgb+depth from each."""

    if verbose:
        print(f"[capture] adding {num_cameras} sphere cameras at radius={radius}, res={resolution}", flush=True)
    cameras = renderer.add_sphere_cameras(name_prefix, center=center, radius=radius, num_cameras=num_cameras, resolution=resolution)
    renderer.set_path_tracing(spp > 1, spp=spp)
    if verbose:
        print(f"[capture] path-tracing spp={spp}; rendering {num_cameras} frames...", flush=True)
    views: list[CapturedView] = []
    t0 = time.time()
    for idx, name in enumerate(cameras):
        frame = renderer.capture_frame(name, rgb=True, depth=True)
        views.append(
            CapturedView(
                rgb=frame["rgb"],
                depth=frame.get("depth"),
                intrinsics=np.asarray(frame["camera_intrinsics"], dtype=np.float32),
                world_T_cam=np.asarray(frame["camera_extrinsics"], dtype=np.float32),
                image_name=f"{idx:04d}.png",
            )
        )
        if verbose and (idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / max(elapsed, 1e-3)
            eta = (num_cameras - (idx + 1)) / max(rate, 1e-3)
            print(f"[capture] {idx + 1}/{num_cameras} frames ({rate:.2f} fps, eta {eta:.1f}s)", flush=True)
    if verbose:
        print(f"[capture] done in {time.time() - t0:.1f}s", flush=True)
    return cameras, views


def export_capture_dataset(views: list[CapturedView], output_dir: str | Path) -> Path:
    """Persist the privileged 100-view capture for inspection / external tools."""

    output = Path(output_dir)
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    cameras = []
    for idx, view in enumerate(views):
        save_png(images_dir / f"{idx:04d}.png", view.rgb)
        cameras.append(
            {
                "image": f"{idx:04d}.png",
                "width": int(view.rgb.shape[1]),
                "height": int(view.rgb.shape[0]),
                "intrinsics": view.intrinsics.tolist(),
                "world_T_cam": view.world_T_cam.tolist(),
            }
        )
    save_json(output / "cameras.json", {"cameras": cameras})
    return output


def write_artifact_pairs(
    output_dir: Path,
    reference: StrategyArtifacts,
    degraded: list[StrategyArtifacts],
    count: int,
) -> dict[str, dict[str, str]]:
    """Match each degraded render to the reference render at the same view."""

    output_dir.mkdir(parents=True, exist_ok=True)
    entries: dict[str, dict[str, str]] = {}
    pair_index = 0
    for strategy in degraded:
        for view_idx, degraded_image in strategy.renders.items():
            if pair_index >= count:
                return entries
            target_image = reference.renders.get(view_idx)
            if target_image is None:
                continue
            key = f"artifacts_{pair_id(pair_index)}"
            pair_dir = output_dir / pair_id(pair_index)
            entries[key] = write_pair(
                pair_dir,
                degraded_image,
                target_image,
                {
                    "component": "artifacts_correction",
                    "strategy": strategy.name,
                    "iterations": strategy.iterations,
                    "view_index": int(view_idx),
                    "train_view_indices": strategy.train_view_indices,
                    "render_view_indices": strategy.render_view_indices,
                },
            )
            pair_index += 1
    return entries


def generate_pairs(
    renderer,
    output_dir: str | Path = "data/artifacts_correction/demo",
    count: int = 50,
    sphere_center: tuple[float, float, float] = (0.0, 0.0, 0.5),
    sphere_radius: float = 1.6,
    num_cameras: int = 100,
    resolution: tuple[int, int] = (512, 512),
    full_iterations: int = 30000,
    splat_kind: str = "3dgs",
    seed: int = 42,
    spp: int = 64,
    captured_views: list[CapturedView] | None = None,
    verbose: bool = True,
) -> dict[str, dict[str, str]]:
    output = Path(output_dir)
    if captured_views is None:
        _, captured_views = capture_sphere_views(
            renderer,
            center=sphere_center,
            radius=sphere_radius,
            num_cameras=num_cameras,
            resolution=resolution,
            spp=spp,
            verbose=verbose,
        )
    if verbose:
        print(f"[artifacts] exporting privileged {len(captured_views)}-view capture", flush=True)
    export_capture_dataset(captured_views, output / "privileged_capture")

    cfg = GSplatConfig(iterations=full_iterations, splat_kind=splat_kind, seed=seed)
    reference_strategy = build_full_reference_strategy(len(captured_views), full_iters=full_iterations)
    degraded_strategies = default_degraded_strategies(len(captured_views), full_iters=full_iterations)

    progress_cb = None
    if verbose:
        def progress_cb(strategy_name: str, step: int, loss: float) -> None:
            print(f"[gsplat:{strategy_name}] step {step:>6d}  loss {loss:.4f}", flush=True)
    artifacts = run_strategy_suite(
        captured_views,
        cfg,
        strategies=[reference_strategy] + degraded_strategies,
        output_root=output / "renders",
        progress_cb=progress_cb,
    )
    reference = artifacts[reference_strategy.name]
    degraded = [artifacts[s.name] for s in degraded_strategies if s.name in artifacts]

    save_json(
        output / "strategies.json",
        {
            "reference": {"name": reference.name, "iterations": reference.iterations},
            "degraded": [
                {
                    "name": s.name,
                    "iterations": s.iterations,
                    "train_view_indices": s.train_view_indices,
                    "render_view_indices": s.render_view_indices,
                }
                for s in degraded
            ],
        },
    )
    return write_artifact_pairs(output, reference, degraded, count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/artifacts_correction/demo")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--full_iterations", type=int, default=30000)
    parser.add_argument("--splat_kind", choices=("3dgs", "2dgs"), default="3dgs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    from diffusion_harmonizer.rendering import launch_renderer

    renderer = launch_renderer(headless=True)
    try:
        entries = generate_pairs(
            renderer,
            args.output_dir,
            args.count,
            full_iterations=args.full_iterations,
            splat_kind=args.splat_kind,
            seed=args.seed,
        )
        Path(args.output_dir).joinpath("pairs.json").write_text(json.dumps(entries, indent=2))
    finally:
        renderer.shutdown()


if __name__ == "__main__":
    main()
