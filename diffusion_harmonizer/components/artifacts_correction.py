"""Novel-View Artifacts Correction (DiffusionHarmonizer §3.2 / DIFIX3D+ §3.2).

Privileged 100-Fibonacci-sphere capture from the live RoboLab env stage feeds
an in-process gsplat trainer that produces:

  * a clean reference run (all views, full iterations) — supplies ``target``
  * four handicapped runs — sparse-K / underfit / cycle / cross-ref —
    each providing degraded ``input`` images at matched view indices.

This component does not author any USD; it only reads rgb+depth+camera matrices
from the runtime's spherical Replicator products.
"""

from __future__ import annotations

import time
from pathlib import Path

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
from diffusion_harmonizer.image_io import save_json, save_png, write_pair


def capture_sphere_views(runtime, cameras: list[str], spp: int = 64, verbose: bool = True) -> list[CapturedView]:
    runtime.set_path_tracing(spp > 1, spp=spp)
    if verbose:
        print(f"[artifacts] path-tracing spp={spp}; rendering {len(cameras)} sphere views", flush=True)
    views: list[CapturedView] = []
    t0 = time.time()
    for idx, name in enumerate(cameras):
        frame = runtime.capture_frame(name, rgb=True, depth=True)
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
            eta = (len(cameras) - (idx + 1)) / max(rate, 1e-3)
            print(f"[artifacts] {idx + 1}/{len(cameras)} frames ({rate:.2f} fps, eta {eta:.1f}s)", flush=True)
    if verbose:
        print(f"[artifacts] capture done in {time.time() - t0:.1f}s", flush=True)
    return views


def export_capture_dataset(views: list[CapturedView], output_dir: str | Path) -> Path:
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
            pair_dir = output_dir / pair_id(pair_index)
            entries[f"artifacts_{pair_id(pair_index)}"] = write_pair(
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
    runtime,
    cameras: list[str],
    output_dir: str | Path,
    count: int = 40,
    full_iterations: int = 30000,
    splat_kind: str = "3dgs",
    seed: int = 42,
    spp: int = 64,
    captured_views: list[CapturedView] | None = None,
    verbose: bool = True,
) -> dict[str, dict[str, str]]:
    output = Path(output_dir)
    if captured_views is None:
        captured_views = capture_sphere_views(runtime, cameras, spp=spp, verbose=verbose)
    if verbose:
        print(f"[artifacts] exporting privileged {len(captured_views)}-view capture", flush=True)
    export_capture_dataset(captured_views, output / "privileged_capture")

    cfg = GSplatConfig(iterations=full_iterations, splat_kind=splat_kind, seed=seed)
    reference_strategy = build_full_reference_strategy(len(captured_views), full_iters=full_iterations)
    degraded_strategies = default_degraded_strategies(len(captured_views), full_iters=full_iterations)

    progress_cb = None
    if verbose:
        def progress_cb(strategy_name, step, loss):
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
