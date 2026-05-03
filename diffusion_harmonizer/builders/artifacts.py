"""Post-hoc Novel-View Artifacts Correction builder.

Reads the per-env hemispheric views written by ``online._hemispheric_snapshot``
(``<env>/01_artifacts_correction/views/<NNNN>/{rgb.png, depth.npy,
intrinsics.json, extrinsics.json}``) and runs the four DIFIX3D+ /
DiffusionHarmonizer §3.2 strategies via ``gsplat_trainer``:

  * ``reference_full`` — all views, full iterations → clean target renders.
  * ``sparse_k``      — K of N views, full iterations → blurred / missing.
  * ``underfit``      — all views, fraction of iterations → spurious geom.
  * ``cycle``         — train on even views, render on odd.
  * ``cross_ref``     — train on first half, render on second half.

Pairs each degraded render with the reference render at the same view index
and writes paired-data dirs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from diffusion_harmonizer.components.common import pair_id
from diffusion_harmonizer.components.gsplat_trainer import (
    CapturedView,
    GSplatConfig,
    build_full_reference_strategy,
    default_degraded_strategies,
    run_strategy_suite,
)
from diffusion_harmonizer.image_io import save_json, write_pair


def _load_views(views_dir: Path) -> list[CapturedView]:
    import imageio.v3 as iio

    views: list[CapturedView] = []
    for view_dir in sorted(p for p in views_dir.iterdir() if p.is_dir()):
        rgb_path = view_dir / "rgb.png"
        K_path = view_dir / "intrinsics.json"
        E_path = view_dir / "extrinsics.json"
        if not (rgb_path.exists() and K_path.exists() and E_path.exists()):
            continue
        rgb = np.asarray(iio.imread(rgb_path))
        if rgb.ndim == 3 and rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        depth = None
        depth_path = view_dir / "depth.npy"
        if depth_path.exists():
            depth = np.load(depth_path)
        K = np.asarray(json.loads(K_path.read_text())["K"], dtype=np.float32)
        T = np.asarray(json.loads(E_path.read_text())["world_T_cam_gl"], dtype=np.float64)
        views.append(
            CapturedView(
                rgb=rgb.astype(np.uint8),
                depth=depth,
                intrinsics=K,
                world_T_cam=T,
                image_name=f"{int(view_dir.name):04d}.png",
            )
        )
    return views


def build(
    env_artifacts_dir: Path,
    output_dir: Path,
    count: int = 40,
    full_iterations: int = 30000,
    splat_kind: str = "3dgs",
    seed: int = 42,
    progress_cb=None,
) -> dict[str, dict[str, str]]:
    views_dir = Path(env_artifacts_dir) / "views"
    if not views_dir.exists():
        raise FileNotFoundError(
            f"No hemispheric views under {views_dir}. Run online with "
            f"--hemisphere-cameras > 0 first."
        )
    views = _load_views(views_dir)
    if not views:
        raise FileNotFoundError(f"No view subdirs found under {views_dir}.")
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = GSplatConfig(iterations=full_iterations, splat_kind=splat_kind, seed=seed)
    reference = build_full_reference_strategy(len(views), full_iters=full_iterations)
    degraded_strategies = default_degraded_strategies(len(views), full_iters=full_iterations)

    print(f"[builder:artifacts] running gsplat on {len(views)} views, full_iters={full_iterations}", flush=True)
    artifacts = run_strategy_suite(
        views,
        cfg,
        strategies=[reference] + degraded_strategies,
        output_root=output_dir / "renders",
        progress_cb=progress_cb,
    )
    ref_artifacts = artifacts[reference.name]
    deg_artifacts = [artifacts[s.name] for s in degraded_strategies if s.name in artifacts]

    save_json(
        output_dir / "strategies.json",
        {
            "reference": {"name": ref_artifacts.name, "iterations": ref_artifacts.iterations},
            "degraded": [
                {
                    "name": s.name,
                    "iterations": s.iterations,
                    "train_view_indices": s.train_view_indices,
                    "render_view_indices": s.render_view_indices,
                }
                for s in deg_artifacts
            ],
        },
    )

    entries: dict[str, dict[str, str]] = {}
    pair_index = 0
    for strategy in deg_artifacts:
        for view_idx, degraded_image in strategy.renders.items():
            if pair_index >= count:
                break
            target_image = ref_artifacts.renders.get(view_idx)
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
                },
            )
            pair_index += 1
        if pair_index >= count:
            break

    (output_dir / "pairs.json").write_text(json.dumps(entries, indent=2))
    print(f"[builder:artifacts] wrote {len(entries)} pairs -> {output_dir}", flush=True)
    return entries
