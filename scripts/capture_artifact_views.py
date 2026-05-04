# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# isort: skip_file
"""Hemispheric multi-view capture for the offline gsplat artifacts builder.

Independent from ``scripts/online_harmonizer_pairs.py``: that script does
trajectory-driven ISP + shadow with one external + one wrist camera; this
script teleports an env-side TiledCamera around the workspace 120 times
to feed the offline gsplat builder. Splitting the two avoids the env-camera
pose drift the user observed when they ran in the same session.

Usage::

    PYTHONPATH=. python scripts/capture_artifact_views.py \
        --task RubiksCubeAndBananaTask \
        --cameras 120 --radius-range 0.8 1.3 --spp 16

Output:
    data/diffusion_harmonizer/<env>/01_artifacts_correction/
        views/<NNNN>/{rgb.png, depth.npy, intrinsics.json, extrinsics.json}
        manifest.json

Then run ``scripts/build_artifacts_pairs.py`` to produce paired data via the
four DIFIX3D+ gsplat strategies.
"""

import argparse
import cv2  # noqa: F401  - import before isaaclab
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", nargs="*", default=None)
parser.add_argument("--env", nargs="*", default=None)
parser.add_argument("--tag", nargs="*", default=None)
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--output-root", default="data/diffusion_harmonizer")
parser.add_argument("--cameras", type=int, default=120,
                    help="Number of hemispheric views to render per env.")
parser.add_argument("--radius-range", type=float, nargs=2, default=(0.8, 1.3),
                    help="Per-view radius is sampled uniformly from this range.")
parser.add_argument("--center", type=float, nargs=3, default=(0.4, 0.0, 0.4))
parser.add_argument("--resolution", type=int, nargs=2, default=(512, 512))
parser.add_argument("--spp", type=int, default=16,
                    help="Path-tracing samples per pixel.")
parser.add_argument("--no-path-tracing", dest="use_path_tracing", action="store_false", default=True)
parser.add_argument("--settle-steps", type=int, default=0,
                    help="Random sample-action steps before snapshot. 0 (default) snapshots from "
                         "post-reset pose, matching the trajectory captures.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--physx-buffer-scale", type=float, default=0.1)
parser.add_argument("--no-headless", dest="headless_override", action="store_false", default=True)
parser.add_argument("--marble-scene", default=None,
                    help="Pin a specific marble USD as the BG (skips random pick across "
                         "marble_scene_roots). Use this to avoid landing on assets with broken "
                         "NuRec field references like marble3.usda. Recommended: "
                         "assets/scenes/marble/marblekitchen.usda")

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.headless = bool(args_cli.headless_override)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from diffusion_harmonizer.artifact_capture import ArtifactCaptureConfig, run_artifact_capture  # noqa: E402
from robolab.core.environments.factory import get_envs  # noqa: E402
from robolab.registrations.droid_jointpos.auto_env_registrations import auto_register_droid_envs  # noqa: E402

auto_register_droid_envs()


def _resolve_env_names(args) -> list[str]:
    if args.env:
        return list(args.env)
    if args.task:
        return list(get_envs(task=args.task))
    if args.tag:
        return list(get_envs(tag=args.tag))
    return list(get_envs())


def main() -> None:
    env_names = _resolve_env_names(args_cli)
    if args_cli.limit is not None:
        env_names = env_names[: args_cli.limit]
    if not env_names:
        raise SystemExit("No envs matched the filter; check --task / --env / --tag.")

    cfg = ArtifactCaptureConfig(
        output_root=Path(args_cli.output_root),
        cameras=args_cli.cameras,
        radius_range=tuple(args_cli.radius_range),
        center=tuple(args_cli.center),
        resolution=tuple(args_cli.resolution),
        spp=args_cli.spp,
        settle_steps=args_cli.settle_steps,
        seed=args_cli.seed,
        device=getattr(args_cli, "device", "cuda:0"),
        num_envs=args_cli.num_envs,
        physx_buffer_scale=args_cli.physx_buffer_scale,
        use_path_tracing=args_cli.use_path_tracing,
        marble_scene=Path(args_cli.marble_scene) if args_cli.marble_scene else None,
    )
    print(f"[artifact] capturing {len(env_names)} env(s) -> {cfg.output_root}", flush=True)
    summary = run_artifact_capture(env_names, cfg)
    total = sum(env.get("num_views", 0) for env in summary["envs"].values())
    print(f"[artifact] wrote {cfg.output_root / 'artifact_summary.json'} ({total} views total)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Terminated with error: {exc}")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
    simulation_app.close()
