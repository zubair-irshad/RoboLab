# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# isort: skip_file
"""Online (per-trajectory) DiffusionHarmonizer pair generator.

Modeled directly on ``examples/demo/run_empty.py``: launches Isaac Sim once,
auto-registers tasks, and for each requested env runs N episodes × M
sample-action steps. At every step we capture from 1-2 cameras and emit ISP
+ shadow paired-data — paper Eq. (3) for ISP, target-vs-shadowLink-excluded
delta for shadows.

Use this for ISP / shadow / relighting (the components that benefit from
trajectory diversity). Use ``capture_harmonizer_views.py`` for static
100-camera sphere captures feeding the offline gsplat builders (artifacts +
asset re-insertion).

Usage::

    PYTHONPATH=. python scripts/online_harmonizer_pairs.py \
        --task RubiksCubeAndBananaTask \
        --num-cameras 2 --num-episodes 3 --num-steps 30 --spp 4
"""

import argparse
import cv2  # noqa: F401  - must import before isaaclab on some builds
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
parser.add_argument("--components", nargs="+", default=["isp_modification", "shadow_simulation"],
                    choices=("isp_modification", "shadow_simulation"))
parser.add_argument("--num-cameras", type=int, default=2,
                    help="Orbit-ring cameras placed around the workspace.")
parser.add_argument("--camera-radius", type=float, default=1.6)
parser.add_argument("--camera-center", type=float, nargs=3, default=(0.4, 0.0, 0.4))
parser.add_argument("--camera-height", type=float, default=0.6)
parser.add_argument("--capture-resolution", type=int, nargs=2, default=(512, 512))
parser.add_argument("--spp", type=int, default=4)
parser.add_argument("--num-episodes", type=int, default=3)
parser.add_argument("--num-steps", type=int, default=30,
                    help="Sample-action steps per episode (matches run_empty.py default of 50).")
parser.add_argument("--capture-every-n-steps", type=int, default=1,
                    help="Skip-N-1 frames between captures to thin out highly correlated samples.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--physx-buffer-scale", type=float, default=0.1)
parser.add_argument("--isp-full-frame-fraction", type=float, default=0.2)
parser.add_argument("--isp-strength", type=float, default=0.8)
parser.add_argument("--shadow-min-coverage", type=float, default=0.001)
parser.add_argument("--no-headless", dest="headless_override", action="store_false", default=True)

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.headless = bool(args_cli.headless_override)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from diffusion_harmonizer.online import OnlineConfig, run_online  # noqa: E402
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

    cfg = OnlineConfig(
        output_root=Path(args_cli.output_root),
        num_cameras=args_cli.num_cameras,
        camera_radius=args_cli.camera_radius,
        camera_center=tuple(args_cli.camera_center),
        camera_height=args_cli.camera_height,
        capture_resolution=tuple(args_cli.capture_resolution),
        spp=args_cli.spp,
        num_episodes=args_cli.num_episodes,
        num_steps_per_episode=args_cli.num_steps,
        capture_every_n_steps=args_cli.capture_every_n_steps,
        seed=args_cli.seed,
        device=getattr(args_cli, "device", "cuda:0"),
        num_envs=args_cli.num_envs,
        physx_buffer_scale=args_cli.physx_buffer_scale,
        components=tuple(args_cli.components),
        isp_full_frame_fraction=args_cli.isp_full_frame_fraction,
        isp_strength=args_cli.isp_strength,
        shadow_min_coverage=args_cli.shadow_min_coverage,
    )
    print(f"[online] running on {len(env_names)} env(s) -> {cfg.output_root}", flush=True)
    summary = run_online(env_names, cfg)
    total_isp = sum(env.get("isp_pairs", 0) for env in summary["envs"].values())
    total_shadow = sum(env.get("shadow_pairs", 0) for env in summary["envs"].values())
    print(f"[online] wrote summary {cfg.output_root / 'summary.json'} "
          f"(isp={total_isp}, shadow={total_shadow})")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Terminated with error: {exc}")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
    simulation_app.close()
