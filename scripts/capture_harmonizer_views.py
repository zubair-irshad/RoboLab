# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# isort: skip_file
"""Capture phase only - run Isaac Sim, dump every variant the post-hoc builders need.

For each requested env this script captures, per spherical camera, up to four
rgb variants plus mask + depth + camera matrices, and writes them under
``data/captures/<env>/sphere/<NNNN>/``. Once finished, Isaac Sim shuts down and
``scripts/build_harmonizer_pairs.py`` consumes the disk capture without ever
needing the simulator again.

Usage::

    PYTHONPATH=. python scripts/capture_harmonizer_views.py --task RubiksCubeAndBananaTask \
        --num-sphere-cameras 30 --spp 4 --variants target no_shadow_fg
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
parser.add_argument("--output-root", default="data/captures")
parser.add_argument("--num-sphere-cameras", type=int, default=60)
parser.add_argument("--sphere-radius", type=float, default=1.6)
parser.add_argument("--sphere-center", type=float, nargs=3, default=(0.4, 0.0, 0.4))
parser.add_argument("--capture-resolution", type=int, nargs=2, default=(512, 512))
parser.add_argument("--spp", type=int, default=16)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--physx-buffer-scale", type=float, default=0.1)
parser.add_argument("--variants", nargs="+", default=["target", "no_shadow_fg", "bg_only", "fg_only"],
                    help="Subset of variants to render. ISP/relighting only need target+mask; "
                         "shadow needs target+no_shadow_fg; reinsertion needs target+bg_only+fg_only.")
parser.add_argument("--no-headless", dest="headless_override", action="store_false", default=True)

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.headless = bool(args_cli.headless_override)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from diffusion_harmonizer.capture import CaptureConfig, capture_envs  # noqa: E402
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

    cfg = CaptureConfig(
        output_root=Path(args_cli.output_root),
        num_sphere_cameras=args_cli.num_sphere_cameras,
        sphere_radius=args_cli.sphere_radius,
        sphere_center=tuple(args_cli.sphere_center),
        capture_resolution=tuple(args_cli.capture_resolution),
        spp=args_cli.spp,
        seed=args_cli.seed,
        device=getattr(args_cli, "device", "cuda:0"),
        num_envs=args_cli.num_envs,
        physx_buffer_scale=args_cli.physx_buffer_scale,
        variants=tuple(args_cli.variants),
    )
    print(f"[capture] capturing {len(env_names)} env(s) -> {cfg.output_root}", flush=True)
    summary = capture_envs(env_names, cfg)
    print(f"[capture] wrote summary to {cfg.output_root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Terminated with error: {exc}")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
    simulation_app.close()
