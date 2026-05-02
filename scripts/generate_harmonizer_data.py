# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# isort: skip_file
"""Generate DiffusionHarmonizer paired-data over registered RoboLab tasks.

Resolves env names through ``robolab.core.environments.factory.get_envs`` and
runs the five paired-data components (artifacts / ISP / relighting / shadow /
asset re-insertion) in one Isaac Sim session. Outputs land in
``data/diffusion_harmonizer/<env_name>/<NN_component>/...``.

Usage::

    # smoke test - 1 env, low iterations / spp / view count
    python scripts/generate_harmonizer_data.py --task BananaInBowlTask \
        --components artifacts_correction --num-sphere-cameras 30 \
        --full-iterations 1000 --spp 4

    # full run on every registered env
    python scripts/generate_harmonizer_data.py
"""

import argparse
import cv2  # noqa: F401  - must import before isaaclab on some builds
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", nargs="*", default=None,
                    help="One or more task class names (e.g. BananaInBowlTask).")
parser.add_argument("--env", nargs="*", default=None,
                    help="Exact env names (variants). Use either --task or --env, not both.")
parser.add_argument("--tag", nargs="*", default=None,
                    help="Tag(s) to filter envs by.")
parser.add_argument("--limit", type=int, default=None, help="Cap the number of envs processed.")
parser.add_argument("--output-root", default="data/diffusion_harmonizer")
parser.add_argument("--components", nargs="+", default=[
    "artifacts_correction",
    "isp_modification",
    "relighting",
    "shadow_simulation",
    "asset_reinsertion",
])
parser.add_argument("--num-sphere-cameras", type=int, default=100)
parser.add_argument("--sphere-radius", type=float, default=1.6)
parser.add_argument("--sphere-center", type=float, nargs=3, default=(0.4, 0.0, 0.4))
parser.add_argument("--capture-resolution", type=int, nargs=2, default=(512, 512))
parser.add_argument("--splat-kind", choices=("3dgs", "2dgs"), default="3dgs")
parser.add_argument("--spp", type=int, default=64)
parser.add_argument("--full-iterations", type=int, default=30000)
parser.add_argument("--artifacts-pairs", type=int, default=40)
parser.add_argument("--isp-pairs", type=int, default=12)
parser.add_argument("--relighting-pairs", type=int, default=8)
parser.add_argument("--shadow-pairs", type=int, default=12)
parser.add_argument("--reinsertion-pairs", type=int, default=12)
parser.add_argument("--relighting-command", default=None,
                    help="Sidecar command for the relighting diffusion model. Skip the component if omitted.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--settle-steps", type=int, default=4)

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports below this point require the app to be running.
from diffusion_harmonizer import PipelineConfig, run_pipeline  # noqa: E402
from robolab.core.environments.factory import get_envs  # noqa: E402
from robolab.registrations.droid_jointpos.auto_env_registrations import auto_register_droid_envs  # noqa: E402

auto_register_droid_envs()


def _resolve_env_names(args) -> list[str]:
    if args.env:
        return list(args.env)
    if args.task:
        envs = get_envs(task=args.task)
    elif args.tag:
        envs = get_envs(tag=args.tag)
    else:
        envs = get_envs()
    return list(envs)


def main() -> None:
    env_names = _resolve_env_names(args_cli)
    if args_cli.limit is not None:
        env_names = env_names[: args_cli.limit]
    if not env_names:
        raise SystemExit("No envs matched the filter; check --task / --env / --tag.")

    cfg = PipelineConfig(
        output_root=Path(args_cli.output_root),
        num_sphere_cameras=args_cli.num_sphere_cameras,
        sphere_radius=args_cli.sphere_radius,
        sphere_center=tuple(args_cli.sphere_center),
        capture_resolution=tuple(args_cli.capture_resolution),
        splat_kind=args_cli.splat_kind,
        spp=args_cli.spp,
        full_iterations=args_cli.full_iterations,
        artifacts_pairs_per_env=args_cli.artifacts_pairs,
        isp_pairs_per_env=args_cli.isp_pairs,
        relighting_pairs_per_env=args_cli.relighting_pairs,
        shadow_pairs_per_env=args_cli.shadow_pairs,
        reinsertion_pairs_per_env=args_cli.reinsertion_pairs,
        relighting_command=args_cli.relighting_command,
        seed=args_cli.seed,
        settle_steps=args_cli.settle_steps,
        components=tuple(args_cli.components),
    )
    print(f"[harmonizer] running on {len(env_names)} env(s) -> {cfg.output_root}")
    summary = run_pipeline(cfg, env_names=env_names)
    total_pairs = sum(
        info.get("num_pairs", 0)
        for env in summary["envs"].values()
        for info in env.get("components", {}).values()
    )
    print(f"[harmonizer] wrote summary {cfg.output_root / 'summary.json'} "
          f"({total_pairs} pairs across {len(summary['envs'])} envs)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Terminated with error: {exc}")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
    simulation_app.close()
