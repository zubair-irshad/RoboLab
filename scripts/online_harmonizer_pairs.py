# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# isort: skip_file
"""Online (per-trajectory) DiffusionHarmonizer pair generator.

Modeled directly on ``examples/demo/run_empty.py``. Reuses the env's own
camera (``over_shoulder_left_camera`` / ``external_cam`` / etc.) — no extra
Replicator products on the stage. For each task: reset -> step N times with
random sample-action commands; at K evenly-spaced steps capture an ISP pair
(software-ISP + visibility-difference mask + Eq. 3 composite) and a shadow
pair (target vs UsdLux.shadowLink-excluded re-render).

Usage::

    PYTHONPATH=. python scripts/online_harmonizer_pairs.py \
        --task RubiksCubeAndBananaTask \
        --num-episodes 2 --num-steps 30 --captures-per-episode 4
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
parser.add_argument("--num-episodes", type=int, default=3)
parser.add_argument("--num-steps", type=int, default=60,
                    help="Sample-action steps per episode. Larger = more robot motion across captures.")
parser.add_argument("--captures-per-episode", type=int, default=4,
                    help="How many evenly-spaced steps per episode produce paired data (2-5 recommended).")
parser.add_argument("--cameras", nargs="+", default=None,
                    help="Subset of camera names to capture from (e.g. --cameras external_cam wrist_cam). "
                         "Default auto-picks every TiledCamera the env exposes.")
parser.add_argument("--action-hold-steps", type=int, default=10,
                    help="Hold each sampled action this many physics steps so the PD controller actually tracks it. "
                         "0 or 1 = re-sample every step (robot chases unreachable targets and barely moves).")
parser.add_argument("--isp-strength", type=float, default=0.25,
                    help="ISP perturbation magnitude (0 = identity, 1 = aggressive). Paper-faithful ~0.25 produces "
                         "subtle tone mismatch (object identity preserved); >0.5 starts swapping object color.")
parser.add_argument("--spp", type=int, default=8,
                    help="Path-tracing samples per pixel during capture. Lower = faster but noisier shadows.")
parser.add_argument("--no-path-tracing", dest="use_path_tracing", action="store_false", default=True,
                    help="Skip path tracing during capture. Shadows will not visibly toggle in the rasterizer.")
parser.add_argument("--sun-intensity-range", type=float, nargs=2, default=(1500.0, 4000.0))
parser.add_argument("--sun-angle-deg-range", type=float, nargs=2, default=(1.0, 6.0))
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--physx-buffer-scale", type=float, default=0.1)
parser.add_argument("--isp-full-frame-fraction", type=float, default=0.0)
parser.add_argument("--shadow-min-coverage", type=float, default=0.0008)
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
        num_episodes=args_cli.num_episodes,
        num_steps_per_episode=args_cli.num_steps,
        captures_per_episode=args_cli.captures_per_episode,
        seed=args_cli.seed,
        device=getattr(args_cli, "device", "cuda:0"),
        num_envs=args_cli.num_envs,
        physx_buffer_scale=args_cli.physx_buffer_scale,
        components=tuple(args_cli.components),
        isp_full_frame_fraction=args_cli.isp_full_frame_fraction,
        isp_strength=args_cli.isp_strength,
        shadow_min_coverage=args_cli.shadow_min_coverage,
        cameras=tuple(args_cli.cameras) if args_cli.cameras else None,
        action_hold_steps=args_cli.action_hold_steps,
        sun_intensity_range=tuple(args_cli.sun_intensity_range),
        sun_angle_deg_range=tuple(args_cli.sun_angle_deg_range),
        use_path_tracing=args_cli.use_path_tracing,
        spp=args_cli.spp,
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
