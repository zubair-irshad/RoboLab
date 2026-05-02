# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Generate DiffusionHarmonizer paired-data over RoboLab scenes.

Iterates ``assets/scenes/*.usda``, references each scene + the Franka robot +
an HDRI dome onto the same Isaac Sim stage, captures 100 spherical views, and
runs all five DiffusionHarmonizer §3.2 components in-process. Output is written
to ``data/diffusion_harmonizer/<scene_id>/`` with one sub-folder per component
plus ``INDEX.txt`` and ``scene_summary.json`` for navigation.

Usage examples::

    # All scenes, default settings (long).
    python scripts/generate_harmonizer_data.py

    # Smoke test on a single scene with reduced gsplat iters.
    python scripts/generate_harmonizer_data.py \
        --scene rubiks_cube_banana_bowl --full-iterations 5000 \
        --num-sphere-cameras 60

    # Restrict to specific components (e.g. skip the relighting sidecar).
    python scripts/generate_harmonizer_data.py \
        --components artifacts_correction shadow_simulation asset_reinsertion
"""

import argparse
from pathlib import Path

from diffusion_harmonizer import PipelineConfig, run_pipeline
from diffusion_harmonizer.scene_templates.robolab_loader import discover_scenes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes-root", default="assets/scenes")
    parser.add_argument("--backgrounds-root", default="assets/backgrounds")
    parser.add_argument("--robot-usd", default="assets/robots/franka_robotiq_2f_85_flattened.usd")
    parser.add_argument("--output-root", default="data/diffusion_harmonizer")
    parser.add_argument("--scene", nargs="*", default=None,
                        help="Subset of scene ids (USDA stems) to process; default = all.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N scenes.")
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
    parser.add_argument("--spp", type=int, default=64,
                        help="Samples-per-pixel for path-traced sphere capture. Lower (e.g. 8) for smoke tests.")
    parser.add_argument("--full-iterations", type=int, default=30000)
    parser.add_argument("--artifacts-pairs", type=int, default=40)
    parser.add_argument("--isp-pairs", type=int, default=12)
    parser.add_argument("--relighting-pairs", type=int, default=8)
    parser.add_argument("--shadow-pairs", type=int, default=12)
    parser.add_argument("--reinsertion-pairs", type=int, default=12)
    parser.add_argument("--relighting-command", default=None,
                        help="Sidecar command for the relighting diffusion model. Skip the component if omitted.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--renderer-type", choices=("raytraced", "pathtraced"), default="raytraced")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = discover_scenes(
        scenes_root=args.scenes_root,
        backgrounds_root=args.backgrounds_root,
        robot_usd=args.robot_usd,
        limit=args.limit,
    )
    if args.scene:
        wanted = set(args.scene)
        specs = [s for s in specs if s.scene_id in wanted]
    if not specs:
        raise SystemExit("No scenes matched the filter; check --scene and --scenes-root.")

    cfg = PipelineConfig(
        output_root=Path(args.output_root),
        num_sphere_cameras=args.num_sphere_cameras,
        sphere_radius=args.sphere_radius,
        sphere_center=tuple(args.sphere_center),
        capture_resolution=tuple(args.capture_resolution),
        splat_kind=args.splat_kind,
        spp=args.spp,
        full_iterations=args.full_iterations,
        artifacts_pairs_per_scene=args.artifacts_pairs,
        isp_pairs_per_scene=args.isp_pairs,
        relighting_pairs_per_scene=args.relighting_pairs,
        shadow_pairs_per_scene=args.shadow_pairs,
        reinsertion_pairs_per_scene=args.reinsertion_pairs,
        relighting_command=args.relighting_command,
        seed=args.seed,
        components=tuple(args.components),
    )
    print(f"Running DiffusionHarmonizer pipeline on {len(specs)} scene(s) -> {cfg.output_root}")
    summary = run_pipeline(cfg, scene_specs=specs, headless=args.headless, renderer_type=args.renderer_type)
    total_pairs = sum(
        info.get("num_pairs", 0)
        for scene in summary["scenes"].values()
        for info in scene["components"].values()
    )
    print(f"Wrote summary to {cfg.output_root / 'summary.json'} ({total_pairs} pairs across {len(summary['scenes'])} scenes)")


if __name__ == "__main__":
    main()
