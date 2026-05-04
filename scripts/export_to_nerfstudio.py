# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Convert hemispheric captures to nerfstudio dataparser format for splatfacto.

Per env, writes a tree under ``<env>/01_artifacts_correction/nerfstudio/``
with one subdirectory per DIFIX3D+ strategy plus a shared ``images/``
pool. Each strategy dir holds a ``transforms.json`` that nerfstudio's
nerfstudio_dataparser consumes directly.

Usage::

    # First capture views (Isaac Sim)
    PYTHONPATH=. python scripts/capture_artifact_views.py --task RubiksCubeAndBananaTask

    # Then export those captures to nerfstudio format (no Isaac Sim)
    python scripts/export_to_nerfstudio.py --output-root data/diffusion_harmonizer

    # Train splatfacto for each strategy. Reference (clean) first:
    ns-train splatfacto --max-num-iterations 30000 \\
        --data data/diffusion_harmonizer/<env>/01_artifacts_correction/nerfstudio/full

    # Each degraded variant uses the same command with a different --data dir
    # and (for ``underfit``) a smaller --max-num-iterations:
    ns-train splatfacto --max-num-iterations 1500 \\
        --data data/diffusion_harmonizer/<env>/01_artifacts_correction/nerfstudio/underfit

The script prints the exact ``ns-train`` and ``ns-render`` commands per
strategy after writing the data.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="data/diffusion_harmonizer")
    parser.add_argument("--env", nargs="*", default=None,
                        help="Restrict to specific env names; default = all envs with a captured snapshot.")
    parser.add_argument("--sparse-k", type=int, default=24,
                        help="Number of views the sparse_k strategy keeps.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-depth-init", dest="with_depth_init", action="store_false", default=True,
                        help="Skip the combined depth-back-projected PLY. Splatfacto will fall back "
                             "to random init, which converges much slower on object-centric captures.")
    parser.add_argument("--depth-stride", type=int, default=8,
                        help="Pixel stride when back-projecting depth to a point cloud (1 = every pixel).")
    parser.add_argument("--depth-target-points", type=int, default=200_000,
                        help="Subsample the combined point cloud to at most this many points.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from diffusion_harmonizer.nerfstudio_export import export_env

    output_root = Path(args.output_root)
    if not output_root.exists():
        raise SystemExit(f"{output_root} not found. Run capture_artifact_views.py first.")

    targets = []
    for env_dir in sorted(p for p in output_root.iterdir() if p.is_dir()):
        if args.env and env_dir.name not in args.env:
            continue
        artifacts_dir = env_dir / "01_artifacts_correction"
        if (artifacts_dir / "views").exists():
            targets.append(artifacts_dir)
    if not targets:
        raise SystemExit(
            f"No envs with hemispheric snapshots under {output_root}. "
            f"Run scripts/capture_artifact_views.py first."
        )

    summary: dict = {"envs": {}}
    for art_dir in targets:
        env_name = art_dir.parent.name
        try:
            ns_root, strategies = export_env(
                art_dir,
                sparse_k=args.sparse_k,
                seed=args.seed,
                with_depth_init=args.with_depth_init,
                depth_stride=args.depth_stride,
                depth_target_points=args.depth_target_points,
            )
        except Exception as exc:
            summary["envs"][env_name] = {"error": str(exc), "traceback": traceback.format_exc()}
            print(f"[export:{env_name}] FAILED: {exc}", flush=True)
            traceback.print_exc()
            continue

        env_summary = {"nerfstudio_root": str(ns_root), "strategies": []}
        print(f"\n[export:{env_name}] -> {ns_root}", flush=True)
        for s in strategies:
            env_summary["strategies"].append(
                {
                    "name": s.name,
                    "transforms": str(s.transforms_path),
                    "num_train": s.num_train,
                    "num_eval": s.num_eval,
                    "note": s.note,
                }
            )
            print(f"  [{s.name}] {s.note}", flush=True)
            print(f"    train: {s.num_train}  eval: {s.num_eval}", flush=True)
            data_dir = s.transforms_path.parent
            print(f"    ns-train {s.suggested_ns_train_args} --data {data_dir}", flush=True)
        summary["envs"][env_name] = env_summary

    (output_root / "nerfstudio_export_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[export] wrote {output_root / 'nerfstudio_export_summary.json'}")
    print(
        "Reminder: install nerfstudio in this venv first — "
        "https://docs.nerf.studio/quickstart/installation.html"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Terminated with error: {exc}")
        traceback.print_exc()
        sys.exit(1)
