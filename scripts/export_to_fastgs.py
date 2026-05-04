# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Convert hemispheric captures to vanilla-3DGS / FastGS COLMAP datasets.

Sibling to ``scripts/export_to_nerfstudio.py``. Re-uses the same captured
``views/<NNNN>/{rgb.png, intrinsics.json, extrinsics.json}`` and the
already-built ``nerfstudio/depth_init.ply`` (so this step is cheap — pure
JSON/PLY rewrites, no Isaac Sim).

Per env, writes::

    <env>/01_artifacts_correction/fastgs/
    ├── images/                          (shared rgb pool)
    ├── fastgs_manifest.json             (read by build_artifacts_via_fastgs.sh)
    ├── full/{train,render}/...          all views, 30k iter reference
    ├── underfit/{train,render}/...      all views, 1500 iter degraded
    └── sparse_arc/{train,render}/...    contiguous-arc holdout (40 train, 120 render)

Each ``train`` / ``render`` subdir is a self-contained COLMAP dataset
(``images/`` + ``sparse/0/{cameras,images,points3D}.txt``).

Usage::

    # First (already done): run capture_artifact_views.py + export_to_nerfstudio.py
    python scripts/export_to_fastgs.py --output-root data/diffusion_harmonizer

    # Then train + render (requires the ``fastgs`` conda env):
    bash scripts/build_artifacts_via_fastgs.sh UtensilsInMugTask
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", default="data/diffusion_harmonizer")
    parser.add_argument("--env", nargs="*", default=None,
                        help="Restrict to specific env names; default = all envs with a captured snapshot.")
    parser.add_argument("--full-iterations", type=int, default=30000)
    parser.add_argument("--underfit-iterations", type=int, default=1500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from diffusion_harmonizer.fastgs_export import export_env

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
            fastgs_root, strategies = export_env(
                art_dir,
                full_iterations=args.full_iterations,
                underfit_iterations=args.underfit_iterations,
            )
        except Exception as exc:
            summary["envs"][env_name] = {"error": str(exc), "traceback": traceback.format_exc()}
            print(f"[fastgs-export:{env_name}] FAILED: {exc}", flush=True)
            traceback.print_exc()
            continue

        env_summary = {"fastgs_root": str(fastgs_root), "strategies": []}
        print(f"\n[fastgs-export:{env_name}] -> {fastgs_root}", flush=True)
        for s in strategies:
            env_summary["strategies"].append(
                {
                    "name": s.name,
                    "train_source_path": str(s.train_source_path),
                    "render_source_path": str(s.render_source_path),
                    "num_train": s.num_train,
                    "num_render": s.num_render,
                    "iterations": s.iterations,
                    "note": s.note,
                }
            )
            print(f"  [{s.name}] {s.note}", flush=True)
            print(f"    train ({s.num_train}): {s.train_source_path}", flush=True)
            print(f"    render ({s.num_render}): {s.render_source_path}", flush=True)
            print(f"    iterations: {s.iterations}", flush=True)
        summary["envs"][env_name] = env_summary

    (output_root / "fastgs_export_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[fastgs-export] wrote {output_root / 'fastgs_export_summary.json'}")
    print(
        "Next: activate the FastGS conda env and run "
        "`bash scripts/build_artifacts_via_fastgs.sh <env_name>`."
    )


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"Terminated with error: {exc}")
        traceback.print_exc()
        sys.exit(1)
