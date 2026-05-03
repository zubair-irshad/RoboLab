# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Offline gsplat builder for Novel-View Artifacts Correction.

Reads the per-env hemispheric snapshot written by ``online._hemispheric_snapshot``
(via ``scripts/online_harmonizer_pairs.py --hemisphere-cameras N``) and runs
the four DIFIX3D+ / DiffusionHarmonizer §3.2 strategies via gsplat:

  * ``reference_full`` — clean target renders (all views, full iterations)
  * ``sparse_k``      — train on K << N views (blurred details, missing regions)
  * ``underfit``      — train on all views but stop early (spurious geometry)
  * ``cycle``         — train on even views, render on odd (held-out interleave)
  * ``cross_ref``     — train on first half, render on second half (camera shift)

No Isaac Sim. Only torch + gsplat.

Usage::

    python scripts/build_artifacts_pairs.py \
        --output-root data/diffusion_harmonizer \
        --full-iterations 30000 \
        --pairs-per-env 40
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="data/diffusion_harmonizer",
                        help="Same directory online_harmonizer_pairs.py wrote into.")
    parser.add_argument("--env", nargs="*", default=None,
                        help="Restrict to specific env names; default = every env with a hemispheric snapshot.")
    parser.add_argument("--full-iterations", type=int, default=30000)
    parser.add_argument("--pairs-per-env", type=int, default=40)
    parser.add_argument("--splat-kind", choices=("3dgs", "2dgs"), default="3dgs")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _enumerate_envs(output_root: Path) -> list[Path]:
    return sorted(
        p / "01_artifacts_correction"
        for p in output_root.iterdir()
        if p.is_dir() and (p / "01_artifacts_correction" / "manifest.json").exists()
    )


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    if not output_root.exists():
        raise SystemExit(f"{output_root} not found. Run online_harmonizer_pairs.py with --hemisphere-cameras first.")

    artifacts_dirs = _enumerate_envs(output_root)
    if args.env:
        wanted = set(args.env)
        artifacts_dirs = [p for p in artifacts_dirs if p.parent.name in wanted]
    if not artifacts_dirs:
        raise SystemExit(
            f"No hemispheric snapshots found under {output_root}/<env>/01_artifacts_correction/. "
            "Run online with --hemisphere-cameras > 0 first."
        )

    from diffusion_harmonizer.builders import artifacts as artifacts_builder

    summary: dict = {"envs": {}}
    for art_dir in artifacts_dirs:
        env_name = art_dir.parent.name
        env_summary: dict = {}

        def _progress(name, step, loss, _env=env_name):
            print(f"[gsplat:{_env}:{name}] step {step:>6d}  loss {loss:.4f}", flush=True)

        try:
            entries = artifacts_builder.build(
                env_artifacts_dir=art_dir,
                output_dir=art_dir,
                count=args.pairs_per_env,
                full_iterations=args.full_iterations,
                splat_kind=args.splat_kind,
                seed=args.seed,
                progress_cb=_progress,
            )
            env_summary["num_pairs"] = len(entries)
        except Exception as exc:
            env_summary["error"] = str(exc)
            env_summary["traceback"] = traceback.format_exc()
            print(f"[builder:artifacts:{env_name}] FAILED: {exc}", flush=True)
            traceback.print_exc()

        summary["envs"][env_name] = env_summary
        (output_root / "artifacts_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"[builder] wrote {output_root / 'artifacts_summary.json'}")


if __name__ == "__main__":
    main()
