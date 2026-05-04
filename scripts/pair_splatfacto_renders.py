# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Pair splatfacto rendered views into DiffusionHarmonizer artifacts-correction data.

Reads renders from each strategy's ``ns-render dataset --output-path ...``
output and emits paired-data dirs of the form::

    <env>/01_artifacts_correction/<NNNN>/{input.png, target.png, comparison.png, metadata.json}

per (degraded_strategy, view_id) tuple. ``target`` always comes from the
reference run (full data, full iterations) so the diffusion model sees a
clean novel view; ``input`` comes from the degraded run for the matching
view id.

Expected layout under ``--ns-root`` (default
``<env>/01_artifacts_correction/nerfstudio/``)::

    full/renders/<image_name>.png            # reference renders
    sparse_k/renders/<image_name>.png        # sparse-strategy renders
    underfit/renders/<image_name>.png        # underfit-strategy renders

If you used ``ns-render dataset --output-path <strategy>/renders --split eval``
(or train), just point this script at ``<strategy>/renders/`` and it'll match
filenames. We pair by file *stem* — frame_0007.png in full pairs with
frame_0007.png in sparse_k.

Usage::

    python scripts/pair_splatfacto_renders.py \\
        --ns-root data/diffusion_harmonizer/RubiksCubeAndBananaTask/01_artifacts_correction/nerfstudio \\
        --output-dir data/diffusion_harmonizer/RubiksCubeAndBananaTask/01_artifacts_correction \\
        --strategies sparse_k underfit
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ns-root", required=True, type=Path,
                        help="Path to the per-env nerfstudio dir (contains full/, sparse_k/, ...).")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Where pair subdirs (0000/, 0001/, ...) are written.")
    parser.add_argument("--strategies", nargs="+", default=("sparse_arc", "underfit"),
                        help="Degraded strategies to pair with the reference 'full' renders. "
                             "sparse_arc is preferred for hemispheric captures (held-out views "
                             "are geometrically separated). sparse_k is a weak alternative.")
    parser.add_argument("--reference", default="full",
                        help="Reference (clean) strategy name; defaults to 'full'.")
    parser.add_argument("--renders-subdir", default="renders",
                        help="Sub-directory under each strategy that holds ns-render output PNGs.")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Cap total pairs across all strategies (per env).")
    return parser.parse_args()


def _index_renders(strategy_dir: Path, renders_subdir: str) -> dict[str, Path]:
    """Walk every PNG under ``<strategy>/<renders_subdir>`` and key by stem.

    ``ns-render dataset`` writes train and eval splits as separate subdirs
    (e.g. ``train/`` and ``eval/`` under the output path). We don't care
    which split a render came from — we just want every rendered viewpoint
    indexed by image stem so we can match across strategies.
    """

    root = strategy_dir / renders_subdir
    if not root.exists():
        return {}
    out: dict[str, Path] = {}
    for png in root.rglob("*.png"):
        out.setdefault(png.stem, png)
    return out


def _save_comparison(pair_dir: Path, input_path: Path, target_path: Path) -> None:
    import imageio.v3 as iio
    import numpy as np

    a = np.asarray(iio.imread(input_path))[..., :3]
    b = np.asarray(iio.imread(target_path))[..., :3]
    if a.shape != b.shape:
        return
    iio.imwrite(pair_dir / "comparison.png", np.concatenate([a, b], axis=1))


def main() -> None:
    args = parse_args()
    ref_dir = args.ns_root / args.reference
    if not (ref_dir / args.renders_subdir).exists():
        raise SystemExit(
            f"Reference renders not found under {ref_dir / args.renders_subdir}. "
            f"Run `ns-render dataset --load-config ... --output-path {ref_dir / args.renders_subdir}` first."
        )

    ref_renders = _index_renders(ref_dir, args.renders_subdir)
    if not ref_renders:
        raise SystemExit(f"No PNGs under {ref_dir / args.renders_subdir}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    import shutil

    pair_index = 0
    summary: dict = {"pairs": []}
    for strategy in args.strategies:
        strat_dir = args.ns_root / strategy
        deg_renders = _index_renders(strat_dir, args.renders_subdir)
        if not deg_renders:
            print(f"[pair] skipping {strategy} — no renders under {strat_dir / args.renders_subdir}", flush=True)
            continue

        matched_stems = sorted(set(ref_renders.keys()) & set(deg_renders.keys()))
        print(f"[pair] {strategy}: {len(matched_stems)} matched view(s) with reference", flush=True)
        for stem in matched_stems:
            if args.max_pairs is not None and pair_index >= args.max_pairs:
                break
            input_path = deg_renders[stem]
            target_path = ref_renders[stem]
            pair_dir = args.output_dir / f"{pair_index:04d}"
            pair_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_path, pair_dir / "input.png")
            shutil.copy2(target_path, pair_dir / "target.png")
            _save_comparison(pair_dir, input_path, target_path)
            (pair_dir / "metadata.json").write_text(json.dumps(
                {
                    "component": "artifacts_correction",
                    "strategy": strategy,
                    "view_stem": stem,
                    "input": str(input_path),
                    "target": str(target_path),
                },
                indent=2,
            ))
            summary["pairs"].append({"pair_id": f"{pair_index:04d}", "strategy": strategy, "stem": stem})
            pair_index += 1

    (args.output_dir / "pairs.json").write_text(json.dumps(summary, indent=2))
    print(f"[pair] wrote {pair_index} pairs -> {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Terminated with error: {exc}")
        traceback.print_exc()
        sys.exit(1)
