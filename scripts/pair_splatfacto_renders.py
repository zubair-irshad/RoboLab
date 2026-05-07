# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Pair splatfacto rendered views into DiffusionHarmonizer artifacts-correction data.

Reads renders from each strategy's ``ns-render dataset --output-path ...``
output and emits paired-data dirs of the form::

    <output-dir>/<NNNN>/{input.png, target.png, comparison.png, metadata.json}

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
import random
import re
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
                        help="Reference (clean) strategy name; defaults to 'full'. "
                             "Only used when --target-source=reference.")
    parser.add_argument("--renders-subdir", default="renders",
                        help="Sub-directory under each strategy that holds ns-render output PNGs.")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Cap total pairs across all strategies (per env).")
    parser.add_argument("--sample-pairs", type=int, default=None,
                        help="Randomly sample this many matched render stems per strategy before writing pairs.")
    parser.add_argument("--sample-seed", type=int, default=42,
                        help="Seed for --sample-pairs.")
    parser.add_argument(
        "--target-source", choices=("reference", "gt"), default="gt",
        help="Where the 'target' (clean) image comes from. "
             "'gt' (default): the captured ground-truth RGB from the artifact-views "
             "directory — the right choice for synthetic data, where we actually "
             "have the photo-real reference. The diffusion model learns 'degraded "
             "GS render → real photo'. "
             "'reference': use the 'full' strategy's GS render as the target — "
             "the diffusion model only learns 'less data → more data', leaving "
             "any GS-vs-photo gap unfixed. Use this if you don't trust the GT "
             "capture (mismatched intrinsics, exposure drift, etc.).",
    )
    parser.add_argument(
        "--gt-views-dir", type=Path, default=None,
        help="When --target-source=gt, where to find the ground-truth views. "
             "Default: <ns-root>/../views (the layout capture_artifact_views.py "
             "and render_dl3dv_with_robot.py --artifact-output-root produce). "
             "Each view is expected at <gt-views-dir>/<NNNN>/rgb.png; the view "
             "id is parsed from the render's stem (digits at the end).",
    )
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


def _gt_path_for_stem(gt_views_dir: Path, stem: str) -> Path | None:
    """Resolve the GT rgb for a render whose filename stem ends in digits.

    Renders from FastGS / nerfstudio carry stems like ``frame_0007`` or
    plain ``0007``. The GT layout (capture_artifact_views.py) stores
    them at ``<views>/<NNNN>/rgb.png``. We strip everything but the
    trailing digits to recover the view id.
    """
    match = re.search(r"(\d+)$", stem)
    if match is None:
        return None
    view_id = int(match.group(1))
    p = gt_views_dir / f"{view_id:04d}" / "rgb.png"
    return p if p.exists() else None


def main() -> None:
    args = parse_args()

    target_mode = args.target_source
    ref_renders: dict = {}
    gt_views_dir: Path | None = None

    if target_mode == "reference":
        ref_dir = args.ns_root / args.reference
        if not (ref_dir / args.renders_subdir).exists():
            raise SystemExit(
                f"Reference renders not found under {ref_dir / args.renders_subdir}. "
                f"Run `ns-render dataset --load-config ... --output-path "
                f"{ref_dir / args.renders_subdir}` first, or pass "
                f"--target-source=gt to use the captured GT instead."
            )
        ref_renders = _index_renders(ref_dir, args.renders_subdir)
        if not ref_renders:
            raise SystemExit(f"No PNGs under {ref_dir / args.renders_subdir}.")
    else:  # gt
        gt_views_dir = (
            args.gt_views_dir.resolve() if args.gt_views_dir is not None
            else (args.ns_root.parent / "views").resolve()
        )
        if not gt_views_dir.is_dir():
            raise SystemExit(
                f"--target-source=gt but GT views dir {gt_views_dir} doesn't exist. "
                f"Pass --gt-views-dir <path> or rerun the capture step (which "
                f"writes <env>/01_artifacts_correction/views/<NNNN>/rgb.png)."
            )
        # Quick existence smoke-check.
        n_gt = sum(1 for d in gt_views_dir.iterdir() if (d / "rgb.png").exists())
        print(f"[pair] target=gt; using {n_gt} GT views under {gt_views_dir}", flush=True)

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

        if target_mode == "reference":
            matched_stems = sorted(set(ref_renders.keys()) & set(deg_renders.keys()))
        else:
            matched_stems = sorted(
                stem for stem in deg_renders.keys()
                if _gt_path_for_stem(gt_views_dir, stem) is not None
            )
        if args.sample_pairs is not None and len(matched_stems) > args.sample_pairs:
            matched_stems = sorted(random.Random(args.sample_seed).sample(matched_stems, args.sample_pairs))
        print(f"[pair] {strategy}: {len(matched_stems)} matched view(s) "
              f"({'reference' if target_mode == 'reference' else 'gt'} target)",
              flush=True)
        for stem in matched_stems:
            if args.max_pairs is not None and pair_index >= args.max_pairs:
                break
            input_path = deg_renders[stem]
            if target_mode == "reference":
                target_path = ref_renders[stem]
                target_kind = f"reference:{args.reference}"
            else:
                target_path = _gt_path_for_stem(gt_views_dir, stem)
                target_kind = "gt"
            pair_dir = args.output_dir / f"{pair_index:04d}"
            pair_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_path, pair_dir / "input.png")
            shutil.copy2(target_path, pair_dir / "target.png")
            _save_comparison(pair_dir, input_path, target_path)
            (pair_dir / "metadata.json").write_text(json.dumps(
                {
                    "component": "artifacts_correction",
                    "strategy": strategy,
                    "target_source": target_kind,
                    "view_stem": stem,
                    "input": str(input_path),
                    "target": str(target_path),
                },
                indent=2,
            ))
            summary["pairs"].append({
                "pair_id": f"{pair_index:04d}", "strategy": strategy,
                "stem": stem, "target_source": target_kind,
            })
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
