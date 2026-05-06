# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Prepare a single DL3DV-Benchmark scene as a robot-environment background.

Pipeline (Phase 1 — scene prep only, no task wiring yet):

    1. download   — pull <hash>/colmaps/ from DL3DV/DL3DV-Benchmark
    2. reconstruct — run FastGS `fast-pgsr` train + mesh extract
    3. align      — gravity-align, floor-zero, metric rescale
    4. emit       — write aligned mesh + metadata.json for downstream phases

Each step is independently skippable so you can iterate without re-doing
the multi-hour reconstruction.

Example::

    python scripts/prepare_dl3dv_scene.py \
        --scene-hash 14eb48a50e37df548894ab6d8cd628a21dae14bbe6c462e894616fc5962e6c49 \
        --cache-dir data/dl3dv_backgrounds/cache \
        --output-dir data/dl3dv_backgrounds/scenes \
        --fastpgsr-repo third_party/FastGS-pgsr \
        --iterations 30000

Add ``--skip-train`` if you've already trained and just want to re-align,
or ``--skip-download`` if the scene is already cached.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python scripts/prepare_dl3dv_scene.py` from repo root without
# needing PYTHONPATH=. — mirrors other scripts in this directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robolab.scene_gen.dl3dv_backgrounds import (  # noqa: E402
    DL3DVSceneRef,
    FastPgsrConfig,
    align_scene_from_mesh,
    download_scene,
    run_fast_pgsr,
)
from robolab.scene_gen.dl3dv_backgrounds.align import write_aligned_mesh  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene-hash", required=True, help="DL3DV scene hash (64-char hex)")
    p.add_argument("--cache-dir", type=Path, default=Path("data/dl3dv_backgrounds/cache"))
    p.add_argument("--output-dir", type=Path, default=Path("data/dl3dv_backgrounds/scenes"))
    p.add_argument(
        "--fastpgsr-repo",
        type=Path,
        default=Path("third_party/FastGS-pgsr"),
        help="local clone of FastGS checked out to the fast-pgsr branch",
    )
    p.add_argument("--conda-env", default="fast-pgsr")
    p.add_argument("--iterations", type=int, default=30000)
    p.add_argument(
        "--metric-scale-hint",
        type=float,
        default=None,
        help="multiplier from COLMAP units to metres; if omitted, derived "
             "from camera-height heuristic (median cam = 1.5 m)",
    )
    p.add_argument("--target-camera-height-m", type=float, default=1.5)

    p.add_argument(
        "--tsdf-voxel-size", type=float, default=0.02,
        help="TSDF voxel size in metres (default 0.02 = 2cm for room-scale).",
    )
    p.add_argument(
        "--tsdf-max-depth", type=float, default=6.0,
        help="TSDF depth cutoff in metres (default 6m).",
    )

    p.add_argument(
        "--images-variant",
        choices=("images", "images_2", "images_4", "images_8"),
        default="images_4",
        help="which image resolution to pull. cameras in sparse/ are "
             "calibrated for full-res images/; we auto-rescale to match.",
    )
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-mesh", action="store_true")
    p.add_argument("--skip-align", action="store_true")
    p.add_argument(
        "--existing-mesh",
        type=Path,
        default=None,
        help="skip reconstruct and align this mesh (for re-aligning with new hints)",
    )
    return p.parse_args()


def _resolve_scene_ref(args: argparse.Namespace) -> DL3DVSceneRef:
    """Return a SceneRef whether or not we just downloaded it."""
    if args.skip_download:
        scene_root = args.cache_dir.resolve() / args.scene_hash
        if not scene_root.is_dir():
            raise FileNotFoundError(
                f"--skip-download but no cached scene at {scene_root}"
            )
        # Even when skipping the network fetch, re-run the local fixups
        # (images symlink + camera rescale) so an incomplete prior cache
        # gets normalized.
        from robolab.scene_gen.dl3dv_backgrounds.download import (
            _autorescale_cameras_to_actual_image_dim,
            _resolve_colmap_source,
        )
        gs = scene_root / "gaussian_splat"
        if gs.is_dir() and args.images_variant != "images":
            target = gs / "images"
            variant_dir = gs / args.images_variant
            if variant_dir.is_dir():
                if target.is_symlink() and target.readlink().name != args.images_variant:
                    target.unlink()
                if not target.exists():
                    target.symlink_to(args.images_variant)
                    print(f"[prepare] symlinked {target} -> {args.images_variant}")
        if gs.is_dir():
            _autorescale_cameras_to_actual_image_dim(gs)
        colmap = _resolve_colmap_source(scene_root)
        ref = DL3DVSceneRef(args.scene_hash, scene_root, colmap)
        ref.assert_colmap_layout()
        return ref
    return download_scene(
        args.scene_hash, args.cache_dir, images_variant=args.images_variant
    )


def main() -> int:
    args = parse_args()
    out_root = args.output_dir.resolve() / args.scene_hash
    out_root.mkdir(parents=True, exist_ok=True)
    model_path = out_root / "fastpgsr"

    print(f"[prepare] scene={args.scene_hash}")
    print(f"[prepare] output={out_root}")

    scene = _resolve_scene_ref(args)
    print(f"[prepare] colmap source: {scene.colmap_source_path}")

    if args.existing_mesh is not None:
        mesh_path = args.existing_mesh.resolve()
        print(f"[prepare] using --existing-mesh {mesh_path}")
    else:
        cfg = FastPgsrConfig(
            repo_path=args.fastpgsr_repo,
            conda_env=args.conda_env,
            iterations=args.iterations,
            tsdf_voxel_size=args.tsdf_voxel_size,
            tsdf_max_depth=args.tsdf_max_depth,
        )
        mesh_path = run_fast_pgsr(
            cfg,
            source_path=scene.colmap_source_path,
            model_path=model_path,
            train=not args.skip_train,
            extract_mesh=not args.skip_mesh,
        )

    if args.skip_align:
        print(f"[prepare] skipping alignment; mesh at {mesh_path}")
        return 0

    aligned = align_scene_from_mesh(
        colmap_source_path=scene.colmap_source_path,
        mesh_path=mesh_path,
        metric_scale_hint=args.metric_scale_hint,
        target_camera_height_m=args.target_camera_height_m,
    )
    print(
        f"[prepare] gravity (colmap frame) = {aligned.gravity_world_colmap.round(3).tolist()}\n"
        f"           scale = {aligned.scale:.4f}\n"
        f"           median camera height (post-align) = {aligned.median_camera_height_m:.2f} m\n"
        f"           floor quality score = {aligned.quality_score:.2f} (0–1, higher is better)"
    )
    if aligned.quality_score < 0.3:
        print(
            "[prepare] WARNING: low floor-quality score — gravity or floor may be misdetected. "
            "Inspect the aligned mesh before using as a robot scene.",
            flush=True,
        )

    aligned_mesh_path = out_root / "mesh_aligned.ply"
    write_aligned_mesh(aligned, src_mesh_path=mesh_path, out_mesh_path=aligned_mesh_path)
    print(f"[prepare] wrote {aligned_mesh_path}")

    metadata = {
        "scene_hash": args.scene_hash,
        "colmap_source_path": str(scene.colmap_source_path),
        "fastpgsr_model_path": str(model_path),
        "raw_mesh_path": str(mesh_path),
        "aligned_mesh_path": str(aligned_mesh_path),
        "world_from_colmap_4x4": aligned.world_from_colmap_4x4().tolist(),
        "scale": aligned.scale,
        "gravity_world_colmap": aligned.gravity_world_colmap.tolist(),
        "median_camera_height_m": aligned.median_camera_height_m,
        "floor_quality_score": aligned.quality_score,
        "metric_scale_hint": args.metric_scale_hint,
        "target_camera_height_m": args.target_camera_height_m,
    }
    meta_path = out_root / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"[prepare] wrote {meta_path}")

    print("\n[prepare] next: visualize with")
    print(f"    python scripts/visualize_dl3dv_scene_in_isaac.py --scene-dir {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
