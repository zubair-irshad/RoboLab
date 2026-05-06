# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Interactive 3D inspection of a prepared DL3DV scene via rerun.

Loads ``mesh_aligned.ply`` + ``metadata.json`` (same scene-dir layout
that ``visualize_dl3dv_alignment.py`` expects), re-runs RANSAC floor
detection on the aligned vertices, and logs the result to a rerun
viewer. Useful for eyeballing whether the floor inlier set actually
covers the floor (vs. e.g. a tabletop or ceiling), and whether the
camera trajectory hovers above z=0 in the gravity-aligned frame.

``rerun-sdk`` is an optional dependency — install with
``pip install rerun-sdk`` if missing.

Usage::

    python scripts/inspect_floor_in_rerun.py --scene-dir <prepared-scene-dir>
    python scripts/inspect_floor_in_rerun.py --scene-dir <dir> --save out.rrd
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robolab.scene_gen.dl3dv_backgrounds.align import (  # noqa: E402
    _ransac_floor_plane,
    _read_images_metadata,
)


def _load_mesh_vertices(mesh_path: Path) -> np.ndarray:
    try:
        import trimesh
    except ImportError as e:  # pragma: no cover
        raise ImportError("trimesh is required; `pip install trimesh`") from e
    mesh = trimesh.load(str(mesh_path), process=False)
    if hasattr(mesh, "dump"):
        try:
            mesh = mesh.dump(concatenate=True)
        except Exception:
            pass
    return np.asarray(mesh.vertices, dtype=np.float64)


def _floor_plane_mesh(verts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices, triangle_indices) for a thin quad at z=0 spanning xy bbox."""
    xy_min = verts[:, :2].min(axis=0)
    xy_max = verts[:, :2].max(axis=0)
    pad = 0.1 * (xy_max - xy_min + 1e-6)
    x0, y0 = xy_min - pad
    x1, y1 = xy_max + pad
    quad_verts = np.array(
        [[x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0]],
        dtype=np.float32,
    )
    tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    return quad_verts, tris


def _subsample(points: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    if len(points) <= n:
        return points
    idx = np.random.default_rng(seed).choice(len(points), size=n, replace=False)
    return points[idx]


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--scene-dir", type=Path, required=True)
    p.add_argument(
        "--save",
        type=Path,
        default=None,
        help="if given, save .rrd recording to this path instead of spawning the viewer",
    )
    p.add_argument(
        "--max-points",
        type=int,
        default=400_000,
        help="cap on point cloud size per channel (random subsample if exceeded)",
    )
    args = p.parse_args()

    try:
        import rerun as rr
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "rerun-sdk is required for this inspector; install with `pip install rerun-sdk`"
        ) from e

    scene_dir = args.scene_dir.resolve()
    meta_path = scene_dir / "metadata.json"
    mesh_path = scene_dir / "mesh_aligned.ply"
    if not meta_path.is_file() or not mesh_path.is_file():
        raise FileNotFoundError(
            f"expected {meta_path} and {mesh_path}; run prepare_dl3dv_scene.py first"
        )

    metadata = json.loads(meta_path.read_text())
    print(f"[inspect] scene_hash = {metadata['scene_hash']}")
    print(f"[inspect] floor quality (stored) = {metadata['floor_quality_score']:.2f}")

    verts = _load_mesh_vertices(mesh_path)
    print(f"[inspect] mesh: {len(verts):,} vertices")

    floor_z, inlier_frac, mask = _ransac_floor_plane(verts, return_mask=True)
    print(
        f"[inspect] re-derived floor: z={floor_z:.4f} "
        f"inlier_frac={inlier_frac:.3f} n_inliers={int(mask.sum()):,}"
    )

    floor_pts = verts[mask]
    non_floor_pts = verts[~mask]
    floor_pts = _subsample(floor_pts, args.max_points, seed=1)
    non_floor_pts = _subsample(non_floor_pts, args.max_points, seed=2)

    # Camera centers in aligned frame (re-derived; metadata has no intrinsics).
    colmap_src = Path(metadata["colmap_source_path"])
    R_wc, t_wc = _read_images_metadata(colmap_src)
    T = np.asarray(metadata["world_from_colmap_4x4"])
    cam_centers = (T[:3, :3] @ t_wc.T).T + T[:3, 3]
    print(f"[inspect] {len(cam_centers)} cameras, "
          f"z range [{cam_centers[:, 2].min():.2f}, {cam_centers[:, 2].max():.2f}] m")

    rec = rr.RecordingStream(application_id="inspect_floor_in_rerun")
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        rec.save(str(args.save))
        print(f"[inspect] saving recording to {args.save}")
    else:
        rec.spawn()
        print("[inspect] spawned rerun viewer (close window to exit)")

    rr.log(
        "scene/mesh/non_floor",
        rr.Points3D(non_floor_pts.astype(np.float32), colors=[120, 120, 120], radii=0.005),
        recording=rec,
    )
    rr.log(
        "scene/mesh/floor",
        rr.Points3D(floor_pts.astype(np.float32), colors=[60, 200, 100], radii=0.008),
        recording=rec,
    )
    plane_v, plane_t = _floor_plane_mesh(verts)
    rr.log(
        "scene/floor_plane",
        rr.Mesh3D(
            vertex_positions=plane_v,
            triangle_indices=plane_t,
            albedo_factor=[60, 200, 100, 60],
        ),
        recording=rec,
    )
    rr.log(
        "scene/cameras",
        rr.Points3D(cam_centers.astype(np.float32), colors=[220, 40, 40], radii=0.04),
        recording=rec,
    )
    axis_origins = np.zeros((3, 3), dtype=np.float32)
    axis_vectors = np.eye(3, dtype=np.float32)
    rr.log(
        "scene/world_axes",
        rr.Arrows3D(
            origins=axis_origins,
            vectors=axis_vectors,
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ),
        recording=rec,
    )
    print("[inspect] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
