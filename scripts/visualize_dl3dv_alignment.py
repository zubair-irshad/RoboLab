# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Sanity-check a prepared DL3DV scene without launching Isaac Sim.

Phase 1 visualization: load ``mesh_aligned.ply`` + ``metadata.json``,
print bbox/floor/camera stats, and dump a top-down + side orthographic
projection of the vertex cloud as PNGs. If the alignment worked, the
top-down view shows a recognizable room footprint and the side view
shows a flat band along z=0 (the floor) with cameras hovering above.

Phase 2's Isaac visualization will replace this once we're authoring
USDZ — at that point we'll also have the GS visual to render properly.

Usage::

    python scripts/visualize_dl3dv_alignment.py \
        --scene-dir data/dl3dv_backgrounds/scenes/<hash>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robolab.scene_gen.dl3dv_backgrounds.align import _read_images_metadata  # noqa: E402


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


def _project(points: np.ndarray, axes: tuple[int, int]) -> np.ndarray:
    return points[:, list(axes)]


def _render_projection(
    pts: np.ndarray,
    axes: tuple[int, int],
    out_path: Path,
    *,
    cameras_xy: np.ndarray | None = None,
    title: str,
    floor_z: float | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    proj = _project(pts, axes)
    # Subsample for speed; meshes can have millions of verts.
    if len(proj) > 200_000:
        idx = np.random.default_rng(0).choice(len(proj), size=200_000, replace=False)
        proj = proj[idx]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(proj[:, 0], proj[:, 1], s=0.3, c="#444", alpha=0.6)
    if cameras_xy is not None:
        ax.scatter(cameras_xy[:, 0], cameras_xy[:, 1], s=20, c="#d62728", label="cameras")
    if floor_z is not None and axes[1] == 2:
        ax.axhline(floor_z, color="#1f77b4", linestyle="--", label=f"floor z={floor_z:.2f}")
    ax.set_aspect("equal")
    labels = ["x (m)", "y (m)", "z (m)"]
    ax.set_xlabel(labels[axes[0]])
    ax.set_ylabel(labels[axes[1]])
    ax.set_title(title)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene-dir", type=Path, required=True)
    args = p.parse_args()

    scene_dir = args.scene_dir.resolve()
    meta_path = scene_dir / "metadata.json"
    mesh_path = scene_dir / "mesh_aligned.ply"
    if not meta_path.is_file() or not mesh_path.is_file():
        raise FileNotFoundError(
            f"expected {meta_path} and {mesh_path}; run prepare_dl3dv_scene.py first"
        )

    metadata = json.loads(meta_path.read_text())
    scene_id = metadata.get("scene_hash") or metadata.get("source_ply") or "<unknown>"
    print(f"[viz] scene = {scene_id}")
    print(f"[viz] scale = {metadata['scale']:.4f}")
    print(f"[viz] median camera height = {metadata.get('median_camera_height_m', 0.0):.2f} m")
    print(f"[viz] floor quality = {metadata['floor_quality_score']:.2f}")

    verts = _load_mesh_vertices(mesh_path)
    print(f"[viz] mesh: {len(verts):,} vertices")
    print(f"[viz] bbox min = {verts.min(axis=0).round(2).tolist()}")
    print(f"[viz] bbox max = {verts.max(axis=0).round(2).tolist()}")

    floor_band_thickness = float(np.std(verts[verts[:, 2] < 0.05][:, 2])) if (verts[:, 2] < 0.05).any() else float("nan")
    print(f"[viz] floor band (z<0.05m) std = {floor_band_thickness:.4f} m")

    # Re-derive camera centers in the aligned frame for the overlay (only
    # when the metadata actually carries COLMAP cameras — marble / echo2
    # scenes have no cameras and metadata.colmap_source_path is null).
    colmap_src_raw = metadata.get("colmap_source_path")
    cam_centers: np.ndarray | None = None
    if colmap_src_raw and Path(colmap_src_raw).exists():
        colmap_src = Path(colmap_src_raw)
        R_wc, t_wc = _read_images_metadata(colmap_src)
        T = np.asarray(metadata["world_from_colmap_4x4"])
        cam_centers = (T[:3, :3] @ t_wc.T).T + T[:3, 3]
        print(f"[viz] camera z range = [{cam_centers[:, 2].min():.2f}, {cam_centers[:, 2].max():.2f}] m")
    else:
        print("[viz] no colmap cameras in metadata — skipping camera overlay")

    out_top = scene_dir / "viz_topdown.png"
    out_side = scene_dir / "viz_side.png"
    _render_projection(
        verts, (0, 1), out_top,
        cameras_xy=cam_centers[:, [0, 1]] if cam_centers is not None else None,
        title="top-down (x, y) — aligned frame",
    )
    _render_projection(
        verts, (0, 2), out_side,
        cameras_xy=cam_centers[:, [0, 2]] if cam_centers is not None else None,
        title="side (x, z) — aligned frame, gravity = -z",
        floor_z=0.0,
    )
    print(f"[viz] wrote {out_top}")
    print(f"[viz] wrote {out_side}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
