# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Sample valid foreground placements on a prepared DL3DV scene + visualize.

Reads ``mesh_aligned.ply`` + metadata, finds free floor cells (top-down
occupancy + erosion by half the foreground footprint), samples N
placement poses, writes ``placements.json`` to the scene dir, and dumps
``viz_placements.png`` overlaying:

  - non-floor mesh vertices (gray dots)
  - sampled placement footprint rectangles (semi-transparent)
  - robot reach disk inside each footprint (orange ring)
  - camera trajectory (red dots) for context
  - origin axes

This is the *no-Isaac* sanity check before running the full multi-view
Isaac render. If the rectangles look right-sized vs the room and they
land in plausible spots, sizing/scale are good.

Usage::

    python scripts/sample_dl3dv_placements.py \
        --scene-dir data/dl3dv_backgrounds/scenes/<hash> \
        --n 12 --footprint-len 1.5 --footprint-wid 1.0 --robot-reach 0.85
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robolab.scene_gen.dl3dv_backgrounds.align import _read_images_metadata  # noqa: E402
from robolab.scene_gen.dl3dv_backgrounds.placement import (  # noqa: E402
    FootprintSpec, sample_placements, write_placements_json,
)


def _load_mesh_vertices(mesh_path: Path) -> np.ndarray:
    import trimesh
    mesh = trimesh.load(str(mesh_path), process=False)
    if hasattr(mesh, "dump"):
        try:
            mesh = mesh.dump(concatenate=True)
        except Exception:
            pass
    return np.asarray(mesh.vertices, dtype=np.float64)


def _render(
    scene_dir: Path,
    *,
    verts: np.ndarray,
    placements: list,
    cam_centers: np.ndarray,
    robot_reach_m: float,
    out_path: Path,
    aligned_mesh_path: Path,
    floor_definition: str = "occupied-column",
    floor_z_tolerance_m: float = 0.15,
    cell_size_m: float = 0.05,
    table_height_m: float = 0.75,
    clearance_m: float = 2.5,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon

    from robolab.scene_gen.dl3dv_backgrounds.align import _ransac_floor_plane
    from robolab.scene_gen.dl3dv_backgrounds.placement import _build_topdown_grids

    # Show the *actual* floor mask used by the placement algorithm — not
    # just RANSAC inliers — so the viz matches what the planner saw.
    # For ``occupied-column`` mode this fills in furniture-occluded
    # floor cells; for ``ransac`` mode it's identical to the inliers.
    _, _, has_floor_used, (xmin, ymin, _, _) = _build_topdown_grids(
        aligned_mesh_path,
        cell_size_m=cell_size_m,
        clearance_m=clearance_m,
        table_height_m=table_height_m,
        floor_definition=floor_definition,
        floor_z_tolerance_m=floor_z_tolerance_m,
    )
    iy, ix = np.where(has_floor_used)
    floor_xy_used = np.column_stack([
        xmin + (ix + 0.5) * cell_size_m,
        ymin + (iy + 0.5) * cell_size_m,
    ])

    # Also overlay the (typically-sparser) RANSAC inliers so you can see
    # at a glance how much "fill-in" the occupied-column rule gave you.
    _, _, ransac_mask = _ransac_floor_plane(verts, return_mask=True)
    ransac_xy = verts[ransac_mask][:, :2]
    if len(ransac_xy) > 100_000:
        idx = np.random.default_rng(0).choice(len(ransac_xy), 100_000, replace=False)
        ransac_xy = ransac_xy[idx]

    # Non-floor obstacles — gray
    pts = verts[(verts[:, 2] > 0.05) & (verts[:, 2] < 2.5)][:, :2]
    if len(pts) > 200_000:
        idx = np.random.default_rng(0).choice(len(pts), 200_000, replace=False)
        pts = pts[idx]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(floor_xy_used[:, 0], floor_xy_used[:, 1], s=0.5, c="#a0d8b3",
               alpha=0.55, label=f"floor mask used ({floor_definition})")
    ax.scatter(ransac_xy[:, 0], ransac_xy[:, 1], s=0.4, c="#117a3a",
               alpha=0.7, label="RANSAC inliers")
    ax.scatter(pts[:, 0], pts[:, 1], s=0.3, c="#444", alpha=0.5, label="non-floor mesh")
    ax.scatter(cam_centers[:, 0], cam_centers[:, 1],
               s=12, c="#d62728", alpha=0.8, label="cameras")

    # Mark each placement with a rectangle + reach disk
    for i, pl in enumerate(placements):
        corners = pl.corners_xy()
        ax.add_patch(Polygon(
            corners, closed=True,
            facecolor="#1f77b4", alpha=0.25, edgecolor="#1f77b4", linewidth=1.5,
        ))
        # Forward arrow showing yaw
        c = np.cos(pl.yaw_rad); s = np.sin(pl.yaw_rad)
        ax.annotate(
            "", xy=(pl.x_m + 0.6 * c, pl.y_m + 0.6 * s),
            xytext=(pl.x_m, pl.y_m),
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5),
        )
        ax.add_patch(Circle(
            (pl.x_m, pl.y_m), robot_reach_m,
            facecolor="none", edgecolor="#ff7f0e", linewidth=1.0, linestyle="--",
            alpha=0.7,
        ))
        ax.text(pl.x_m, pl.y_m, str(i), fontsize=8, ha="center", va="center")

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"sampled placements (n={len(placements)}) — "
        f"footprint {placements[0].footprint.length_m}×{placements[0].footprint.width_m} m, "
        f"reach {robot_reach_m} m"
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene-dir", type=Path, required=True)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--footprint-len", type=float, default=1.5,
                   help="task footprint length along +X (metres)")
    p.add_argument("--footprint-wid", type=float, default=1.0,
                   help="task footprint width along +Y (metres)")
    p.add_argument("--clearance", type=float, default=2.5,
                   help="vertical robot-column clearance (m). Default 2.5 "
                        "covers a Franka with full reach above the table.")
    p.add_argument("--robot-reach", type=float, default=0.85,
                   help="lateral arm reach beyond the footprint edge — used "
                        "BOTH as the visualization disk and as the erosion "
                        "margin when sampling. Default 0.85 (Franka).")
    p.add_argument("--cell-size", type=float, default=0.05,
                   help="occupancy grid cell size (metres)")
    p.add_argument("--table-height", type=float, default=0.75,
                   help="height (m) at which obstacle bands split. Below = "
                        "must clear footprint only; above = must clear "
                        "footprint + reach. Lets the arm sweep over sofas "
                        "while still avoiding walls.")
    p.add_argument("--floor-close-radius", type=float, default=1.0,
                   help="morphological closing radius (m) applied to the "
                        "RANSAC floor mask. Bridges gaps where furniture "
                        "occludes floor capture. Default 1.0 m.")
    p.add_argument(
        "--camera-floor-radius", type=float, default=1.5,
        help="every training camera position contributes a disk of this "
             "radius (m) to the floor mask, on the assumption that the "
             "operator was walking on the floor. Fills in patchy regions "
             "where cameras were dense but RANSAC inliers were sparse. "
             "Pass 0 to disable.",
    )
    p.add_argument("--yaw", choices=("face_centroid", "random"),
                   default="face_centroid")
    p.add_argument("--sampling", choices=("farthest_point", "random"),
                   default="farthest_point",
                   help="how to select N picks from the free region. "
                        "farthest_point spreads picks across the whole free "
                        "area; random clusters when the free region is small.")
    p.add_argument("--min-separation", type=float, default=0.5,
                   help="minimum xy distance (m) between sampled placements. "
                        "Hard floor on cluster collapse — picks closer than "
                        "this get dropped.")
    p.add_argument(
        "--max-distance-to-camera", type=float, default=None,
        help="if set, only sample placements whose xy is within this many "
             "metres of a training camera position. The DL3DV operator only "
             "captured certain regions densely; rendering from a placement "
             "far from any camera produces blurry/smeared walls. Try 1.5–2.5 "
             "metres for typical handheld captures.",
    )
    p.add_argument(
        "--min-floor-area-m2", type=float, default=3.0,
        help="drop floor connected-components smaller than this area "
             "(m²). Removes RANSAC false-positives from adjacent rooms / "
             "horizontal furniture surfaces / outdoor patches that the "
             "morphological closing might bridge to the real room. "
             "Pass 0 to disable. Default 3 m² ≈ smallest plausible room.",
    )
    p.add_argument(
        "--min-free-area-m2", type=float, default=1.0,
        help="drop tiny components from the FINAL free-region mask "
             "(after low/high obstacle erosion). Catches the 'doorway "
             "peninsula' failure mode where a thin passage between "
             "rooms survives the floor-component filter but only "
             "contributes a thin sliver of valid placement cells. "
             "Pass 0 to disable. Default 1 m² ≈ a few placements' worth.",
    )
    p.add_argument(
        "--floor-definition", choices=("occupied-column", "ransac"),
        default="occupied-column",
        help="how to decide which xy cells count as floor. "
             "'occupied-column' (default): every cell whose lowest "
             "mesh point is within --floor-z-tolerance of the RANSAC "
             "floor — fills in furniture-occluded floor. Best for "
             "synthesis-based scenes (Marble, Echo2). 'ransac': only "
             "cells where a RANSAC floor inlier landed — conservative; "
             "use when synthesis fill-in produces false positives.",
    )
    p.add_argument(
        "--floor-z-tolerance-m", type=float, default=0.15,
        help="how far above the RANSAC floor z a column's lowest mesh "
             "point may be and still count as floor. 0.15 m allows "
             "thin rugs / floor noise without flagging table tops "
             "(typically ≥ 0.4 m up).",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    scene_dir = args.scene_dir.resolve()
    metadata = json.loads((scene_dir / "metadata.json").read_text())
    aligned_mesh = scene_dir / "mesh_aligned.ply"
    if not aligned_mesh.is_file():
        raise FileNotFoundError(f"missing {aligned_mesh}")

    footprint = FootprintSpec(
        length_m=args.footprint_len,
        width_m=args.footprint_wid,
        clearance_m=args.clearance,
        robot_reach_m=args.robot_reach,
        table_height_m=args.table_height,
    )
    # Derive camera centres in aligned world frame. We need them whenever
    # camera-floor-radius > 0 (uses cam positions as floor evidence) OR
    # max-distance-to-camera is set (filters placements near cameras).
    # Marble / Echo2 scenes have no cameras (metadata.colmap_source_path is
    # null); silently degrade those flags to no-op rather than crashing.
    colmap_src = metadata.get("colmap_source_path")
    has_cams = bool(colmap_src) and Path(colmap_src).exists()
    needs_cams = args.camera_floor_radius > 0 or args.max_distance_to_camera is not None
    cam_centers_world = None
    if needs_cams and has_cams:
        R_wc, t_wc = _read_images_metadata(Path(colmap_src))
        T = np.asarray(metadata["world_from_colmap_4x4"])
        cam_centers_world = (T[:3, :3] @ t_wc.T).T + T[:3, 3]
    elif needs_cams and not has_cams:
        print(
            "[placements] no colmap_source_path in metadata "
            "(marble / echo2 scene?); ignoring --camera-floor-radius and "
            "--max-distance-to-camera."
        )

    placements = sample_placements(
        aligned_mesh_path=aligned_mesh,
        n_placements=args.n,
        footprint=footprint,
        cell_size_m=args.cell_size,
        yaw_strategy=args.yaw,
        rng_seed=args.seed,
        floor_close_radius_m=args.floor_close_radius,
        sampling=args.sampling,
        min_separation_m=args.min_separation,
        camera_centers_world=cam_centers_world,
        max_camera_distance_m=args.max_distance_to_camera,
        camera_floor_radius_m=args.camera_floor_radius,
        min_floor_area_m2=args.min_floor_area_m2,
        min_free_area_m2=args.min_free_area_m2,
        floor_definition=args.floor_definition,
        floor_z_tolerance_m=args.floor_z_tolerance_m,
    )

    out_json = scene_dir / "placements.json"
    write_placements_json(placements, out_json)
    print(f"[placements] wrote {len(placements)} poses → {out_json}")

    # Re-derive camera centres in aligned frame for the overlay (empty
    # array when there are no cameras, e.g. marble / echo2).
    if has_cams:
        R_wc, t_wc = _read_images_metadata(Path(colmap_src))
        T = np.asarray(metadata["world_from_colmap_4x4"])
        cam_centers = (T[:3, :3] @ t_wc.T).T + T[:3, 3]
    else:
        cam_centers = np.empty((0, 3), dtype=np.float64)

    verts = _load_mesh_vertices(aligned_mesh)
    out_png = scene_dir / "viz_placements.png"
    _render(
        scene_dir, verts=verts, placements=placements,
        cam_centers=cam_centers, robot_reach_m=args.robot_reach,
        out_path=out_png,
        aligned_mesh_path=aligned_mesh,
        floor_definition=args.floor_definition,
        floor_z_tolerance_m=args.floor_z_tolerance_m,
        cell_size_m=args.cell_size,
        table_height_m=args.table_height,
        clearance_m=args.clearance,
    )
    print(f"[placements] wrote viz → {out_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
