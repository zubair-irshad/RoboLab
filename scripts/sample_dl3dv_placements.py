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
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon

    from robolab.scene_gen.dl3dv_backgrounds.align import _ransac_floor_plane

    # Floor inliers (where actual captured floor lives) — show in green
    # so you can see at a glance whether placements land *inside* the
    # captured room or outside it.
    _, _, floor_mask = _ransac_floor_plane(verts, return_mask=True)
    floor_pts = verts[floor_mask][:, :2]
    if len(floor_pts) > 100_000:
        idx = np.random.default_rng(0).choice(len(floor_pts), 100_000, replace=False)
        floor_pts = floor_pts[idx]

    # Non-floor obstacles — gray
    pts = verts[(verts[:, 2] > 0.05) & (verts[:, 2] < 2.5)][:, :2]
    if len(pts) > 200_000:
        idx = np.random.default_rng(0).choice(len(pts), 200_000, replace=False)
        pts = pts[idx]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(floor_pts[:, 0], floor_pts[:, 1], s=0.3, c="#3cb371",
               alpha=0.4, label="captured floor (RANSAC)")
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
    p.add_argument("--yaw", choices=("face_centroid", "random"),
                   default="face_centroid")
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
    )
    placements = sample_placements(
        aligned_mesh_path=aligned_mesh,
        n_placements=args.n,
        footprint=footprint,
        cell_size_m=args.cell_size,
        yaw_strategy=args.yaw,
        rng_seed=args.seed,
    )

    out_json = scene_dir / "placements.json"
    write_placements_json(placements, out_json)
    print(f"[placements] wrote {len(placements)} poses → {out_json}")

    # Re-derive camera centres in aligned frame for the overlay
    R_wc, t_wc = _read_images_metadata(Path(metadata["colmap_source_path"]))
    T = np.asarray(metadata["world_from_colmap_4x4"])
    cam_centers = (T[:3, :3] @ t_wc.T).T + T[:3, 3]

    verts = _load_mesh_vertices(aligned_mesh)
    out_png = scene_dir / "viz_placements.png"
    _render(
        scene_dir, verts=verts, placements=placements,
        cam_centers=cam_centers, robot_reach_m=args.robot_reach,
        out_path=out_png,
    )
    print(f"[placements] wrote viz → {out_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
