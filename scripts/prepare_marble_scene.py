# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Prepare a single Gaussian-splat PLY (Marble / Echo2 / etc.) as a robot-environment background.

This is the COLMAP-free analogue of ``prepare_dl3dv_scene.py``: it
takes a single ``.ply`` (gaussians or a regular point cloud) and emits
the same downstream artifact set the existing render pipeline already
consumes:

    <out_dir>/
      mesh_aligned.ply     # proxy collider mesh, gravity-aligned, floor at z=0
      metadata.json        # world_from_colmap_4x4, scale, etc.
      source.ply           # symlink (or copy) to the input PLY for 3DGUT

Pipeline:

  1. Load gaussian centres (filter by opacity if the PLY carries it).
  2. Apply an initial gravity rotation from ``--up-axis`` (Marble
     defaults to +Y up; pass ``+z`` if your generator already uses
     z-up).
  3. Determine metric scale:
       --metric-scale K          -> use K directly
       --ceiling-height-m H      -> derive scale so 95th-pct-z above
                                     the RANSAC floor maps to H metres
       (default)                 -> ceiling-height heuristic with H = 2.7 m
  4. RANSAC floor on the metric, rotated cloud; refine gravity from
     the fitted plane normal (capped by ``--max-gravity-correction-deg``).
  5. Translate so the floor sits at z = 0.
  6. Build a coarse triangle mesh from the aligned cloud (Open3D
     Poisson by default; alpha-shape fallback). This is the **collider**
     used by ``render_dl3dv_with_robot.py``'s depth pass — it doesn't
     need to be photoreal, just topologically reasonable.
  7. Write ``mesh_aligned.ply`` + ``metadata.json``. The metadata is
     intentionally schema-compatible with ``prepare_dl3dv_scene.py`` so
     ``sample_dl3dv_placements.py`` and ``render_dl3dv_with_robot.py``
     work unchanged.

The original PLY is left untouched on disk; pass it directly to
``dl3dv_gs_to_usdz.py --ply <ply>`` (3DGUT keeps the source frame, and
the render script applies ``world_from_colmap_4x4`` to bring the GS
visual into the aligned frame at runtime).

Usage::

    python scripts/prepare_marble_scene.py \
        --ply data/echo-2/Designer_Bath_Laundry_Nook.ply \
        --out-dir data/marble_backgrounds/scenes/Designer_Bath_Laundry_Nook \
        --up-axis +y --ceiling-height-m 2.7
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robolab.scene_gen.dl3dv_backgrounds.align import (  # noqa: E402
    AlignedScene,
    _R_align_up_to_z,
    _fit_plane_normal_svd,
    _ransac_floor_plane,
)


_UP_AXES: dict[str, np.ndarray] = {
    "+x": np.array([1.0, 0.0, 0.0]),
    "-x": np.array([-1.0, 0.0, 0.0]),
    "+y": np.array([0.0, 1.0, 0.0]),
    "-y": np.array([0.0, -1.0, 0.0]),
    "+z": np.array([0.0, 0.0, 1.0]),
    "-z": np.array([0.0, 0.0, -1.0]),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ply", type=Path, required=True,
                   help="input Gaussian-splat (or plain point-cloud) PLY")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="scene directory to populate (created if missing)")
    p.add_argument(
        "--up-axis", choices=("auto",) + tuple(_UP_AXES), default="auto",
        help="which axis of the input PLY points up. ``auto`` (default) "
             "scans all 6 candidates and picks the one whose lower-third "
             "RANSAC floor wins on inlier fraction — works regardless of "
             "the generator's convention. Pass +y for Marble / Three.js, "
             "+z for COLMAP-z-up, etc. when you want to override.",
    )
    p.add_argument(
        "--metric-scale", type=float, default=None,
        help="multiplier from PLY units to metres. Mutually exclusive with "
             "--ceiling-height-m. If neither is set, ceiling-height heuristic "
             "is used with the default target height.",
    )
    p.add_argument(
        "--ceiling-height-m", type=float, default=2.7,
        help="when --metric-scale is not given, derive scale so the cloud's "
             "95th-percentile-z above the floor maps to this many metres. "
             "2.7 m matches a typical residential ceiling. Ignored when "
             "--metric-scale is provided.",
    )
    p.add_argument(
        "--opacity-threshold", type=float, default=0.05,
        help="filter gaussians with sigmoid(opacity) below this threshold "
             "before alignment + meshing. 0 to disable. Only applies when the "
             "PLY actually carries an `opacity` field (i.e. is a 3DGS PLY).",
    )
    p.add_argument(
        "--max-gravity-correction-deg", type=float, default=25.0,
        help="cap on how much gravity may be refined from the RANSAC plane "
             "normal. Set to 0 to disable refinement (trust --up-axis exactly).",
    )
    p.add_argument(
        "--floor-inlier-thresh-m", type=float, default=0.05,
        help="RANSAC plane inlier distance (metres, post-scale).",
    )
    p.add_argument(
        "--proxy-method",
        choices=("poisson", "alpha", "voxel-points"),
        default="poisson",
        help="how to build the proxy collider mesh. poisson = Open3D Poisson "
             "surface reconstruction (smooth, requires open3d). alpha = "
             "Open3D alpha-shape (faster, holier). voxel-points = no faces, "
             "useful only for placement sampling (USD writer will fail).",
    )
    p.add_argument(
        "--voxel-size-m", type=float, default=0.05,
        help="voxel-downsample resolution before mesh reconstruction.",
    )
    p.add_argument(
        "--poisson-depth", type=int, default=9,
        help="octree depth for Poisson reconstruction (8-10 typical).",
    )
    p.add_argument(
        "--poisson-density-quantile", type=float, default=0.05,
        help="trim Poisson vertices in the lowest density quantile (removes "
             "balloon-y triangles outside the actual surface).",
    )
    p.add_argument(
        "--alpha-m", type=float, default=0.20,
        help="alpha radius for alpha-shape reconstruction (only when "
             "--proxy-method alpha).",
    )
    p.add_argument(
        "--max-points", type=int, default=500_000,
        help="random subsample down to this many points before alignment. "
             "Marble PLYs can have millions of gaussians — RANSAC + Poisson "
             "are quadratic-ish in size.",
    )
    p.add_argument(
        "--symlink-source", action="store_true",
        help="symlink the input PLY to <out-dir>/source.ply.original instead "
             "of copying it. Symlinks save disk but break if the source moves.",
    )
    p.add_argument(
        "--gs-sh-mode",
        choices=("pad-to-3", "drop", "preserve"),
        default="pad-to-3",
        help="how to normalize SH coefficients when writing source.ply "
             "(consumed by 3DGUT, which only accepts SH degree 0 or 3). "
             "pad-to-3 (default): keep DC + existing degree-1/2 bands, "
             "zero-pad up to degree 3 — preserves view-dependent colour "
             "from the source. drop: keep DC only, set f_rest=zeros — "
             "safe but loses any view-dependent shading. preserve: copy "
             "f_rest as-is — only valid when source already has 0 or 45 "
             "f_rest_* properties.",
    )
    return p.parse_args()


# ---- ply IO ----------------------------------------------------------------

def _read_ply_points(
    ply_path: Path, *, opacity_threshold: float = 0.05,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (Nx3 positions, Nx3 RGB-uint8 or None).

    Handles three flavours:
      * 3DGS PLYs (have ``opacity``, ``f_dc_*``, ``scale_*``, ``rot_*``).
        Filters by sigmoid(opacity) ≥ threshold, computes RGB from
        ``f_dc_0..2`` (the SH degree-0 colour band) via the 3DGS
        convention ``rgb = 0.5 + SH_C0 * f_dc``.
      * Standard point-cloud PLYs with ``red,green,blue`` uint8.
      * Plain xyz PLYs.

    Falls back from plyfile -> trimesh if plyfile isn't installed.
    """
    try:
        from plyfile import PlyData
    except ImportError:
        # Fallback: trimesh PointCloud, no opacity filtering possible.
        import trimesh  # type: ignore[import-not-found]
        pc = trimesh.load(str(ply_path), process=False)
        verts = np.asarray(pc.vertices, dtype=np.float64)
        colors = None
        vc = getattr(getattr(pc, "visual", None), "vertex_colors", None)
        if vc is not None and len(vc) == len(verts):
            colors = np.asarray(vc, dtype=np.uint8)[:, :3]
        print(f"[prepare-marble] plyfile not installed; fell back to trimesh "
              f"({len(verts):,} pts, no opacity filter)")
        return verts, colors

    pd = PlyData.read(str(ply_path))
    v = pd["vertex"]
    names = set(v.data.dtype.names)
    xyz = np.stack(
        [np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1
    ).astype(np.float64)

    keep = np.ones(len(xyz), dtype=bool)
    if "opacity" in names and opacity_threshold > 0:
        # Stored as logit (pre-sigmoid) in the standard 3DGS PLY layout.
        opacity_raw = np.asarray(v["opacity"], dtype=np.float64)
        opacity = 1.0 / (1.0 + np.exp(-opacity_raw))
        keep = opacity >= opacity_threshold
        print(
            f"[prepare-marble] opacity filter: kept {int(keep.sum()):,} / "
            f"{len(keep):,} gaussians (sigmoid(opacity) ≥ {opacity_threshold})"
        )

    colors: np.ndarray | None = None
    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        # 3DGS SH degree-0 -> RGB.  SH_C0 = 0.28209479177387814
        sh_c0 = 0.28209479177387814
        f_dc = np.stack(
            [np.asarray(v["f_dc_0"]), np.asarray(v["f_dc_1"]), np.asarray(v["f_dc_2"])],
            axis=1,
        ).astype(np.float64)
        rgb = np.clip(0.5 + sh_c0 * f_dc, 0.0, 1.0)
        colors = (rgb * 255.0 + 0.5).astype(np.uint8)
    elif {"red", "green", "blue"}.issubset(names):
        colors = np.stack(
            [np.asarray(v["red"]), np.asarray(v["green"]), np.asarray(v["blue"])],
            axis=1,
        ).astype(np.uint8)

    xyz = xyz[keep]
    if colors is not None:
        colors = colors[keep]

    print(f"[prepare-marble] loaded {len(xyz):,} points from {ply_path.name}")
    return xyz, colors


# ---- up-axis auto-detection -------------------------------------------------

def auto_detect_up_axis(
    points: np.ndarray, *,
    floor_inlier_thresh_frac: float = 0.01,
    near_floor_band_frac: float = 0.15,
    rng_seed: int = 0,
) -> tuple[str, dict[str, dict[str, float]]]:
    """Pick the axis most consistent with floor-down / ceiling-up.

    For each of the 6 candidate world-up directions (±x, ±y, ±z) we
    rotate the cloud so the candidate becomes +Z, fit a RANSAC floor,
    and score the candidate by combining two independent signals:

      1. **Inlier fraction** — how good a horizontal plane the RANSAC
         floor actually is. Both the true floor *and* the true ceiling
         tend to score high here (they're both flat, large, horizontal
         planes), so this signal alone isn't enough — kitchens with
         flat ceilings routinely have inlier_frac(ceiling) ≈
         inlier_frac(floor).

      2. **Density-near-floor ratio** — count points in a band just
         above the fitted "floor" plane vs a band just below the 95th
         percentile of z (the "ceiling"). In a typical room *much*
         more geometry sits near the floor (furniture bases, table
         legs, baseboards, kitchen counters) than near the ceiling
         (a few fixtures, maybe a fan). If we picked the wrong axis
         and the fitted "floor" is actually the ceiling, this ratio
         flips — most of the geometry is now near our "ceiling" (the
         true floor).

    Final score = ``inlier_frac * density_ratio``. Both signals must
    be high for a candidate to win, which rules out side-axes (low
    RANSAC inliers when the room is "tipped on its side") and the
    inverted up-axis (low density ratio).

    The RANSAC threshold and density band auto-scale with the cloud's
    vertical extent so this works in arbitrary unscaled units.
    """
    extent = float(np.percentile(points, 95, axis=0).max() -
                   np.percentile(points, 5, axis=0).min())
    thresh = max(extent * floor_inlier_thresh_frac, 1e-6)
    band = max(extent * near_floor_band_frac, 1e-6)

    candidates: dict[str, dict[str, float]] = {}
    for axis_name, world_up in _UP_AXES.items():
        R = _R_align_up_to_z(world_up)
        rotated = points @ R.T
        rotated_z = rotated[:, 2]
        floor_z, inlier_frac = _ransac_floor_plane(
            rotated, inlier_thresh=thresh, rng_seed=rng_seed,
        )
        z_top = float(np.percentile(rotated_z, 95))
        margin = float(z_top - floor_z)

        # Density bands: just above the picked floor vs just below the
        # picked ceiling. Exclude the floor inliers themselves so we
        # measure the "stuff resting on the floor" not the floor itself.
        n_near_floor = int(np.sum(
            (rotated_z > floor_z + thresh) & (rotated_z <= floor_z + band)
        ))
        n_near_ceiling = int(np.sum(
            (rotated_z >= z_top - band) & (rotated_z < z_top - thresh)
        ))
        denom = max(n_near_floor + n_near_ceiling, 1)
        density_ratio = float(n_near_floor) / float(denom)

        score = float(inlier_frac) * density_ratio
        if margin <= 0:
            score *= 0.01  # impossible orientation; cloud doesn't extend up

        candidates[axis_name] = {
            "inlier_frac": float(inlier_frac),
            "density_ratio": density_ratio,
            "n_near_floor": float(n_near_floor),
            "n_near_ceiling": float(n_near_ceiling),
            "upward_margin": margin,
            "score": score,
        }

    ranked = sorted(candidates.items(), key=lambda kv: kv[1]["score"], reverse=True)
    best = ranked[0][0]
    print("[auto-up] candidates (axis: score = inlier_frac × density_ratio):")
    for axis_name, info in ranked:
        print(
            f"           {axis_name:>2s}: score={info['score']:.3f}  "
            f"(inliers={info['inlier_frac']:.3f}, "
            f"density_ratio={info['density_ratio']:.3f}  "
            f"[{int(info['n_near_floor'])} near floor / "
            f"{int(info['n_near_ceiling'])} near ceiling], "
            f"upward_margin={info['upward_margin']:+.3f})"
        )
    print(f"[auto-up] selected: {best}")
    return best, candidates


# ---- alignment (no-COLMAP variant) -----------------------------------------

def align_pointcloud(
    *,
    points: np.ndarray,
    initial_world_up: np.ndarray,
    metric_scale: float | None,
    target_ceiling_height_m: float,
    refine_gravity: bool = True,
    max_gravity_correction_deg: float = 25.0,
    floor_inlier_thresh_m: float = 0.05,
) -> AlignedScene:
    """COLMAP-free analogue of ``align_scene_from_mesh``.

    Differences from the DL3DV alignment:
      * Initial gravity comes from a user-supplied axis, not from
        averaged camera-up vectors (we don't have cameras).
      * Metric scale falls back to a *ceiling-height* heuristic instead
        of a *camera-height* heuristic — the only height feature we can
        observe in a single PLY is the cloud's vertical extent.
      * ``median_camera_height_m`` is reported as 0 (no cameras).
    """
    R_align_initial = _R_align_up_to_z(initial_world_up)

    pts_rot = points @ R_align_initial.T

    if metric_scale is not None:
        scale = float(metric_scale)
    else:
        # Ceiling-height heuristic: rough RANSAC floor on the unscaled
        # rotated cloud, then choose scale so that
        # 95th-percentile-z minus floor-z = target_ceiling_height_m.
        # Threshold here is in *PLY units*; we don't know the unit, so
        # use a fraction of the cloud's vertical extent.
        z_unscaled = pts_rot[:, 2]
        extent = float(np.percentile(z_unscaled, 95) - np.percentile(z_unscaled, 5))
        rough_thresh = max(extent * 0.01, 1e-6)
        rough_floor_z, _ = _ransac_floor_plane(
            pts_rot, inlier_thresh=rough_thresh,
        )
        height_in_units = float(np.percentile(z_unscaled, 95) - rough_floor_z)
        if height_in_units <= 1e-9:
            raise RuntimeError(
                "ceiling-height heuristic failed: 95th-pct-z is below the "
                "rough floor. Probably gravity is flipped — try a different "
                "--up-axis (or its negative)."
            )
        scale = float(target_ceiling_height_m) / height_in_units
        print(
            f"[align] ceiling-height heuristic: 95th-pct-height = "
            f"{height_in_units:.3f} (PLY units); target = "
            f"{target_ceiling_height_m:.2f} m; scale = {scale:.4f}"
        )

    pts_metric = scale * pts_rot

    floor_z_m, inlier_frac, floor_mask = _ransac_floor_plane(
        pts_metric, inlier_thresh=floor_inlier_thresh_m, return_mask=True,
    )

    R_align = R_align_initial
    refinement_deg = 0.0
    if refine_gravity and int(floor_mask.sum()) >= 100:
        plane_normal = _fit_plane_normal_svd(pts_metric[floor_mask])
        cos_a = float(np.clip(plane_normal[2], -1.0, 1.0))
        refinement_deg = float(np.degrees(np.arccos(cos_a)))
        if refinement_deg <= max_gravity_correction_deg:
            R_correction = _R_align_up_to_z(plane_normal)
            R_align = R_correction @ R_align_initial
            pts_metric = scale * (points @ R_align.T)
            floor_z_m, inlier_frac, floor_mask = _ransac_floor_plane(
                pts_metric, inlier_thresh=floor_inlier_thresh_m, return_mask=True,
            )
        else:
            print(
                f"[align] gravity correction {refinement_deg:.1f}° exceeds "
                f"max ({max_gravity_correction_deg}°); skipping refinement"
            )
            refinement_deg = 0.0

    t_align_metric = np.array([0.0, 0.0, -floor_z_m])
    gravity_world_colmap = -R_align.T @ np.array([0.0, 0.0, 1.0])

    return AlignedScene(
        R_align=R_align,
        t_align=t_align_metric,
        scale=scale,
        gravity_world_colmap=gravity_world_colmap,
        floor_z_pre_translate_m=floor_z_m,
        median_camera_height_m=0.0,  # no cameras
        quality_score=inlier_frac,
        gravity_refinement_deg=refinement_deg,
    )


# ---- proxy mesh ------------------------------------------------------------

@dataclass
class ProxyMesh:
    vertices: np.ndarray  # (N, 3) float64
    faces: np.ndarray | None  # (M, 3) int32 or None for points-only
    colors: np.ndarray | None  # (N, 3) uint8 or None


def build_proxy_mesh(
    *,
    points_aligned: np.ndarray,
    colors: np.ndarray | None,
    method: str,
    voxel_size_m: float,
    poisson_depth: int,
    poisson_density_quantile: float,
    alpha_m: float,
) -> ProxyMesh:
    """Construct a coarse triangle mesh from an aligned point cloud.

    The mesh is used as the depth-pass collider in
    ``render_dl3dv_with_robot.py``. It only has to give plausible
    background depth; it doesn't need to be photoreal (the GS USDZ
    handles the rgb pass).
    """
    if method == "voxel-points":
        # Vertices only, no faces. Sufficient for placement sampling but
        # ``dl3dv_mesh_to_usd.py`` will refuse this — we use it only for
        # debugging.
        try:
            import open3d as o3d  # type: ignore[import-not-found]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_aligned)
            if colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
            ds = pcd.voxel_down_sample(voxel_size=voxel_size_m)
            verts = np.asarray(ds.points)
            cols = (np.asarray(ds.colors) * 255.0).astype(np.uint8) if ds.has_colors() else None
        except ImportError:
            # Numpy-only voxel downsample fallback.
            keys = np.floor(points_aligned / voxel_size_m).astype(np.int64)
            _, idx = np.unique(keys, axis=0, return_index=True)
            verts = points_aligned[np.sort(idx)]
            cols = colors[np.sort(idx)] if colors is not None else None
        return ProxyMesh(vertices=verts, faces=None, colors=cols)

    try:
        import open3d as o3d  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "open3d is required for Poisson / alpha-shape reconstruction; "
            "`pip install open3d`. Or use --proxy-method voxel-points."
        ) from e

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_aligned)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

    pcd = pcd.voxel_down_sample(voxel_size=voxel_size_m)
    n_after_ds = len(pcd.points)
    print(f"[proxy] voxel-downsampled to {n_after_ds:,} points "
          f"(voxel = {voxel_size_m * 100:.1f} cm)")

    # Estimate normals oriented upward — Poisson needs consistent normals.
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxel_size_m * 4.0, max_nn=30,
    ))
    pcd.orient_normals_to_align_with_direction(np.array([0.0, 0.0, 1.0]))

    if method == "poisson":
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=poisson_depth, scale=1.1, linear_fit=False,
        )
        if poisson_density_quantile > 0:
            d = np.asarray(densities)
            keep = d >= np.quantile(d, poisson_density_quantile)
            mesh.remove_vertices_by_mask(~keep)
        mesh.remove_unreferenced_vertices()
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.triangles, dtype=np.int32)
        cols = (np.asarray(mesh.vertex_colors) * 255.0).astype(np.uint8) \
            if mesh.has_vertex_colors() else None
        print(f"[proxy] Poisson(depth={poisson_depth}) -> "
              f"{len(verts):,} verts, {len(faces):,} faces")
        return ProxyMesh(vertices=verts, faces=faces, colors=cols)

    if method == "alpha":
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd, alpha_m,
        )
        mesh.remove_unreferenced_vertices()
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.triangles, dtype=np.int32)
        print(f"[proxy] alpha-shape(alpha={alpha_m}) -> "
              f"{len(verts):,} verts, {len(faces):,} faces")
        return ProxyMesh(vertices=verts, faces=faces, colors=None)

    raise ValueError(f"unknown --proxy-method {method!r}")


def write_proxy_ply(proxy: ProxyMesh, out_path: Path) -> None:
    """Write a PLY trimesh / Open3D / our USD writer can all consume."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if proxy.faces is None:
        # Vertices-only PLY (Points). Use plyfile to keep it simple.
        from plyfile import PlyData, PlyElement  # type: ignore[import-not-found]
        v = np.empty(
            len(proxy.vertices),
            dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],
        )
        v["x"], v["y"], v["z"] = (
            proxy.vertices[:, 0].astype(np.float32),
            proxy.vertices[:, 1].astype(np.float32),
            proxy.vertices[:, 2].astype(np.float32),
        )
        PlyData([PlyElement.describe(v, "vertex")], text=False).write(str(out_path))
        return

    import trimesh  # type: ignore[import-not-found]
    mesh = trimesh.Trimesh(
        vertices=proxy.vertices, faces=proxy.faces, process=False,
    )
    if proxy.colors is not None:
        # trimesh wants RGBA uint8.
        rgba = np.concatenate(
            [proxy.colors, 255 * np.ones((len(proxy.colors), 1), dtype=np.uint8)],
            axis=1,
        )
        mesh.visual.vertex_colors = rgba
    mesh.export(str(out_path))


# ---- 3DGUT-compatible PLY normalizer ---------------------------------------
#
# 3DGUT's loader (threedgrut.model.model.GaussianModel.init_from_ply) only
# accepts SH layouts where the f_rest_* count is exactly 0 (degree 0) or
# exactly 45 (degree 3). Many generators (Marble, some Echo2 variants)
# emit degree-1 PLYs with 9 f_rest_* properties — those are rejected with
# a 'found 9, expected 45 or 0' ValueError.
#
# We rewrite the PLY to a 3DGUT-acceptable layout. The vertex layout
# (positions, normals, opacity, scales, rotations, colours) is preserved
# byte-for-byte; only the f_rest_* properties are rewritten according to
# --gs-sh-mode.

def _channel_major_pad(src: np.ndarray, src_deg: int, dst_deg: int) -> np.ndarray:
    """Pad an (N, K_src*3) f_rest array to (N, K_dst*3), zero-fill new bands.

    INRIA's 3DGS PLY layout for f_rest_* is channel-major,
    coefficient-minor:

        for c in 0..2:
            for k in 0..(K-1):
                f_rest_{c*K + k}

    where K = (deg+1)^2 - 1 = 3, 8, or 15 for degrees 1, 2, 3. Padding to
    a higher degree means inserting (K_dst - K_src) zeros after each
    channel's existing block.
    """
    k_src = (src_deg + 1) ** 2 - 1
    k_dst = (dst_deg + 1) ** 2 - 1
    assert src.shape[1] == k_src * 3, src.shape
    n = src.shape[0]
    out = np.zeros((n, k_dst * 3), dtype=src.dtype)
    src_per_ch = src.reshape(n, 3, k_src)  # (N, channels, coeffs)
    out_per_ch = out.reshape(n, 3, k_dst)
    out_per_ch[:, :, :k_src] = src_per_ch
    return out_per_ch.reshape(n, k_dst * 3)


def _f_rest_count_to_degree(n_f_rest: int) -> int | None:
    """Map an f_rest_* count to the SH degree it represents (or None)."""
    # n_f_rest = 3 * ((deg+1)^2 - 1)
    for deg in (0, 1, 2, 3):
        if 3 * ((deg + 1) ** 2 - 1) == n_f_rest:
            return deg
    return None


def normalize_gs_ply_for_3dgut(
    src_ply: Path, dst_ply: Path, *, mode: str = "pad-to-3",
) -> dict:
    """Rewrite ``src_ply`` to ``dst_ply`` with a 3DGUT-compatible SH layout.

    Modes:
      * ``pad-to-3`` (default) — pad existing f_rest_* up to 45 with
        zeros (preserves source's degree-1/2 colour bands).
      * ``drop`` — drop f_rest_* entirely (degree 0; DC colour only).
      * ``preserve`` — copy as-is; raises if not already 0 or 45.

    Returns a small dict describing what changed (for metadata).
    """
    try:
        from plyfile import PlyData, PlyElement
    except ImportError as e:
        raise ImportError(
            "plyfile required for 3DGUT PLY normalization; `pip install plyfile`"
        ) from e

    pd = PlyData.read(str(src_ply))
    if "vertex" not in [el.name for el in pd.elements]:
        raise ValueError(f"PLY {src_ply} has no 'vertex' element")
    v = pd["vertex"]
    names = list(v.data.dtype.names)
    f_rest_names = [n for n in names if n.startswith("f_rest_")]
    n_src = len(f_rest_names)
    src_deg = _f_rest_count_to_degree(n_src)

    print(f"[normalize] source has {n_src} f_rest_* properties "
          f"(degree {src_deg if src_deg is not None else '??'})")

    # Decide target degree.
    if mode == "preserve":
        if n_src not in (0, 45):
            raise ValueError(
                f"source has {n_src} f_rest_* properties; "
                f"--gs-sh-mode preserve only accepts 0 or 45. "
                f"Use pad-to-3 or drop instead."
            )
        if src_ply.resolve() == dst_ply.resolve():
            return {"mode": "preserve", "src_degree": src_deg, "dst_degree": src_deg, "rewrote": False}
        # Even in "preserve" mode we still copy (so callers can rely on
        # dst_ply existing).
        import shutil
        shutil.copy2(src_ply, dst_ply)
        return {"mode": "preserve", "src_degree": src_deg, "dst_degree": src_deg, "rewrote": True}

    if mode == "drop":
        dst_deg = 0
    elif mode == "pad-to-3":
        if src_deg is None:
            raise ValueError(
                f"source has {n_src} f_rest_* properties — not a recognized "
                f"SH degree count. Try --gs-sh-mode drop."
            )
        dst_deg = 3
    else:
        raise ValueError(f"unknown --gs-sh-mode {mode!r}")

    n_dst = 3 * ((dst_deg + 1) ** 2 - 1)

    # Carry over every non-f_rest property unchanged.
    keep_names = [n for n in names if not n.startswith("f_rest_")]

    # Build the new dtype.
    dtype_lookup = dict(v.data.dtype.descr)  # name -> dtype string
    new_dtype = [(n, dtype_lookup[n]) for n in keep_names]
    f_rest_dtype = "<f4"  # standard 3DGS PLYs use float32 for SH
    if n_src > 0:
        f_rest_dtype = dtype_lookup[f_rest_names[0]]
    new_dtype += [(f"f_rest_{i}", f_rest_dtype) for i in range(n_dst)]

    n_verts = len(v.data)
    new_arr = np.empty(n_verts, dtype=new_dtype)
    for n in keep_names:
        new_arr[n] = v[n]

    if n_dst > 0:
        if n_src == 0:
            # Source has no SH beyond DC. Pad with zeros.
            for i in range(n_dst):
                new_arr[f"f_rest_{i}"] = 0.0
        else:
            src_block = np.stack([np.asarray(v[n]) for n in f_rest_names], axis=1)
            padded = _channel_major_pad(
                src_block, src_deg=src_deg, dst_deg=dst_deg,
            ).astype(np.dtype(f_rest_dtype))
            for i in range(n_dst):
                new_arr[f"f_rest_{i}"] = padded[:, i]

    # Reassemble. Preserve other elements (e.g. 'face', though splat PLYs
    # rarely have faces) and the source endianness/format choice.
    new_vertex_el = PlyElement.describe(new_arr, "vertex")
    other_els = [el for el in pd.elements if el.name != "vertex"]
    out_pd = PlyData([new_vertex_el, *other_els], text=pd.text, byte_order=pd.byte_order)
    dst_ply.parent.mkdir(parents=True, exist_ok=True)
    out_pd.write(str(dst_ply))

    print(f"[normalize] rewrote SH degree {src_deg} -> {dst_deg} "
          f"({n_src} -> {n_dst} f_rest_* properties); wrote {dst_ply}")
    return {
        "mode": mode,
        "src_degree": src_deg,
        "dst_degree": dst_deg,
        "src_f_rest_count": n_src,
        "dst_f_rest_count": n_dst,
        "rewrote": True,
    }


# ---- main ------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    ply_path = args.ply.resolve()
    if not ply_path.is_file():
        raise FileNotFoundError(ply_path)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[prepare-marble] input  : {ply_path}")
    print(f"[prepare-marble] output : {out_dir}")

    points, colors = _read_ply_points(
        ply_path, opacity_threshold=args.opacity_threshold,
    )

    if len(points) == 0:
        raise RuntimeError(
            "no points after opacity filtering — try lowering "
            "--opacity-threshold (currently "
            f"{args.opacity_threshold})."
        )

    # Optional subsample (alignment + Poisson are quadratic-ish in N).
    if args.max_points and len(points) > args.max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(points), args.max_points, replace=False)
        points = points[idx]
        if colors is not None:
            colors = colors[idx]
        print(f"[prepare-marble] subsampled to {len(points):,} points")

    # Up-axis: either explicit (+x, -x, ..., +z, -z) or 'auto' which scans
    # all 6 candidates and picks the best floor (combined RANSAC inlier
    # fraction × density-asymmetry).
    auto_up_scores: dict[str, dict[str, float]] | None = None
    if args.up_axis == "auto":
        chosen_axis, auto_up_scores = auto_detect_up_axis(points)
    else:
        chosen_axis = args.up_axis

    aligned = align_pointcloud(
        points=points,
        initial_world_up=_UP_AXES[chosen_axis],
        metric_scale=args.metric_scale,
        target_ceiling_height_m=args.ceiling_height_m,
        refine_gravity=(args.max_gravity_correction_deg > 0),
        max_gravity_correction_deg=args.max_gravity_correction_deg,
        floor_inlier_thresh_m=args.floor_inlier_thresh_m,
    )
    print(
        f"[align]   gravity (PLY frame) = {aligned.gravity_world_colmap.round(3).tolist()}\n"
        f"          scale = {aligned.scale:.4f} m / PLY-unit\n"
        f"          gravity refinement = {aligned.gravity_refinement_deg:.2f}°\n"
        f"          floor-z (pre-translate, metric) = {aligned.floor_z_pre_translate_m:.3f} m\n"
        f"          floor quality score = {aligned.quality_score:.2f} (0–1, higher is better)"
    )
    if aligned.quality_score < 0.2:
        print(
            "[prepare-marble] WARNING: low floor-quality score — gravity or "
            "floor may be misdetected. Visualize mesh_aligned.ply before "
            "trusting the placements.",
            flush=True,
        )

    # Apply the transform to the (filtered, subsampled) points and build
    # the proxy mesh in aligned-world frame.
    M = aligned.world_from_colmap_4x4()
    pts_aligned = (M[:3, :3] @ points.T).T + M[:3, 3]

    proxy = build_proxy_mesh(
        points_aligned=pts_aligned,
        colors=colors,
        method=args.proxy_method,
        voxel_size_m=args.voxel_size_m,
        poisson_depth=args.poisson_depth,
        poisson_density_quantile=args.poisson_density_quantile,
        alpha_m=args.alpha_m,
    )

    aligned_mesh_path = out_dir / "mesh_aligned.ply"
    write_proxy_ply(proxy, aligned_mesh_path)
    print(f"[prepare-marble] wrote {aligned_mesh_path}")

    # Write a 3DGUT-compatible source.ply to the scene dir. 3DGUT's PLY
    # loader only accepts SH degree 0 (no f_rest_*) or 3 (45 f_rest_*).
    # Marble PLYs commonly carry degree 1 (9 f_rest_*) which 3DGUT
    # rejects with 'found 9, expected 45 or 0'. We normalize here so
    # the downstream `dl3dv_gs_to_usdz.py --ply <out_dir>/source.ply`
    # call always succeeds.
    #
    # Also keep a `source.ply.original` symlink/copy so the unmodified
    # source is available next to the normalized one.
    src_normalized = out_dir / "source.ply"
    src_original = out_dir / "source.ply.original"
    if src_original.exists() or src_original.is_symlink():
        src_original.unlink()
    if args.symlink_source:
        try:
            src_original.symlink_to(ply_path)
        except OSError:
            import shutil
            shutil.copy2(ply_path, src_original)
    else:
        import shutil
        shutil.copy2(ply_path, src_original)

    sh_info = normalize_gs_ply_for_3dgut(
        ply_path, src_normalized, mode=args.gs_sh_mode,
    )

    metadata = {
        # NB: keys named *_colmap_* are kept for compatibility with the
        # existing renderer / placement / mesh-to-usd scripts. They mean
        # "raw input frame" here, not literal COLMAP.
        "scene_hash": ply_path.stem,
        "source_ply": str(ply_path),
        "source_ply_original": str(src_original),
        "source_ply_normalized": str(src_normalized),
        "source_format": "marble_gs_ply",
        "colmap_source_path": None,  # signal: no COLMAP cameras
        "aligned_mesh_path": str(aligned_mesh_path),
        "world_from_colmap_4x4": M.tolist(),
        "scale": aligned.scale,
        "gravity_world_colmap": aligned.gravity_world_colmap.tolist(),
        "gravity_refinement_deg": aligned.gravity_refinement_deg,
        "median_camera_height_m": aligned.median_camera_height_m,
        "floor_quality_score": aligned.quality_score,
        "floor_z_pre_translate_m": aligned.floor_z_pre_translate_m,
        "metric_scale_hint": args.metric_scale,
        "ceiling_height_target_m": args.ceiling_height_m,
        "scale_source": "user" if args.metric_scale is not None else "ceiling_height_heuristic",
        "up_axis_requested": args.up_axis,
        "up_axis_chosen": chosen_axis,
        "auto_up_scores": auto_up_scores,
        "gs_sh_normalization": sh_info,
        "proxy_mesh": {
            "method": args.proxy_method,
            "voxel_size_m": args.voxel_size_m,
            "poisson_depth": args.poisson_depth,
            "alpha_m": args.alpha_m,
            "n_vertices": int(len(proxy.vertices)),
            "n_faces": int(len(proxy.faces)) if proxy.faces is not None else 0,
        },
    }
    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"[prepare-marble] wrote {meta_path}")

    print("\n[prepare-marble] next steps:")
    print(f"    python scripts/dl3dv_mesh_to_usd.py --scene-dir {out_dir}")
    print(f"    python scripts/dl3dv_gs_to_usdz.py  --scene-dir {out_dir} "
          f"--ply {src_normalized}")
    print(f"    python scripts/sample_dl3dv_placements.py "
          f"--scene-dir {out_dir} --camera-floor-radius 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
