# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Gravity-align and metric-scale a reconstructed DL3DV scene.

COLMAP (and therefore fast-pgsr's output) lives in an arbitrary world
frame: gravity is unknown, scale is unknown, the floor is wherever it
happens to land. To use a scene as a robot environment we need

    +Z = up (gravity = -Z)
    floor at z = 0
    units approximately metric

Gravity estimate: average COLMAP camera up-vectors. For handheld
captures the operator's "up" is consistent across views, and COLMAP's
images.txt gives us the per-frame world-from-cam rotation directly. The
mean of -R[:, 1] (camera +Y points down in COLMAP/OpenCV → world-up is
-R[:,1]) is a robust gravity prior.

Sanity check: after rotating the mesh so gravity = -Z, the lowest large
horizontal cluster of vertices should have a near-zero z spread. We
return a quality score so the caller can flag scenes that need manual
review.

Metric scale: hard from images alone. Default heuristic — set scale so
median camera height above the detected floor = 1.5 m (typical handheld
capture). Caller can override via ``metric_scale_hint``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ---- COLMAP images.txt parsing (minimal) -----------------------------------

def _parse_colmap_images_txt(path: Path) -> np.ndarray:
    """Return an (N, 3, 3) array of world-from-cam rotations and (N, 3) translations.

    COLMAP images.txt format (per the official spec) — every odd
    non-comment line is::

        IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME

    where (Q, T) is the *world-to-cam* extrinsic (i.e. cam_from_world).
    We invert to get cam-from-world's transpose = world-from-cam.

    Returns ``(R_world_cam, t_world_cam)`` stacks.
    """
    rotations: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    with path.open() as f:
        line_idx = 0
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            line_idx += 1
            if line_idx % 2 == 0:  # the second line is 2D-3D matches; skip
                continue
            parts = line.split()
            qw, qx, qy, qz = (float(parts[i]) for i in range(1, 5))
            tx, ty, tz = (float(parts[i]) for i in range(5, 8))
            R_cw = _quat_to_R(qw, qx, qy, qz)  # cam-from-world
            t_cw = np.array([tx, ty, tz])
            R_wc = R_cw.T
            t_wc = -R_wc @ t_cw
            rotations.append(R_wc)
            translations.append(t_wc)
    if not rotations:
        raise ValueError(f"no image entries parsed from {path}")
    return np.stack(rotations), np.stack(translations)


def _quat_to_R(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


def _parse_colmap_images_bin(path: Path) -> np.ndarray:
    """Parse COLMAP images.bin → (R_world_cam, t_world_cam) stacks.

    Format (from COLMAP's read_write_model.py, BSD-licensed):

        uint64  num_images
        for each image:
            uint32  image_id
            double  qw, qx, qy, qz
            double  tx, ty, tz
            uint32  camera_id
            char[]  name (null-terminated)
            uint64  num_points2D
            (double x, double y, int64 point3D_id) * num_points2D
    """
    import struct

    rotations: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    with path.open("rb") as f:
        (num_images,) = struct.unpack("<Q", f.read(8))
        for _ in range(num_images):
            f.read(4)  # image_id
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            f.read(4)  # camera_id
            # name: read until null byte
            name_chars: list[bytes] = []
            while True:
                ch = f.read(1)
                if ch == b"\x00" or ch == b"":
                    break
                name_chars.append(ch)
            (num_points2d,) = struct.unpack("<Q", f.read(8))
            f.seek(num_points2d * 24, 1)  # skip 2D-3D matches
            R_cw = _quat_to_R(qw, qx, qy, qz)
            t_cw = np.array([tx, ty, tz])
            R_wc = R_cw.T
            t_wc = -R_wc @ t_cw
            rotations.append(R_wc)
            translations.append(t_wc)
    if not rotations:
        raise ValueError(f"no image entries parsed from {path}")
    return np.stack(rotations), np.stack(translations)


def _read_images_metadata(colmap_source_path: Path) -> tuple[np.ndarray, np.ndarray]:
    sparse0 = colmap_source_path / "sparse" / "0"
    txt = sparse0 / "images.txt"
    if txt.is_file():
        return _parse_colmap_images_txt(txt)
    binp = sparse0 / "images.bin"
    if binp.is_file():
        return _parse_colmap_images_bin(binp)
    raise FileNotFoundError(
        f"neither images.txt nor images.bin found under {sparse0}"
    )


# ---- alignment math --------------------------------------------------------

def _R_align_up_to_z(world_up: np.ndarray) -> np.ndarray:
    """Rotation R such that R @ world_up = +Z, with minimal roll."""
    u = world_up / np.linalg.norm(world_up)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(u, z)
    s = np.linalg.norm(v)
    c = float(u @ z)
    if s < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array(
        [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]
    )
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


@dataclass
class AlignedScene:
    """Result of aligning a DL3DV reconstruction to a robot-friendly frame.

    Convention: ``transform_points(p) = scale * R_align @ p + t_align``
    where ``t_align`` is **already in metric units** (post-scale). This
    differs from earlier revisions where t_align was in COLMAP units.
    """

    R_align: np.ndarray  # (3, 3) world_aligned_from_world_colmap (refined)
    t_align: np.ndarray  # (3,) metric translation, applied after scale*R
    scale: float         # COLMAP units → metres
    gravity_world_colmap: np.ndarray  # final gravity dir in COLMAP frame
    floor_z_pre_translate_m: float    # floor z (in metric, post-scale, pre-translate)
    median_camera_height_m: float
    quality_score: float               # RANSAC inlier fraction in [0, 1]
    gravity_refinement_deg: float      # angle by which gravity was refined; 0 if not

    def transform_points(self, pts: np.ndarray) -> np.ndarray:
        return self.scale * (pts @ self.R_align.T) + self.t_align

    def world_from_colmap_4x4(self) -> np.ndarray:
        """Single 4x4 that maps COLMAP-world points to aligned-world."""
        T = np.eye(4)
        T[:3, :3] = self.scale * self.R_align
        T[:3, 3] = self.t_align
        return T


def _ransac_floor_plane(
    vertices_aligned: np.ndarray,
    *,
    lower_pct: float = 30.0,
    inlier_thresh: float = 0.03,
    max_normal_dev_deg: float = 25.0,
    max_iters: int = 400,
    rng_seed: int = 0,
    return_mask: bool = False,
) -> tuple[float, float] | tuple[float, float, np.ndarray]:
    """Find the floor z + inlier fraction by RANSAC plane fit.

    Restricts to vertices in the lower ``lower_pct`` of z so we don't
    accidentally lock onto a tabletop. Rejects plane candidates whose
    normal deviates from +Z by more than ``max_normal_dev_deg`` (so we
    reject vertical walls). The largest inlier set wins.

    Returns ``(floor_z, inlier_fraction)`` where ``inlier_fraction`` is
    the fraction of *lower-portion* vertices on the winning plane —
    high values (>0.4) indicate a strong, dense floor; low values
    indicate the lower portion is dominated by floaters.

    If ``return_mask`` is True, also returns a boolean array of length
    ``len(vertices_aligned)`` flagging which input vertices are floor
    inliers (i.e. in the lower portion AND within ``inlier_thresh`` of
    the winning plane).

    Robust to TSDF "shadow" artifacts: those are sparse and unaligned,
    so they accumulate few inliers and lose to the real floor.
    """
    z = vertices_aligned[:, 2]
    z_thresh = float(np.percentile(z, lower_pct))
    lower_mask = z <= z_thresh
    candidates_full = vertices_aligned[lower_mask]
    if len(candidates_full) < 100:
        if return_mask:
            return float(np.median(z)), 0.0, np.zeros(len(vertices_aligned), dtype=bool)
        return float(np.median(z)), 0.0

    rng = np.random.default_rng(rng_seed)
    if len(candidates_full) > 20_000:
        sub_idx = rng.choice(len(candidates_full), 20_000, replace=False)
        candidates = candidates_full[sub_idx]
    else:
        candidates = candidates_full

    cos_thresh = float(np.cos(np.deg2rad(max_normal_dev_deg)))
    up = np.array([0.0, 0.0, 1.0])

    best_inliers = 0
    best_z = float(np.median(candidates[:, 2]))
    best_n: np.ndarray | None = None
    best_d: float = 0.0
    for _ in range(max_iters):
        idx = rng.choice(len(candidates), 3, replace=False)
        p0, p1, p2 = candidates[idx]
        n = np.cross(p1 - p0, p2 - p0)
        nn = float(np.linalg.norm(n))
        if nn < 1e-9:
            continue
        n = n / nn
        if abs(float(n @ up)) < cos_thresh:
            continue  # skip non-horizontal candidates
        d = -float(n @ p0)
        dists = np.abs(candidates @ n + d)
        n_in = int((dists < inlier_thresh).sum())
        if n_in > best_inliers:
            best_inliers = n_in
            inlier_pts = candidates[dists < inlier_thresh]
            best_z = float(np.median(inlier_pts[:, 2]))
            best_n = n
            best_d = d

    inlier_frac = best_inliers / len(candidates)
    if not return_mask:
        return best_z, inlier_frac

    full_mask = np.zeros(len(vertices_aligned), dtype=bool)
    if best_n is not None:
        all_dists = np.abs(vertices_aligned @ best_n + best_d)
        full_mask = lower_mask & (all_dists < inlier_thresh)
    return best_z, inlier_frac, full_mask


def _fit_plane_normal_svd(points: np.ndarray) -> np.ndarray:
    """Best-fit plane normal via SVD. Returns a unit vector with z>0."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    # Smallest singular vector of the centered cloud is the plane normal.
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    n = Vt[-1]
    n = n / (np.linalg.norm(n) + 1e-12)
    if n[2] < 0:
        n = -n
    return n


def align_scene_from_mesh(
    *,
    colmap_source_path: Path,
    mesh_path: Path,
    metric_scale_hint: float | None = None,
    target_camera_height_m: float = 1.5,
    refine_gravity: bool = True,
    max_gravity_correction_deg: float = 25.0,
    floor_inlier_thresh_m: float = 0.03,
) -> AlignedScene:
    """Gravity-align + metric-rescale + floor-zero a reconstruction.

    Pipeline (in order):

      1. Initial gravity from COLMAP camera-up averaging (~5-10° error).
      2. Determine metric scale: from ``metric_scale_hint`` if given,
         else camera-height heuristic with a rough percentile floor.
      3. Apply scale + initial gravity → mesh in approximately metric,
         approximately upright frame.
      4. RANSAC floor with **metric** inlier threshold (3 cm by default).
      5. (Optional) refine gravity from the fitted plane's SVD normal,
         compose a small correction with initial R, re-RANSAC for the
         final floor.
      6. Translate so floor sits at z = 0.

    The 4×4 returned via ``AlignedScene.world_from_colmap_4x4()`` maps
    raw COLMAP-frame points to the aligned, metric, floor-at-zero frame.
    """
    try:
        import trimesh
    except ImportError as e:  # pragma: no cover
        raise ImportError("trimesh is required; `pip install trimesh`") from e

    R_wc, t_wc = _read_images_metadata(colmap_source_path)
    # 1. Initial gravity from camera ups (COLMAP cam +Y is down → world-up = -R[:,1])
    cam_ups = -R_wc[:, :, 1]
    world_up = cam_ups.mean(axis=0)
    norm = float(np.linalg.norm(world_up))
    if norm < 1e-3:
        raise RuntimeError("camera up-vectors did not agree on a gravity direction")
    world_up /= norm
    R_align_initial = _R_align_up_to_z(world_up)

    mesh = trimesh.load(str(mesh_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    verts = np.asarray(mesh.vertices, dtype=np.float64)

    # Apply initial gravity (no scale yet)
    verts_rot = verts @ R_align_initial.T
    cam_centers_rot = t_wc @ R_align_initial.T

    # 2. Metric scale
    if metric_scale_hint is not None:
        scale = float(metric_scale_hint)
    else:
        # Camera-height heuristic needs a rough floor in COLMAP units.
        # We use RANSAC on the unscaled rotated mesh — its threshold is
        # "fuzzy" here because we don't know units yet, but for the
        # purpose of getting a floor estimate accurate enough for the
        # median-camera-height calculation, that's fine.
        rough_floor_z, _ = _ransac_floor_plane(
            verts_rot, inlier_thresh=floor_inlier_thresh_m * 2.0,
        )
        median_h = float(np.median(cam_centers_rot[:, 2] - rough_floor_z))
        if median_h <= 1e-6:
            raise RuntimeError(
                "median camera height above rough floor is non-positive — "
                "gravity may be flipped"
            )
        scale = target_camera_height_m / median_h

    # 3. Apply scale → mesh now in roughly metric, roughly upright frame.
    verts_metric = scale * verts_rot
    cam_centers_metric = scale * cam_centers_rot

    # 4. RANSAC with proper metric threshold
    floor_z_m, inlier_frac, floor_mask = _ransac_floor_plane(
        verts_metric,
        inlier_thresh=floor_inlier_thresh_m,
        return_mask=True,
    )

    # 5. Optional gravity refinement from fitted plane normal
    R_align = R_align_initial
    refinement_deg = 0.0
    if refine_gravity and int(floor_mask.sum()) >= 100:
        plane_normal = _fit_plane_normal_svd(verts_metric[floor_mask])
        # Plane normal in current rotated frame; ideally [0,0,1]. The
        # angle between it and +Z is the residual gravity error.
        cos_a = float(np.clip(plane_normal[2], -1.0, 1.0))
        refinement_deg = float(np.degrees(np.arccos(cos_a)))
        if refinement_deg <= max_gravity_correction_deg:
            R_correction = _R_align_up_to_z(plane_normal)
            R_align = R_correction @ R_align_initial
            # Re-rotate (and re-scale) and re-RANSAC for the FINAL floor
            verts_metric = scale * (verts @ R_align.T)
            cam_centers_metric = scale * (t_wc @ R_align.T)
            floor_z_m, inlier_frac, floor_mask = _ransac_floor_plane(
                verts_metric,
                inlier_thresh=floor_inlier_thresh_m,
                return_mask=True,
            )
        else:
            print(
                f"[align] gravity correction {refinement_deg:.1f}° exceeds "
                f"max ({max_gravity_correction_deg}°); skipping refinement"
            )
            refinement_deg = 0.0

    # 6. Translate so floor at z=0 (in metric)
    t_align_metric = np.array([0.0, 0.0, -floor_z_m])

    # Final gravity vector in COLMAP frame: gravity_aligned = (0,0,-1)
    # In COLMAP frame: gravity_colmap = R_align.T @ (0,0,-1)
    gravity_world_colmap = -R_align.T @ np.array([0.0, 0.0, 1.0])

    median_cam_height_m = float(np.median(cam_centers_metric[:, 2] - floor_z_m))

    return AlignedScene(
        R_align=R_align,
        t_align=t_align_metric,
        scale=scale,
        gravity_world_colmap=gravity_world_colmap,
        floor_z_pre_translate_m=floor_z_m,
        median_camera_height_m=median_cam_height_m,
        quality_score=inlier_frac,
        gravity_refinement_deg=refinement_deg,
    )


def write_aligned_mesh(
    aligned: AlignedScene,
    *,
    src_mesh_path: Path,
    out_mesh_path: Path,
) -> None:
    """Apply the alignment transform to a mesh and write the result."""
    try:
        import trimesh
    except ImportError as e:  # pragma: no cover - install hint
        raise ImportError("trimesh is required") from e
    mesh = trimesh.load(str(src_mesh_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    mesh.apply_transform(aligned.world_from_colmap_4x4())
    out_mesh_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_mesh_path))
