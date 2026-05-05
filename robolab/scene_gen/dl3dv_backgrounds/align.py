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
    """Result of aligning a DL3DV reconstruction to a robot-friendly frame."""

    R_align: np.ndarray  # (3, 3) world_aligned_from_world_colmap
    t_align: np.ndarray  # (3,) translation applied after rotation
    scale: float         # uniform scale applied last
    gravity_world_colmap: np.ndarray  # estimated gravity dir in COLMAP frame
    floor_z_pre_translate: float
    median_camera_height_m: float
    quality_score: float  # 1 = great, 0 = bad. See _quality_score.

    def transform_points(self, pts: np.ndarray) -> np.ndarray:
        return self.scale * (pts @ self.R_align.T + self.t_align)

    def world_from_colmap_4x4(self) -> np.ndarray:
        """Single 4x4 that maps COLMAP-world points to aligned-world."""
        T = np.eye(4)
        T[:3, :3] = self.scale * self.R_align
        T[:3, 3] = self.scale * self.t_align
        return T


def _floor_z_from_mesh(vertices_aligned: np.ndarray, low_pct: float = 1.0) -> float:
    return float(np.percentile(vertices_aligned[:, 2], low_pct))


def _quality_score(
    vertices_aligned: np.ndarray,
    floor_z: float,
    floor_band: float = 0.05,
) -> float:
    """How planar is the lowest 5cm slab in the aligned frame? Higher = better.

    A real floor produces a thin slab; a mis-aligned scene produces a
    wedge. We measure inverse-thickness in the slab with vertices in the
    bottom 10% by z.
    """
    z = vertices_aligned[:, 2]
    lo = floor_z
    hi = lo + max(floor_band, 0.5 * (np.percentile(z, 10) - lo))
    band = vertices_aligned[(z >= lo) & (z <= hi)]
    if len(band) < 100:
        return 0.0
    spread = float(np.std(band[:, 2]))
    return float(np.clip(1.0 - spread / floor_band, 0.0, 1.0))


def align_scene_from_mesh(
    *,
    colmap_source_path: Path,
    mesh_path: Path,
    metric_scale_hint: float | None = None,
    target_camera_height_m: float = 1.5,
) -> AlignedScene:
    """Compute (R, t, s) that gravity-aligns and metrically rescales a scene.

    Parameters
    ----------
    colmap_source_path
        Directory with sparse/0/images.txt — used to read camera poses.
    mesh_path
        PLY mesh from fast-pgsr — used for floor detection & quality
        scoring. Vertices only; we don't need faces here.
    metric_scale_hint
        If provided, overrides the camera-height heuristic. Useful when
        you have a known-size object in the scene.
    target_camera_height_m
        Used when ``metric_scale_hint is None``: scene is scaled so the
        median camera ends up this many metres above the detected floor.
    """
    try:
        import trimesh
    except ImportError as e:  # pragma: no cover - install hint
        raise ImportError(
            "trimesh is required for scene alignment; install with `pip install trimesh`"
        ) from e

    R_wc, t_wc = _read_images_metadata(colmap_source_path)
    # COLMAP cam frame: +X right, +Y down, +Z forward → world-up = -R[:, 1]
    cam_ups = -R_wc[:, :, 1]
    world_up = cam_ups.mean(axis=0)
    norm = float(np.linalg.norm(world_up))
    if norm < 1e-3:
        raise RuntimeError("camera up-vectors did not agree on a gravity direction")
    world_up /= norm

    R_align = _R_align_up_to_z(world_up)

    mesh = trimesh.load(str(mesh_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    verts_rot = verts @ R_align.T
    cam_centers_rot = t_wc @ R_align.T

    floor_z = _floor_z_from_mesh(verts_rot)
    t_align = np.array([0.0, 0.0, -floor_z])

    median_cam_height_pre_scale = float(np.median(cam_centers_rot[:, 2] - floor_z))
    if median_cam_height_pre_scale <= 1e-3:
        raise RuntimeError(
            "median camera height above floor is non-positive after gravity alignment "
            "— likely the floor was misdetected (try inverting gravity or check mesh)"
        )

    if metric_scale_hint is not None:
        scale = float(metric_scale_hint)
    else:
        scale = target_camera_height_m / median_cam_height_pre_scale

    verts_final = scale * (verts_rot + t_align)
    quality = _quality_score(verts_final, floor_z=0.0)

    return AlignedScene(
        R_align=R_align,
        t_align=t_align,
        scale=scale,
        gravity_world_colmap=-world_up,
        floor_z_pre_translate=floor_z,
        median_camera_height_m=scale * median_cam_height_pre_scale,
        quality_score=quality,
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
