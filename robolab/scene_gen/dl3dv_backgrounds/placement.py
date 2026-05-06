# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Sample placements for a RoboLab foreground inside a DL3DV background.

Pipeline:

    1. Re-derive RANSAC floor mask on the aligned mesh vertices.
    2. Project non-floor vertices to a 2D top-down occupancy grid (5cm
       cells by default). Each cell is occupied if any non-floor vertex
       projects into it within a clearance height window.
    3. Erode by half the foreground footprint so the task table won't
       overlap walls/furniture.
    4. Sample N free pixels uniformly → ``(x, y, z=0, yaw)`` poses.

Footprint convention: a (length, width) rectangle aligned to the +X axis
*before* yaw rotation. For typical RoboLab tasks, ~1.5 m × 1.0 m covers
table + robot mount; the robot's reach disk extends another ~0.85 m above.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from .align import _ransac_floor_plane


@dataclass
class FootprintSpec:
    """3D volume the task occupies, before yaw rotation.

    The robot's actual working volume is a column above the placement:
    table footprint (length × width) at the base, sweeping up to the
    full reach of the arm. Both clearance_m (vertical extent) and
    robot_reach_m (lateral margin around the footprint for the arm)
    matter for whether a placement is collision-free.
    """
    length_m: float = 1.5   # along +X (task footprint at floor level)
    width_m: float = 1.0    # along +Y
    clearance_m: float = 2.5  # vertical reach of the robot column
    robot_reach_m: float = 0.85  # lateral arm reach beyond the footprint edge

    @property
    def erosion_radius_m(self) -> float:
        """Conservative outer radius for free-space erosion.

        ``half_diag`` covers any rotation of the rectangle; adding
        ``robot_reach`` keeps the arm's swept lateral volume clear of
        obstacles regardless of yaw.
        """
        half_diag = 0.5 * float(np.hypot(self.length_m, self.width_m))
        return half_diag + max(0.0, self.robot_reach_m)


@dataclass
class Placement:
    """A sampled (x, y, yaw) pose with its footprint on the floor."""
    x_m: float
    y_m: float
    yaw_rad: float
    footprint: FootprintSpec

    def to_jsonable(self) -> dict:
        d = asdict(self)
        d["footprint"] = asdict(self.footprint)
        return d

    def corners_xy(self) -> np.ndarray:
        """4×2 array of footprint corners in world frame, after yaw rotation."""
        L, W = self.footprint.length_m, self.footprint.width_m
        local = np.array([[+L / 2, +W / 2], [+L / 2, -W / 2],
                          [-L / 2, -W / 2], [-L / 2, +W / 2]])
        c, s = np.cos(self.yaw_rad), np.sin(self.yaw_rad)
        R = np.array([[c, -s], [s, c]])
        return (local @ R.T) + np.array([self.x_m, self.y_m])


def _build_topdown_occupancy(
    aligned_mesh_path: Path,
    *,
    cell_size_m: float = 0.05,
    clearance_m: float = 1.5,
    floor_band_m: float = 0.05,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Return ``(occupied[H, W], (xmin, ymin, xmax, ymax))``.

    A cell is occupied if any vertex with z in ``(floor_band, clearance)``
    projects into it. Vertices on the floor (z<=floor_band) and vertices
    above the clearance ceiling are ignored — we don't care about the
    ceiling and we want the floor itself to be free.

    Note: because the aligned mesh has TSDF "shadow" floaters below z=0,
    we also exclude any vertex with z < -0.1 m (those are clearly noise).
    """
    try:
        import trimesh
    except ImportError as e:  # pragma: no cover
        raise ImportError("trimesh required; `pip install trimesh`") from e
    mesh = trimesh.load(str(aligned_mesh_path), process=False)
    if hasattr(mesh, "dump"):
        try:
            mesh = mesh.dump(concatenate=True)
        except Exception:
            pass
    verts = np.asarray(mesh.vertices, dtype=np.float64)

    z = verts[:, 2]
    keep = (z > floor_band_m) & (z < clearance_m) & (z > -0.1)
    pts = verts[keep, :2]
    if len(pts) == 0:
        raise RuntimeError(
            f"no vertices in clearance band ({floor_band_m}, {clearance_m}) m — "
            f"mesh empty or alignment broken?"
        )

    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    # Pad bbox by 1 m so the erosion has somewhere to retreat to.
    xmin, ymin = xmin - 1.0, ymin - 1.0
    xmax, ymax = xmax + 1.0, ymax + 1.0

    W = int(np.ceil((xmax - xmin) / cell_size_m))
    H = int(np.ceil((ymax - ymin) / cell_size_m))
    occ = np.zeros((H, W), dtype=bool)
    ix = np.clip(((pts[:, 0] - xmin) / cell_size_m).astype(int), 0, W - 1)
    iy = np.clip(((pts[:, 1] - ymin) / cell_size_m).astype(int), 0, H - 1)
    occ[iy, ix] = True
    return occ, (float(xmin), float(ymin), float(xmax), float(ymax))


def _erode(occupancy: np.ndarray, radius_cells: int) -> np.ndarray:
    """Binary dilation of occupancy = erosion of free space.

    Uses scipy if available; falls back to a numpy convolution.
    """
    if radius_cells <= 0:
        return occupancy.copy()
    try:
        from scipy.ndimage import binary_dilation
        struct = np.ones((2 * radius_cells + 1, 2 * radius_cells + 1), dtype=bool)
        return binary_dilation(occupancy, structure=struct)
    except ImportError:
        from numpy.lib.stride_tricks import sliding_window_view
        pad = radius_cells
        padded = np.pad(occupancy, pad, mode="constant", constant_values=False)
        windowed = sliding_window_view(padded, (2 * pad + 1, 2 * pad + 1))
        return windowed.any(axis=(-1, -2))


def sample_placements(
    *,
    aligned_mesh_path: Path,
    n_placements: int = 20,
    footprint: FootprintSpec | None = None,
    cell_size_m: float = 0.05,
    yaw_strategy: str = "face_centroid",
    rng_seed: int = 0,
) -> list[Placement]:
    """Sample ``n_placements`` valid foreground poses on the aligned floor.

    Parameters
    ----------
    aligned_mesh_path
        Path to ``mesh_aligned.ply`` (output of prepare_dl3dv_scene).
    footprint
        Task footprint. Default: 1.5 m × 1.0 m, 1.5 m clearance.
    yaw_strategy
        ``"face_centroid"`` — task +X faces the room centroid (cameras
        captured the room from inside, so this faces the robot toward
        the visually-rich part).
        ``"random"`` — uniform random yaw.
    """
    if footprint is None:
        footprint = FootprintSpec()

    occ, (xmin, ymin, xmax, ymax) = _build_topdown_occupancy(
        aligned_mesh_path,
        cell_size_m=cell_size_m,
        clearance_m=footprint.clearance_m,
    )

    # Erode by an outer-radius that covers both rotation of the rectangle
    # AND the robot's lateral reach beyond the footprint edge. This is
    # conservative (a Minkowski sum approximation), but cheap and ensures
    # the robot can sweep its working volume without hitting walls or
    # furniture regardless of yaw.
    erode_cells = int(np.ceil(footprint.erosion_radius_m / cell_size_m))
    occ_eroded = _erode(occ, erode_cells)
    free = ~occ_eroded

    # Free cells, converted back to world-xy.
    iy, ix = np.where(free)
    if len(ix) == 0:
        raise RuntimeError(
            f"no free cells after erosion (footprint half-diag = "
            f"{footprint.half_diag_m:.2f} m); scene may be too cluttered "
            f"or footprint too large"
        )

    centroid = np.array([0.5 * (xmin + xmax), 0.5 * (ymin + ymax)])

    rng = np.random.default_rng(rng_seed)
    pick = rng.choice(len(ix), size=min(n_placements, len(ix)), replace=False)
    placements: list[Placement] = []
    for k in pick:
        x = xmin + (ix[k] + 0.5) * cell_size_m
        y = ymin + (iy[k] + 0.5) * cell_size_m
        if yaw_strategy == "random":
            yaw = float(rng.uniform(-np.pi, np.pi))
        elif yaw_strategy == "face_centroid":
            d = centroid - np.array([x, y])
            yaw = float(np.arctan2(d[1], d[0]))
        else:
            raise ValueError(f"unknown yaw_strategy {yaw_strategy!r}")
        placements.append(Placement(x_m=float(x), y_m=float(y),
                                    yaw_rad=yaw, footprint=footprint))
    return placements


def write_placements_json(placements: list[Placement], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"n": len(placements), "placements": [p.to_jsonable() for p in placements]},
        indent=2,
    ))
