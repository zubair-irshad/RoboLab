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


def _build_topdown_grids(
    aligned_mesh_path: Path,
    *,
    cell_size_m: float = 0.05,
    clearance_m: float = 2.5,
    floor_band_m: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """Return ``(obstacle, has_floor, (xmin, ymin, xmax, ymax))`` grids.

    ``obstacle[H, W]``: True if any vertex with z ∈ (floor_band, clearance)
    projects into the cell. Walls, furniture, hanging fixtures, etc.

    ``has_floor[H, W]``: True if any RANSAC-inlier floor vertex (i.e.
    actual captured floor) projects into the cell. Critically,
    ``has_floor=False`` outside the room — DL3DV captures stop at room
    boundaries, so anywhere the operator didn't aim at the floor has no
    floor evidence and isn't a valid placement (the simulator floor would
    be physically there, but visually there's nothing — the DL3DV mesh
    has no walls, ceiling, or floor in that region).

    A cell is a valid placement iff ``has_floor AND NOT obstacle`` after
    the appropriate erosion in the caller.

    TSDF "shadow" floaters at z < -0.1 are ignored on both sides.
    """
    try:
        import trimesh
    except ImportError as e:  # pragma: no cover
        raise ImportError("trimesh required; `pip install trimesh`") from e
    from .align import _ransac_floor_plane

    mesh = trimesh.load(str(aligned_mesh_path), process=False)
    if hasattr(mesh, "dump"):
        try:
            mesh = mesh.dump(concatenate=True)
        except Exception:
            pass
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    z = verts[:, 2]

    # Re-derive the floor mask (RANSAC) on the aligned mesh. The floor
    # should be at z≈0, but in case the alignment quality is imperfect,
    # we recompute.
    _, _, floor_mask = _ransac_floor_plane(verts, return_mask=True)

    obstacle_keep = (z > floor_band_m) & (z < clearance_m) & (z > -0.1)
    obstacle_pts = verts[obstacle_keep, :2]
    floor_pts = verts[floor_mask, :2]

    if len(obstacle_pts) == 0 and len(floor_pts) == 0:
        raise RuntimeError("no usable vertices found — mesh empty or alignment broken?")

    # Build a common bbox covering both — that way any cell can be looked
    # up in either grid with the same indexing.
    all_pts = np.concatenate([obstacle_pts, floor_pts]) if (
        len(obstacle_pts) and len(floor_pts)
    ) else (obstacle_pts if len(obstacle_pts) else floor_pts)
    xmin, ymin = all_pts.min(axis=0)
    xmax, ymax = all_pts.max(axis=0)
    xmin, ymin = float(xmin) - 1.0, float(ymin) - 1.0
    xmax, ymax = float(xmax) + 1.0, float(ymax) + 1.0

    W = int(np.ceil((xmax - xmin) / cell_size_m))
    H = int(np.ceil((ymax - ymin) / cell_size_m))

    def _rasterize(points: np.ndarray) -> np.ndarray:
        grid = np.zeros((H, W), dtype=bool)
        if len(points) == 0:
            return grid
        ix = np.clip(((points[:, 0] - xmin) / cell_size_m).astype(int), 0, W - 1)
        iy = np.clip(((points[:, 1] - ymin) / cell_size_m).astype(int), 0, H - 1)
        grid[iy, ix] = True
        return grid

    obstacle = _rasterize(obstacle_pts)
    has_floor = _rasterize(floor_pts)
    return obstacle, has_floor, (xmin, ymin, xmax, ymax)


# Backward-compat alias kept for any callers that imported the old name.
def _build_topdown_occupancy(*args, **kwargs):
    obs, _, bbox = _build_topdown_grids(*args, **kwargs)
    return obs, bbox


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

    obstacle, has_floor, (xmin, ymin, xmax, ymax) = _build_topdown_grids(
        aligned_mesh_path,
        cell_size_m=cell_size_m,
        clearance_m=footprint.clearance_m,
    )

    # Two erosions:
    #   - obstacles dilate by the full erosion radius so any rotation of
    #     the footprint + robot reach stays clear of furniture/walls.
    #   - floor coverage erodes (== free space dilates) so we only sample
    #     from cells where the FULL footprint sits over captured floor.
    #     The half-diagonal is enough here — we don't need the robot's
    #     reach to be over floor (it can extend over a sofa visually).
    erode_cells_obs = int(np.ceil(footprint.erosion_radius_m / cell_size_m))
    half_diag = 0.5 * float(np.hypot(footprint.length_m, footprint.width_m))
    erode_cells_floor = int(np.ceil(half_diag / cell_size_m))

    obstacle_eroded = _erode(obstacle, erode_cells_obs)
    # Floor erosion: we want cells where every pixel within half-diag is
    # ALSO has_floor. Equivalent: free_floor = NOT _erode(NOT has_floor).
    floor_eroded = ~_erode(~has_floor, erode_cells_floor)

    free = floor_eroded & (~obstacle_eroded)

    iy, ix = np.where(free)
    if len(ix) == 0:
        n_floor = int(has_floor.sum())
        n_floor_eroded = int(floor_eroded.sum())
        n_free_no_floor_check = int((~obstacle_eroded).sum())
        raise RuntimeError(
            f"no valid placement cells. has_floor: {n_floor} → eroded "
            f"{n_floor_eroded}; obstacle-eroded free: "
            f"{n_free_no_floor_check}; intersection: 0. "
            f"Footprint may be too large for the captured floor area, or "
            f"the floor mask is too sparse — try smaller --footprint-* or "
            f"a less aggressive --robot-reach."
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
