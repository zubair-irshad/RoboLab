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

    Obstacles get checked in TWO height bands separately:

      - Below ``table_height_m``: must clear the footprint half-diag
        (the task's own table can't overlap furniture down here).
      - Between ``table_height_m`` and ``clearance_m``: must clear the
        footprint half-diag PLUS ``robot_reach_m`` (the arm sweeps
        laterally up here).

    A sofa (low) within reach of the robot is fine — the arm goes over.
    A wall or tall bookcase within reach is not — the arm hits it.
    """
    length_m: float = 1.5
    width_m: float = 1.0
    clearance_m: float = 2.5         # top of robot working volume
    robot_reach_m: float = 0.85      # lateral reach beyond footprint edge
    table_height_m: float = 0.75     # split between low / high obstacle bands

    @property
    def half_diag_m(self) -> float:
        return 0.5 * float(np.hypot(self.length_m, self.width_m))

    @property
    def low_erosion_radius_m(self) -> float:
        """Below table top: only the table footprint matters."""
        return self.half_diag_m

    @property
    def high_erosion_radius_m(self) -> float:
        """Above table top: footprint + arm reach (any yaw)."""
        return self.half_diag_m + max(0.0, self.robot_reach_m)

    # Backward-compat alias used by older callers; equivalent to
    # ``high_erosion_radius_m`` (the more conservative of the two).
    @property
    def erosion_radius_m(self) -> float:
        return self.high_erosion_radius_m


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
    table_height_m: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """Return ``(low_obs, high_obs, has_floor, (xmin, ymin, xmax, ymax))``.

    Obstacles split into two height bands so the caller can apply
    different erosion radii:

      ``low_obs`` — vertices in (floor_band, table_height). Furniture
      that the task table cannot overlap (sofas, coffee tables).

      ``high_obs`` — vertices in (table_height, clearance). Things that
      block the robot's swung arm (walls, tall bookcases, hanging
      fixtures). Most low furniture doesn't reach this band, so a
      placement next to a sofa stays valid as long as the airspace
      above the sofa is clear.

      ``has_floor`` — RANSAC-inlier floor vertices. Required so we
      don't sample placements outside the captured room boundary.

    TSDF "shadow" floaters at z < -0.1 are filtered everywhere.
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

    _, _, floor_mask = _ransac_floor_plane(verts, return_mask=True)

    valid_height = (z > floor_band_m) & (z < clearance_m) & (z > -0.1)
    is_low = valid_height & (z <= table_height_m)
    is_high = valid_height & (z > table_height_m)

    low_pts = verts[is_low, :2]
    high_pts = verts[is_high, :2]
    floor_pts = verts[floor_mask, :2]

    if len(low_pts) == 0 and len(high_pts) == 0 and len(floor_pts) == 0:
        raise RuntimeError("no usable vertices found — mesh empty or alignment broken?")

    pieces = [a for a in (low_pts, high_pts, floor_pts) if len(a) > 0]
    all_pts = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
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

    low_obs = _rasterize(low_pts)
    high_obs = _rasterize(high_pts)
    has_floor = _rasterize(floor_pts)
    return low_obs, high_obs, has_floor, (xmin, ymin, xmax, ymax)


# Backward-compat alias kept for any callers that imported the old name.
def _build_topdown_occupancy(*args, **kwargs):
    low, high, _, bbox = _build_topdown_grids(*args, **kwargs)
    return (low | high), bbox


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


def _close(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    """Morphological closing: fill holes / bridge gaps up to ``radius_cells``.

    Closing = dilate then erode with the same kernel. Connects fragments
    of a sparse mask (e.g. RANSAC floor inliers occluded by furniture)
    without extending the overall extent — outside the original mask's
    convex hull, dilate-then-erode is a no-op.
    """
    if radius_cells <= 0:
        return mask.copy()
    try:
        from scipy.ndimage import binary_closing
        struct = np.ones((2 * radius_cells + 1, 2 * radius_cells + 1), dtype=bool)
        return binary_closing(mask, structure=struct)
    except ImportError:
        # Compose dilate(erode^c)^c via _erode.
        dilated = _erode(mask, radius_cells)
        return ~_erode(~dilated, radius_cells)


def _farthest_point_sample(
    candidates_xy: np.ndarray, n: int, rng: np.random.Generator,
) -> np.ndarray:
    """Greedy farthest-point sampling. Returns indices into ``candidates_xy``.

    Picks the first point at random, then iteratively the point with
    maximum min-distance to already-selected points. Spreads samples
    across the free region — better than uniform random when the free
    region has multiple disconnected components or strong local clusters.
    """
    n = min(n, len(candidates_xy))
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    selected = [int(rng.integers(0, len(candidates_xy)))]
    dists = np.linalg.norm(candidates_xy - candidates_xy[selected[0]], axis=1)
    for _ in range(n - 1):
        i = int(np.argmax(dists))
        if dists[i] <= 1e-9:
            break  # all remaining cells are duplicates of selected ones
        selected.append(i)
        new_d = np.linalg.norm(candidates_xy - candidates_xy[i], axis=1)
        dists = np.minimum(dists, new_d)
    return np.asarray(selected, dtype=np.int64)


def sample_placements(
    *,
    aligned_mesh_path: Path,
    n_placements: int = 20,
    footprint: FootprintSpec | None = None,
    cell_size_m: float = 0.05,
    yaw_strategy: str = "face_centroid",
    rng_seed: int = 0,
    floor_close_radius_m: float = 0.5,
    sampling: str = "farthest_point",
    min_separation_m: float = 0.5,
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

    low_obs, high_obs, has_floor, (xmin, ymin, xmax, ymax) = _build_topdown_grids(
        aligned_mesh_path,
        cell_size_m=cell_size_m,
        clearance_m=footprint.clearance_m,
        table_height_m=footprint.table_height_m,
    )

    # Close the floor mask first to bridge gaps caused by furniture
    # occluding floor capture. Doesn't extend the floor outside its
    # original convex hull.
    close_cells = int(np.ceil(floor_close_radius_m / cell_size_m))
    has_floor_closed = _close(has_floor, close_cells)

    # Three independent erosions, AND'd together:
    #   - low obstacles dilate by half_diag (table-only clearance)
    #   - high obstacles dilate by half_diag + reach (arm sweep)
    #   - floor coverage erodes by half_diag (footprint over captured floor)
    erode_low = int(np.ceil(footprint.low_erosion_radius_m / cell_size_m))
    erode_high = int(np.ceil(footprint.high_erosion_radius_m / cell_size_m))
    erode_floor = int(np.ceil(footprint.half_diag_m / cell_size_m))

    low_eroded = _erode(low_obs, erode_low)
    high_eroded = _erode(high_obs, erode_high)
    floor_eroded = ~_erode(~has_floor_closed, erode_floor)

    free = floor_eroded & (~low_eroded) & (~high_eroded)

    iy, ix = np.where(free)
    if len(ix) == 0:
        raise RuntimeError(
            f"no valid placement cells. Diagnostics:\n"
            f"  has_floor (raw):      {int(has_floor.sum()):8d} cells\n"
            f"  has_floor (closed):   {int(has_floor_closed.sum()):8d} cells\n"
            f"  floor footprint-fit:  {int(floor_eroded.sum()):8d} cells\n"
            f"  low-obstacle free:    {int((~low_eroded).sum()):8d} cells\n"
            f"  high-obstacle free:   {int((~high_eroded).sum()):8d} cells\n"
            f"  intersection:                0 cells\n"
            f"Most common causes:\n"
            f"  - Footprint+reach larger than the room: shrink --footprint-* "
            f"or --robot-reach.\n"
            f"  - Floor mask too sparse: increase --floor-close-radius "
            f"(default 0.5 m, try 1.0 or 1.5).\n"
            f"  - High-obstacle band picks up walls everywhere: lower "
            f"--clearance (default 2.5 m, try 1.5)."
        )

    centroid = np.array([0.5 * (xmin + xmax), 0.5 * (ymin + ymax)])

    rng = np.random.default_rng(rng_seed)
    cell_xy = np.column_stack([
        xmin + (ix + 0.5) * cell_size_m,
        ymin + (iy + 0.5) * cell_size_m,
    ])

    if sampling == "farthest_point":
        pick = _farthest_point_sample(cell_xy, n_placements, rng)
    elif sampling == "random":
        pick = rng.choice(len(ix), size=min(n_placements, len(ix)), replace=False)
    else:
        raise ValueError(f"unknown sampling strategy {sampling!r}")

    # Optional min-separation pruning: enforce that picks are at least
    # min_separation_m apart in xy. FPS already maximizes spread, but a
    # hard floor stops two picks from collapsing onto a 5cm cluster when
    # the free region is degenerate.
    if min_separation_m > 0:
        kept: list[int] = []
        for idx in pick:
            p = cell_xy[idx]
            if all(np.linalg.norm(p - cell_xy[k]) >= min_separation_m for k in kept):
                kept.append(int(idx))
        pick = np.asarray(kept, dtype=np.int64)
        if len(pick) < min(n_placements, len(cell_xy)):
            print(
                f"[placement] only {len(pick)} placements satisfy "
                f"min_separation={min_separation_m} m (requested {n_placements}); "
                f"the free region is small or fragmented"
            )

    placements: list[Placement] = []
    for k in pick:
        x = float(cell_xy[k, 0])
        y = float(cell_xy[k, 1])
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
