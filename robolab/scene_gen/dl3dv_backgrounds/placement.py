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
    floor_definition: str = "occupied-column",
    floor_z_tolerance_m: float = 0.15,
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

      ``has_floor`` — boolean per-cell mask of where the floor is
      believed to exist. Two definitions are supported:

        ``ransac`` — only cells where a RANSAC floor inlier landed.
        Conservative; works well when cameras directly captured the
        floor (DL3DV). Fragile on synthesis-based scenes (Marble,
        Echo2) where gaussians cluster on furniture and leave the
        actual floor sparsely covered.

        ``occupied-column`` (default) — every xy cell whose **lowest**
        mesh point sits within ``floor_z_tolerance_m`` of the RANSAC
        floor z. Fills in occluded floor: a vanity at xy=(1,1) has its
        base on the floor, so the lowest point at that xy is at
        floor_z and the cell is correctly marked as floor (the vanity
        itself is also marked as a low obstacle in ``low_obs``, which
        the caller's erosion handles separately). Walls at floor cells
        likewise become floor *and* high obstacles — the high-obstacle
        erosion still removes them from valid placements.

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

    floor_z, _, ransac_mask = _ransac_floor_plane(verts, return_mask=True)

    valid_height = (z > floor_band_m) & (z < clearance_m) & (z > -0.1)
    is_low = valid_height & (z <= table_height_m)
    is_high = valid_height & (z > table_height_m)

    low_pts = verts[is_low, :2]
    high_pts = verts[is_high, :2]
    ransac_floor_pts = verts[ransac_mask, :2]

    if len(low_pts) == 0 and len(high_pts) == 0 and len(ransac_floor_pts) == 0:
        raise RuntimeError("no usable vertices found — mesh empty or alignment broken?")

    # Bounding box: take all relevant points so the grid covers everything.
    pieces = [a for a in (low_pts, high_pts, ransac_floor_pts) if len(a) > 0]
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

    if floor_definition == "ransac":
        has_floor = _rasterize(ransac_floor_pts)
    elif floor_definition == "occupied-column":
        # Per-cell minimum-z reduction. We fill an HxW grid with +inf,
        # then for every vertex update the cell to min(current, z).
        # numpy.minimum.at handles repeated indices correctly.
        valid = z > -0.1  # drop TSDF shadows
        v_xy = verts[valid, :2]
        v_z = z[valid]
        ix = np.clip(((v_xy[:, 0] - xmin) / cell_size_m).astype(int), 0, W - 1)
        iy = np.clip(((v_xy[:, 1] - ymin) / cell_size_m).astype(int), 0, H - 1)
        cell_min_z = np.full((H, W), np.inf, dtype=np.float64)
        np.minimum.at(cell_min_z, (iy, ix), v_z)
        # Floor exists wherever the column's lowest mesh point is at
        # floor level (within tolerance). +inf cells (no mesh in column)
        # stay False naturally.
        has_floor = cell_min_z <= (floor_z + floor_z_tolerance_m)
        n_ransac_cells = int(_rasterize(ransac_floor_pts).sum())
        n_column_cells = int(has_floor.sum())
        print(
            f"[placement] floor: occupied-column at floor_z={floor_z:.3f} m "
            f"± {floor_z_tolerance_m:.2f} m -> {n_column_cells:,} cells "
            f"(vs RANSAC-only: {n_ransac_cells:,} cells)"
        )
    else:
        raise ValueError(
            f"unknown floor_definition {floor_definition!r}; "
            f"expected 'ransac' or 'occupied-column'"
        )
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


def _keep_components_above_area(
    mask: np.ndarray, min_cells: int,
) -> tuple[np.ndarray, list[int]]:
    """Drop connected components in ``mask`` smaller than ``min_cells``.

    Returns ``(filtered_mask, kept_areas)`` where ``kept_areas`` is a
    sorted-descending list of the surviving component areas (in cells).

    4-connectivity. Uses scipy when available, falls back to an
    iterative numpy flood fill.
    """
    if min_cells <= 1 or not mask.any():
        return mask.copy(), [int(mask.sum())] if mask.any() else []
    try:
        from scipy.ndimage import label  # type: ignore[import-not-found]
        labels, n_components = label(mask)
    except ImportError:
        labels = np.zeros_like(mask, dtype=np.int32)
        n_components = 0
        h, w = mask.shape
        for sy in range(h):
            for sx in range(w):
                if not mask[sy, sx] or labels[sy, sx] != 0:
                    continue
                n_components += 1
                stack = [(sy, sx)]
                labels[sy, sx] = n_components
                while stack:
                    y, x = stack.pop()
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = n_components
                            stack.append((ny, nx))

    sizes = np.bincount(labels.ravel())  # sizes[0] = background
    keep_labels = np.where(sizes >= min_cells)[0]
    keep_labels = keep_labels[keep_labels != 0]
    if len(keep_labels) == 0:
        # Nothing meets the threshold; fall back to keeping the largest
        # so the caller still has something to work with.
        if n_components == 0:
            return np.zeros_like(mask), []
        largest = int(np.argmax(sizes[1:]) + 1)
        out = labels == largest
        return out, [int(sizes[largest])]
    out = np.isin(labels, keep_labels)
    kept_areas = sorted((int(sizes[k]) for k in keep_labels), reverse=True)
    return out, kept_areas


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
    camera_centers_world: np.ndarray | None = None,
    max_camera_distance_m: float | None = None,
    camera_floor_radius_m: float = 1.5,
    min_floor_area_m2: float = 3.0,
    min_free_area_m2: float = 1.0,
    floor_definition: str = "occupied-column",
    floor_z_tolerance_m: float = 0.15,
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
        floor_definition=floor_definition,
        floor_z_tolerance_m=floor_z_tolerance_m,
    )

    # Close the floor mask to bridge gaps caused by furniture occluding
    # floor capture (closing doesn't extend floor outside its convex hull).
    close_cells = int(np.ceil(floor_close_radius_m / cell_size_m))
    has_floor_closed = _close(has_floor, close_cells)

    # Drop tiny floor components (RANSAC false positives in adjacent
    # rooms / outside / on furniture). Without this, the closing step
    # can bridge real-room floor with stray planar regions and produce
    # placements that land outside the actual room.
    if min_floor_area_m2 > 0:
        min_cells = int(np.ceil(min_floor_area_m2 / (cell_size_m ** 2)))
        n_before = int(has_floor_closed.sum())
        has_floor_closed, kept_areas = _keep_components_above_area(
            has_floor_closed, min_cells,
        )
        n_after = int(has_floor_closed.sum())
        kept_m2 = [a * cell_size_m ** 2 for a in kept_areas]
        print(
            f"[placement] floor connected-components filter: kept "
            f"{len(kept_areas)} component(s) ≥ {min_floor_area_m2:.1f} m² "
            f"(areas = {[round(a, 2) for a in kept_m2]} m²); "
            f"{n_before - n_after} cells dropped"
        )

    # Camera-position floor evidence: anywhere the operator walked, the
    # floor was directly underneath them — even if the camera was
    # pointed at a sofa rather than the floor. Project camera xy to the
    # grid, dilate by ``camera_floor_radius_m``, and union with the
    # RANSAC-derived floor. This fixes the common pattern where a
    # camera-dense area has SPARSE RANSAC coverage because the operator
    # was looking at furniture there.
    if camera_centers_world is not None and camera_floor_radius_m > 0:
        cam_xy = np.asarray(camera_centers_world)[:, :2]
        cam_floor = np.zeros_like(has_floor)
        ix_cam = np.clip(((cam_xy[:, 0] - xmin) / cell_size_m).astype(int), 0, has_floor.shape[1] - 1)
        iy_cam = np.clip(((cam_xy[:, 1] - ymin) / cell_size_m).astype(int), 0, has_floor.shape[0] - 1)
        cam_floor[iy_cam, ix_cam] = True
        cam_dilate_cells = int(np.ceil(camera_floor_radius_m / cell_size_m))
        # _erode dilates True regions when applied to a binary mask.
        cam_floor = _erode(cam_floor, cam_dilate_cells)
        n_added = int((cam_floor & ~has_floor_closed).sum())
        has_floor_closed = has_floor_closed | cam_floor
        print(
            f"[placement] camera-position floor evidence: dilated {len(cam_xy)} "
            f"cameras by {camera_floor_radius_m} m, added {n_added} cells "
            f"to the floor mask"
        )

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

    # Drop tiny components in the FINAL free mask. This catches the
    # "peninsula" failure mode: a doorway / open passage between rooms
    # is connected to the main floor (so it survives the floor-mask
    # component filter), but the eroded `free` cells in that passage
    # form only a thin strip (~0.5 m² total) — much smaller than the
    # main room's free area. Without this filter, FPS sampling can
    # bias the very first pick into the passage.
    if min_free_area_m2 > 0 and free.any():
        min_free_cells = int(np.ceil(min_free_area_m2 / (cell_size_m ** 2)))
        n_free_before = int(free.sum())
        free, free_areas = _keep_components_above_area(free, min_free_cells)
        n_free_after = int(free.sum())
        free_areas_m2 = [a * cell_size_m ** 2 for a in free_areas]
        print(
            f"[placement] free-region filter: kept "
            f"{len(free_areas)} component(s) ≥ {min_free_area_m2:.2f} m² "
            f"(areas = {[round(a, 2) for a in free_areas_m2]} m²); "
            f"{n_free_before - n_free_after} cells dropped"
        )

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

    # Optionally bias to placements near training-camera positions —
    # the GS/NuRec rendering quality is best where the training cameras
    # actually saw the scene. Filtering placements to within
    # ``max_camera_distance_m`` of a training camera puts the rendering
    # hemisphere in the well-covered sweet spot, avoiding smeared
    # walls/ceiling from out-of-distribution view angles.
    if camera_centers_world is not None and max_camera_distance_m is not None:
        cam_xy = np.asarray(camera_centers_world)[:, :2]
        # Compute min xy distance from each candidate cell to any camera.
        # Vectorized — fine up to a few thousand cells × cameras.
        min_dists = np.linalg.norm(
            cell_xy[:, None, :] - cam_xy[None, :, :], axis=-1
        ).min(axis=1)
        near = min_dists <= max_camera_distance_m
        n_kept = int(near.sum())
        n_total = len(cell_xy)
        if n_kept == 0:
            raise RuntimeError(
                f"no free cells within {max_camera_distance_m} m of any "
                f"training camera ({n_total} candidates were considered). "
                f"Increase --max-distance-to-camera or drop the flag."
            )
        cell_xy = cell_xy[near]
        print(
            f"[placement] camera-distance filter: kept {n_kept}/{n_total} cells "
            f"within {max_camera_distance_m} m of a training camera"
        )

    if sampling == "farthest_point":
        pick = _farthest_point_sample(cell_xy, n_placements, rng)
    elif sampling == "random":
        pick = rng.choice(len(cell_xy), size=min(n_placements, len(cell_xy)),
                          replace=False)
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
