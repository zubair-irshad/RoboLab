"""Export hemispheric captures to nerfstudio dataparser format.

Reads ``<env>/01_artifacts_correction/views/<NNNN>/{rgb.png, intrinsics.json,
extrinsics.json}`` and writes a tree consumable by ``ns-train splatfacto``:

  ``<env>/01_artifacts_correction/nerfstudio/``
    ├── images/                     ← shared rgb pool (frame_<NNNN>.png)
    ├── full/transforms.json        all 120 frames        (reference)
    ├── sparse_k/transforms.json    K of N frames         (sparse)
    ├── underfit/transforms.json    all frames            (early-stop variant)
    ├── cycle/transforms.json       even=train odd=eval   (cycle)
    └── cross_ref/transforms.json   1st half=train, 2nd=eval (cross-ref)

Each strategy directory symlinks its ``images`` to the shared pool so you only
keep one copy of the rgb on disk. ``transforms.json`` follows nerfstudio's
schema (camera_model="OPENCV", per-frame ``transform_matrix`` in OpenGL
convention which matches Isaac Sim's Replicator OpenGL pose).

For the strategies that hold out frames as eval (cycle / cross_ref) we write
``train_filenames`` and ``eval_filenames`` arrays into transforms.json so
nerfstudio's nerfstudio_dataparser routes the splits accordingly. For sparse
and underfit there's no held-out split — train_split_fraction stays default
and the strategy variation is in the iteration count or train-frame count
controlled at ``ns-train`` time.

No torch / no Isaac Sim — pure stdlib + numpy + imageio.
"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class StrategyExport:
    name: str
    transforms_path: Path
    num_train: int
    num_eval: int
    note: str
    suggested_ns_train_args: str


def _read_view(view_dir: Path) -> dict | None:
    rgb_path = view_dir / "rgb.png"
    K_path = view_dir / "intrinsics.json"
    E_path = view_dir / "extrinsics.json"
    if not (rgb_path.exists() and K_path.exists() and E_path.exists()):
        return None
    K = np.asarray(json.loads(K_path.read_text())["K"], dtype=np.float64)
    T = np.asarray(json.loads(E_path.read_text())["world_T_cam_gl"], dtype=np.float64)
    depth_path = view_dir / "depth.npy"
    return {
        "view_id": int(view_dir.name),
        "rgb_path": rgb_path,
        "depth_path": depth_path if depth_path.exists() else None,
        "K": K,
        "T": T,
    }


def _back_project_view(view: dict, stride: int = 8, depth_min: float = 0.05, depth_max: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    """Back-project a view's depth map to world-space colored points.

    Pixel (u, v) -> camera-frame OpenCV point (x_cv, y_cv, z_cv) via pinhole,
    then to OpenGL camera frame (flip Y and Z) so the world transform we
    saved (``world_T_cam_gl`` in OpenGL convention) maps it correctly to
    world coordinates. Output: (N, 3) points in world frame, (N, 3) uint8
    colors.
    """

    if view.get("depth_path") is None:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    import imageio.v3 as iio

    rgb = np.asarray(iio.imread(view["rgb_path"]))
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    depth = np.load(view["depth_path"]).astype(np.float32).squeeze()
    if depth.ndim != 2:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    h, w = depth.shape[:2]
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[ys, xs]
    valid = np.isfinite(z) & (z > depth_min) & (z < depth_max)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    K = view["K"]
    T = view["T"]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # OpenCV pinhole back-projection (image y goes down, depth = +Z forward)
    x_cv = (xs[valid] - cx) * z[valid] / fx
    y_cv = (ys[valid] - cy) * z[valid] / fy
    # Convert to the OpenGL camera frame our world_T_cam_gl was built for
    # (Y up, Z back): flip Y and Z signs.
    points_cam = np.stack(
        [x_cv, -y_cv, -z[valid], np.ones_like(z[valid])], axis=-1
    )
    points_world = (T @ points_cam.T).T[:, :3]
    colors = rgb[ys[valid], xs[valid]].astype(np.uint8)
    if colors.ndim == 1:
        colors = np.repeat(colors[:, None], 3, axis=1)
    return points_world.astype(np.float32), colors


def _write_ply_xyz_rgb(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    n = int(points.shape[0])
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    verts = np.empty(
        n,
        dtype=[
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    verts["x"] = points[:, 0]
    verts["y"] = points[:, 1]
    verts["z"] = points[:, 2]
    verts["red"] = colors[:, 0]
    verts["green"] = colors[:, 1]
    verts["blue"] = colors[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        verts.tofile(f)


def _bg_sphere_seed(
    center: np.ndarray,
    radius: float,
    n: int,
    seed: int,
    color: tuple[int, int, int] = (160, 160, 160),
) -> tuple[np.ndarray, np.ndarray]:
    """Random points on a sphere of ``radius`` around ``center``.

    Reason these exist: vanilla 3DGS / FastGS densifies *from existing
    gaussians*. With zero seed coverage of the kitchen BG (depth pass
    returns +inf for marble pixels in our captures), the BG region has no
    gaussians at all and 3DGS invents floaters via random splits. Seeding
    a thin neutral-grey shell around the expected scene extent gives 3DGS
    starting gaussians that the photometric loss can then pull onto the
    actual marble geometry visible in rgb.
    """

    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3)).astype(np.float32)
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    pts = center.astype(np.float32) + radius * v
    cols = np.tile(np.array(color, dtype=np.uint8), (n, 1))
    return pts, cols


def _build_combined_pointcloud(
    views: list[dict],
    output_path: Path,
    stride: int = 8,
    target_points: int = 50_000,
    seed: int = 42,
    depth_max: float = 50.0,
    bg_sphere_center: tuple[float, float, float] = (0.4, 0.0, 0.4),
    bg_sphere_radius: float = 4.0,
    bg_sphere_points: int = 20_000,
) -> int:
    """Concat back-projected points from every view, subsample, write PLY.

    Returns the number of points written. Returns 0 if no view has depth.

    Two changes vs the original implementation:
      * ``depth_max`` raised to 50 m and inf/nan now logged. The old 20 m
        clip silently dropped any marble wall pixels that did get a finite
        depth past 20 m.
      * ``bg_sphere_points`` random points on a sphere of ``bg_sphere_radius``
        around ``bg_sphere_center`` are concatenated *before* subsampling.
        This guarantees BG seed coverage even if the depth pass missed the
        marble entirely (returning +inf).
    """

    points_list: list[np.ndarray] = []
    colors_list: list[np.ndarray] = []
    finite_depths: list[np.ndarray] = []
    inf_frac_acc = 0.0
    n_views_with_depth = 0
    for view in views:
        pts, cols = _back_project_view(view, stride=stride, depth_max=depth_max)
        if pts.size:
            points_list.append(pts)
            colors_list.append(cols)
        # Lightweight stats so the user can sanity-check how much BG is
        # registering after the marble-visibility fix in runtime.py.
        dpath = view.get("depth_path")
        if dpath is not None:
            try:
                z = np.load(dpath).astype(np.float32).squeeze()
                if z.ndim == 2:
                    inf_frac_acc += float(np.isinf(z).mean())
                    finite = z[np.isfinite(z) & (z > 0)]
                    if finite.size:
                        finite_depths.append(np.array([finite.min(), np.median(finite), finite.max()]))
                    n_views_with_depth += 1
            except Exception:
                pass
    if n_views_with_depth:
        avg_inf = inf_frac_acc / n_views_with_depth
        if finite_depths:
            stats = np.stack(finite_depths)
            print(
                f"[depth-stats] views={n_views_with_depth} inf_frac_avg={avg_inf:.3f} "
                f"finite_min_med_max=[{stats[:,0].min():.2f}, "
                f"{np.median(stats[:,1]):.2f}, {stats[:,2].max():.2f}] m",
                flush=True,
            )

    # Inject the BG sphere seed regardless of whether back-projection found
    # anything — the whole point is to backstop the depth pass when it misses.
    if bg_sphere_points > 0:
        bg_pts, bg_cols = _bg_sphere_seed(
            np.asarray(bg_sphere_center, dtype=np.float64),
            radius=float(bg_sphere_radius),
            n=int(bg_sphere_points),
            seed=seed + 1,
        )
        points_list.append(bg_pts)
        colors_list.append(bg_cols)
        print(
            f"[depth-init] +{bg_sphere_points} BG sphere seeds "
            f"@ r={bg_sphere_radius:.2f} m around {tuple(bg_sphere_center)}",
            flush=True,
        )

    if not points_list:
        return 0
    points = np.concatenate(points_list, axis=0)
    colors = np.concatenate(colors_list, axis=0)
    if points.shape[0] > target_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(points.shape[0], size=target_points, replace=False)
        points = points[idx]
        colors = colors[idx]
    _write_ply_xyz_rgb(output_path, points, colors)
    return int(points.shape[0])


def _stage_image_pool(views: list[dict], images_dir: Path) -> None:
    """Symlink (or copy if symlinks aren't allowed) each rgb into a shared pool."""

    images_dir.mkdir(parents=True, exist_ok=True)
    for v in views:
        dst = images_dir / f"frame_{v['view_id']:04d}.png"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            dst.symlink_to(v["rgb_path"].resolve())
        except (OSError, NotImplementedError):
            shutil.copy2(v["rgb_path"], dst)


def _build_transforms(
    views: list[dict],
    *,
    width: int,
    height: int,
    train_ids: list[int] | None,
    eval_ids: list[int] | None,
) -> dict:
    """Compose a nerfstudio transforms.json dict.

    All views go into ``frames``; if ``train_ids`` / ``eval_ids`` are
    provided we also emit ``train_filenames`` and ``eval_filenames``
    so nerfstudio's nerfstudio_dataparser respects the split.
    """

    K = views[0]["K"]
    transforms = {
        "camera_model": "OPENCV",
        "fl_x": float(K[0, 0]),
        "fl_y": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "w": int(width),
        "h": int(height),
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "frames": [
            {
                "file_path": f"images/frame_{v['view_id']:04d}.png",
                "transform_matrix": v["T"].tolist(),
            }
            for v in views
        ],
    }
    if train_ids is not None:
        transforms["train_filenames"] = [
            f"images/frame_{vid:04d}.png" for vid in sorted(train_ids)
        ]
    if eval_ids is not None:
        transforms["eval_filenames"] = [
            f"images/frame_{vid:04d}.png" for vid in sorted(eval_ids)
        ]
    return transforms


def _ensure_images_link(strategy_dir: Path, images_pool: Path) -> None:
    link = strategy_dir / "images"
    if link.exists() or link.is_symlink():
        link.unlink()
    try:
        link.symlink_to(Path("..") / images_pool.name, target_is_directory=True)
    except (OSError, NotImplementedError):
        shutil.copytree(images_pool, link, dirs_exist_ok=True)


def export_env(
    env_artifacts_dir: Path,
    output_dir: Path | None = None,
    sparse_k: int = 24,
    seed: int = 42,
    with_depth_init: bool = True,
    depth_stride: int = 8,
    depth_target_points: int = 50_000,
    depth_max: float = 50.0,
    bg_sphere_radius: float = 4.0,
    bg_sphere_points: int = 20_000,
) -> tuple[Path, list[StrategyExport]]:
    """Export one env's captures into the nerfstudio strategy tree.

    Returns ``(nerfstudio_root, [StrategyExport, ...])``.
    """

    import imageio.v3 as iio

    views_dir = Path(env_artifacts_dir) / "views"
    if not views_dir.exists():
        raise FileNotFoundError(f"No views dir under {env_artifacts_dir}.")
    if output_dir is None:
        output_dir = Path(env_artifacts_dir) / "nerfstudio"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Capture center for the BG sphere comes from manifest.json so the seed
    # is co-located with the camera shell. Falls back to the default the
    # capture script uses.
    bg_center = (0.4, 0.0, 0.4)
    manifest_path = Path(env_artifacts_dir) / "manifest.json"
    if manifest_path.exists():
        try:
            mf = json.loads(manifest_path.read_text())
            bg_center = tuple(mf.get("center", bg_center))
        except Exception:
            pass

    views = sorted(
        (v for v in (_read_view(d) for d in views_dir.iterdir() if d.is_dir()) if v is not None),
        key=lambda v: v["view_id"],
    )
    if not views:
        raise FileNotFoundError(f"No usable view subdirs under {views_dir}.")

    img = np.asarray(iio.imread(views[0]["rgb_path"]))
    height, width = int(img.shape[0]), int(img.shape[1])
    if img.shape[-1] == 4:
        # nerfstudio handles RGBA but we drop alpha for cleanness.
        for v in views:
            arr = np.asarray(iio.imread(v["rgb_path"]))[..., :3]
            iio.imwrite(v["rgb_path"], arr)

    images_pool = output_dir / "images"
    _stage_image_pool(views, images_pool)

    # Optional combined depth-back-projected PLY for splatfacto init.
    # Splatfacto's nerfstudio_dataparser reads ``ply_file_path`` from
    # transforms.json and uses it as the initial Gaussian centers + colors.
    # Skips silently if no view has depth.npy.
    ply_root = output_dir / "depth_init.ply"
    ply_filename: str | None = None
    n_init_points = 0
    if with_depth_init:
        n_init_points = _build_combined_pointcloud(
            views,
            ply_root,
            stride=depth_stride,
            target_points=depth_target_points,
            seed=seed,
            depth_max=depth_max,
            bg_sphere_center=bg_center,
            bg_sphere_radius=bg_sphere_radius,
            bg_sphere_points=bg_sphere_points,
        )
        if n_init_points > 0:
            ply_filename = ply_root.name
            print(
                f"[export] {env_artifacts_dir.parent.name}: depth_init.ply -> "
                f"{n_init_points} points",
                flush=True,
            )

    del seed  # kept for backward compat; we now pick spatially deterministic subsets
    all_ids = sorted(v["view_id"] for v in views)
    n = len(all_ids)
    k = min(max(1, sparse_k), n)
    # Equally-spaced sparse: every (n/k)-th view of the Fibonacci spiral.
    # Deterministic and gives the same hemispheric coverage as random sampling
    # but with predictable spacing.
    spaced_indices = [int(round(i * (n - 1) / max(k - 1, 1))) for i in range(k)]
    sparse_ids = sorted({all_ids[i] for i in spaced_indices})
    sparse_eval_ids = [vid for vid in all_ids if vid not in set(sparse_ids)]
    # Contiguous-arc holdout: training views form a continuous block of the
    # Fibonacci spiral (which traces a connected hemispheric path), so the
    # held-out indices sit on a completely different angular sector and
    # splatfacto genuinely has to extrapolate — gives much stronger
    # paired-data signal on overlapping-view datasets per the DIFIX3D+
    # paper's note about sparse reconstruction.
    arc_ids = all_ids[: max(1, n // 3)]  # first third of the spiral as train
    arc_eval_ids = all_ids[n // 3 :]

    strategies: list[StrategyExport] = []

    def write_variant(name: str, frames: list[dict], train_ids, eval_ids, note, ns_train_args) -> None:
        strat_dir = output_dir / name
        strat_dir.mkdir(parents=True, exist_ok=True)
        _ensure_images_link(strat_dir, images_pool)
        transforms = _build_transforms(
            frames, width=width, height=height, train_ids=train_ids, eval_ids=eval_ids,
        )
        if ply_filename is not None:
            ply_link = strat_dir / ply_filename
            if ply_link.exists() or ply_link.is_symlink():
                ply_link.unlink()
            try:
                ply_link.symlink_to(Path("..") / ply_filename)
            except (OSError, NotImplementedError):
                shutil.copy2(ply_root, ply_link)
            transforms["ply_file_path"] = ply_filename
        (strat_dir / "transforms.json").write_text(json.dumps(transforms, indent=2))
        strategies.append(
            StrategyExport(
                name=name,
                transforms_path=strat_dir / "transforms.json",
                num_train=len(train_ids) if train_ids is not None else len(frames),
                num_eval=len(eval_ids) if eval_ids is not None else 0,
                note=note,
                suggested_ns_train_args=ns_train_args,
            )
        )

    write_variant(
        "full",
        views,
        train_ids=None,
        eval_ids=None,
        note="All views as train. Clean reference splatfacto run; pairs as the target.",
        ns_train_args="splatfacto --max-num-iterations 30000",
    )
    write_variant(
        "sparse_k",
        views,  # all frames in transforms.json; train_filenames picks the subset
        train_ids=sparse_ids,
        eval_ids=sparse_eval_ids,
        note=(
            f"{len(sparse_ids)} equally-spaced views from the Fibonacci spiral as "
            f"train_filenames; remaining {len(sparse_eval_ids)} as eval_filenames. "
            "WARNING: per DIFIX3D+ §3.2, sparse reconstruction is suboptimal when "
            "held-out views observe the same region as training views — which is the "
            "case here on a tightly-sampled hemisphere. Use sparse_arc instead for "
            "stronger paired-data signal."
        ),
        ns_train_args="splatfacto --max-num-iterations 30000",
    )
    write_variant(
        "sparse_arc",
        views,
        train_ids=arc_ids,
        eval_ids=arc_eval_ids,
        note=(
            f"Train on the first {len(arc_ids)} contiguous views of the Fibonacci "
            f"spiral; render the remaining {len(arc_eval_ids)} as held-out. The held-"
            "out views sit on a different angular sector of the hemisphere, so "
            "splatfacto cannot interpolate them well — strong DIFIX3D+ sparse-"
            "reconstruction supervision signal even on overlapping-view captures."
        ),
        ns_train_args="splatfacto --max-num-iterations 30000",
    )
    write_variant(
        "underfit",
        views,
        train_ids=None,
        eval_ids=None,
        note=(
            "All views as train, deliberately undertrained. Pass --max-num-iterations "
            "150 so densification stops before the model converges; renders show "
            "spurious geometry and missing details across the whole hemisphere."
        ),
        ns_train_args="splatfacto --max-num-iterations 150",
    )

    return output_dir, strategies


def export_all(
    output_root: Path,
    sparse_k: int = 24,
    seed: int = 42,
) -> dict[str, list[StrategyExport]]:
    """Run :func:`export_env` over every env that has a hemispheric snapshot."""

    output_root = Path(output_root)
    summary: dict[str, list[StrategyExport]] = {}
    for env_dir in sorted(p for p in output_root.iterdir() if p.is_dir()):
        artifacts_dir = env_dir / "01_artifacts_correction"
        if not (artifacts_dir / "views").exists():
            continue
        _, strats = export_env(artifacts_dir, sparse_k=sparse_k, seed=seed)
        summary[env_dir.name] = strats
    return summary
