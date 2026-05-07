"""Convert hemispheric captures to vanilla-3DGS / FastGS COLMAP datasets.

FastGS (https://github.com/fastgs/FastGS) is a fork of graphdeco-inria's
3DGS reference implementation. It is *not* a nerfstudio plugin: it expects
its own conda env, its own ``train.py`` / ``render.py``, and a COLMAP-style
source path::

    <source_path>/
    ├── images/                          (input rgb)
    └── sparse/0/
        ├── cameras.txt
        ├── images.txt
        └── points3D.txt

Pose conversion vs the nerfstudio export
----------------------------------------

The hemispheric capture (``capture_artifact_views.py``) saves each view's
``world_T_cam_gl`` in **OpenGL** convention (cam axes: +X right, +Y up,
+Z back). COLMAP / vanilla 3DGS expects the **OpenCV world-to-camera**
extrinsic (cam axes: +X right, +Y down, +Z forward), serialized as a
unit quaternion (qw, qx, qy, qz) plus a translation (tx, ty, tz).

Handled in ``_opengl_world_T_cam_to_colmap_qt`` below.

Per-strategy layout
-------------------

Some DIFIX3D+ strategies hold out part of the spiral as eval frames. The
held-out frames must (a) be *excluded* from the training source_path so
3DGS doesn't fit them, but (b) still rendered post-training so we can
build paired data. Vanilla 3DGS' ``--eval`` flag splits by every-8th
index, which doesn't match our sparse holdout — so we emit two sibling
COLMAP datasets per strategy::

    <strategy_dir>/
    ├── train/
    │   ├── images/             (only train frames)
    │   └── sparse/0/{cameras,images,points3D}.txt
    └── render/
        ├── images/             (render frames)
        └── sparse/0/{cameras,images,points3D}.txt

The bash runner trains against ``train/`` and re-renders against
``render/``. ``underfit`` can train on all frames while rendering only an
evenly spaced subset for paired data. ``points3D.txt`` is seeded from the
depth-init PLY produced
by ``nerfstudio_export.py``; we re-use that file rather than re-running
the back-projection.

No torch / no Isaac Sim — pure stdlib + numpy + imageio.
"""

from __future__ import annotations

import json
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FastgsStrategyExport:
    name: str
    train_source_path: Path
    render_source_path: Path
    num_train: int
    num_render: int
    iterations: int
    note: str


# ---- pose conversion --------------------------------------------------------

_GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def _opengl_world_T_cam_to_colmap_qt(T_world_cam_gl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OpenGL world-from-cam (4x4) -> COLMAP world-to-cam (quat_wxyz, tvec).

    Steps: convert camera frame OpenGL->OpenCV (flip Y/Z columns of the
    rotation), invert to get world-to-cam, extract quaternion + translation.
    """

    T_world_cam_cv = T_world_cam_gl @ _GL_TO_CV
    T_cam_world_cv = np.linalg.inv(T_world_cam_cv)
    R = T_cam_world_cv[:3, :3]
    t = T_cam_world_cv[:3, 3]
    return _rotmat_to_quat_wxyz(R), t


def _rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a unit quaternion (w, x, y, z)."""

    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    return q


# ---- COLMAP txt writers -----------------------------------------------------


def _write_cameras_txt(path: Path, fx: float, fy: float, cx: float, cy: float, width: int, height: int) -> None:
    """One shared PINHOLE camera entry (camera_id=1)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"1 PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n"
    )


def _write_images_txt(path: Path, entries: list[tuple[int, np.ndarray, np.ndarray, str]]) -> None:
    """Each entry: (image_id, quat_wxyz, tvec, filename).

    COLMAP images.txt format: one image takes two lines — the pose line and
    a (possibly empty) POINTS2D line. Vanilla 3DGS reads only the pose; we
    leave the second line empty.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Image list with two lines of data per image:\n",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)\n",
    ]
    for image_id, q, t, name in entries:
        lines.append(
            f"{image_id} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} 1 {name}\n"
        )
        lines.append("\n")  # empty POINTS2D line
    path.write_text("".join(lines))


def _read_ply_xyz_rgb(ply_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the binary little-endian PLY produced by ``nerfstudio_export``."""

    with open(ply_path, "rb") as f:
        header_lines: list[bytes] = []
        while True:
            line = f.readline()
            header_lines.append(line)
            if line.strip() == b"end_header":
                break
        header = b"".join(header_lines).decode("ascii")
        n = 0
        for line in header.splitlines():
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
                break
        # struct: float x,y,z; uchar r,g,b — 3*4 + 3 = 15 bytes per vertex
        raw = f.read(15 * n)
    arr = np.frombuffer(
        raw,
        dtype=np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ]),
        count=n,
    )
    points = np.stack([arr["x"], arr["y"], arr["z"]], axis=-1).astype(np.float32)
    colors = np.stack([arr["red"], arr["green"], arr["blue"]], axis=-1).astype(np.uint8)
    return points, colors


def _write_points3d_txt(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """COLMAP points3D.txt: ID X Y Z R G B ERROR TRACK[]."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 3D point list with one line of data per point:\n",
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n",
    ]
    for i, (xyz, rgb) in enumerate(zip(points, colors), start=1):
        lines.append(
            f"{i} {xyz[0]} {xyz[1]} {xyz[2]} "
            f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])} 0\n"
        )
    path.write_text("".join(lines))


def _write_empty_points3d_txt(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# 3D point list with one line of data per point:\n"
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
    )


def _random_seed_points(
    center: np.ndarray,
    radius: float,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(count, 3)).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    radii = (rng.random((count, 1), dtype=np.float32) ** (1.0 / 3.0)) * float(radius)
    points = center.astype(np.float32) + dirs * radii
    colors = np.full((count, 3), 128, dtype=np.uint8)
    return points, colors


# ---- view loading (mirrors nerfstudio_export) -------------------------------


def _read_view(view_dir: Path) -> dict | None:
    rgb_path = view_dir / "rgb.png"
    K_path = view_dir / "intrinsics.json"
    E_path = view_dir / "extrinsics.json"
    if not (rgb_path.exists() and K_path.exists() and E_path.exists()):
        return None
    K = np.asarray(json.loads(K_path.read_text())["K"], dtype=np.float64)
    T = np.asarray(json.loads(E_path.read_text())["world_T_cam_gl"], dtype=np.float64)
    return {
        "view_id": int(view_dir.name),
        "rgb_path": rgb_path,
        "K": K,
        "T": T,
    }


def _link_or_copy(dst: Path, src: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def _link_or_copy_dir(dst: Path, src: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    try:
        dst.symlink_to(src.resolve(), target_is_directory=True)
    except (OSError, NotImplementedError):
        shutil.copytree(src, dst, dirs_exist_ok=True)


# ---- main entry -------------------------------------------------------------


def _stage_dataset(
    dataset_dir: Path,
    views: list[dict],
    width: int,
    height: int,
    points_xyz: np.ndarray | None,
    points_rgb: np.ndarray | None,
    image_pool: Path,
) -> None:
    """Write images/ + sparse/0/ for one COLMAP dataset."""

    if dataset_dir.exists() or dataset_dir.is_symlink():
        if dataset_dir.is_symlink() or dataset_dir.is_file():
            dataset_dir.unlink()
        else:
            shutil.rmtree(dataset_dir)
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    # Stage per-view symlinks pointing into the shared pool. Frame names match
    # what's referenced inside ``images.txt``.
    for v in views:
        name = f"frame_{v['view_id']:04d}.png"
        _link_or_copy(images_dir / name, image_pool / name)

    K = views[0]["K"]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    sparse_dir = dataset_dir / "sparse" / "0"
    _write_cameras_txt(sparse_dir / "cameras.txt", fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)

    entries: list[tuple[int, np.ndarray, np.ndarray, str]] = []
    for image_id, v in enumerate(views, start=1):
        q, t = _opengl_world_T_cam_to_colmap_qt(v["T"])
        entries.append((image_id, q, t, f"frame_{v['view_id']:04d}.png"))
    _write_images_txt(sparse_dir / "images.txt", entries)

    if points_xyz is not None and points_xyz.size:
        _write_points3d_txt(sparse_dir / "points3D.txt", points_xyz, points_rgb)
    else:
        _write_empty_points3d_txt(sparse_dir / "points3D.txt")


def export_env(
    env_artifacts_dir: Path,
    output_dir: Path | None = None,
    sparse_arc_train_fraction: float | None = None,
    full_iterations: int = 7000,
    underfit_iterations: int = 3000,
    sparse_arc_train_count: int | None = 20,
    underfit_render_count: int | None = 40,
    depth_target_points: int | None = 0,
    random_seed_points: int = 500,
    seed: int = 42,
) -> tuple[Path, list[FastgsStrategyExport]]:
    """Export one env's hemispheric captures into the FastGS COLMAP tree.

    Mirrors the strategy choices in ``nerfstudio_export.export_env`` so
    paired data lines up frame-for-frame between the splatfacto and FastGS
    runs. We only emit ``full``, ``underfit``, ``sparse_arc`` — the
    sparse_k / cycle / cross_ref strategies need every-8th-style holdout
    that doesn't translate cleanly to vanilla 3DGS.

    Returns ``(fastgs_root, [FastgsStrategyExport, ...])``.
    """

    import imageio.v3 as iio

    views_dir = Path(env_artifacts_dir) / "views"
    if not views_dir.exists():
        raise FileNotFoundError(f"No views dir under {env_artifacts_dir}.")
    if output_dir is None:
        output_dir = Path(env_artifacts_dir) / "fastgs"
    output_dir.mkdir(parents=True, exist_ok=True)

    views = sorted(
        (v for v in (_read_view(d) for d in views_dir.iterdir() if d.is_dir()) if v is not None),
        key=lambda v: v["view_id"],
    )
    if not views:
        raise FileNotFoundError(f"No usable view subdirs under {views_dir}.")

    img = np.asarray(iio.imread(views[0]["rgb_path"]))
    height, width = int(img.shape[0]), int(img.shape[1])

    # Shared image pool — every strategy/dataset's images/ symlinks back here
    # so we keep one rgb copy on disk.
    image_pool = output_dir / "images"
    image_pool.mkdir(parents=True, exist_ok=True)
    for v in views:
        _link_or_copy(image_pool / f"frame_{v['view_id']:04d}.png", v["rgb_path"])

    # Optionally re-use the depth-back-projected PLY that ``nerfstudio_export``
    # built. By default this is disabled so sparse/underfit train without a
    # depth-derived point seed and expose stronger rendering artifacts.
    ns_root = Path(env_artifacts_dir) / "nerfstudio"
    ply_path = ns_root / "depth_init.ply"
    if depth_target_points is not None and depth_target_points <= 0:
        if random_seed_points > 0:
            manifest_path = Path(env_artifacts_dir) / "manifest.json"
            center = np.array([0.4, 0.0, 0.4], dtype=np.float32)
            if manifest_path.exists():
                try:
                    center = np.asarray(json.loads(manifest_path.read_text()).get("center", center), dtype=np.float32)
                except Exception:
                    pass
            cam_centers = np.stack([v["T"][:3, 3] for v in views], axis=0).astype(np.float32)
            radius = float(np.median(np.linalg.norm(cam_centers - center[None, :], axis=1)))
            points_xyz, points_rgb = _random_seed_points(center, max(radius, 0.25), random_seed_points, seed)
            seed_note = f"using {len(points_xyz)} random non-depth seed point(s)"
        else:
            points_xyz = points_rgb = None
            seed_note = "points3D.txt will be empty"
        print(
            f"[fastgs-export] {env_artifacts_dir.parent.name}: "
            f"depth point seed disabled — {seed_note}",
            flush=True,
        )
    elif ply_path.exists():
        points_xyz, points_rgb = _read_ply_xyz_rgb(ply_path)
        if depth_target_points is not None and len(points_xyz) > depth_target_points:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(points_xyz), size=depth_target_points, replace=False)
            points_xyz = points_xyz[idx]
            points_rgb = points_rgb[idx]
        print(
            f"[fastgs-export] {env_artifacts_dir.parent.name}: "
            f"seeding points3D.txt from {ply_path.name} ({len(points_xyz)} points)",
            flush=True,
        )
    else:
        points_xyz = points_rgb = None
        print(
            f"[fastgs-export] {env_artifacts_dir.parent.name}: "
            "no depth_init.ply — points3D.txt will be empty (3DGS random init)",
            flush=True,
        )

    n = len(views)
    if sparse_arc_train_count is not None:
        arc_n = min(n, max(1, int(sparse_arc_train_count)))
    else:
        fraction = 1.0 / 6.0 if sparse_arc_train_fraction is None else sparse_arc_train_fraction
        arc_n = min(n, max(1, int(round(n * fraction))))
    arc_train_indices = set(np.linspace(0, n - 1, arc_n, dtype=int).tolist())
    arc_train = [views[i] for i in sorted(arc_train_indices)]
    arc_render = [v for i, v in enumerate(views) if i not in arc_train_indices]
    if underfit_render_count is not None:
        underfit_render_n = min(n, max(1, int(underfit_render_count)))
        underfit_render_indices = set(np.linspace(0, n - 1, underfit_render_n, dtype=int).tolist())
        underfit_render = [views[i] for i in sorted(underfit_render_indices)]
    else:
        underfit_render = views
    sorted_ids = [v["view_id"] for v in views]

    strategies: list[FastgsStrategyExport] = []

    def write_strategy(
        name: str,
        train_views: list[dict],
        render_views: list[dict],
        iterations: int,
        note: str,
    ) -> None:
        strat_dir = output_dir / name
        strat_dir.mkdir(parents=True, exist_ok=True)
        train_dir = strat_dir / "train"
        render_dir = strat_dir / "render"
        _stage_dataset(train_dir, train_views, width, height, points_xyz, points_rgb, image_pool)
        # Same gaussian model is rendered from both source paths post-training
        # so the render dataset uses the same point cloud seed.
        _stage_dataset(render_dir, render_views, width, height, points_xyz, points_rgb, image_pool)
        strategies.append(
            FastgsStrategyExport(
                name=name,
                train_source_path=train_dir,
                render_source_path=render_dir,
                num_train=len(train_views),
                num_render=len(render_views),
                iterations=iterations,
                note=note,
            )
        )

    write_strategy(
        "full",
        train_views=views,
        render_views=views,
        iterations=full_iterations,
        note=f"All {n} hemispheric views as train. Reference run.",
    )
    write_strategy(
        "underfit",
        train_views=views,
        render_views=underfit_render,
        iterations=underfit_iterations,
        note=(
            f"All {n} views as train, but stop after {underfit_iterations} "
            f"iterations and render {len(underfit_render)} evenly spaced "
            "poses — densification cuts off early so renders show "
            "missing fine detail. Degraded variant for paired data."
        ),
    )
    write_strategy(
        "sparse_arc",
        train_views=arc_train,
        render_views=arc_render,
        iterations=full_iterations,
        note=(
            f"Train on {len(arc_train)}/{n} evenly spaced views of the "
            f"Fibonacci spiral; render only the {len(arc_render)} held-out "
            "poses. The held-out views are hallucinated by the model — strong "
            "DIFIX3D+ sparse-reconstruction supervision signal."
        ),
    )

    # Drop a small JSON manifest the bash runner reads to know what to launch.
    manifest = {
        "env_name": Path(env_artifacts_dir).parent.name,
        "view_count": n,
        "view_ids": sorted_ids,
        "image_pool": str(image_pool.resolve()),
        "strategies": [
            {
                "name": s.name,
                "train_source_path": str(s.train_source_path.resolve()),
                "render_source_path": str(s.render_source_path.resolve()),
                "num_train": s.num_train,
                "num_render": s.num_render,
                "iterations": s.iterations,
                "note": s.note,
            }
            for s in strategies
        ],
    }
    (output_dir / "fastgs_manifest.json").write_text(json.dumps(manifest, indent=2))

    return output_dir, strategies


def export_all(
    output_root: Path,
    full_iterations: int = 7000,
    underfit_iterations: int = 3000,
) -> dict[str, list[FastgsStrategyExport]]:
    output_root = Path(output_root)
    summary: dict[str, list[FastgsStrategyExport]] = {}
    for env_dir in sorted(p for p in output_root.iterdir() if p.is_dir()):
        artifacts_dir = env_dir / "01_artifacts_correction"
        if not (artifacts_dir / "views").exists():
            continue
        _, strats = export_env(
            artifacts_dir,
            full_iterations=full_iterations,
            underfit_iterations=underfit_iterations,
        )
        summary[env_dir.name] = strats
    return summary
