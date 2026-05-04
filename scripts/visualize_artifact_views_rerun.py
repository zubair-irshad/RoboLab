# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Visualize a hemispheric capture in rerun.

Points at the same tree ``capture_artifact_views.py`` writes:

    <root>/<env>/01_artifacts_correction/
        views/<NNNN>/{rgb.png, depth.npy, intrinsics.json, extrinsics.json}
        manifest.json
        nerfstudio/depth_init.ply              (optional, for the seed cloud)
        fastgs/<strategy>/render/...           (optional)
        <NNNN>/{input.png, target.png, metadata.json}   (pair output of pair_splatfacto_renders.py)
        pairs.json

Logs to rerun:
  - world axes + capture-center marker
  - each camera's pinhole frustum + rgb (under ``world/cameras/<NNNN>``)
  - per-view back-projected depth points (uniform stride)
  - merged ``depth_init.ply`` if present (this is the seed used by 3DGS / FastGS)
  - histogram of camera-to-center distance, so you can sanity-check the radius shell
  - pair input/target renders if they exist (top-level <NNNN>/ dirs from
    pair_splatfacto_renders.py), attached to the matching camera frustum so
    you can scrub views and see the degraded render vs the clean target

Usage:
    pip install rerun-sdk imageio numpy
    PYTHONPATH=. python scripts/visualize_artifact_views_rerun.py \\
        --root data/diffusion_harmonizer \\
        --env  UtensilsInMugTask \\
        --stride 12 \\
        --max-views 60      # log every other view to keep things light

Open the rerun viewer it spawns (or pass --save my.rrd to dump and view later).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--root", default="data/diffusion_harmonizer",
                   help="Output root used by capture_artifact_views.py.")
    p.add_argument("--env", required=True, help="Env name (subdir under --root).")
    p.add_argument("--stride", type=int, default=12,
                   help="Pixel stride for per-view back-projection (smaller = denser).")
    p.add_argument("--max-views", type=int, default=120,
                   help="Cap on number of views logged (uniformly spaced).")
    p.add_argument("--depth-min", type=float, default=0.05)
    p.add_argument("--depth-max", type=float, default=20.0)
    p.add_argument("--no-per-view-points", action="store_true",
                   help="Skip per-view back-projected clouds; only show depth_init.ply.")
    p.add_argument("--save", default=None,
                   help="Optional .rrd path to dump the recording instead of spawning the viewer.")
    p.add_argument("--app-id", default=None,
                   help="Override the rerun application id.")
    return p.parse_args()


# ---- pose conventions --------------------------------------------------------

# capture_artifact_views.py saves world_T_cam in OpenGL convention
# (cam axes: +X right, +Y up, +Z back). rerun's Pinhole assumes camera-frame
# +X right, +Y down, +Z forward — i.e. OpenCV. We feed rerun the OpenCV pose
# by post-multiplying world_T_cam_gl with diag(1,-1,-1,1).
_GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def _world_T_cam_cv(world_T_cam_gl: np.ndarray) -> np.ndarray:
    return world_T_cam_gl @ _GL_TO_CV


# ---- I/O ---------------------------------------------------------------------


def _read_view(view_dir: Path, _missing: list | None = None):
    rgb_path = view_dir / "rgb.png"
    K_path = view_dir / "intrinsics.json"
    E_path = view_dir / "extrinsics.json"
    missing = [p.name for p in (rgb_path, K_path, E_path) if not p.exists()]
    if missing:
        if _missing is not None:
            _missing.append((view_dir, missing, sorted(p.name for p in view_dir.iterdir())))
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


def _back_project(view, stride: int, depth_min: float, depth_max: float):
    if view["depth_path"] is None:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)
    import imageio.v3 as iio
    rgb = np.asarray(iio.imread(view["rgb_path"]))
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    depth = np.load(view["depth_path"]).astype(np.float32).squeeze()
    if depth.ndim != 2:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)
    h, w = depth.shape[:2]
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[ys, xs]
    valid = np.isfinite(z) & (z > depth_min) & (z < depth_max)
    if not np.any(valid):
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)
    K = view["K"]; T = view["T"]
    fx, fy = K[0, 0], K[1, 1]; cx, cy = K[0, 2], K[1, 2]
    x_cv = (xs[valid] - cx) * z[valid] / fx
    y_cv = (ys[valid] - cy) * z[valid] / fy
    # OpenGL camera frame: flip Y, Z so world_T_cam_gl applies directly.
    pts_cam = np.stack([x_cv, -y_cv, -z[valid], np.ones_like(z[valid])], axis=-1)
    pts_world = (T @ pts_cam.T).T[:, :3]
    cols = rgb[ys[valid], xs[valid]].astype(np.uint8)
    if cols.ndim == 1:
        cols = np.repeat(cols[:, None], 3, axis=1)
    return pts_world.astype(np.float32), cols


def _read_ply_xyz_rgb(path: Path):
    """Minimal PLY reader matching nerfstudio_export._write_ply_xyz_rgb."""
    with open(path, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            header += line
            if line.strip() == b"end_header":
                break
        n = 0
        for line in header.splitlines():
            if line.startswith(b"element vertex"):
                n = int(line.split()[-1])
        dtype = np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ])
        verts = np.fromfile(f, dtype=dtype, count=n)
    pts = np.stack([verts["x"], verts["y"], verts["z"]], axis=-1).astype(np.float32)
    cols = np.stack([verts["red"], verts["green"], verts["blue"]], axis=-1).astype(np.uint8)
    return pts, cols


# ---- main --------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    try:
        import rerun as rr
    except ImportError:
        sys.exit("rerun not installed. `pip install rerun-sdk`.")
    import imageio.v3 as iio

    art_dir = Path(args.root) / args.env / "01_artifacts_correction"
    if not art_dir.exists():
        sys.exit(f"{art_dir} not found. Run capture_artifact_views.py first.")
    views_dir = art_dir / "views"
    if not views_dir.exists():
        sys.exit(f"{views_dir} not found.")

    missing_log: list = []
    raw_views = []
    for d in sorted(views_dir.iterdir()):
        if not d.is_dir():
            continue
        v = _read_view(d, missing_log)
        if v is not None:
            raw_views.append(v)
    views = sorted(raw_views, key=lambda v: v["view_id"])
    if not views:
        msg = [f"No usable view subdirs under {views_dir}."]
        if missing_log:
            d, miss, present = missing_log[0]
            msg.append(f"  example: {d}")
            msg.append(f"    missing : {miss}")
            msg.append(f"    present : {present}")
        sys.exit("\n".join(msg))

    # Honor --max-views by uniform subsampling
    if args.max_views and len(views) > args.max_views:
        idx = np.linspace(0, len(views) - 1, args.max_views).round().astype(int)
        views = [views[i] for i in idx]

    # Manifest hints (center, radius range)
    center = np.array([0.4, 0.0, 0.4], dtype=np.float64)
    radius_range = (0.8, 1.3)
    manifest_path = art_dir / "manifest.json"
    if manifest_path.exists():
        try:
            mf = json.loads(manifest_path.read_text())
            center = np.asarray(mf.get("center", center), dtype=np.float64)
            radius_range = tuple(mf.get("radius_range", radius_range))
        except Exception:
            pass

    app_id = args.app_id or f"artifact-views::{args.env}"
    rr.init(app_id, spawn=args.save is None)
    if args.save:
        rr.save(args.save)

    # World basis (right-handed, +Z up — matches Isaac Lab world).
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "world/origin",
        rr.Arrows3D(
            origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            vectors=[[0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]],
            colors=[[255, 64, 64], [64, 255, 64], [64, 64, 255]],
        ),
        static=True,
    )
    rr.log(
        "world/center",
        rr.Points3D([center.tolist()], colors=[[255, 255, 0]], radii=0.02, labels=["capture center"]),
        static=True,
    )

    # Reference shells visualizing the camera radius range.
    n_ring = 64
    theta = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    for r, name in ((radius_range[0], "r_lo"), (radius_range[1], "r_hi")):
        ring = np.stack(
            [center[0] + r * np.cos(theta), center[1] + r * np.sin(theta),
             np.full(n_ring, center[2])],
            axis=-1,
        )
        rr.log(
            f"world/shell/{name}",
            rr.LineStrips3D([np.vstack([ring, ring[:1]])], colors=[[120, 120, 120]]),
            static=True,
        )

    # Cameras
    h_w_logged = False
    cam_positions = []
    for view in views:
        T_gl = view["T"]
        T_cv = _world_T_cam_cv(T_gl)
        cam_positions.append(T_gl[:3, 3].copy())

        path = f"world/cameras/{view['view_id']:04d}"
        rr.log(
            path,
            rr.Transform3D(
                translation=T_cv[:3, 3],
                mat3x3=T_cv[:3, :3],
            ),
        )
        K = view["K"]
        rgb = np.asarray(iio.imread(view["rgb_path"]))
        if rgb.ndim == 3 and rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        h, w = rgb.shape[:2]
        rr.log(
            f"{path}/image",
            rr.Pinhole(
                image_from_camera=K.astype(np.float32),
                width=w, height=h,
                camera_xyz=rr.ViewCoordinates.RDF,  # OpenCV: +X right, +Y down, +Z forward
            ),
        )
        rr.log(f"{path}/image/rgb", rr.Image(rgb))

        if view["depth_path"] is not None:
            depth = np.load(view["depth_path"]).astype(np.float32).squeeze()
            if depth.ndim == 2:
                rr.log(
                    f"{path}/image/depth",
                    rr.DepthImage(depth, meter=1.0),
                )
        h_w_logged = True

    cam_positions = np.asarray(cam_positions, dtype=np.float32)
    rr.log(
        "world/camera_centers",
        rr.Points3D(cam_positions, colors=[[80, 200, 255]], radii=0.012),
        static=True,
    )

    # Camera-to-center distance histogram (on the timeline so it persists)
    if cam_positions.size:
        dists = np.linalg.norm(cam_positions - center.astype(np.float32), axis=-1)
        rr.log("stats/cam_distance_to_center",
               rr.BarChart(np.histogram(dists, bins=20)[0]), static=True)
        print(f"[viz] camera radius: min={dists.min():.3f} median={np.median(dists):.3f} "
              f"max={dists.max():.3f}  (configured shell {radius_range[0]:.2f}-{radius_range[1]:.2f})")

    # Pair outputs (top-level <NNNN>/ dirs from pair_splatfacto_renders.py).
    # metadata.json carries view_stem like "frame_0007" — we match that back
    # to a view_id and attach input/target under the same camera frustum so
    # the rerun viewer lets you scrub view-by-view between gt rgb, the
    # degraded render (input), and the reference render (target).
    pair_dirs = sorted(p for p in art_dir.iterdir()
                       if p.is_dir() and p.name.isdigit() and p.name != "views")
    n_pairs_logged = 0
    for pair_dir in pair_dirs:
        meta_path = pair_dir / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        stem = str(meta.get("view_stem", ""))
        # Expected stems: "frame_0007" — extract the numeric view id.
        digits = "".join(ch for ch in stem if ch.isdigit())
        if not digits:
            continue
        view_id = int(digits)
        cam_path = f"world/cameras/{view_id:04d}"
        strategy = str(meta.get("strategy", "pair"))
        for kind in ("input", "target"):
            png = pair_dir / f"{kind}.png"
            if not png.exists():
                continue
            img = np.asarray(iio.imread(png))
            if img.ndim == 3 and img.shape[-1] == 4:
                img = img[..., :3]
            rr.log(f"{cam_path}/image/{strategy}_{kind}", rr.Image(img))
        n_pairs_logged += 1
    if n_pairs_logged:
        print(f"[viz] attached {n_pairs_logged} pair(s) to their camera frustums")

    # Per-view back-projected points (subsampled stride)
    if not args.no_per_view_points:
        for view in views:
            pts, cols = _back_project(view, args.stride, args.depth_min, args.depth_max)
            if pts.size:
                rr.log(
                    f"world/depth_points/{view['view_id']:04d}",
                    rr.Points3D(pts, colors=cols, radii=0.004),
                    static=True,
                )

    # Combined depth_init.ply (the actual seed for 3DGS / FastGS densification)
    ply_paths = [
        art_dir / "nerfstudio" / "depth_init.ply",
        art_dir / "depth_init.ply",
    ]
    for ply_path in ply_paths:
        if ply_path.exists():
            try:
                pts, cols = _read_ply_xyz_rgb(ply_path)
                rr.log(
                    "world/depth_init_ply",
                    rr.Points3D(pts, colors=cols, radii=0.006),
                    static=True,
                )
                print(f"[viz] depth_init.ply: {ply_path} ({len(pts)} pts) "
                      f"extent x[{pts[:,0].min():.2f},{pts[:,0].max():.2f}] "
                      f"y[{pts[:,1].min():.2f},{pts[:,1].max():.2f}] "
                      f"z[{pts[:,2].min():.2f},{pts[:,2].max():.2f}]")
                break
            except Exception as exc:
                print(f"[viz] failed to read {ply_path}: {exc}")

    print(f"[viz] logged {len(views)} views (of {sum(1 for _ in views_dir.iterdir())} on disk).")


if __name__ == "__main__":
    main()
