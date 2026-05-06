# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# isort: skip_file
"""Render a RoboLab task at a DL3DV-sampled placement, multi-view.

Visual sanity check for the alignment + placement pipeline. Pipeline:

    1. Read a placement (x, y, yaw) from the prepared scene's
       placements.json.
    2. Launch Isaac Sim with the requested RoboLab task env.
    3. Reference ``mesh_aligned.usd`` as the background, transformed so
       that the chosen placement ends up at world origin (where the
       task's table naturally spawns). The DL3DV scene is rotated by
       -yaw so the foreground +X faces the room centroid.
    4. Hemispheric Fibonacci capture of N views around the workspace
       (similar to ``capture_artifact_views.py`` but simpler — just RGB
       PNGs, no depth/intrinsics output).

This validates *visually* whether the robot's size matches the room.
If the robot looks tiny → scale is too large; huge → scale is too
small; clipped by walls → placement landed in a tight spot.

Prerequisites (run in this order on a prepared scene):

    python scripts/prepare_dl3dv_scene.py   ...               # alignment
    python scripts/sample_dl3dv_placements.py --scene-dir ...  # placements
    python scripts/dl3dv_mesh_to_usd.py      --scene-dir ...  # USD bridge

Usage::

    PYTHONPATH=. python scripts/render_dl3dv_with_robot.py \
        --task UtensilsInMugTask \
        --scene-dir data/dl3dv_backgrounds/scenes/<hash> \
        --placement-idx 0 \
        --num-views 12 \
        --output-subdir dl3dv_render_check

Output:
    <scene-dir>/<output_subdir>/<task>/placement_<idx>/<NNNN>_rgb.png
"""

import argparse
import json
import sys
from pathlib import Path

import cv2  # noqa: F401  imported before isaaclab on purpose
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", required=True,
                    help="RoboLab task / env name (e.g. UtensilsInMugTask)")
parser.add_argument("--scene-dir", type=Path, required=True,
                    help="prepared DL3DV scene directory")
parser.add_argument("--placement-idx", type=int, default=0,
                    help="which placement (0-based) from placements.json")
parser.add_argument("--num-views", type=int, default=12)
parser.add_argument("--radius-range", type=float, nargs=2, default=(1.2, 2.0))
parser.add_argument("--center", type=float, nargs=3, default=(0.4, 0.0, 0.5),
                    help="hemisphere center in world frame (above the task table)")
parser.add_argument("--resolution", type=int, nargs=2, default=(720, 720))
parser.add_argument("--spp", type=int, default=8,
                    help="path-tracing samples per pixel (lower = faster)")
parser.add_argument("--no-path-tracing", dest="use_path_tracing",
                    action="store_false", default=True)
parser.add_argument("--output-subdir", default="dl3dv_render_check")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--no-headless", dest="headless_override",
                    action="store_false", default=True)
parser.add_argument("--num-envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.headless = bool(args_cli.headless_override)

# Launch Isaac BEFORE importing anything that pulls Omni
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ------- post-launch imports -----------------------------------------------
import torch  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

from diffusion_harmonizer.runtime import HarmonizerRuntime  # noqa: E402
from diffusion_harmonizer.image_io import save_png  # noqa: E402
from robolab.registrations.droid_jointpos.auto_env_registrations import (  # noqa: E402
    auto_register_droid_envs,
)

auto_register_droid_envs()

_CAMERA_CANDIDATES = (
    "over_shoulder_left_camera",
    "over_shoulder_right_camera",
    "external_cam",
    "head_camera",
    "egocentric_mirrored_camera",
    "wrist_cam",
)


# ---- pose math --------------------------------------------------------------

def _fibonacci_hemisphere_eyes(n: int, radius_range, center, rng) -> np.ndarray:
    """Evenly spaced eye points on the upper hemisphere around ``center``."""
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle
    eyes = np.zeros((n, 3))
    rng_state = np.random.default_rng(rng)
    radii = rng_state.uniform(radius_range[0], radius_range[1], size=n)
    for i in range(n):
        # Map i ∈ [0, n) → upper hemisphere (z >= 0)
        z = 1.0 - (i + 0.5) / n  # in (0, 1]
        r_xy = np.sqrt(max(0.0, 1.0 - z * z))
        theta = phi * i
        x = r_xy * np.cos(theta)
        y = r_xy * np.sin(theta)
        eyes[i] = np.array([x, y, max(z, 0.05)]) * radii[i] + np.asarray(center)
    return eyes


def _look_at_quat_wxyz(eye, target, up=np.array([0, 0, 1.0])):
    """OpenGL look-at quaternion (wxyz). Cam: +Z back, +Y up, +X right."""
    fwd = (np.asarray(target) - np.asarray(eye)).astype(np.float64)
    fwd /= max(float(np.linalg.norm(fwd)), 1e-12)
    z = -fwd  # OpenGL +Z is back
    x = np.cross(up, z)
    x /= max(float(np.linalg.norm(x)), 1e-12)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    # rotation matrix → quaternion (wxyz)
    tr = R.trace()
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0); qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    return np.array([qw, qx, qy, qz])


def _build_bg_transform(placement_x: float, placement_y: float,
                        placement_yaw: float) -> np.ndarray:
    """4×4 that maps DL3DV-aligned points so the placement lands at origin.

    World point p_world = R_z(-yaw) · (p_dl3dv - [x, y, 0]).
    """
    c, s = np.cos(-placement_yaw), np.sin(-placement_yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = -R @ np.array([placement_x, placement_y, 0.0])
    return M


def _set_bg_transform(stage, prim_path: str, M: np.ndarray) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"BG prim {prim_path} not on stage")
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    op = xformable.AddTransformOp()
    op.Set(Gf.Matrix4d(*M.flatten().tolist()))


# ---- main flow --------------------------------------------------------------

def main() -> int:
    scene_dir = args_cli.scene_dir.resolve()
    placements_path = scene_dir / "placements.json"
    bg_usd_path = scene_dir / "mesh_aligned.usd"
    if not placements_path.is_file():
        raise FileNotFoundError(
            f"missing {placements_path}; run sample_dl3dv_placements.py first"
        )
    if not bg_usd_path.is_file():
        raise FileNotFoundError(
            f"missing {bg_usd_path}; run dl3dv_mesh_to_usd.py first"
        )

    placements = json.loads(placements_path.read_text())["placements"]
    if args_cli.placement_idx >= len(placements):
        raise IndexError(
            f"placement_idx {args_cli.placement_idx} out of range "
            f"(have {len(placements)} placements)"
        )
    pl = placements[args_cli.placement_idx]
    print(
        f"[render] placement[{args_cli.placement_idx}] = "
        f"x={pl['x_m']:.2f}, y={pl['y_m']:.2f}, yaw={np.degrees(pl['yaw_rad']):.1f}°"
    )

    runtime = HarmonizerRuntime(
        env_name=args_cli.task,
        seed=args_cli.seed,
        device="cuda:0",
        num_envs=args_cli.num_envs,
        physx_buffer_scale=0.1,
        enable_depth=False,
    )
    env = runtime.env

    bg_prim_path = "/World/DL3DVScene"
    runtime.set_background_scene(
        bg_usd_path, prim_path=bg_prim_path,
        # Same mesh acts as collider (it's polygons, not NuRec)
        collider_path=bg_usd_path,
        collider_visible_to_path_tracer=True,
    )
    M = _build_bg_transform(pl["x_m"], pl["y_m"], pl["yaw_rad"])
    _set_bg_transform(runtime.stage, bg_prim_path, M)
    print(f"[render] BG transformed: placement_xy → world origin, yaw zeroed")

    if args_cli.use_path_tracing:
        runtime.set_path_tracing(True, spp=args_cli.spp)

    # Pick the env-side camera we'll teleport
    sensors = list(getattr(env.scene, "sensors", {}).keys())
    cam_name = next((c for c in _CAMERA_CANDIDATES if c in sensors), None)
    if cam_name is None:
        raise RuntimeError(f"no env camera available; sensors={sensors}")
    print(f"[render] using camera {cam_name}")
    base_cam = env.scene[cam_name]
    env_ids = torch.tensor([0], device=env.device, dtype=torch.long)

    out_dir = scene_dir / args_cli.output_subdir / args_cli.task / f"placement_{args_cli.placement_idx:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "placement.json").write_text(json.dumps(pl, indent=2))

    eyes = _fibonacci_hemisphere_eyes(
        args_cli.num_views, args_cli.radius_range,
        np.asarray(args_cli.center), args_cli.seed,
    )
    target = np.asarray(args_cli.center)

    for i, eye in enumerate(eyes):
        quat = _look_at_quat_wxyz(eye, target)
        pos_t = torch.tensor(eye, device=env.device, dtype=torch.float32).unsqueeze(0)
        quat_t = torch.tensor(quat, device=env.device, dtype=torch.float32).unsqueeze(0)
        base_cam.set_world_poses(
            positions=pos_t, orientations=quat_t,
            env_ids=env_ids, convention="opengl",
        )
        env.sim.render()
        env.scene.update(0.0)
        rgb = base_cam.data.output.get("rgb")
        if rgb is None:
            print(f"[render] view {i:03d}: no rgb buffer (??), skipping")
            continue
        arr = rgb[0].detach().cpu().numpy()
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        save_png(out_dir / f"{i:04d}_rgb.png", arr)

    print(f"[render] wrote {len(eyes)} views to {out_dir}")
    runtime.set_background_scene(None)
    runtime.close()
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    finally:
        simulation_app.close()
    sys.exit(rc)
