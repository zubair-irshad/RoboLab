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
parser.add_argument(
    "--bg-z-offset", type=float, default=None,
    help="vertical shift applied to the DL3DV background (metres). If not "
         "set (default), we auto-detect the task's floor z from the spawned "
         "stage and use it — generalizes across tasks with different table "
         "heights, no table at all, custom workdesks, etc. Pass an explicit "
         "number (e.g. -0.75) to override the auto-detection.",
)
parser.add_argument(
    "--bg-z-offset-fallback", type=float, default=-0.75,
    help="fallback z-offset (m) used only when auto-detection finds nothing "
         "useful (e.g. task has no static geometry). Default -0.75 "
         "(Franka-on-table convention).",
)
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


def _make_task_scan_tolerant() -> None:
    """Skip tasks whose module-level import fails (e.g. missing assets).

    Some RoboLab tasks call ``import_scene("foo.usda", ...)`` at class
    definition time. If that asset isn't resolvable in the current env
    (missing file, working-directory mismatch, asset-registry not yet
    primed), the bare ``auto_register_droid_envs()`` call aborts on the
    first failure and we never get to the task we actually want.

    We wrap ``EnvironmentFactory.create_env_cfg`` so each broken task
    just logs a warning and is skipped, leaving the rest registered.
    """
    try:
        import robolab.core.environments.factory as factory_mod
    except ImportError:
        return
    fac_cls = getattr(factory_mod, "EnvironmentFactory", None)
    if fac_cls is None or not hasattr(fac_cls, "create_env_cfg"):
        return
    if getattr(fac_cls.create_env_cfg, "_dl3dv_tolerant", False):
        return
    _orig = fac_cls.create_env_cfg

    def _tolerant(self, task, *args, **kwargs):
        try:
            return _orig(self, task, *args, **kwargs)
        except Exception as exc:
            print(f"[render] auto-register skipping {task!r}: "
                  f"{type(exc).__name__}: {exc}")
            return None

    _tolerant._dl3dv_tolerant = True  # type: ignore[attr-defined]
    fac_cls.create_env_cfg = _tolerant


_make_task_scan_tolerant()
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


def _build_bg_transform(
    placement_x: float, placement_y: float,
    placement_yaw: float, z_offset: float = 0.0,
) -> np.ndarray:
    """4×4 that maps DL3DV-aligned points so the placement lands above origin.

    World point p_world = R_z(-yaw) · (p_dl3dv - [x, y, 0]) + [0, 0, z_offset].

    With ``z_offset = -table_height``, the DL3DV floor (z=0 in DL3DV frame)
    ends up at z = -table_height in world — under the task's table top
    (which is at world z=0). So:

      - task table top → world z=0
      - task table base → world z = -table_height = DL3DV floor
      - DL3DV's natural-height furniture (sofas, etc.) → reasonable
        world heights between DL3DV floor and ceiling
    """
    c, s = np.cos(-placement_yaw), np.sin(-placement_yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = -R @ np.array([placement_x, placement_y, 0.0])
    M[2, 3] += z_offset
    return M


def _set_bg_transform(stage, prim_path: str, M: np.ndarray) -> None:
    """Author ``xformOp:transform`` on a prim from a numpy 4×4.

    USD's ``GfMatrix4d`` uses row-vector / row-major convention with
    translation in the LAST ROW (world = local · M). numpy convention is
    column-vector with translation in the last COLUMN (world = M · local).
    They differ by a transpose.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"BG prim {prim_path} not on stage")
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    op = xformable.AddTransformOp()
    op.Set(Gf.Matrix4d(*M.T.flatten().tolist()))


def _verify_bg_transform(stage, prim_path: str, expected_M: np.ndarray) -> None:
    """Read back the prim's local transform and compare to what we authored."""
    from pxr import UsdGeom
    prim = stage.GetPrimAtPath(prim_path)
    xformable = UsdGeom.Xformable(prim)
    actual = xformable.GetLocalTransformation()
    actual_np = np.array([[actual[i][j] for j in range(4)] for i in range(4)])
    # USD row-major → transpose to compare with our column-vector numpy form
    actual_np = actual_np.T
    if np.allclose(actual_np, expected_M, atol=1e-5):
        print(f"[render] BG xformOp verified: matrix landed on {prim_path}")
    else:
        diff = np.abs(actual_np - expected_M).max()
        print(
            f"[render] BG xformOp MISMATCH on {prim_path} (max abs diff = {diff:.4f}). "
            f"Expected translation = {expected_M[:3, 3].round(3).tolist()}, "
            f"actual translation = {actual_np[:3, 3].round(3).tolist()}"
        )


def _detect_task_floor_z(
    stage,
    *,
    env_root: str = "/World/envs/env_0",
    robot_name_hints: tuple[str, ...] = ("robot", "franka", "panda", "ur5", "ur10", "arm"),
    skip_dl3dv: bool = True,
) -> float | None:
    """Find the lowest z of static task geometry under ``env_root``.

    Skips:
      - the robot's prim subtree (heuristic: name contains a known hint)
      - the DL3DV background prim (we don't want to detect ourselves)
      - prims with empty / invalid bboxes

    Returns the lowest world-frame z encountered, or ``None`` if nothing
    useful was found. The caller decides what to do with ``None``
    (typically: fall back to a sensible default).

    Why this works as "task floor": RoboLab task scenes load workdesks /
    breakfast tables / etc. as static USD references under the env. The
    table's USD includes the table base sitting on the task's expected
    floor, so the bbox's min-z is exactly the floor height in world
    frame. For tasks with no table, the lowest static body is typically
    a ground plane, which gives floor=0 — also correct.
    """
    from pxr import Usd, UsdGeom

    env_prim = stage.GetPrimAtPath(env_root)
    if not env_prim.IsValid():
        return None
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
    )

    lowest_z: float | None = None
    skipped: list[str] = []
    for child in env_prim.GetChildren():
        name = child.GetName()
        name_lc = name.lower()
        if any(h in name_lc for h in robot_name_hints):
            skipped.append(f"{name} (robot)")
            continue
        if skip_dl3dv and "dl3dv" in name_lc:
            skipped.append(f"{name} (background)")
            continue
        try:
            bbox = bbox_cache.ComputeWorldBound(child)
            r = bbox.ComputeAlignedRange()
            if r.IsEmpty():
                continue
            zmin = float(r.GetMin()[2])
        except Exception:
            continue
        if lowest_z is None or zmin < lowest_z:
            lowest_z = zmin
            chosen_name = name

    if lowest_z is None:
        print(f"[render] task-floor auto-detect: no usable static prim under {env_root}; "
              f"skipped={skipped}")
        return None
    print(
        f"[render] task-floor auto-detect: lowest static prim = "
        f"{chosen_name} @ z={lowest_z:.3f} m  (skipped robot/bg: {len(skipped)})"
    )
    return lowest_z


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
    if args_cli.bg_z_offset is not None:
        bg_z_offset = float(args_cli.bg_z_offset)
        print(f"[render] using user-supplied bg_z_offset = {bg_z_offset:+.3f} m")
    else:
        detected = _detect_task_floor_z(runtime.stage)
        if detected is not None:
            bg_z_offset = detected
        else:
            bg_z_offset = float(args_cli.bg_z_offset_fallback)
            print(f"[render] auto-detect failed; using fallback {bg_z_offset:+.3f} m")
    M = _build_bg_transform(
        pl["x_m"], pl["y_m"], pl["yaw_rad"], z_offset=bg_z_offset,
    )
    print(f"[render] computed BG transform M (numpy convention):")
    for row in M:
        print(f"           [{row[0]: 8.4f} {row[1]: 8.4f} {row[2]: 8.4f} {row[3]: 8.4f}]")
    _set_bg_transform(runtime.stage, bg_prim_path, M)
    _verify_bg_transform(runtime.stage, bg_prim_path, M)
    print(
        f"[render] BG transformed: placement_xy → world origin, "
        f"yaw zeroed, z_offset = {bg_z_offset:+.3f} m"
    )

    # Also print the BG prim's COLLIDER sibling's transform — same matrix
    # should apply to it. If the collider has its own (uncoupled) transform,
    # depth and visual won't agree. set_background_scene uses
    # /World/MarbleBackgroundCollider for the collider.
    collider_path = "/World/MarbleBackgroundCollider"
    if runtime.stage.GetPrimAtPath(collider_path).IsValid():
        _set_bg_transform(runtime.stage, collider_path, M)
        _verify_bg_transform(runtime.stage, collider_path, M)
        print(f"[render] applied same BG transform to collider {collider_path}")

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
