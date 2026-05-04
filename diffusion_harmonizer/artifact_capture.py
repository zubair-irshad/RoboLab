"""Hemispheric multi-view snapshot for offline gsplat artifacts training.

Lives in its own module / CLI because:

  * Repositioning the env-side camera 100+ times around the workspace is a
    different concern from the trajectory-driven ISP / shadow captures
    that ``online_harmonizer_pairs.py`` handles.
  * Mixing the two leaves the env-camera in a stale pose and chews capture
    time inside a session that should focus on trajectory replays.

Implementation pattern: reuse the env's existing TiledCamera, teleport it
to each Fibonacci-hemisphere position via ``set_world_poses``, force a
``env.sim.render() + env.scene.update(0)`` to refresh the camera buffer
without advancing physics, and read rgb / depth / intrinsics / extrinsics
from ``base_cam.data``. No Replicator render products attached, so no
first-frame warmup hang.

Output: ``<output_root>/<env>/01_artifacts_correction/views/<NNNN>/{rgb.png,
depth.npy, intrinsics.json, extrinsics.json}`` plus a ``manifest.json``.
``scripts/build_artifacts_pairs.py`` reads these views and runs the four
DIFIX3D+ gsplat strategies post-hoc.
"""

from __future__ import annotations

import gc
import random
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from diffusion_harmonizer.image_io import save_json, save_png
from diffusion_harmonizer.runtime import HarmonizerRuntime


_CAMERA_CANDIDATES = (
    "over_shoulder_left_camera",
    "over_shoulder_right_camera",
    "external_cam",
    "head_camera",
    "egocentric_mirrored_camera",
    "wrist_cam",
)


@dataclass
class ArtifactCaptureConfig:
    output_root: Path = Path("data/diffusion_harmonizer")
    cameras: int = 120
    radius_range: tuple[float, float] = (0.8, 1.3)
    center: tuple[float, float, float] = (0.4, 0.0, 0.4)
    resolution: tuple[int, int] = (512, 512)
    spp: int = 16
    settle_steps: int = 0
    seed: int = 42
    device: str = "cuda:0"
    num_envs: int = 1
    physx_buffer_scale: float = 0.1
    use_path_tracing: bool = True
    # Marble (3D) scenes referenced as the visible background instead of dome
    # HDRIs, to test the hypothesis that latlong HDR projection hurts
    # background rendering quality. Set to () to fall back to plain HDRI.
    marble_scene_roots: tuple[str, ...] = ("assets/scenes/marble",)


def run_artifact_capture(env_names: list[str], cfg: ArtifactCaptureConfig) -> dict:
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    summary: dict = {"envs": {}, "started_at": time.time(), "config": _config_to_json(cfg)}
    save_json(cfg.output_root / "artifact_config.json", summary["config"])

    for env_name in env_names:
        t0 = time.time()
        try:
            env_summary = _capture_env(env_name, cfg)
        except Exception as exc:
            env_summary = {"error": str(exc), "traceback": traceback.format_exc()}
            print(f"[artifact:{env_name}] FAILED: {exc}", flush=True)
            traceback.print_exc()
        env_summary["wall_seconds"] = time.time() - t0
        summary["envs"][env_name] = env_summary
        save_json(cfg.output_root / "artifact_summary.json", summary)
    summary["finished_at"] = time.time()
    save_json(cfg.output_root / "artifact_summary.json", summary)
    return summary


def _capture_env(env_name: str, cfg: ArtifactCaptureConfig) -> dict:
    print(f"\n[artifact] === env {env_name} ===", flush=True)
    runtime = HarmonizerRuntime(
        env_name=env_name,
        seed=cfg.seed,
        device=cfg.device,
        num_envs=cfg.num_envs,
        physx_buffer_scale=cfg.physx_buffer_scale,
        # Force depth into the camera's data_types BEFORE env construction
        # so cam.data.output["distance_to_image_plane"] is populated when we
        # capture each hemispheric pose. Without this, depth.npy was silently
        # never written and the depth-init PLY exporter had nothing to read.
        enable_depth=True,
    )
    output_dir = cfg.output_root / env_name / "01_artifacts_correction"
    marble_loaded = False
    try:
        marble_scenes = _gather_marble_scenes(cfg.marble_scene_roots)
        if marble_scenes:
            chosen = random.Random(cfg.seed + hash(env_name)).choice(marble_scenes)
            collider = _find_collider_for(chosen)
            runtime.set_background_scene(chosen, collider_path=collider)
            marble_loaded = True
            print(
                f"[artifact:{env_name}] marble background: {chosen}"
                + (f"  (collider: {collider.name})" if collider else "  (no collider found)"),
                flush=True,
            )
        return _hemispheric_snapshot(runtime, runtime.env, output_dir, cfg, env_name)
    finally:
        if marble_loaded:
            try:
                runtime.set_background_scene(None)
            except Exception:
                pass
        runtime.close()
        _release_cuda_memory()


def _find_collider_for(marble_path: Path) -> Path | None:
    """Find the polygonal collider USD that ships next to a marble NuRec asset.

    Marble NuRec packs typically include a sibling ``<stem>_collider.usd``
    or generic ``*_collider.usd`` in the same directory. We use that as
    the depth-pass proxy because rasterized ``distance_to_image_plane``
    cannot intersect NuRec volumes.
    """
    parent = Path(marble_path).parent
    stem = Path(marble_path).stem
    candidates = [
        parent / f"{stem}_collider.usd",
        parent / f"{stem}_collider.usda",
        parent / f"{stem}_collider.usdc",
    ]
    # Case-insensitive sibling match if exact stem doesn't hit (e.g. asset
    # named ``marblekitchen.usda`` but collider is ``MarbleKitchen_collider.usd``).
    for sibling in parent.glob("*collider*"):
        candidates.append(sibling)
    for c in candidates:
        if c.exists() and c.is_file():
            return c.resolve()
    return None


def _gather_marble_scenes(roots) -> list[Path]:
    suffixes = {".usda", ".usdc", ".usdz", ".usd"}
    out: list[Path] = []
    for root in roots:
        path = Path(root)
        if not path.exists():
            continue
        for candidate in path.rglob("*"):
            if not candidate.is_file() or candidate.suffix.lower() not in suffixes:
                continue
            if "collider" in candidate.stem.lower():
                continue
            out.append(candidate.resolve())
    return sorted(out)


def _hemispheric_snapshot(runtime, env, output_dir: Path, cfg: ArtifactCaptureConfig, env_name: str) -> dict:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    views_dir = output_dir / "views"
    views_dir.mkdir(parents=True, exist_ok=True)

    available = list(getattr(env.scene, "sensors", {}).keys())
    cam_name = next((c for c in _CAMERA_CANDIDATES if c in available), None)
    if cam_name is None:
        return {"env_name": env_name, "skipped": f"no env camera available; sensors={available}"}
    base_cam = env.scene[cam_name]
    env_ids = torch.tensor([0], device=env.device, dtype=torch.long)

    original_pos = base_cam.data.pos_w[0].clone()
    original_quat = base_cam.data.quat_w_world[0].clone()

    r_lo, r_hi = cfg.radius_range
    print(
        f"[artifact:{env_name}] driving {cam_name}: {cfg.cameras} views radius∈[{r_lo:.2f},{r_hi:.2f}] spp={cfg.spp}",
        flush=True,
    )

    if cfg.settle_steps > 0:
        runtime.step(cfg.settle_steps)

    if cfg.use_path_tracing:
        runtime.set_path_tracing(True, spp=cfg.spp)
        _kit_update()

    center = np.asarray(cfg.center, dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    rng = random.Random(cfg.seed)
    eyes = _fibonacci_hemisphere_eyes(cfg.cameras, cfg.radius_range, center, rng)

    # Two-pass capture when the marble is NuRec + has a polygon collider.
    # Pass 1: NuRec visible, collider invisible, path tracing on -> rgb.
    # Pass 2: NuRec invisible, collider visible, path tracing off -> depth.
    # Detected by presence of MarbleBackgroundCollider on the stage. If it
    # isn't there (e.g. plain HDRI run), we fall through to the original
    # single-pass capture and depth simply comes back as +inf in the BG.
    has_collider = bool(
        runtime.stage.GetPrimAtPath("/World/MarbleBackgroundCollider").IsValid()
    )

    manifest_views = []
    t0 = time.time()
    try:
        for idx, eye in enumerate(eyes):
            quat_wxyz = _look_at_quat_opengl(eye, center, up)
            pos_t = torch.tensor(eye, device=env.device, dtype=torch.float32).unsqueeze(0)
            quat_t = torch.tensor(quat_wxyz, device=env.device, dtype=torch.float32).unsqueeze(0)
            base_cam.set_world_poses(
                positions=pos_t, orientations=quat_t, env_ids=env_ids, convention="opengl",
            )

            # ---- rgb pass ----
            if has_collider:
                runtime.set_collider_for_depth_pass(False)
                # path tracing should already be on from the loop entry; no
                # need to toggle every iteration unless the depth pass below
                # turned it off.
                if cfg.use_path_tracing:
                    runtime.set_path_tracing(True, spp=cfg.spp)
            env.sim.render()
            env.scene.update(0.0)

            out = base_cam.data.output
            rgb_t = out.get("rgb")
            if rgb_t is None:
                continue
            rgb = rgb_t[0].detach().cpu().numpy()
            if rgb.ndim == 3 and rgb.shape[-1] == 4:
                rgb = rgb[..., :3]
            if rgb.dtype != np.uint8:
                if float(rgb.max()) <= 1.0 + 1e-6:
                    rgb = rgb * 255.0
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)

            depth_arr = None
            if has_collider:
                # ---- depth pass ----
                # Hide NuRec, show collider, drop path tracing (rasterized
                # depth runs much faster and we don't need PT for distance).
                runtime.set_collider_for_depth_pass(True)
                if cfg.use_path_tracing:
                    runtime.set_path_tracing(False)
                env.sim.render()
                env.scene.update(0.0)
                depth_t = out.get("distance_to_image_plane")
                if depth_t is None:
                    depth_t = out.get("depth")
                if depth_t is not None:
                    depth_arr = depth_t[0].detach().cpu().numpy().astype(np.float32).squeeze()
            else:
                depth_t = out.get("distance_to_image_plane")
                if depth_t is None:
                    depth_t = out.get("depth")
                if depth_t is not None:
                    depth_arr = depth_t[0].detach().cpu().numpy().astype(np.float32).squeeze()

            K = base_cam.data.intrinsic_matrices[0].detach().cpu().numpy().astype(np.float32)
            # Build the camera-to-world matrix directly from our look-at math
            # (already in OpenGL convention: -Z forward, +Y up, +X right).
            # We CANNOT read it back from Isaac Lab's ``data.pos_w`` /
            # ``data.quat_w_world`` because those return the pose in Isaac's
            # "world" convention (+X forward, +Y left, +Z up), so even though
            # we set the pose with convention="opengl" the read-back is in a
            # different frame. Saving Isaac-world as if it were OpenGL was
            # making nerfstudio interpret every camera rotated by ~90°,
            # capping training PSNR at ~13 dB.
            world_T_cam = _build_world_T_cam_from_lookat(eye, center, up)

            view_dir = views_dir / f"{idx:04d}"
            view_dir.mkdir(parents=True, exist_ok=True)
            save_png(view_dir / "rgb.png", rgb)
            if depth_arr is not None:
                np.save(view_dir / "depth.npy", depth_arr)
            save_json(view_dir / "intrinsics.json", {"K": K.tolist()})
            save_json(view_dir / "extrinsics.json", {"world_T_cam_gl": world_T_cam.tolist()})
            manifest_views.append({"view_id": idx, "dir": str(view_dir.relative_to(output_dir))})

            if (idx + 1) % 10 == 0:
                elapsed = time.time() - t0
                rate = (idx + 1) / max(elapsed, 1e-3)
                print(
                    f"[artifact:{env_name}]   {idx + 1}/{len(eyes)} ({rate:.2f} fps)",
                    flush=True,
                )
    finally:
        # Make sure the stage doesn't get left in "depth pass" state if we
        # bail mid-loop (NuRec invisible, collider visible).
        if has_collider:
            runtime.set_collider_for_depth_pass(False)
        base_cam.set_world_poses(
            positions=original_pos.unsqueeze(0),
            orientations=original_quat.unsqueeze(0),
            env_ids=env_ids,
            convention="opengl",
        )
        env.sim.render()
        env.scene.update(0.0)
        if cfg.use_path_tracing:
            runtime.set_path_tracing(False)

    save_json(
        output_dir / "manifest.json",
        {
            "env_name": env_name,
            "driving_camera": cam_name,
            "num_views": len(manifest_views),
            "resolution": list(cfg.resolution),
            "spp": cfg.spp,
            "center": list(cfg.center),
            "radius_range": list(cfg.radius_range),
            "views": manifest_views,
        },
    )
    print(
        f"[artifact:{env_name}] done in {time.time() - t0:.1f}s ({len(manifest_views)} views written)",
        flush=True,
    )
    return {
        "env_name": env_name,
        "driving_camera": cam_name,
        "num_views": len(manifest_views),
        "manifest": str(output_dir / "manifest.json"),
    }


def _fibonacci_hemisphere_eyes(num: int, radius_range: tuple[float, float], center: np.ndarray, rng: random.Random) -> np.ndarray:
    golden = np.pi * (3.0 - np.sqrt(5.0))
    r_lo, r_hi = float(radius_range[0]), float(radius_range[1])
    pts = []
    for i in range(num):
        z = 0.05 + 0.9 * (i / max(num - 1, 1))
        r_unit = np.sqrt(max(0.0, 1.0 - z * z))
        theta = golden * i
        x = r_unit * np.cos(theta)
        y = r_unit * np.sin(theta)
        radius = rng.uniform(r_lo, r_hi)
        pts.append(center + radius * np.array([x, y, z], dtype=np.float64))
    return np.stack(pts, axis=0)


def _look_at_quat_opengl(eye, target, up):
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    f = target - eye
    f /= np.linalg.norm(f) + 1e-12
    r = np.cross(f, up)
    r /= np.linalg.norm(r) + 1e-12
    u = np.cross(r, f)
    R = np.stack([r, u, -f], axis=1)
    t = R.trace()
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
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
    q /= np.linalg.norm(q)
    return q


def _build_world_T_cam_from_lookat(eye, target, up) -> np.ndarray:
    """Compose a (4, 4) world-from-camera matrix in OpenGL convention.

    OpenGL camera basis: +X right, +Y up, +Z back (-Z forward). The
    rotation columns are therefore (right, up, -forward) where ``forward``
    is the unit vector from eye toward target. Translation is just ``eye``.
    """

    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    f = target - eye
    f /= np.linalg.norm(f) + 1e-12
    r = np.cross(f, up)
    r /= np.linalg.norm(r) + 1e-12
    u = np.cross(r, f)
    R = np.stack([r, u, -f], axis=1)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = eye
    return T


def _build_world_T_cam(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])
    n = (w * w + x * x + y * y + z * z) ** 0.5
    if n < 1e-9:
        R = np.eye(3, dtype=np.float64)
    else:
        w, x, y, z = w / n, x / n, y / n, z / n
        R = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return T


def _kit_update() -> None:
    try:
        import omni.kit.app

        omni.kit.app.get_app().update()
    except Exception:
        pass


def _release_cuda_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _config_to_json(cfg) -> dict:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.__dict__.items()}
