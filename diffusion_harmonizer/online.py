"""Online (per-trajectory) DiffusionHarmonizer pair generator.

Models ``examples/demo/run_empty.py``: for each registered task, create the
env, add 1-2 cameras, then run ``num_episodes`` × ``num_steps_per_episode``
steps. At every step we capture from each camera and emit:

  * **ISP modification** pair  — software-ISP applied through the foreground
    mask, composited with Eq. (3).
  * **Shadow simulation** pair — same view rendered twice (shadows on, then
    foreground excluded from ``UsdLux.shadowLink``), paired as
    ``input=no-shadow``, ``target=full-shadow``.

Trajectory diversity (random sampled actions, fresh contacts, varying robot
poses) gives many distinct paired examples per task per minute of Isaac Sim
runtime — far cheaper than re-launching the simulator for each scene.

Use the offline ``capture_harmonizer_views.py`` + gsplat builders for the
artifacts-correction and asset-reinsertion components. Those need a static
100-camera sphere capture and are not amortizable into a trajectory loop.
"""

from __future__ import annotations

import random
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from diffusion_harmonizer.components.common import (
    feather_mask,
    foreground_mask_with_fallback,
)
from diffusion_harmonizer.components.isp_modification import (
    apply_software_isp,
    sample_isp_params,
)
from diffusion_harmonizer.image_io import save_json, save_png, write_pair
from diffusion_harmonizer.runtime import HarmonizerRuntime


@dataclass
class OnlineConfig:
    output_root: Path = Path("data/diffusion_harmonizer")
    # Camera placement around the workspace. Pick 1-2 angles that see the
    # full table + robot arm. Default is two over-shoulder-style orbit cams.
    num_cameras: int = 2
    camera_radius: float = 1.6
    camera_center: tuple[float, float, float] = (0.4, 0.0, 0.4)
    camera_height: float = 0.6
    capture_resolution: tuple[int, int] = (512, 512)
    spp: int = 4
    # Trajectory geometry. ``num_steps_per_episode`` matches run_empty's default.
    num_episodes: int = 3
    num_steps_per_episode: int = 30
    capture_every_n_steps: int = 1
    seed: int = 42
    device: str = "cuda:0"
    num_envs: int = 1
    physx_buffer_scale: float = 0.1
    components: tuple[str, ...] = ("isp_modification", "shadow_simulation")
    isp_full_frame_fraction: float = 0.2
    isp_strength: float = 0.8
    shadow_min_coverage: float = 0.001


def run_online(env_names: list[str], cfg: OnlineConfig) -> dict:
    """Run the per-trajectory paired-data loop over every requested env."""

    cfg.output_root.mkdir(parents=True, exist_ok=True)
    summary: dict = {"envs": {}, "started_at": time.time(), "config": _config_to_json(cfg)}
    save_json(cfg.output_root / "config.json", summary["config"])

    for env_name in env_names:
        t0 = time.time()
        try:
            env_summary = _run_env_online(env_name, cfg)
        except Exception as exc:
            env_summary = {"error": str(exc), "traceback": traceback.format_exc()}
            print(f"[online:{env_name}] FAILED: {exc}", flush=True)
            traceback.print_exc()
        env_summary["wall_seconds"] = time.time() - t0
        summary["envs"][env_name] = env_summary
        save_json(cfg.output_root / "summary.json", summary)
    summary["finished_at"] = time.time()
    save_json(cfg.output_root / "summary.json", summary)
    return summary


def _run_env_online(env_name: str, cfg: OnlineConfig) -> dict:
    from isaaclab.envs.utils.spaces import sample_space

    env_dir = cfg.output_root / env_name
    isp_dir = env_dir / "02_isp_modification"
    shadow_dir = env_dir / "04_shadow_simulation"

    print(f"\n[online] === env {env_name} ===", flush=True)
    runtime = HarmonizerRuntime(
        env_name=env_name,
        seed=cfg.seed,
        device=cfg.device,
        num_envs=cfg.num_envs,
        physx_buffer_scale=cfg.physx_buffer_scale,
    )
    try:
        foreground_paths = runtime.foreground_prim_paths()
        print(f"[online:{env_name}] foreground: {foreground_paths}", flush=True)
        print(f"[online:{env_name}] receivers:  {runtime.receiver_prim_paths()}", flush=True)

        cameras = runtime.add_orbit_cameras(
            center=cfg.camera_center,
            radius=cfg.camera_radius,
            height=cfg.camera_height,
            num_cameras=cfg.num_cameras,
            resolution=cfg.capture_resolution,
            name_prefix=f"online_{env_name}",
        )
        print(f"[online:{env_name}] added {len(cameras)} cameras", flush=True)
        runtime.set_path_tracing(cfg.spp > 1, spp=cfg.spp)

        rng = random.Random(cfg.seed)
        isp_count = 0
        shadow_count = 0

        for episode in range(cfg.num_episodes):
            print(f"[online:{env_name}] episode {episode + 1}/{cfg.num_episodes}", flush=True)
            runtime.env.reset()
            for step in range(cfg.num_steps_per_episode):
                actions = sample_space(
                    runtime.env.single_action_space,
                    device=runtime.env.device,
                    batch_size=runtime.env.num_envs,
                )
                runtime.env.step(actions)

                if step % cfg.capture_every_n_steps != 0:
                    continue

                for cam_idx, camera in enumerate(cameras):
                    tag = f"e{episode:02d}_s{step:03d}_c{cam_idx:01d}"
                    if "isp_modification" in cfg.components:
                        if _build_isp_pair(runtime, camera, isp_dir / tag, foreground_paths, rng, cfg):
                            isp_count += 1
                    if "shadow_simulation" in cfg.components:
                        if _build_shadow_pair(runtime, camera, shadow_dir / tag, foreground_paths, cfg):
                            shadow_count += 1
                if (step + 1) % 10 == 0:
                    print(
                        f"[online:{env_name}]   step {step + 1}/{cfg.num_steps_per_episode}"
                        f" — isp={isp_count} shadow={shadow_count}",
                        flush=True,
                    )

        print(f"[online:{env_name}] DONE — isp={isp_count} shadow={shadow_count}", flush=True)
        return {
            "env_name": env_name,
            "isp_pairs": isp_count,
            "shadow_pairs": shadow_count,
            "episodes": cfg.num_episodes,
            "steps_per_episode": cfg.num_steps_per_episode,
            "cameras": len(cameras),
            "foreground_prim_paths": foreground_paths,
        }
    finally:
        runtime.close()
        _release_cuda_memory()


def _build_isp_pair(runtime, camera: str, pair_dir: Path, foreground_paths: list[str], rng: random.Random, cfg: OnlineConfig) -> bool:
    """One ISP pair from one captured frame. Returns True if a pair was written."""

    frame = runtime.capture_frame(camera, rgb=True, segmentation=True)
    target = frame["rgb"]
    use_full_frame = cfg.isp_full_frame_fraction > 0.0 and rng.random() < cfg.isp_full_frame_fraction
    params = sample_isp_params(rng, scale=0.3 if use_full_frame else cfg.isp_strength)
    isp = apply_software_isp(target, params, rng)

    if use_full_frame:
        mask = np.ones(target.shape[:2], dtype=np.float32)
        mode = "full_frame_mild"
    else:
        mask, mask_source = foreground_mask_with_fallback(
            frame["segmentation"], frame["segmentation_mapping"] or {}, foreground_paths,
        )
        if float(np.mean(mask > 0.05)) < 0.002:
            return False
        mask = feather_mask(mask, sigma=3.0)
        mode = mask_source

    mixed = (
        mask[..., None] * isp.astype(np.float32)
        + (1.0 - mask[..., None]) * target.astype(np.float32)
    ).astype(np.uint8)
    save_png(pair_dir / "isp_full.png", isp)
    write_pair(
        pair_dir, mixed, target,
        {
            "component": "isp_modification",
            "mode": mode,
            "camera": camera,
            "params": params.__dict__,
            "mask_coverage": float(np.mean(mask > 0.05)),
            "foreground_prim_paths": foreground_paths,
        },
        mask=mask,
    )
    return True


def _build_shadow_pair(runtime, camera: str, pair_dir: Path, foreground_paths: list[str], cfg: OnlineConfig) -> bool:
    """One shadow pair from one frame: capture target, toggle shadowLink, capture no_shadow_fg."""

    target = runtime.capture_frame(camera, rgb=True, segmentation=True)
    runtime.set_shadow_link_excludes(foreground_paths, enabled=True)
    try:
        no_shadow = runtime.capture_frame(camera, rgb=True)
    finally:
        runtime.set_shadow_link_excludes(foreground_paths, enabled=False)

    target_rgb = target["rgb"]
    no_shadow_rgb = no_shadow["rgb"]
    fg_mask, mask_source = foreground_mask_with_fallback(
        target["segmentation"], target["segmentation_mapping"] or {}, foreground_paths,
    )

    diff = np.abs(target_rgb.astype(np.int16) - no_shadow_rgb.astype(np.int16)).astype(np.uint8)
    if float(np.mean(fg_mask > 0.05)) >= 0.002:
        shadow_mask = _shadow_delta(target_rgb, no_shadow_rgb, _dilate(fg_mask))
    else:
        shadow_mask = np.max(diff.astype(np.float32), axis=-1) / 255.0
        mask_source = "shadow_delta_no_foreground_mask"

    if float(np.mean(shadow_mask > 0.03)) < cfg.shadow_min_coverage:
        return False

    save_png(pair_dir / "shadow_diff.png", diff)
    save_png(pair_dir / "shadow_mask.png", np.repeat((shadow_mask * 255).astype(np.uint8)[..., None], 3, axis=-1))
    write_pair(
        pair_dir, no_shadow_rgb, target_rgb,
        {
            "component": "shadow_simulation",
            "camera": camera,
            "mask_source": mask_source,
            "shadow_mask_coverage": float(np.mean(shadow_mask > 0.03)),
            "foreground_prim_paths": foreground_paths,
        },
    )
    return True


def _dilate(mask: np.ndarray, pixels: int = 7) -> np.ndarray:
    import cv2

    kernel = np.ones((pixels, pixels), dtype=np.uint8)
    return cv2.dilate((mask > 0.05).astype(np.uint8), kernel, iterations=1).astype(np.float32)


def _shadow_delta(target: np.ndarray, no_shadow: np.ndarray, fg_dilated: np.ndarray) -> np.ndarray:
    diff = np.max(np.abs(target.astype(np.float32) - no_shadow.astype(np.float32)), axis=-1) / 255.0
    return np.clip(diff * (1.0 - np.clip(fg_dilated, 0.0, 1.0)), 0.0, 1.0)


def _release_cuda_memory() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _config_to_json(cfg) -> dict:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.__dict__.items()}
