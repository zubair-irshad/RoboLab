"""Online (per-trajectory) DiffusionHarmonizer pair generator.

Models ``examples/demo/run_empty.py`` exactly:

  * one Isaac Sim launch (the CLI driver does it via ``AppLauncher``)
  * for each registered task: ``create_env(task)`` -> ``env.reset()`` ->
    ``env.step(sample_space_action)`` for ``num_steps`` per ``num_episodes``
  * pull rgb out of the env's own camera via ``unpack_image_obs(obs)`` —
    *no* Replicator, *no* extra cameras attached to the stage. The TiledCamera
    that RoboLab already wires into ``env.scene`` is what we read from.

Per capture we generate two paired-data items:

  * ISP modification — composite Eq. (3) using a software-ISP and a foreground
    mask derived from a visibility-difference (foreground hidden + a hold-pose
    re-render). No segmentation annotator needed.
  * Shadow simulation — toggle every light's ``UsdLux.shadowLink`` to exclude
    the foreground, take a hold-pose re-render, pair as
    ``input=no_shadow`` / ``target=full_shadow``.

We sample only ``captures_per_episode`` evenly-spaced steps per episode so the
data isn't dominated by near-duplicate consecutive frames.
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
    foreground_mask_from_visibility_difference,
)
from diffusion_harmonizer.components.isp_modification import (
    apply_software_isp,
    sample_isp_params,
)
from diffusion_harmonizer.image_io import save_json, save_png, write_pair
from diffusion_harmonizer.runtime import HarmonizerRuntime


# RoboLab tasks are typically registered with one or both of these. We pick
# the first one that's actually present on the env scene.
_CAMERA_CANDIDATES = (
    "over_shoulder_left_camera",
    "external_cam",
    "head_camera",
    "egocentric_mirrored_camera",
    "wrist_cam",
)


@dataclass
class OnlineConfig:
    output_root: Path = Path("data/diffusion_harmonizer")
    num_episodes: int = 3
    num_steps_per_episode: int = 30
    captures_per_episode: int = 4  # 2-5 spaced evenly across the episode
    seed: int = 42
    device: str = "cuda:0"
    num_envs: int = 1
    physx_buffer_scale: float = 0.1
    components: tuple[str, ...] = ("isp_modification", "shadow_simulation")
    isp_full_frame_fraction: float = 0.2
    isp_strength: float = 0.8
    shadow_min_coverage: float = 0.001
    camera_name: str | None = None  # auto-pick from env.scene if None


def run_online(env_names: list[str], cfg: OnlineConfig) -> dict:
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
    from robolab.core.observations.observation_utils import unpack_image_obs

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
        env = runtime.env
        foreground_paths = runtime.foreground_prim_paths()
        camera_name = _pick_camera(env, cfg.camera_name)
        print(f"[online:{env_name}] foreground: {foreground_paths}", flush=True)
        print(f"[online:{env_name}] camera:     {camera_name}", flush=True)

        rng = random.Random(cfg.seed)
        isp_count = 0
        shadow_count = 0

        for episode in range(cfg.num_episodes):
            print(f"[online:{env_name}] episode {episode + 1}/{cfg.num_episodes}", flush=True)
            obs, _ = env.reset()
            capture_steps = _capture_steps(cfg.num_steps_per_episode, cfg.captures_per_episode)

            last_actions = None
            for step in range(cfg.num_steps_per_episode):
                actions = sample_space(env.single_action_space, device=env.device, batch_size=env.num_envs)
                obs, _, _, _, _ = env.step(actions)
                last_actions = actions

                if step in capture_steps:
                    print(f"[online:{env_name}]   capturing at step {step}", flush=True)
                    target_rgb = _unpack_rgb(obs, camera_name)
                    if target_rgb is None:
                        print(f"[online:{env_name}]   !! camera '{camera_name}' missing from obs at step {step}", flush=True)
                        continue

                    # Mask: hide foreground, hold-pose step, read camera again.
                    runtime.set_prims_visibility(foreground_paths, False)
                    obs_bg, _, _, _, _ = env.step(last_actions)
                    bg_rgb = _unpack_rgb(obs_bg, camera_name)
                    runtime.set_prims_visibility(foreground_paths, True)
                    mask = foreground_mask_from_visibility_difference(target_rgb, bg_rgb, threshold=0.10)
                    mask = feather_mask(mask, sigma=2.0)

                    if "isp_modification" in cfg.components:
                        if _build_isp_pair(
                            target_rgb, mask, isp_dir / f"e{episode:02d}_s{step:03d}",
                            rng, cfg, camera_name,
                        ):
                            isp_count += 1

                    if "shadow_simulation" in cfg.components:
                        runtime.set_shadow_link_excludes(foreground_paths, enabled=True)
                        obs_ns, _, _, _, _ = env.step(last_actions)
                        no_shadow_rgb = _unpack_rgb(obs_ns, camera_name)
                        runtime.set_shadow_link_excludes(foreground_paths, enabled=False)
                        if _build_shadow_pair(
                            target_rgb, no_shadow_rgb, mask,
                            shadow_dir / f"e{episode:02d}_s{step:03d}",
                            cfg, camera_name,
                        ):
                            shadow_count += 1

        print(f"[online:{env_name}] DONE — isp={isp_count} shadow={shadow_count}", flush=True)
        return {
            "env_name": env_name,
            "isp_pairs": isp_count,
            "shadow_pairs": shadow_count,
            "episodes": cfg.num_episodes,
            "steps_per_episode": cfg.num_steps_per_episode,
            "captures_per_episode": cfg.captures_per_episode,
            "camera": camera_name,
            "foreground_prim_paths": foreground_paths,
        }
    finally:
        runtime.close()
        _release_cuda_memory()


def _pick_camera(env, requested: str | None) -> str:
    if requested:
        return requested
    for name in _CAMERA_CANDIDATES:
        if name in env.scene.sensors:
            return name
    raise RuntimeError(
        f"None of {_CAMERA_CANDIDATES} found on env.scene.sensors. "
        f"Available: {list(env.scene.sensors.keys())}"
    )


def _unpack_rgb(obs, camera_name: str) -> np.ndarray | None:
    """Pull a (H, W, 3) uint8 numpy array from the env's image observation."""

    from robolab.core.observations.observation_utils import unpack_image_obs

    images = unpack_image_obs(obs, obs_group_name="image_obs", camera_suffix="_camera")
    rgb = images.get(camera_name)
    if rgb is None and camera_name.endswith("_camera"):
        rgb = images.get(camera_name[: -len("_camera")])
    if rgb is None:
        return None
    arr = np.asarray(rgb)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr.astype(np.uint8, copy=False)


def _capture_steps(num_steps: int, captures: int) -> set[int]:
    captures = max(1, min(captures, num_steps))
    # Cluster captures away from the very start — first few frames are visually
    # identical to the reset state. Quartile-onward placement is fine.
    start = max(1, num_steps // 6)
    end = num_steps - 1
    return set(int(round(s)) for s in np.linspace(start, end, num=captures))


def _build_isp_pair(target: np.ndarray, mask: np.ndarray, pair_dir: Path, rng: random.Random, cfg: OnlineConfig, camera_name: str) -> bool:
    use_full_frame = cfg.isp_full_frame_fraction > 0.0 and rng.random() < cfg.isp_full_frame_fraction
    params = sample_isp_params(rng, scale=0.3 if use_full_frame else cfg.isp_strength)
    isp = apply_software_isp(target, params, rng)
    if use_full_frame:
        m = np.ones(target.shape[:2], dtype=np.float32)
        mode = "full_frame_mild"
    else:
        if float(np.mean(mask > 0.05)) < 0.002:
            return False
        m = mask
        mode = "masked_foreground"
    mixed = (
        m[..., None] * isp.astype(np.float32)
        + (1.0 - m[..., None]) * target.astype(np.float32)
    ).astype(np.uint8)
    save_png(pair_dir / "isp_full.png", isp)
    write_pair(
        pair_dir, mixed, target,
        {
            "component": "isp_modification",
            "mode": mode,
            "camera": camera_name,
            "params": params.__dict__,
            "mask_coverage": float(np.mean(m > 0.05)),
        },
        mask=m,
    )
    return True


def _build_shadow_pair(target: np.ndarray, no_shadow: np.ndarray | None, fg_mask: np.ndarray, pair_dir: Path, cfg: OnlineConfig, camera_name: str) -> bool:
    if no_shadow is None:
        return False
    diff = np.abs(target.astype(np.int16) - no_shadow.astype(np.int16)).astype(np.uint8)
    if float(np.mean(fg_mask > 0.05)) >= 0.002:
        shadow_mask = _shadow_delta(target, no_shadow, _dilate(fg_mask))
    else:
        shadow_mask = np.max(diff.astype(np.float32), axis=-1) / 255.0
    if float(np.mean(shadow_mask > 0.03)) < cfg.shadow_min_coverage:
        return False
    save_png(pair_dir / "shadow_diff.png", diff)
    save_png(pair_dir / "shadow_mask.png", np.repeat((shadow_mask * 255).astype(np.uint8)[..., None], 3, axis=-1))
    write_pair(
        pair_dir, no_shadow, target,
        {
            "component": "shadow_simulation",
            "camera": camera_name,
            "shadow_mask_coverage": float(np.mean(shadow_mask > 0.03)),
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
