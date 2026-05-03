"""Online (per-trajectory) DiffusionHarmonizer pair generator.

Models ``examples/demo/run_empty.py``: one Isaac Sim launch, one
``create_env(task)``, then ``num_episodes × num_steps_per_episode`` random
sample-action steps. We capture from every TiledCamera the env already
exposes (typically ``external_cam`` and ``wrist_cam``) at K evenly-spaced
steps per episode.

Per capture we generate:

  * **ISP modification** — applies a software-ISP and composites Eq. (3)
    using a mask that covers **only the manipulable objects** (not the
    robot). The paper applies ISP to inserted objects so their tone
    mismatches the background; the robot is "native" to the scene and stays
    untouched.
  * **Shadow simulation** — at episode start we add a randomized distant
    sun light that produces crisp cast shadows; per capture we render the
    same view twice (shadow on / foreground excluded from ``UsdLux.shadowLink``)
    and pair the delta as the supervision signal. Without a distant sun the
    dome HDRI's ambient light makes shadowLink toggles invisible.

A single random sample-action target makes the robot chase a different pose
every step, so it never tracks any of them. We hold each sampled action for
``action_hold_steps`` (default 5) so the PD controller actually reaches the
commanded pose, giving visibly different captures across the trajectory.
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


# Tasks register one or more of these. We capture from every one that's
# actually present on the env.scene (typically external_cam + wrist_cam).
_CAMERA_CANDIDATES = (
    "over_shoulder_left_camera",
    "over_shoulder_right_camera",
    "external_cam",
    "head_camera",
    "egocentric_mirrored_camera",
    "wrist_cam",
)


@dataclass
class OnlineConfig:
    output_root: Path = Path("data/diffusion_harmonizer")
    num_episodes: int = 3
    num_steps_per_episode: int = 60
    captures_per_episode: int = 4
    action_hold_steps: int = 10  # held long enough for the PD controller to actually track each target
    seed: int = 42
    device: str = "cuda:0"
    num_envs: int = 1
    physx_buffer_scale: float = 0.1
    components: tuple[str, ...] = ("isp_modification", "shadow_simulation", "asset_reinsertion")
    isp_full_frame_fraction: float = 0.0
    # Linear-pipeline ISP. 0.5 -> half-amplitude around identity for every
    # knob (exposure, gains, CCM noise, gamma, contrast, brightness, sat).
    # Paper-faithful subtle tone mismatch; raise toward 1.0 for sharper
    # "different camera ISP" look.
    isp_strength: float = 0.5
    shadow_min_coverage: float = 0.0008
    # Per-CAPTURE randomized sun for visible cast shadows. Smaller angular size
    # = sharper shadow edges; higher intensity = stronger contrast vs the dome.
    sun_intensity_range: tuple[float, float] = (3000.0, 8000.0)
    sun_angle_deg_range: tuple[float, float] = (0.5, 3.0)
    # Per-CAPTURE dome rotation. The paper "randomly varies environment maps"
    # for shadow simulation; rotating the existing HDRI is a cheap way to
    # change overall illumination direction without swapping textures.
    randomize_dome_rotation: bool = True
    # Cameras used by asset re-insertion (paper-faithful: external + wrist).
    reinsertion_cameras: tuple[str, ...] = ("external_cam", "over_shoulder_left_camera", "wrist_cam")
    # Optional hemispheric multi-view snapshot for offline gsplat artifacts
    # training. 0 = skip; ~30-60 is a reasonable starter for DIFIX3D+ strategies.
    hemisphere_cameras: int = 0
    hemisphere_radius: float = 1.6
    hemisphere_center: tuple[float, float, float] = (0.4, 0.0, 0.4)
    hemisphere_resolution: tuple[int, int] = (512, 512)
    hemisphere_spp: int = 16
    hemisphere_settle_steps: int = 5
    # Force path tracing during capture so the rasterizer's shadowLink
    # limitations don't make shadow toggles invisible.
    use_path_tracing: bool = True
    spp: int = 8
    cameras: tuple[str, ...] | None = None
    # When set, replay actions from this directory's per-task HDF5 demos
    # (matches examples/demo/run_recorded.py). Gives smooth feasible
    # robot trajectories instead of clamped-at-joint-limits flailing.
    playback_data_root: Path | None = Path("examples/demo/recorded_data")


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
    import torch
    from isaaclab.envs.utils.spaces import sample_space

    env_dir = cfg.output_root / env_name
    isp_dir = env_dir / "02_isp_modification"
    shadow_dir = env_dir / "04_shadow_simulation"
    reinsert_dir = env_dir / "05_asset_reinsertion"

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
        full_foreground = runtime.foreground_prim_paths()    # robot + objects (for shadow)
        objects_only = runtime.object_prim_paths()           # objects only (for ISP)
        cameras = _pick_cameras(env, cfg.cameras)
        print(f"[online:{env_name}] foreground (full):    {full_foreground}", flush=True)
        print(f"[online:{env_name}] objects (ISP scope):  {objects_only}", flush=True)
        print(f"[online:{env_name}] cameras:              {cameras}", flush=True)

        # TiledCamera rasterization barely respects USD shadowLink. Path
        # tracing does. Switch the renderer once at startup so every env.step
        # below produces a path-traced frame.
        # Stay rasterized for trajectory steps (fast); switch to path tracing
        # only inside each capture's three-step block (slow but correct
        # shadowLink behaviour).
        if cfg.use_path_tracing:
            runtime.set_path_tracing(False)
            print(
                f"[online:{env_name}] path tracing toggled per-capture only (spp={cfg.spp})",
                flush=True,
            )
        else:
            print(f"[online:{env_name}] rasterization throughout (--no-path-tracing)", flush=True)

        playback_actions = _load_playback_actions(env_name, cfg.playback_data_root, cfg.num_episodes)
        if playback_actions:
            print(
                f"[online:{env_name}] playback: {len(playback_actions)} demo episode(s), "
                f"length(s)={[len(a) for a in playback_actions]}",
                flush=True,
            )
        else:
            print(
                f"[online:{env_name}] playback: NONE (falling back to sample_space + action_hold_steps)",
                flush=True,
            )

        rng = random.Random(cfg.seed)
        isp_count = 0
        shadow_count = 0
        reinsertion_count = 0

        # One-shot hemispheric multi-view snapshot for the offline gsplat
        # artifacts builder. Done before the trajectory loop starts so the
        # scene is in its reset pose with the robot at default config.
        if "artifacts_correction" in cfg.components and cfg.hemisphere_cameras > 0:
            _hemispheric_snapshot(runtime, env, env_dir / "01_artifacts_correction", cfg, env_name)

        for episode in range(cfg.num_episodes):
            print(
                f"[online:{env_name}] episode {episode + 1}/{cfg.num_episodes}",
                flush=True,
            )

            obs, _ = env.reset()
            episode_actions = (
                playback_actions[episode % len(playback_actions)] if playback_actions else None
            )
            num_steps = (
                min(cfg.num_steps_per_episode, len(episode_actions))
                if episode_actions is not None
                else cfg.num_steps_per_episode
            )
            capture_steps = _capture_steps(num_steps, cfg.captures_per_episode)

            held_actions = None
            for step in range(num_steps):
                if episode_actions is not None:
                    actions_np = episode_actions[step]
                    held_actions = (
                        torch.tensor(actions_np, device=env.device, dtype=torch.float32)
                        .unsqueeze(0)
                        .repeat(env.num_envs, 1)
                    )
                else:
                    if held_actions is None or step % max(1, cfg.action_hold_steps) == 0:
                        held_actions = sample_space(
                            env.single_action_space, device=env.device, batch_size=env.num_envs
                        )

                # Trajectory steps run rasterized (cheap, deterministic).
                obs, _, _, _, _ = env.step(held_actions)

                if step not in capture_steps:
                    continue

                joint_pos = env.scene["robot"].data.joint_pos[0].detach().cpu().numpy()
                print(
                    f"[online:{env_name}]   capturing at step {step} — "
                    f"robot joint_pos[:3]={tuple(round(float(j), 3) for j in joint_pos[:3])}",
                    flush=True,
                )

                # Mask via visibility-difference in RASTERIZATION.
                # Path-traced renders have ~5-10% per-pixel noise at spp=8;
                # the diff threshold of 0.15 then catches noise everywhere
                # and the mask leaks onto receivers (table, plate). The
                # rasterized vis-diff is deterministic — diff = exactly the
                # object pixels minus a few edge stragglers we morph away.
                target_rgb_raster = _unpack_rgb(obs, cameras[0])  # any camera works as a sentinel
                object_masks: dict[str, np.ndarray] = {}
                if objects_only:
                    runtime.set_prims_visibility(objects_only, False)
                    _kit_update()
                    obs_bg, _, _, _, _ = env.step(held_actions)
                    runtime.set_prims_visibility(objects_only, True)
                    _kit_update()
                    for camera_name in cameras:
                        tgt = _unpack_rgb(obs, camera_name)
                        bg = _unpack_rgb(obs_bg, camera_name)
                        if tgt is None or bg is None:
                            continue
                        m = foreground_mask_from_visibility_difference(tgt, bg, threshold=0.15)
                        object_masks[camera_name] = feather_mask(m, sigma=2.0)
                else:
                    for camera_name in cameras:
                        tgt = _unpack_rgb(obs, camera_name)
                        if tgt is not None:
                            object_masks[camera_name] = np.zeros(tgt.shape[:2], dtype=np.float32)

                # Build ISP pairs from the (rasterized) target — software ISP
                # doesn't need PT and keeps the mask self-consistent.
                if "isp_modification" in cfg.components:
                    for camera_name in cameras:
                        target_rgb = _unpack_rgb(obs, camera_name)
                        if target_rgb is None:
                            continue
                        tag = f"e{episode:02d}_s{step:03d}_{camera_name}"
                        if _build_isp_pair(
                            target_rgb,
                            object_masks.get(camera_name, np.zeros(target_rgb.shape[:2], dtype=np.float32)),
                            isp_dir / tag, rng, cfg, camera_name,
                        ):
                            isp_count += 1

                # PT block: shadow + asset re-insertion both consume PT renders
                # under matched lighting. Fresh sun direction + dome rotation
                # per capture (paper §3.2 randomizes light direction/softness).
                want_shadow = "shadow_simulation" in cfg.components and full_foreground
                want_reinsert = "asset_reinsertion" in cfg.components and objects_only
                if (want_shadow or want_reinsert) and cfg.use_path_tracing:
                    sun_dir = _random_sun_direction(rng)
                    sun_intensity = rng.uniform(*cfg.sun_intensity_range)
                    sun_angle = rng.uniform(*cfg.sun_angle_deg_range)
                    runtime.set_distant_light(
                        intensity=sun_intensity, angle_deg=sun_angle, direction=sun_dir,
                    )
                    if cfg.randomize_dome_rotation:
                        runtime.set_dome_hdri(None, rotation_deg=rng.uniform(0.0, 360.0))
                    runtime.set_path_tracing(True, spp=cfg.spp)
                    _kit_update()

                    obs_t, _, _, _, _ = env.step(held_actions)  # target_pt: full PBR

                    obs_ns = None
                    if want_shadow:
                        runtime.set_shadow_link_excludes(full_foreground, enabled=True)
                        _kit_update()
                        obs_ns, _, _, _, _ = env.step(held_actions)
                        runtime.set_shadow_link_excludes(full_foreground, enabled=False)
                        _kit_update()

                    obs_bg_pt = None
                    if want_reinsert:
                        runtime.set_prims_visibility(objects_only, False)
                        _kit_update()
                        obs_bg_pt, _, _, _, _ = env.step(held_actions)
                        runtime.set_prims_visibility(objects_only, True)
                        _kit_update()

                    runtime.set_path_tracing(False)

                    for camera_name in cameras:
                        target_pt = _unpack_rgb(obs_t, camera_name)
                        if target_pt is None:
                            continue
                        mask = object_masks.get(
                            camera_name, np.zeros(target_pt.shape[:2], dtype=np.float32)
                        )
                        tag = f"e{episode:02d}_s{step:03d}_{camera_name}"

                        if want_shadow and obs_ns is not None:
                            no_shadow = _unpack_rgb(obs_ns, camera_name)
                            if no_shadow is not None and _build_shadow_pair(
                                target_pt, no_shadow, mask, shadow_dir / tag, cfg, camera_name,
                            ):
                                shadow_count += 1

                        if (
                            want_reinsert
                            and obs_bg_pt is not None
                            and camera_name in cfg.reinsertion_cameras
                        ):
                            bg_pt = _unpack_rgb(obs_bg_pt, camera_name)
                            if bg_pt is not None and _build_reinsertion_pair(
                                target_pt, bg_pt, mask, reinsert_dir / tag, cfg, camera_name,
                            ):
                                reinsertion_count += 1

        print(
            f"[online:{env_name}] DONE — isp={isp_count} shadow={shadow_count} "
            f"reinsertion={reinsertion_count}",
            flush=True,
        )
        return {
            "env_name": env_name,
            "isp_pairs": isp_count,
            "shadow_pairs": shadow_count,
            "reinsertion_pairs": reinsertion_count,
            "episodes": cfg.num_episodes,
            "steps_per_episode": cfg.num_steps_per_episode,
            "captures_per_episode": cfg.captures_per_episode,
            "cameras": cameras,
            "reinsertion_cameras": list(cfg.reinsertion_cameras),
            "object_prim_paths": objects_only,
            "foreground_prim_paths": full_foreground,
        }
    finally:
        runtime.close()
        _release_cuda_memory()


def _pick_cameras(env, requested: tuple[str, ...] | None) -> list[str]:
    available = list(getattr(env.scene, "sensors", {}).keys())
    if requested:
        return [c for c in requested if c in available]
    return [c for c in _CAMERA_CANDIDATES if c in available] or available[:1]


def _random_sun_direction(rng: random.Random) -> tuple[float, float, float]:
    """Sample a sun direction above the horizon (z < 0 in world-space)."""

    az = rng.uniform(0, 2 * np.pi)
    elev = rng.uniform(np.radians(20), np.radians(70))
    return (
        float(np.cos(elev) * np.cos(az)),
        float(np.cos(elev) * np.sin(az)),
        float(-np.sin(elev)),
    )


def _unpack_rgb(obs, camera_name: str) -> np.ndarray | None:
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
        mode = "object_foreground"
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


def _build_reinsertion_pair(target_pt: np.ndarray, bg_pt: np.ndarray, fg_mask: np.ndarray, pair_dir: Path, cfg: OnlineConfig, camera_name: str) -> bool:
    """Asset re-insertion (DiffusionHarmonizer §3.2).

    Composite ``bg_pt`` (foreground hidden, no object cast shadows) with the
    foreground pixels lifted from ``target_pt`` (full PBR). Object pixels:
    same as the real render. Receiver pixels: bg-only render with no object
    cast shadows on the table — the "synthetically inserted" look the paper
    flags as missing harmonization. Pair this with the full PBR target so
    the model learns to add the missing shadow + harmonization.
    """

    if float(np.mean(fg_mask > 0.05)) < 0.002:
        return False
    m = fg_mask[..., None]
    composite = (m * target_pt.astype(np.float32) + (1.0 - m) * bg_pt.astype(np.float32)).astype(np.uint8)
    save_png(pair_dir / "bg_render.png", bg_pt)
    write_pair(
        pair_dir, composite, target_pt,
        {
            "component": "asset_reinsertion",
            "camera": camera_name,
            "mask_coverage": float(np.mean(fg_mask > 0.05)),
            "note": "input = mask*target_pt + (1-mask)*bg_pt; target = full PBR with shadows.",
        },
        mask=fg_mask,
    )
    return True


def _hemispheric_snapshot(runtime, env, output_dir: Path, cfg: OnlineConfig, env_name: str) -> None:
    """One-shot multi-view rgb+depth+camera matrix dump for offline gsplat.

    The DIFIX3D+ artifacts-correction strategies (sparse_k / underfit / cycle
    / cross_ref) need rgb + intrinsics + extrinsics across N hemispheric
    views of a static scene. We do that here once per env, before the
    trajectory loop touches anything, and write to disk:

      ``<output_dir>/views/<NNNN>/{rgb.png, depth.npy, intrinsics.json,
      extrinsics.json}``

    The post-hoc ``scripts/build_artifacts_pairs.py`` reads these views and
    runs the gsplat strategies; no Isaac Sim needed there.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    views_dir = output_dir / "views"
    views_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[online:{env_name}] hemispheric snapshot: {cfg.hemisphere_cameras} cameras "
        f"radius={cfg.hemisphere_radius} spp={cfg.hemisphere_spp}",
        flush=True,
    )

    # Settle the env so default joint positions & object resting state are stable.
    if cfg.hemisphere_settle_steps > 0:
        runtime.step(cfg.hemisphere_settle_steps)

    cameras = runtime.add_sphere_cameras(
        center=cfg.hemisphere_center,
        radius=cfg.hemisphere_radius,
        num_cameras=cfg.hemisphere_cameras,
        resolution=cfg.hemisphere_resolution,
        name_prefix=f"hemi_{env_name}",
    )
    runtime.set_path_tracing(True, spp=cfg.hemisphere_spp)
    _kit_update()

    manifest_views = []
    import time as _time

    t0 = _time.time()
    for idx, cam in enumerate(cameras):
        frame = runtime.capture_frame(cam, rgb=True, depth=True)
        view_dir = views_dir / f"{idx:04d}"
        view_dir.mkdir(parents=True, exist_ok=True)
        save_png(view_dir / "rgb.png", frame["rgb"])
        if frame.get("depth") is not None:
            np.save(view_dir / "depth.npy", frame["depth"])
        save_json(view_dir / "intrinsics.json", {"K": frame["camera_intrinsics"].tolist()})
        save_json(view_dir / "extrinsics.json", {"world_T_cam_gl": frame["camera_extrinsics"].tolist()})
        manifest_views.append({"view_id": idx, "dir": str(view_dir.relative_to(output_dir))})
        if (idx + 1) % 10 == 0:
            elapsed = _time.time() - t0
            rate = (idx + 1) / max(elapsed, 1e-3)
            print(
                f"[online:{env_name}]   hemi {idx + 1}/{len(cameras)} ({rate:.2f} fps)",
                flush=True,
            )

    save_json(
        output_dir / "manifest.json",
        {
            "env_name": env_name,
            "num_views": len(cameras),
            "resolution": list(cfg.hemisphere_resolution),
            "spp": cfg.hemisphere_spp,
            "views": manifest_views,
        },
    )
    runtime.set_path_tracing(False)
    print(
        f"[online:{env_name}] hemispheric snapshot done in {_time.time() - t0:.1f}s",
        flush=True,
    )


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


def _load_playback_actions(env_name: str, root: Path | None, num_episodes: int) -> list[np.ndarray]:
    """Load up to ``num_episodes`` HDF5 demo trajectories.

    Tries ``<root>/<env_name>/data.hdf5`` first. If the requested env has no
    recorded demos, falls back to the first sibling task that does — the
    Droid action space is identical across tasks, so a borrowed trajectory
    still gives feasible robot motion (it just won't be task-specific).
    """

    if root is None:
        return []
    root = Path(root)
    if not root.exists():
        return []
    try:
        from robolab.core.utils.file_utils import load_hdf5_episode_data
    except Exception:
        return []

    candidates: list[Path] = []
    own = root / env_name / "data.hdf5"
    if own.exists():
        candidates.append(own)
    for sibling in sorted(root.iterdir()):
        sibling_path = sibling / "data.hdf5"
        if sibling.name == env_name or not sibling_path.exists():
            continue
        candidates.append(sibling_path)

    for hdf5_path in candidates:
        actions: list[np.ndarray] = []
        for episode in range(num_episodes):
            try:
                arr = load_hdf5_episode_data(str(hdf5_path), episode, "actions")
            except Exception:
                break
            if arr is None or len(arr) == 0:
                break
            actions.append(np.asarray(arr))
        if actions:
            if hdf5_path.parent.name != env_name:
                print(
                    f"[playback] {env_name} has no recorded demos; borrowing "
                    f"{hdf5_path.parent.name} ({len(actions)} episodes) — "
                    f"action space is identical across Droid tasks.",
                    flush=True,
                )
            return actions
    return []


def _kit_update() -> None:
    """Pump the Omniverse Kit app once so USD attribute changes propagate to the renderer."""

    try:
        import omni.kit.app

        omni.kit.app.get_app().update()
    except Exception:
        pass


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
