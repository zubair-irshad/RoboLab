"""Live Isaac Sim capture: dump every variant the post-hoc builders need.

This module owns the *only* phase that needs Isaac Sim. For each requested env
it captures three rgb variants per spherical camera plus the instance mask,
depth, and camera matrices, and saves them to disk:

  ``data/captures/<env>/sphere/<NNNN>/``
    target.png         - full PBR (foreground + cast shadows + dome lighting)
    no_shadow_fg.png   - same, but foreground excluded from UsdLux shadowLink
    bg_only.png        - foreground hidden (table/shelf/dome only)
    fg_only.png        - receivers hidden, shadows off (foreground albedo)
    mask.png           - foreground instance-segmentation mask
    depth.npy          - distance_to_camera depth
    intrinsics.json    - (3, 3) camera intrinsics
    extrinsics.json    - (4, 4) world-from-camera in Replicator/OpenGL convention

Plus a `manifest.json` per env listing all view ids, foreground/receiver prim
paths, dome HDRI used, and ISP / shadow / reinsertion handle which subset they
need. Builders read from these directories; Isaac Sim is never re-launched.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from diffusion_harmonizer.components.common import (
    feather_mask,
    foreground_mask_with_fallback,
)
from diffusion_harmonizer.image_io import save_json, save_png
from diffusion_harmonizer.runtime import HarmonizerRuntime


@dataclass
class CaptureConfig:
    output_root: Path = Path("data/captures")
    num_sphere_cameras: int = 60
    sphere_radius: float = 1.6
    sphere_center: tuple[float, float, float] = (0.4, 0.0, 0.4)
    capture_resolution: tuple[int, int] = (512, 512)
    spp: int = 16
    seed: int = 42
    device: str = "cuda:0"
    num_envs: int = 1
    physx_buffer_scale: float = 0.1
    # Which variants to save. ``no_shadow_fg`` powers the shadow builder;
    # ``bg_only`` + ``fg_only`` power asset re-insertion. Skip the ones you
    # don't need to halve render time.
    variants: tuple[str, ...] = ("target", "no_shadow_fg", "bg_only", "fg_only")


def capture_envs(env_names: list[str], cfg: CaptureConfig) -> dict:
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    summary: dict = {"envs": {}, "started_at": time.time(), "config": _config_to_json(cfg)}
    for env_name in env_names:
        t0 = time.time()
        try:
            env_summary = _capture_env(env_name, cfg)
        except Exception as exc:
            import traceback as tb

            env_summary = {"env_name": env_name, "error": str(exc), "traceback": tb.format_exc()}
            print(f"[capture:{env_name}] FAILED: {exc}", flush=True)
        env_summary["wall_seconds"] = time.time() - t0
        summary["envs"][env_name] = env_summary
        save_json(cfg.output_root / "summary.json", summary)
    summary["finished_at"] = time.time()
    save_json(cfg.output_root / "summary.json", summary)
    return summary


def _capture_env(env_name: str, cfg: CaptureConfig) -> dict:
    env_dir = cfg.output_root / env_name
    sphere_dir = env_dir / "sphere"
    sphere_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[capture] === env {env_name} ===", flush=True)
    runtime = HarmonizerRuntime(
        env_name=env_name,
        seed=cfg.seed,
        device=cfg.device,
        num_envs=cfg.num_envs,
        physx_buffer_scale=cfg.physx_buffer_scale,
    )
    try:
        foreground_paths = runtime.foreground_prim_paths()
        receiver_paths = runtime.receiver_prim_paths()
        print(f"[capture:{env_name}] foreground: {foreground_paths}", flush=True)
        print(f"[capture:{env_name}] receivers:  {receiver_paths}", flush=True)

        cameras = runtime.add_sphere_cameras(
            center=cfg.sphere_center,
            radius=cfg.sphere_radius,
            num_cameras=cfg.num_sphere_cameras,
            resolution=cfg.capture_resolution,
            name_prefix=f"capture_{env_name}",
        )
        print(f"[capture:{env_name}] added {len(cameras)} sphere cameras at radius={cfg.sphere_radius}", flush=True)
        runtime.set_path_tracing(cfg.spp > 1, spp=cfg.spp)

        manifest = {
            "env_name": env_name,
            "instruction": getattr(runtime.env_cfg, "instruction", None),
            "foreground_prim_paths": foreground_paths,
            "receiver_prim_paths": receiver_paths,
            "num_views": len(cameras),
            "resolution": list(cfg.capture_resolution),
            "spp": cfg.spp,
            "variants_saved": list(cfg.variants),
            "views": [],
        }

        # 1) Target pass — full lighting + shadows + foreground.
        if "target" in cfg.variants:
            _capture_pass(runtime, cameras, sphere_dir, "target", with_segmentation=True, with_depth=True, verbose=env_name)

        # 2) No-shadow-foreground pass — same lighting, foreground excluded
        #    from every light's shadowLink. Diff with target = shadow signal.
        if "no_shadow_fg" in cfg.variants:
            runtime.set_shadow_link_excludes(foreground_paths, enabled=True)
            try:
                _capture_pass(runtime, cameras, sphere_dir, "no_shadow_fg", verbose=env_name)
            finally:
                runtime.set_shadow_link_excludes(foreground_paths, enabled=False)

        # 3) Background-only pass — foreground hidden. Used by the asset
        #    re-insertion builder to train a clean background gsplat.
        if "bg_only" in cfg.variants:
            runtime.set_prims_visibility(foreground_paths, False)
            try:
                _capture_pass(runtime, cameras, sphere_dir, "bg_only", with_depth=True, verbose=env_name)
            finally:
                runtime.set_prims_visibility(foreground_paths, True)

        # 4) Foreground-only pass — receivers hidden, shadows off, dome on.
        #    Foreground albedo ready for compositing.
        if "fg_only" in cfg.variants and receiver_paths:
            runtime.set_prims_visibility(receiver_paths, False)
            runtime.set_shadows_enabled(False)
            try:
                _capture_pass(runtime, cameras, sphere_dir, "fg_only", verbose=env_name)
            finally:
                runtime.set_shadows_enabled(True)
                runtime.set_prims_visibility(receiver_paths, True)

        # 5) Save mask + camera matrices once (target pass already had segmentation).
        for view_id in range(len(cameras)):
            view_dir = sphere_dir / f"{view_id:04d}"
            manifest["views"].append({"view_id": view_id, "dir": str(view_dir.relative_to(cfg.output_root))})

        save_json(env_dir / "manifest.json", manifest)
        print(f"[capture:{env_name}] manifest written: {env_dir / 'manifest.json'}", flush=True)
        return {"env_name": env_name, "num_views": len(cameras), "variants": list(cfg.variants), "manifest": str(env_dir / "manifest.json")}
    finally:
        runtime.close()
        _release_cuda_memory()


def _capture_pass(
    runtime,
    cameras: list[str],
    sphere_dir: Path,
    variant: str,
    *,
    with_segmentation: bool = False,
    with_depth: bool = False,
    verbose: str = "",
) -> None:
    print(f"[capture:{verbose}] >>> pass={variant} ({len(cameras)} cameras)", flush=True)
    t0 = time.time()
    foreground_paths = runtime.foreground_prim_paths()
    for idx, camera in enumerate(cameras):
        view_dir = sphere_dir / f"{idx:04d}"
        view_dir.mkdir(parents=True, exist_ok=True)
        frame = runtime.capture_frame(
            camera, rgb=True, depth=with_depth, segmentation=with_segmentation
        )
        save_png(view_dir / f"{variant}.png", frame["rgb"])
        if with_depth and frame.get("depth") is not None:
            np.save(view_dir / "depth.npy", frame["depth"])
        if with_segmentation:
            mask, mask_source = foreground_mask_with_fallback(
                frame["segmentation"], frame["segmentation_mapping"] or {}, foreground_paths
            )
            mask = feather_mask(mask, sigma=2.0)
            save_png(view_dir / "mask.png", np.repeat((mask * 255).astype(np.uint8)[..., None], 3, axis=-1))
            (view_dir / "mask_source.txt").write_text(mask_source)
        # Camera matrices once per view (target pass writes them).
        if variant == "target":
            save_json(view_dir / "intrinsics.json", {"K": frame["camera_intrinsics"].tolist()})
            save_json(view_dir / "extrinsics.json", {"world_T_cam_gl": frame["camera_extrinsics"].tolist()})
        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / max(elapsed, 1e-3)
            eta = (len(cameras) - (idx + 1)) / max(rate, 1e-3)
            print(f"[capture:{verbose}]   {variant} {idx + 1}/{len(cameras)} ({rate:.2f} fps, eta {eta:.1f}s)", flush=True)
    print(f"[capture:{verbose}] <<< pass={variant} done in {time.time() - t0:.1f}s", flush=True)


def _release_cuda_memory() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _config_to_json(cfg: CaptureConfig) -> dict:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.__dict__.items()}


def load_view(view_dir: Path, variant: str) -> np.ndarray:
    import imageio.v3 as iio

    return np.asarray(iio.imread(view_dir / f"{variant}.png"))[..., :3]


def load_mask(view_dir: Path) -> np.ndarray:
    import imageio.v3 as iio

    arr = np.asarray(iio.imread(view_dir / "mask.png"))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float32) / 255.0


def load_view_dirs(env_capture_dir: Path) -> list[Path]:
    sphere_dir = env_capture_dir / "sphere"
    return sorted(p for p in sphere_dir.iterdir() if p.is_dir())
