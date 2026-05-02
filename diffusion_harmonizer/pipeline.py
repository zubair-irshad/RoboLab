"""End-to-end DiffusionHarmonizer paired-data engine over registered RoboLab tasks.

For every requested env name (returned by ``robolab.core.environments.factory.get_envs``)
we:

1. ``create_env(env_name)`` — Isaac Lab managers spawn the scene, robot,
   cameras, and dome HDRI on the live stage.
2. Wrap it in ``HarmonizerRuntime``, add 100 Fibonacci-sphere Replicator cameras,
   and step the env so physics settles.
3. Run all five DiffusionHarmonizer §3.2 paired-data components against the
   live stage. Foreground prim paths are derived from
   ``env_cfg.contact_object_list`` (entries that don't look like
   table/shelf/bin/ground are foreground; the robot is always foreground).
4. Save inputs / targets / masks / per-component intermediates to disk.
5. ``runtime.close()`` and move on to the next env.

The orchestrator does not own the Isaac Sim app — the CLI driver launches it
once via ``isaaclab.app.AppLauncher`` and the orchestrator runs inside.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from diffusion_harmonizer.components import (
    artifacts_correction,
    asset_reinsertion,
    isp_modification,
    relighting,
    shadow_simulation,
)
from diffusion_harmonizer.image_io import save_json, save_png
from diffusion_harmonizer.runtime import HarmonizerRuntime


@dataclass
class PipelineConfig:
    output_root: Path = Path("data/diffusion_harmonizer")
    num_sphere_cameras: int = 100
    sphere_radius: float = 1.6
    sphere_center: tuple[float, float, float] = (0.4, 0.0, 0.4)
    capture_resolution: tuple[int, int] = (512, 512)
    splat_kind: str = "3dgs"  # or "2dgs"
    spp: int = 64
    full_iterations: int = 30000
    artifacts_pairs_per_env: int = 40
    isp_pairs_per_env: int = 12
    relighting_pairs_per_env: int = 8
    shadow_pairs_per_env: int = 12
    reinsertion_pairs_per_env: int = 12
    relighting_command: str | None = None
    seed: int = 42
    settle_steps: int = 0  # 0 = skip stepping; env.reset() inside HarmonizerRuntime is enough
    device: str = "cuda:0"  # pass "cpu" to fall back to CPU PhysX on memory-constrained GPUs
    num_envs: int = 1
    orbit_cameras: int = 8  # used by ISP/relighting/shadow when sphere cameras aren't built
    physx_buffer_scale: float = 0.1  # 0.1 = 10% of Isaac Lab's parallel-training defaults; raise for big scenes
    components: tuple[str, ...] = (
        "artifacts_correction",
        "isp_modification",
        "relighting",
        "shadow_simulation",
        "asset_reinsertion",
    )
    hdri_roots: tuple[str, ...] = ("assets/backgrounds/indoors", "assets/backgrounds/default")
    preview_cameras: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...] = field(
        default_factory=lambda: (
            ((1.6, -1.2, 1.2), (0.4, 0.0, 0.5)),
            ((1.6, 1.2, 1.2), (0.4, 0.0, 0.5)),
            ((-0.2, -2.0, 1.4), (0.4, 0.0, 0.6)),
            ((0.4, 0.0, 2.4), (0.4, 0.0, 0.5)),
        )
    )


def run_pipeline(cfg: PipelineConfig, env_names: list[str]) -> dict[str, Any]:
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"envs": {}, "started_at": time.time()}
    save_json(cfg.output_root / "config.json", _config_to_json(cfg, env_names))

    for env_name in env_names:
        t0 = time.time()
        try:
            env_summary = _run_env(env_name, cfg)
        except Exception as exc:
            env_summary = {"env_name": env_name, "error": str(exc), "traceback": traceback.format_exc()}
            print(f"[pipeline:{env_name}] FAILED: {exc}\n{env_summary['traceback']}", flush=True)
        env_summary["wall_seconds"] = time.time() - t0
        summary["envs"][env_name] = env_summary
        save_json(cfg.output_root / "summary.json", summary)

    summary["finished_at"] = time.time()
    save_json(cfg.output_root / "summary.json", summary)
    return summary


def _run_env(env_name: str, cfg: PipelineConfig) -> dict[str, Any]:
    env_dir = cfg.output_root / env_name
    env_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[pipeline] === env {env_name} ===", flush=True)

    runtime = HarmonizerRuntime(
        env_name=env_name,
        seed=cfg.seed,
        device=cfg.device,
        num_envs=cfg.num_envs,
        physx_buffer_scale=cfg.physx_buffer_scale,
    )
    summary: dict[str, Any] = {
        "env_name": env_name,
        "instruction": getattr(runtime.env_cfg, "instruction", None),
        "contact_object_list": runtime.contact_object_names(),
        "foreground_prim_paths": runtime.foreground_prim_paths(),
        "receiver_prim_paths": runtime.receiver_prim_paths(),
        "components": {},
    }
    print(f"[pipeline:{env_name}] instruction: {summary['instruction']}", flush=True)
    print(f"[pipeline:{env_name}] contact_object_list: {summary['contact_object_list']}", flush=True)
    print(f"[pipeline:{env_name}] foreground_prim_paths: {summary['foreground_prim_paths']}", flush=True)

    try:
        if cfg.settle_steps > 0:
            print(f"[pipeline:{env_name}] stepping {cfg.settle_steps} settle steps", flush=True)
            runtime.step(cfg.settle_steps)
        _save_preview(runtime, env_dir / "preview", cfg.preview_cameras)

        # Cheap single-view components only need an orbit ring of cameras.
        orbit_cameras: list[str] = []
        cheap_components = ("isp_modification", "relighting", "shadow_simulation")
        if any(c in cfg.components for c in cheap_components):
            orbit_cameras = runtime.add_orbit_cameras(
                center=cfg.sphere_center,
                radius=cfg.sphere_radius,
                num_cameras=cfg.orbit_cameras,
                resolution=cfg.capture_resolution,
                name_prefix=f"orbit_{env_name}",
            )
            print(f"[pipeline:{env_name}] added {len(orbit_cameras)} orbit cameras", flush=True)

        # Cheap components first: failing on heavy components later still leaves output.
        if "isp_modification" in cfg.components:
            print(f"[pipeline:{env_name}] >>> 02_isp_modification", flush=True)
            entries = isp_modification.generate_pairs(
                runtime,
                cameras=orbit_cameras,
                output_dir=env_dir / "02_isp_modification",
                count=cfg.isp_pairs_per_env,
                seed=cfg.seed,
            )
            summary["components"]["isp_modification"] = {"num_pairs": len(entries)}
            print(f"[pipeline:{env_name}] <<< 02_isp_modification wrote {len(entries)} pairs", flush=True)

        if "relighting" in cfg.components:
            print(f"[pipeline:{env_name}] >>> 03_relighting", flush=True)
            try:
                entries = relighting.generate_pairs(
                    runtime,
                    cameras=orbit_cameras,
                    output_dir=env_dir / "03_relighting",
                    count=cfg.relighting_pairs_per_env,
                    relighting_command=cfg.relighting_command,
                    seed=cfg.seed,
                )
                summary["components"]["relighting"] = {"num_pairs": len(entries)}
                print(f"[pipeline:{env_name}] <<< 03_relighting wrote {len(entries)} pairs", flush=True)
            except RuntimeError as exc:
                summary["components"]["relighting"] = {"skipped": str(exc)}
                print(f"[pipeline:{env_name}] !!! 03_relighting skipped: {exc}", flush=True)

        if "shadow_simulation" in cfg.components:
            print(f"[pipeline:{env_name}] >>> 04_shadow_simulation", flush=True)
            entries = shadow_simulation.generate_pairs(
                runtime,
                cameras=orbit_cameras,
                output_dir=env_dir / "04_shadow_simulation",
                count=cfg.shadow_pairs_per_env,
                seed=cfg.seed,
                hdri_roots=cfg.hdri_roots,
                spp=cfg.spp,
            )
            summary["components"]["shadow_simulation"] = {"num_pairs": len(entries)}
            print(f"[pipeline:{env_name}] <<< 04_shadow_simulation wrote {len(entries)} pairs", flush=True)

        # Heavy components: build the 100-camera sphere only when needed.
        sphere_components = ("artifacts_correction", "asset_reinsertion")
        sphere_cameras: list[str] = []
        if any(c in cfg.components for c in sphere_components):
            sphere_cameras = runtime.add_sphere_cameras(
                center=cfg.sphere_center,
                radius=cfg.sphere_radius,
                num_cameras=cfg.num_sphere_cameras,
                resolution=cfg.capture_resolution,
                name_prefix=f"sphere_{env_name}",
            )
            print(f"[pipeline:{env_name}] added {len(sphere_cameras)} spherical cameras", flush=True)

        captured_views = None
        if "artifacts_correction" in cfg.components:
            print(f"[pipeline:{env_name}] >>> 01_artifacts_correction", flush=True)
            captured_views = artifacts_correction.capture_sphere_views(runtime, sphere_cameras, spp=cfg.spp)
            entries = artifacts_correction.generate_pairs(
                runtime,
                cameras=sphere_cameras,
                output_dir=env_dir / "01_artifacts_correction",
                count=cfg.artifacts_pairs_per_env,
                full_iterations=cfg.full_iterations,
                splat_kind=cfg.splat_kind,
                seed=cfg.seed,
                spp=cfg.spp,
                captured_views=captured_views,
            )
            summary["components"]["artifacts_correction"] = {"num_pairs": len(entries)}
            print(f"[pipeline:{env_name}] <<< 01_artifacts_correction wrote {len(entries)} pairs", flush=True)

        if "asset_reinsertion" in cfg.components:
            print(f"[pipeline:{env_name}] >>> 05_asset_reinsertion", flush=True)

            def _gs_progress(name, step, loss):
                print(f"[gsplat:{name}] step {step:>6d}  loss {loss:.4f}", flush=True)

            entries = asset_reinsertion.generate_pairs(
                runtime,
                cameras=sphere_cameras,
                output_dir=env_dir / "05_asset_reinsertion",
                count=cfg.reinsertion_pairs_per_env,
                full_iterations=cfg.full_iterations,
                splat_kind=cfg.splat_kind,
                seed=cfg.seed,
                spp=cfg.spp,
                progress_cb=_gs_progress,
            )
            summary["components"]["asset_reinsertion"] = {"num_pairs": len(entries)}
            print(f"[pipeline:{env_name}] <<< 05_asset_reinsertion wrote {len(entries)} pairs", flush=True)

        save_json(env_dir / "env_summary.json", summary)
        _write_index(env_dir, summary)
    finally:
        runtime.close()
        _release_cuda_memory()
    return summary


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


def _save_preview(runtime, preview_dir: Path, presets) -> None:
    preview_dir.mkdir(parents=True, exist_ok=True)
    for idx, (eye, look_at) in enumerate(presets):
        name = f"preview_{idx:02d}"
        if name not in runtime.cameras:
            runtime._add_camera(name=name, position=eye, look_at=look_at, resolution=(800, 600))
        frame = runtime.capture_frame(name, rgb=True)
        save_png(preview_dir / f"{idx:02d}_rgb.png", frame["rgb"])


def _write_index(env_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        f"Env: {summary['env_name']}",
        f"Instruction: {summary.get('instruction')}",
        f"Foreground prims: {summary['foreground_prim_paths']}",
        f"Receiver prims:   {summary['receiver_prim_paths']}",
        "",
    ]
    for component, info in summary["components"].items():
        lines.append(f"[{component}]")
        for key, value in info.items():
            lines.append(f"  {key}: {value}")
        lines.append("")
    (env_dir / "INDEX.txt").write_text("\n".join(lines))


def _config_to_json(cfg: PipelineConfig, env_names: list[str]) -> dict[str, Any]:
    return {
        "env_names": env_names,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.__dict__.items()},
    }
