"""End-to-end DiffusionHarmonizer paired-data engine over RoboLab scenes.

For every scene in ``assets/scenes/`` we:

1. Reference the scene USDA + Franka USD + dome-light HDRI onto a single Isaac
   Sim stage (mirrors ``robolab/core/environments`` prim layout).
2. Capture privileged 100-Fibonacci-sphere views (rgb + depth).
3. Run all five DiffusionHarmonizer §3.2 paired-data components:
     a. Novel-view artifacts correction  (in-process gsplat: sparse-K / underfit
        / cycle / cross-ref handicaps vs full-quality reference)
     b. ISP modification                 (software-ISP foreground / background mix)
     c. Relighting                       (sidecar diffusion model crop relight)
     d. Physically based shadow simulation
        (USD shadow-link excludes foreground casters; lighting unchanged)
     e. Asset re-insertion               (background gsplat composite vs full PBR)
4. Save inputs / targets / masks / per-component intermediates to disk in a
   navigable directory layout. No HTML.

The orchestrator owns the Isaac Sim app and the renderer; the components are
plain functions invoked sequentially with a shared ``foreground_paths`` list
that always includes the Franka root + every manipulable scene object.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from diffusion_harmonizer.components import (
    artifacts_correction,
    asset_reinsertion,
    isp_modification,
    relighting,
    shadow_simulation,
)
from diffusion_harmonizer.data.image_io import save_json, save_png
from diffusion_harmonizer.scene_templates.robolab_loader import (
    RoboLabSceneSpec,
    discover_scenes,
    reference_scene_into_renderer,
)


@dataclass
class PipelineConfig:
    output_root: Path = Path("data/diffusion_harmonizer")
    num_sphere_cameras: int = 100
    sphere_radius: float = 1.6
    sphere_center: tuple[float, float, float] = (0.4, 0.0, 0.4)
    capture_resolution: tuple[int, int] = (512, 512)
    splat_kind: str = "3dgs"  # or "2dgs"
    full_iterations: int = 30000
    artifacts_pairs_per_scene: int = 40
    isp_pairs_per_scene: int = 12
    relighting_pairs_per_scene: int = 8
    shadow_pairs_per_scene: int = 12
    reinsertion_pairs_per_scene: int = 12
    relighting_command: str | None = None  # required for the relighting component
    seed: int = 42
    components: tuple[str, ...] = (
        "artifacts_correction",
        "isp_modification",
        "relighting",
        "shadow_simulation",
        "asset_reinsertion",
    )


def run_pipeline(
    cfg: PipelineConfig,
    scene_specs: list[RoboLabSceneSpec] | None = None,
    headless: bool = True,
    renderer_type: str = "raytraced",
) -> dict[str, Any]:
    """Top-level entry point. Constructs the renderer, iterates scenes."""

    from diffusion_harmonizer.rendering import launch_renderer

    cfg.output_root.mkdir(parents=True, exist_ok=True)
    if scene_specs is None:
        scene_specs = discover_scenes()
    if not scene_specs:
        raise RuntimeError("No RoboLab scenes discovered. Check assets/scenes/.")

    save_json(
        cfg.output_root / "config.json",
        {
            "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.__dict__.items()},
            "scenes": [
                {
                    "scene_id": s.scene_id,
                    "scene_usda": str(s.scene_usda),
                    "robot_usd": str(s.robot_usd),
                    "hdri_path": str(s.hdri_path),
                    "object_prim_paths": s.object_prim_paths,
                    "receiver_prim_paths": s.receiver_prim_paths,
                }
                for s in scene_specs
            ],
        },
    )

    renderer = launch_renderer(headless=headless, renderer_type=renderer_type)
    summary: dict[str, Any] = {"scenes": {}, "started_at": time.time()}
    try:
        for spec in scene_specs:
            t0 = time.time()
            scene_summary = _run_scene(renderer, spec, cfg)
            scene_summary["wall_seconds"] = time.time() - t0
            summary["scenes"][spec.scene_id] = scene_summary
            save_json(cfg.output_root / "summary.json", summary)
    finally:
        renderer.shutdown()
    summary["finished_at"] = time.time()
    save_json(cfg.output_root / "summary.json", summary)
    return summary


def _run_scene(renderer, spec: RoboLabSceneSpec, cfg: PipelineConfig) -> dict[str, Any]:
    scene_dir = cfg.output_root / spec.scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "scene_id": spec.scene_id,
        "scene_usda": str(spec.scene_usda),
        "hdri_path": str(spec.hdri_path),
        "foreground_prim_paths": spec.foreground_prim_paths,
        "components": {},
    }

    # Fresh stage per scene so prim paths and camera lists don't leak between scenes.
    renderer.world.clear() if hasattr(renderer.world, "clear") else None
    renderer.cameras.clear()
    renderer.render_products.clear()
    renderer.annotators.clear()
    reference_scene_into_renderer(renderer, spec)
    renderer.step(4)

    _save_scene_preview(renderer, spec, scene_dir / "preview")

    foreground_paths = spec.foreground_prim_paths

    if "artifacts_correction" in cfg.components:
        ac_dir = scene_dir / "01_artifacts_correction"
        cameras, captured_views = artifacts_correction.capture_sphere_views(
            renderer,
            center=cfg.sphere_center,
            radius=cfg.sphere_radius,
            num_cameras=cfg.num_sphere_cameras,
            resolution=cfg.capture_resolution,
            name_prefix=f"{spec.scene_id}_sphere",
        )
        entries = artifacts_correction.generate_pairs(
            renderer,
            output_dir=ac_dir,
            count=cfg.artifacts_pairs_per_scene,
            full_iterations=cfg.full_iterations,
            splat_kind=cfg.splat_kind,
            seed=cfg.seed,
            captured_views=captured_views,
        )
        summary["components"]["artifacts_correction"] = {"output_dir": str(ac_dir), "num_pairs": len(entries)}
    else:
        cameras = []
        captured_views = []

    if "isp_modification" in cfg.components:
        isp_dir = scene_dir / "02_isp_modification"
        entries = isp_modification.generate_pairs(
            renderer,
            output_dir=isp_dir,
            count=cfg.isp_pairs_per_scene,
            foreground_paths=foreground_paths,
            seed=cfg.seed,
            full_frame_fraction=0.2,
        )
        summary["components"]["isp_modification"] = {"output_dir": str(isp_dir), "num_pairs": len(entries)}

    if "relighting" in cfg.components:
        rl_dir = scene_dir / "03_relighting"
        try:
            entries = relighting.generate_pairs(
                renderer,
                output_dir=rl_dir,
                count=cfg.relighting_pairs_per_scene,
                foreground_paths=foreground_paths,
                relighting_command=cfg.relighting_command,
                seed=cfg.seed,
            )
            summary["components"]["relighting"] = {"output_dir": str(rl_dir), "num_pairs": len(entries)}
        except RuntimeError as exc:
            summary["components"]["relighting"] = {"output_dir": str(rl_dir), "skipped": str(exc)}

    if "shadow_simulation" in cfg.components:
        shadow_dir = scene_dir / "04_shadow_simulation"
        entries = shadow_simulation.generate_pairs(
            renderer,
            output_dir=shadow_dir,
            count=cfg.shadow_pairs_per_scene,
            seed=cfg.seed,
            foreground_paths=foreground_paths,
        )
        summary["components"]["shadow_simulation"] = {"output_dir": str(shadow_dir), "num_pairs": len(entries)}

    if "asset_reinsertion" in cfg.components:
        reinsert_dir = scene_dir / "05_asset_reinsertion"
        entries = asset_reinsertion.generate_pairs(
            renderer,
            foreground_paths=foreground_paths,
            output_dir=reinsert_dir,
            count=cfg.reinsertion_pairs_per_scene,
            sphere_center=cfg.sphere_center,
            sphere_radius=cfg.sphere_radius,
            num_cameras=cfg.num_sphere_cameras,
            resolution=cfg.capture_resolution,
            full_iterations=cfg.full_iterations,
            splat_kind=cfg.splat_kind,
            seed=cfg.seed,
        )
        summary["components"]["asset_reinsertion"] = {"output_dir": str(reinsert_dir), "num_pairs": len(entries)}

    save_json(scene_dir / "scene_summary.json", summary)
    _write_step_overview(scene_dir, summary)
    return summary


def _save_scene_preview(renderer, spec: RoboLabSceneSpec, preview_dir: Path) -> None:
    """Render four canonical preview cameras so each scene is visually verifiable."""

    preview_dir.mkdir(parents=True, exist_ok=True)
    presets = [
        ((1.6, -1.2, 1.2), (0.4, 0.0, 0.5)),
        ((1.6, 1.2, 1.2), (0.4, 0.0, 0.5)),
        ((-0.2, -2.0, 1.4), (0.4, 0.0, 0.6)),
        ((0.4, 0.0, 2.4), (0.4, 0.0, 0.5)),
    ]
    for idx, (eye, look_at) in enumerate(presets):
        name = f"{spec.scene_id}_preview_{idx:02d}"
        renderer.add_camera(name, position=eye, look_at=look_at, resolution=(800, 600))
        frame = renderer.capture_frame(name, rgb=True, segmentation=True)
        save_png(preview_dir / f"{idx:02d}_rgb.png", frame["rgb"])


def _write_step_overview(scene_dir: Path, summary: dict[str, Any]) -> None:
    """Plain-text per-step index pointing to every component's outputs on disk."""

    lines = [f"Scene: {summary['scene_id']}", f"Foreground prims: {summary['foreground_prim_paths']}", ""]
    for component, info in summary["components"].items():
        lines.append(f"[{component}]")
        for key, value in info.items():
            lines.append(f"  {key}: {value}")
        lines.append("")
    (scene_dir / "INDEX.txt").write_text("\n".join(lines))
