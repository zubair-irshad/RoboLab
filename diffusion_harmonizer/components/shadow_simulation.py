from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from diffusion_harmonizer.asset_manager import AssetIndex
from diffusion_harmonizer.components.common import (
    discover_demo_cameras,
    foreground_mask_with_fallback,
    pair_id,
)
from diffusion_harmonizer.image_io import save_png, write_pair


def _random_unit_vector_above_horizon(rng: random.Random) -> tuple[float, float, float]:
    az = rng.uniform(0.0, 2.0 * np.pi)
    elev = rng.uniform(np.radians(15.0), np.radians(75.0))
    return (
        float(np.cos(elev) * np.cos(az)),
        float(np.cos(elev) * np.sin(az)),
        float(-np.sin(elev)),
    )


def _filtered_hdris(index: AssetIndex, query: str | None = None) -> list[Path]:
    texture_suffixes = {".hdr", ".exr", ".png", ".jpg", ".jpeg"}
    hdris = [path for path in index.hdris() if path.suffix.lower() in texture_suffixes]
    if not query:
        return hdris
    include = [token.strip().lower() for token in query.split(",") if token.strip()]
    bad = ("outdoor", "forest", "field", "park", "street", "road", "airport", "beach", "mountain", "sky")
    ranked = []
    for path in hdris:
        low = str(path).lower()
        if any(token in low for token in bad) and not any(token in low for token in include):
            continue
        score = sum(token in low for token in include)
        ranked.append((score, str(path), path))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in ranked] or hdris


def _shadow_delta_mask(target: np.ndarray, no_shadow_input: np.ndarray, foreground: np.ndarray) -> np.ndarray:
    diff = np.max(np.abs(target.astype(np.float32) - no_shadow_input.astype(np.float32)), axis=-1) / 255.0
    return np.clip(diff * (1.0 - np.clip(foreground, 0.0, 1.0)), 0.0, 1.0)


def _dilate_mask(mask: np.ndarray, pixels: int = 5) -> np.ndarray:
    import cv2

    kernel = np.ones((pixels, pixels), dtype=np.uint8)
    return cv2.dilate((mask > 0.05).astype(np.uint8), kernel, iterations=1).astype(np.float32)


def generate_pairs(
    renderer,
    assets_root: str | Path = "assets/geniesim",
    output_dir: str | Path = "data/shadow_simulation/demo",
    count: int = 30,
    seed: int = 42,
    pre_pair_callback=None,
    hdri_query: str | None = "indoor,studio,kitchen,office,warehouse,room",
    foreground_paths: list[str] | None = None,
) -> dict[str, dict[str, str]]:
    rng = random.Random(seed)
    index = AssetIndex(assets_root)
    hdris = _filtered_hdris(index, hdri_query)
    if not hdris:
        raise FileNotFoundError(f"No HDRI files found under {assets_root}; finish the GenieSimAssets download first.")

    output = Path(output_dir)
    cameras = discover_demo_cameras(renderer, count=min(5, count))
    entries: dict[str, dict[str, str]] = {}
    foreground_paths = foreground_paths or ["/World/Robot", "/World/Object_"]
    for idx in range(count):
        camera = cameras[idx % len(cameras)]
        scene_state = pre_pair_callback(idx, camera) if pre_pair_callback else {}
        hdri = str(rng.choice(hdris))
        dome_intensity = rng.uniform(800.0, 1600.0)
        dome_rotation = rng.uniform(0.0, 360.0)
        sun = {
            "intensity": rng.uniform(500.0, 1500.0),
            "angle_deg": rng.uniform(2.0, 8.0),
            "direction": _random_unit_vector_above_horizon(rng),
        }

        renderer.set_dome_light(hdri, intensity=dome_intensity, rotation_deg=dome_rotation)
        renderer.set_distant_light(intensity=sun["intensity"], angle_deg=sun["angle_deg"], direction=sun["direction"])
        renderer.set_shadows_enabled(True)
        renderer.set_path_tracing(True, spp=64)
        target_frame = renderer.capture_frame(camera, rgb=True, segmentation=True)
        target = target_frame["rgb"]
        hard_fg_mask, mask_source = foreground_mask_with_fallback(
            target_frame["segmentation"],
            target_frame["segmentation_mapping"] or {},
            foreground_paths,
        )

        # Same camera, same lights, same materials. USD shadow-linking removes
        # foreground prims from the light shadow-caster collection while leaving
        # them visible and illuminated. No post-hoc color or tone correction.
        shadow_link_excludes = renderer.set_shadow_link_excludes(list(foreground_paths), enabled=True)
        renderer.set_path_tracing(True, spp=64)
        degraded = renderer.capture_frame(camera, rgb=True)["rgb"]
        renderer.set_shadow_link_excludes(list(foreground_paths), enabled=False)

        diff = np.abs(target.astype(np.int16) - degraded.astype(np.int16)).astype(np.uint8)
        if float(np.mean(hard_fg_mask > 0.05)) >= 0.002:
            fg_exclusion = _dilate_mask(hard_fg_mask, pixels=7)
            shadow_mask = _shadow_delta_mask(target, degraded, fg_exclusion)
        else:
            shadow_mask = np.max(diff.astype(np.float32), axis=-1) / 255.0
            mask_source = "shadow_delta_no_foreground_mask"
        pair_dir = output / pair_id(idx)
        save_png(pair_dir / "shadow_diff.png", diff)
        save_png(pair_dir / "shadow_mask.png", np.repeat((shadow_mask * 255).astype(np.uint8)[..., None], 3, axis=-1))
        key = f"shadow_{pair_id(idx)}"
        entries[key] = write_pair(
            pair_dir,
            degraded,
            target,
            {
                "component": "shadow_simulation",
                "camera": camera,
                "hdri": hdri,
                "dome_intensity": dome_intensity,
                "dome_rotation_deg": dome_rotation,
                "distant_light": sun,
                "degradation": "UsdLux shadowLink excludes foreground casters; lighting unchanged",
                "shadow_link_excludes": shadow_link_excludes,
                "mask_source": mask_source,
                "shadow_mask_coverage": float(np.mean(shadow_mask > 0.03)),
                "scene_state": scene_state,
            },
        )
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets_root", default="assets/geniesim")
    parser.add_argument("--output_dir", default="data/shadow_simulation/demo")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hdri_query", default="indoor,studio,kitchen,office,warehouse,room")
    args = parser.parse_args()
    from diffusion_harmonizer.rendering import launch_renderer

    renderer = launch_renderer(headless=True)
    try:
        generate_pairs(renderer, args.assets_root, args.output_dir, args.count, args.seed, hdri_query=args.hdri_query)
    finally:
        renderer.shutdown()


if __name__ == "__main__":
    main()
