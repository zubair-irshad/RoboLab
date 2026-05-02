from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path

import numpy as np

from diffusion_harmonizer.components.common import discover_demo_cameras, feather_mask, foreground_mask, pair_id
from diffusion_harmonizer.data.image_io import save_png, write_pair


class RelightingModel:
    """Sidecar adapter for a real relighting diffusion model.

    DiffusionHarmonizer cites DiffusionRenderer [19] for relighting. This
    adapter keeps Isaac Sim isolated from model dependency conflicts: pass a
    command that reads a foreground crop and mask and writes a relit crop.
    """

    def __init__(self, command: str | None = None, cache_dir: str | Path = "assets/models/relighting"):
        self.command = command
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def relight(self, crop: np.ndarray, mask: np.ndarray, prompt: str, work_dir: Path) -> np.ndarray:
        if not self.command:
            raise RuntimeError(
                "No relighting diffusion command configured. Use DiffusionRenderer or another real relighting "
                "model sidecar and pass --relighting_command."
            )
        crop_path = work_dir / "relight_crop.png"
        mask_path = work_dir / "relight_mask.png"
        out_path = work_dir / "relight_output.png"
        save_png(crop_path, crop)
        save_png(mask_path, np.repeat((mask * 255).astype(np.uint8)[..., None], 3, axis=-1))
        subprocess.run(
            [
                self.command,
                "--input",
                str(crop_path),
                "--mask",
                str(mask_path),
                "--output",
                str(out_path),
                "--prompt",
                prompt,
                "--cache_dir",
                str(self.cache_dir),
            ],
            check=True,
        )
        import imageio.v3 as iio

        return np.asarray(iio.imread(out_path))[..., :3]


def _bbox(mask: np.ndarray, pad: int, width: int, height: int) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0.05)
    if len(xs) == 0:
        return None
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, width)
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, height)
    return x0, y0, x1, y1


def _lighting_prompt(rng: random.Random) -> str:
    az = rng.uniform(0, 360)
    elev = rng.uniform(10, 75)
    temp = rng.randint(2800, 8500)
    fill = rng.choice(["no fill", "cool blue side fill", "warm amber side fill", "green industrial side fill"])
    return f"relight foreground from azimuth {az:.1f} elevation {elev:.1f}, {temp}K key light, {fill}"


def generate_pairs(
    renderer,
    output_dir: str | Path = "data/relighting/demo",
    count: int = 30,
    foreground_paths: list[str] | None = None,
    relighting_command: str | None = None,
    seed: int = 42,
) -> dict[str, dict[str, str]]:
    rng = random.Random(seed)
    output = Path(output_dir)
    model = RelightingModel(command=relighting_command)
    cameras = discover_demo_cameras(renderer, count=min(5, count))
    foreground_paths = foreground_paths or ["/World/Robot", "/World/Object_"]
    entries: dict[str, dict[str, str]] = {}

    for idx in range(count):
        camera = cameras[idx % len(cameras)]
        frame = renderer.capture_frame(camera, rgb=True, segmentation=True)
        target = frame["rgb"]
        mask = foreground_mask(frame["segmentation"], frame["segmentation_mapping"] or {}, foreground_paths)
        mask = feather_mask(mask, sigma=3.0)
        h, w = mask.shape
        box = _bbox(mask, pad=24, width=w, height=h)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        prompt = _lighting_prompt(rng)
        pair_dir = output / pair_id(idx)
        relit_crop = model.relight(target[y0:y1, x0:x1], mask[y0:y1, x0:x1], prompt, pair_dir)
        relit_full = target.copy()
        relit_full[y0:y1, x0:x1] = relit_crop
        degraded = (mask[..., None] * relit_full.astype(np.float32) + (1.0 - mask[..., None]) * target.astype(np.float32)).astype(np.uint8)
        key = f"relighting_{pair_id(idx)}"
        entries[key] = write_pair(
            pair_dir,
            degraded,
            target,
            {"component": "relighting", "camera": camera, "prompt": prompt, "bbox": [x0, y0, x1, y1]},
            mask=mask,
        )
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/relighting/demo")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--relighting_command", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    from diffusion_harmonizer.rendering import launch_renderer

    renderer = launch_renderer(headless=True)
    try:
        entries = generate_pairs(renderer, args.output_dir, args.count, relighting_command=args.relighting_command, seed=args.seed)
        Path(args.output_dir).joinpath("pairs.json").write_text(json.dumps(entries, indent=2))
    finally:
        renderer.shutdown()


if __name__ == "__main__":
    main()
