"""Relighting (DiffusionHarmonizer §3.2).

The paper relights only the foreground crop with a diffusion relighting model
(DiffusionRenderer [19]) under a randomly sampled lighting prompt, then
composites the relit crop back over the original frame so global lighting
becomes inconsistent. We keep the diffusion model behind a sidecar command so
this repo doesn't pull in conflicting torch / diffusers stacks.

Pass ``relighting_command`` pointing at a script that reads ``--input``, ``--mask``
and writes ``--output``. If unset, the component is skipped with a clear message.
"""

from __future__ import annotations

import random
import subprocess
from pathlib import Path

import numpy as np

from diffusion_harmonizer.components.common import feather_mask, foreground_mask, pair_id
from diffusion_harmonizer.image_io import save_png, write_pair


class RelightingModel:
    def __init__(self, command: str | None = None, cache_dir: str | Path = "assets/models/relighting"):
        self.command = command
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def relight(self, crop: np.ndarray, mask: np.ndarray, prompt: str, work_dir: Path) -> np.ndarray:
        if not self.command:
            raise RuntimeError("No relighting diffusion command configured. Pass --relighting-command.")
        crop_path = work_dir / "relight_crop.png"
        mask_path = work_dir / "relight_mask.png"
        out_path = work_dir / "relight_output.png"
        save_png(crop_path, crop)
        save_png(mask_path, np.repeat((mask * 255).astype(np.uint8)[..., None], 3, axis=-1))
        subprocess.run(
            [
                self.command,
                "--input", str(crop_path),
                "--mask", str(mask_path),
                "--output", str(out_path),
                "--prompt", prompt,
                "--cache_dir", str(self.cache_dir),
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
    runtime,
    cameras: list[str],
    output_dir: str | Path,
    count: int = 12,
    relighting_command: str | None = None,
    seed: int = 42,
) -> dict[str, dict[str, str]]:
    rng = random.Random(seed)
    output = Path(output_dir)
    model = RelightingModel(command=relighting_command)
    foreground_paths = runtime.foreground_prim_paths()
    entries: dict[str, dict[str, str]] = {}

    for idx in range(min(count, len(cameras))):
        camera = cameras[idx]
        frame = runtime.capture_frame(camera, rgb=True, segmentation=True)
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
        pair_dir.mkdir(parents=True, exist_ok=True)
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
