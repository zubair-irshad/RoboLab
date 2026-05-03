"""Post-hoc ISP-modification pair builder (DiffusionHarmonizer §3.2 Eq. 3).

Reads ``target.png`` + ``mask.png`` from each captured view, applies a random
software-ISP, composites I_mix = M ⊙ I_ISP + (1−M) ⊙ I_orig, and writes the
pair to disk. Pure NumPy + OpenCV — no Isaac Sim, no torch, no GPU.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np

from diffusion_harmonizer.capture import load_mask, load_view, load_view_dirs
from diffusion_harmonizer.components.common import pair_id
from diffusion_harmonizer.components.isp_modification import (
    apply_software_isp,
    sample_isp_params,
)
from diffusion_harmonizer.image_io import save_png, write_pair


def build(
    env_capture_dir: Path,
    output_dir: Path,
    count: int = 12,
    seed: int = 42,
    full_frame_fraction: float = 0.2,
    strength: float = 0.8,
) -> dict[str, dict[str, str]]:
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    view_dirs = load_view_dirs(env_capture_dir)
    if not view_dirs:
        raise FileNotFoundError(f"No captured views under {env_capture_dir}/sphere")

    entries: dict[str, dict[str, str]] = {}
    pair_index = 0
    for view_dir in view_dirs:
        if pair_index >= count:
            break
        target = load_view(view_dir, "target")
        try:
            mask = load_mask(view_dir)
        except FileNotFoundError:
            continue
        use_full_frame = full_frame_fraction > 0.0 and rng.random() < full_frame_fraction
        params = sample_isp_params(rng, scale=0.3 if use_full_frame else strength)
        isp = apply_software_isp(target, params, rng)
        if use_full_frame:
            mask_arr = np.ones(target.shape[:2], dtype=np.float32)
            mode = "full_frame_mild"
        else:
            if float(np.mean(mask > 0.05)) < 0.002:
                continue
            mask_arr = mask
            mode = "masked_foreground"
        mixed = (
            mask_arr[..., None] * isp.astype(np.float32)
            + (1.0 - mask_arr[..., None]) * target.astype(np.float32)
        ).astype(np.uint8)
        pair_dir = output_dir / pair_id(pair_index)
        save_png(pair_dir / "isp_full.png", isp)
        entries[f"isp_{pair_id(pair_index)}"] = write_pair(
            pair_dir,
            mixed,
            target,
            {
                "component": "isp_modification",
                "mode": mode,
                "view_dir": str(view_dir),
                "params": asdict(params),
                "mask_coverage": float(np.mean(mask_arr > 0.05)),
            },
            mask=mask_arr,
        )
        pair_index += 1
    (output_dir / "pairs.json").write_text(json.dumps(entries, indent=2))
    print(f"[builder:isp] {env_capture_dir.name}: wrote {len(entries)} pairs -> {output_dir}", flush=True)
    return entries
