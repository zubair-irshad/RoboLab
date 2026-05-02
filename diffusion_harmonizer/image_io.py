from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def ensure_uint8_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, 0.0, 1.0) * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def save_png(path: str | Path, image: np.ndarray) -> None:
    import imageio.v3 as iio

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output, ensure_uint8_rgb(image))


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))


def visualize_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return np.zeros((*depth.shape[:2], 3), dtype=np.uint8)
    lo, hi = np.percentile(depth[valid], [2, 98])
    norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return np.repeat(((1.0 - norm) * 255).astype(np.uint8)[..., None], 3, axis=-1)


def visualize_normals(normals: np.ndarray) -> np.ndarray:
    arr = np.asarray(normals, dtype=np.float32)
    arr = (np.clip(arr, -1.0, 1.0) + 1.0) * 0.5
    return (arr * 255).astype(np.uint8)


def visualize_segmentation(segmentation: np.ndarray) -> np.ndarray:
    seg = np.asarray(segmentation, dtype=np.int64)
    out = np.zeros((*seg.shape[:2], 3), dtype=np.uint8)
    ids = np.unique(seg)
    for idx in ids:
        if idx == 0:
            continue
        color = np.array(
            [
                (idx * 37 + 17) % 255,
                (idx * 67 + 29) % 255,
                (idx * 97 + 53) % 255,
            ],
            dtype=np.uint8,
        )
        out[seg == idx] = color
    return out


def comparison(input_image: np.ndarray, target_image: np.ndarray) -> np.ndarray:
    return np.concatenate([ensure_uint8_rgb(input_image), ensure_uint8_rgb(target_image)], axis=1)


def write_pair(
    pair_dir: str | Path,
    input_image: np.ndarray,
    target_image: np.ndarray,
    metadata: dict[str, Any],
    mask: np.ndarray | None = None,
) -> dict[str, str]:
    pair_path = Path(pair_dir)
    pair_path.mkdir(parents=True, exist_ok=True)
    save_png(pair_path / "input.png", input_image)
    save_png(pair_path / "target.png", target_image)
    save_png(pair_path / "comparison.png", comparison(input_image, target_image))
    if mask is not None:
        mask_img = (np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)
        save_png(pair_path / "mask.png", np.repeat(mask_img[..., None], 3, axis=-1))
    save_json(pair_path / "metadata.json", metadata)
    return {
        "image": str(pair_path / "input.png"),
        "target_image": str(pair_path / "target.png"),
        "prompt": "remove degradation",
    }
