from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def ids_matching_paths(mapping: dict[int, str], path_needles: Iterable[str]) -> list[int]:
    needles = [needle.lower() for needle in path_needles if needle]
    if not needles:
        return [idx for idx in mapping if idx != 0]
    matched = []
    for idx, path in mapping.items():
        low = path.lower()
        if any(needle in low for needle in needles):
            matched.append(idx)
    return matched


def foreground_mask(segmentation: np.ndarray, mapping: dict[int, str], path_needles: Iterable[str]) -> np.ndarray:
    ids = ids_matching_paths(mapping, path_needles)
    if not ids:
        return np.zeros(segmentation.shape[:2], dtype=np.float32)
    return np.isin(segmentation, ids).astype(np.float32)


def foreground_mask_by_instance_area(
    segmentation: np.ndarray,
    min_coverage: float = 0.0002,
    max_coverage: float = 0.30,
) -> np.ndarray:
    """Fallback for Isaac instance masks when id->prim-path labels are absent.

    The generated demos have a large procedural table/background plus smaller
    robot/object foreground instances. This fallback keeps visible nonzero
    instance IDs in a conservative area range and drops very large receiver IDs.
    """

    seg = np.asarray(segmentation)
    if seg.size == 0:
        return np.zeros(seg.shape[:2], dtype=np.float32)
    selected = []
    total = float(seg.shape[0] * seg.shape[1])
    for idx in np.unique(seg):
        if int(idx) == 0:
            continue
        coverage = float(np.count_nonzero(seg == idx)) / max(total, 1.0)
        if min_coverage <= coverage <= max_coverage:
            selected.append(idx)
    if not selected:
        return np.zeros(seg.shape[:2], dtype=np.float32)
    return np.isin(seg, selected).astype(np.float32)


def foreground_mask_with_fallback(
    segmentation: np.ndarray,
    mapping: dict[int, str],
    path_needles: Iterable[str],
) -> tuple[np.ndarray, str]:
    mask = foreground_mask(segmentation, mapping, path_needles)
    if float(np.mean(mask > 0.05)) >= 0.002:
        return mask, "replicator_mapping"
    fallback = foreground_mask_by_instance_area(segmentation)
    if float(np.mean(fallback > 0.05)) >= 0.002:
        return fallback, "instance_area_fallback"
    return np.zeros(segmentation.shape[:2], dtype=np.float32), "empty_foreground_mask"


def foreground_mask_from_visibility_difference(
    target_rgb: np.ndarray,
    receiver_rgb: np.ndarray,
    threshold: float = 0.10,
) -> np.ndarray:
    """Estimate visible foreground pixels from target vs foreground-hidden render.

    This is a fallback for Isaac/Replicator builds where instance segmentation
    produces no useful IDs. The threshold is intentionally fairly high so soft
    cast shadows on the receiver are mostly excluded.
    """

    import cv2

    target = target_rgb.astype(np.float32) / 255.0
    receiver = receiver_rgb.astype(np.float32) / 255.0
    diff = np.max(np.abs(target - receiver), axis=-1)
    mask = (diff > threshold).astype(np.uint8)
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask.astype(np.float32)


def feather_mask(mask: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    import cv2

    return cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)


def discover_demo_cameras(renderer, count: int = 5) -> list[str]:
    if renderer.cameras:
        return list(renderer.cameras.keys())[:count]
    presets = [
        ((1.7, -1.2, 1.35), (-0.20, 0.0, 0.82)),
        ((1.6, 1.2, 1.25), (-0.20, 0.0, 0.82)),
        ((-1.7, -1.1, 1.35), (-0.35, 0.0, 0.95)),
        ((-1.8, 1.0, 1.25), (-0.35, 0.0, 0.95)),
        ((0.0, -2.0, 1.55), (-0.15, 0.0, 0.88)),
        ((-2.2, 0.0, 1.65), (-0.55, 0.0, 1.0)),
    ]
    names = []
    for idx in range(count):
        position, look_at = presets[idx % len(presets)]
        name = f"demo_cam_{idx:03d}"
        renderer.add_camera(name, position=position, look_at=look_at, resolution=(640, 480), focal_length=20.0)
        names.append(name)
    return names


def pair_id(index: int) -> str:
    return f"{index:04d}"


def relative_to_cwd(path: str | Path) -> str:
    try:
        return str(Path(path).relative_to(Path.cwd()))
    except Exception:
        return str(path)
