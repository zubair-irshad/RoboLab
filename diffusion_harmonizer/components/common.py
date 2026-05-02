"""Mask helpers shared by ISP / Relighting / Shadow / Asset-Reinsertion."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def ids_matching_paths(mapping: dict[int, str], path_needles: Iterable[str]) -> list[int]:
    needles = [n.lower() for n in path_needles if n]
    if not needles:
        return [idx for idx in mapping if idx != 0]
    return [idx for idx, path in mapping.items() if any(n in path.lower() for n in needles)]


def foreground_mask(segmentation: np.ndarray, mapping: dict[int, str], path_needles: Iterable[str]) -> np.ndarray:
    ids = ids_matching_paths(mapping, path_needles)
    if not ids:
        return np.zeros(segmentation.shape[:2], dtype=np.float32)
    return np.isin(segmentation, ids).astype(np.float32)


def foreground_mask_with_fallback(
    segmentation: np.ndarray,
    mapping: dict[int, str],
    path_needles: Iterable[str],
) -> tuple[np.ndarray, str]:
    """Mask via Replicator's prim-path mapping, falling back to instance-area heuristic."""

    mask = foreground_mask(segmentation, mapping, path_needles)
    if float(np.mean(mask > 0.05)) >= 0.002:
        return mask, "replicator_mapping"

    seg = np.asarray(segmentation)
    total = float(seg.shape[0] * seg.shape[1])
    selected = []
    for idx in np.unique(seg):
        if int(idx) == 0:
            continue
        coverage = float(np.count_nonzero(seg == idx)) / max(total, 1.0)
        # Manipulable foreground objects are typically modest-coverage instances.
        if 0.0002 <= coverage <= 0.30:
            selected.append(idx)
    if not selected:
        return np.zeros(seg.shape[:2], dtype=np.float32), "empty_foreground_mask"
    return np.isin(seg, selected).astype(np.float32), "instance_area_fallback"


def foreground_mask_from_visibility_difference(
    target_rgb: np.ndarray,
    receiver_rgb: np.ndarray,
    threshold: float = 0.10,
) -> np.ndarray:
    """Pixel-difference fallback when instance segmentation has no useful labels."""

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


def pair_id(index: int) -> str:
    return f"{index:04d}"
