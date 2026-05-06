# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Metric-scale estimation for a COLMAP scene using MoGe-v2 monocular depth.

Why this beats the camera-height heuristic
------------------------------------------
The default ``align_scene_from_mesh`` derives metric scale from
"median camera = 1.5 m above the floor" — robust to within ~20 % for
typical handheld captures, but biased if the operator stood/crouched
unusually. MoGe-v2 emits *metric* monocular depth: for any pixel in any
sampled frame we get d_metric (m), and COLMAP gives us d_colmap (in
arbitrary scene units) for the same pixel via 2D↔3D correspondences.
The ratio d_metric / d_colmap is constant across pixels (up to noise)
and equals the scene's metric-per-COLMAP scale factor.

Median-across-correspondences-then-median-across-frames is robust to
outliers (a few mismatched correspondences, MoGe failing on textureless
walls, etc).

Optional dependency
-------------------
MoGe is a heavy dep (PyTorch + a ViT-L checkpoint). Install with::

    pip install git+https://github.com/microsoft/MoGe.git

The estimator is opt-in: nothing else in this package imports it.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ---- COLMAP readers (binary, sufficient for DL3DV) ------------------------

def _quat_to_R(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


def _read_points3D_bin(path: Path) -> dict[int, np.ndarray]:
    """Return {point3D_id: xyz}. Discards rgb, error, and track."""
    pts: dict[int, np.ndarray] = {}
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            (pid,) = struct.unpack("<Q", f.read(8))
            xyz = np.array(struct.unpack("<3d", f.read(24)), dtype=np.float64)
            f.read(3)  # rgb
            f.read(8)  # error
            (track_len,) = struct.unpack("<Q", f.read(8))
            f.seek(track_len * 8, 1)  # skip track
            pts[int(pid)] = xyz
    return pts


def _read_points3D_txt(path: Path) -> dict[int, np.ndarray]:
    pts: dict[int, np.ndarray] = {}
    for raw in path.read_text().splitlines():
        if not raw.strip() or raw.startswith("#"):
            continue
        parts = raw.split()
        pts[int(parts[0])] = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
    return pts


def _read_points3D(sparse_dir: Path) -> dict[int, np.ndarray]:
    binp = sparse_dir / "points3D.bin"
    if binp.is_file():
        return _read_points3D_bin(binp)
    txt = sparse_dir / "points3D.txt"
    if txt.is_file():
        return _read_points3D_txt(txt)
    raise FileNotFoundError(f"no points3D.{{bin,txt}} under {sparse_dir}")


@dataclass
class _ImageRecord:
    name: str
    R_cw: np.ndarray   # cam-from-world rotation
    t_cw: np.ndarray   # cam-from-world translation
    point2D_xy: np.ndarray  # (N, 2)
    point3D_ids: np.ndarray  # (N,) int64; -1 means "no 3D match"


def _read_images_bin(path: Path) -> list[_ImageRecord]:
    recs: list[_ImageRecord] = []
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            f.read(4)  # image_id
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            f.read(4)  # camera_id
            chars: list[bytes] = []
            while True:
                ch = f.read(1)
                if ch == b"\x00" or ch == b"":
                    break
                chars.append(ch)
            name = b"".join(chars).decode("utf-8", errors="replace")
            (npts,) = struct.unpack("<Q", f.read(8))
            xy = np.zeros((npts, 2), dtype=np.float64)
            pids = np.zeros(npts, dtype=np.int64)
            for i in range(npts):
                x, y, pid = struct.unpack("<ddq", f.read(24))
                xy[i] = (x, y)
                pids[i] = pid
            recs.append(_ImageRecord(
                name=name,
                R_cw=_quat_to_R(qw, qx, qy, qz),
                t_cw=np.array([tx, ty, tz]),
                point2D_xy=xy,
                point3D_ids=pids,
            ))
    return recs


def _read_images_txt(path: Path) -> list[_ImageRecord]:
    recs: list[_ImageRecord] = []
    lines = [ln for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    for i in range(0, len(lines), 2):
        head = lines[i].split()
        # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        qw, qx, qy, qz = (float(head[k]) for k in range(1, 5))
        tx, ty, tz = (float(head[k]) for k in range(5, 8))
        name = head[9]
        body = lines[i + 1].split() if i + 1 < len(lines) else []
        # body: x y point3D_id repeated
        npts = len(body) // 3
        xy = np.zeros((npts, 2), dtype=np.float64)
        pids = np.zeros(npts, dtype=np.int64)
        for k in range(npts):
            xy[k] = (float(body[3 * k]), float(body[3 * k + 1]))
            pids[k] = int(body[3 * k + 2])
        recs.append(_ImageRecord(
            name=name,
            R_cw=_quat_to_R(qw, qx, qy, qz),
            t_cw=np.array([tx, ty, tz]),
            point2D_xy=xy,
            point3D_ids=pids,
        ))
    return recs


def _read_images(sparse_dir: Path) -> list[_ImageRecord]:
    binp = sparse_dir / "images.bin"
    if binp.is_file():
        return _read_images_bin(binp)
    txt = sparse_dir / "images.txt"
    if txt.is_file():
        return _read_images_txt(txt)
    raise FileNotFoundError(f"no images.{{bin,txt}} under {sparse_dir}")


# ---- public API ------------------------------------------------------------

@dataclass
class MoGeScaleResult:
    scale: float                # meters per COLMAP unit
    per_frame_scales: list[float]
    num_correspondences: int
    num_frames_used: int


def estimate_metric_scale_via_moge(
    *,
    colmap_source_path: Path,
    num_frames: int = 5,
    device: str = "cuda",
    model_name: str = "Ruicheng/moge-2-vitl-normal",
    min_correspondences_per_frame: int = 20,
    rng_seed: int = 0,
) -> MoGeScaleResult:
    """Estimate metric-per-COLMAP scale by comparing MoGe depth to COLMAP depth.

    Returns a multiplier suitable as ``metric_scale_hint`` in
    ``align_scene_from_mesh``.

    Sampling: ``num_frames`` evenly spaced across the trajectory (not
    random — neighbouring frames look similar, even spacing maximizes
    geometric coverage).
    """
    sparse_dir = colmap_source_path / "sparse" / "0"
    images_dir = colmap_source_path / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"missing images/ under {colmap_source_path}")

    image_recs = _read_images(sparse_dir)
    points3D = _read_points3D(sparse_dir)

    # Drop records whose image file isn't on disk (DL3DV variant filtering may
    # leave images.bin entries pointing at unshipped frames).
    image_recs = [r for r in image_recs if (images_dir / r.name).is_file()]
    if len(image_recs) < num_frames:
        num_frames = max(1, len(image_recs))
    if not image_recs:
        raise RuntimeError(
            f"no on-disk images matched COLMAP records under {images_dir}"
        )

    # Even-spaced frame sampling
    indices = np.linspace(0, len(image_recs) - 1, num_frames).round().astype(int)
    sampled = [image_recs[int(i)] for i in indices]

    try:
        import torch
        from moge.model.v2 import MoGeModel
    except ImportError as e:  # pragma: no cover - install hint
        raise ImportError(
            "MoGe is required for metric-scale estimation. Install with "
            "`pip install git+https://github.com/microsoft/MoGe.git`"
        ) from e

    try:
        import cv2
    except ImportError as e:  # pragma: no cover
        raise ImportError("opencv-python required; `pip install opencv-python`") from e

    print(f"[moge] loading {model_name} on {device}…")
    model = MoGeModel.from_pretrained(model_name).to(device).eval()

    per_frame_scales: list[float] = []
    total_corr = 0
    for rec in sampled:
        img_path = images_dir / rec.name
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"[moge] WARN: could not read {img_path}, skipping")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]

        img_t = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0).to(device)
        with torch.no_grad():
            output = model.infer(img_t)
        depth_metric = output["depth"].detach().cpu().numpy()
        mask = output.get("mask")
        if mask is not None:
            mask = mask.detach().cpu().numpy().astype(bool)

        # Per-pixel COLMAP cam depth: project each 3D point into this cam.
        ratios: list[float] = []
        for k in range(len(rec.point3D_ids)):
            pid = int(rec.point3D_ids[k])
            if pid <= 0 or pid not in points3D:
                continue
            x, y = rec.point2D_xy[k]
            xi, yi = int(round(float(x))), int(round(float(y)))
            if not (0 <= xi < W and 0 <= yi < H):
                continue
            if mask is not None and not mask[yi, xi]:
                continue
            d_metric = float(depth_metric[yi, xi])
            if not np.isfinite(d_metric) or d_metric <= 0:
                continue
            p_cam = rec.R_cw @ points3D[pid] + rec.t_cw
            d_colmap = float(p_cam[2])  # +Z forward
            if d_colmap <= 0:
                continue
            ratios.append(d_metric / d_colmap)

        if len(ratios) >= min_correspondences_per_frame:
            s = float(np.median(ratios))
            per_frame_scales.append(s)
            total_corr += len(ratios)
            print(f"[moge] {rec.name}: {len(ratios):4d} pairs, scale ≈ {s:.4f} m/unit")
        else:
            print(f"[moge] {rec.name}: only {len(ratios)} usable pairs, skipping")

    if not per_frame_scales:
        raise RuntimeError(
            "no frame yielded enough usable correspondences — check "
            "MoGe mask quality / COLMAP point density"
        )

    final = float(np.median(per_frame_scales))
    spread = float(np.std(per_frame_scales))
    print(
        f"[moge] final scale = {final:.4f} m/unit "
        f"(per-frame std = {spread:.4f}, n_frames = {len(per_frame_scales)}, "
        f"n_corr = {total_corr})"
    )
    return MoGeScaleResult(
        scale=final,
        per_frame_scales=per_frame_scales,
        num_correspondences=total_corr,
        num_frames_used=len(per_frame_scales),
    )
