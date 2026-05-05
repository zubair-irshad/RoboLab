# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Single-scene downloader for DL3DV/DL3DV-Benchmark.

Each scene under ``DL3DV/DL3DV-Benchmark`` ships a 3DGS-style folder::

    <hash>/gaussian_splat/
      sparse/              <- post-undistortion COLMAP (pinhole)
      images/              <- full-res undistorted (~several GB)
      images_2/, images_4/, images_8/  <- downsampled
      distorted/           <- pre-undistortion COLMAP + database (huge)
      stereo/              <- COLMAP MVS depth maps (huge)
      input/               <- raw camera frames (huge)
      run-colmap-*.sh, transforms.json

We pull *only* ``sparse/`` and one chosen image variant — typically
``images_4`` (270p, sufficient for fast-pgsr prototyping) — and skip
everything else. Camera intrinsics in ``sparse/`` are calibrated for
full-res ``images/``, so when using a downsampled variant we rescale
fx/fy/cx/cy and W/H to match by reading actual image dimensions.

Auth: gated. Run ``huggingface-cli login`` and accept terms at
https://huggingface.co/datasets/DL3DV/DL3DV-Benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DL3DV_REPO_ID = "DL3DV/DL3DV-Benchmark"

VALID_IMAGE_VARIANTS = ("images", "images_2", "images_4", "images_8")


@dataclass(frozen=True)
class DL3DVSceneRef:
    """Locator for a downloaded DL3DV scene."""

    scene_hash: str
    root: Path
    colmap_source_path: Path

    def assert_colmap_layout(self) -> None:
        images = self.colmap_source_path / "images"
        sparse = self.colmap_source_path / "sparse" / "0"
        if not images.is_dir():
            raise FileNotFoundError(f"missing images/ under {self.colmap_source_path}")
        if not sparse.is_dir():
            raise FileNotFoundError(f"missing sparse/0/ under {self.colmap_source_path}")


def _resolve_colmap_source(scene_root: Path) -> Path:
    """Return the path FastGS should use as ``--source_path``.

    For DL3DV: ``<scene_root>/gaussian_splat/`` after we've added the
    ``images → <variant>`` symlink. Falls back to other layouts.
    """
    gs = scene_root / "gaussian_splat"
    if (gs / "images").exists() and (gs / "sparse").is_dir():
        return gs
    # distorted-only fallback (older code path; kept for robustness)
    if (gs / "input").is_dir() and (gs / "distorted" / "sparse").is_dir():
        return _build_distorted_source(gs)
    for c in (scene_root, scene_root / "colmap", scene_root / "colmaps"):
        if (c / "images").exists() and (c / "sparse").is_dir():
            return c
    raise FileNotFoundError(
        f"no usable COLMAP layout under {scene_root}; expected "
        f"{gs}/{{images, sparse}}"
    )


def _build_distorted_source(gs_root: Path) -> Path:
    """Pair distorted COLMAP with input/ frames into a derived source dir.

    Used only when ``sparse/`` (post-undistorter) wasn't downloaded but
    ``distorted/sparse/`` and ``input/`` exist.
    """
    src = gs_root / "_fastgs_src"
    src.mkdir(exist_ok=True)
    for name, target_rel in (
        ("images", Path("..") / "input"),
        ("sparse", Path("..") / "distorted" / "sparse"),
    ):
        link = src / name
        if not (link.is_symlink() or link.exists()):
            link.symlink_to(target_rel)
    print(f"[download] assembled distorted-COLMAP source at {src}")
    return src


# ---- COLMAP camera autorescale --------------------------------------------

# COLMAP camera-model param counts (colmap/src/base/camera_models.h).
_COLMAP_MODEL_NUM_PARAMS: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),     # f, cx, cy
    1: ("PINHOLE", 4),             # fx, fy, cx, cy
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}
# Number of leading params that are in pixel units (need rescaling).
_PIXEL_PARAM_COUNT: dict[str, int] = {
    "SIMPLE_PINHOLE": 3, "SIMPLE_RADIAL": 3, "RADIAL": 3,
    "SIMPLE_RADIAL_FISHEYE": 3, "RADIAL_FISHEYE": 3,
    "PINHOLE": 4, "OPENCV": 4, "OPENCV_FISHEYE": 4,
    "FULL_OPENCV": 4, "FOV": 4, "THIN_PRISM_FISHEYE": 4,
}


def _read_cameras_bin(path: Path) -> list[dict]:
    import struct
    cams: list[dict] = []
    with path.open("rb") as f:
        (num_cameras,) = struct.unpack("<Q", f.read(8))
        for _ in range(num_cameras):
            cam_id, model_id = struct.unpack("<iI", f.read(8))
            width, height = struct.unpack("<QQ", f.read(16))
            if model_id not in _COLMAP_MODEL_NUM_PARAMS:
                raise ValueError(f"unknown COLMAP camera model id {model_id}")
            model_name, n = _COLMAP_MODEL_NUM_PARAMS[model_id]
            params = list(struct.unpack(f"<{n}d", f.read(8 * n)))
            cams.append(dict(
                id=cam_id, model=model_name,
                width=int(width), height=int(height), params=params,
            ))
    return cams


def _read_cameras_txt(path: Path) -> list[dict]:
    cams: list[dict] = []
    for raw in path.read_text().splitlines():
        if not raw.strip() or raw.startswith("#"):
            continue
        parts = raw.split()
        cams.append(dict(
            id=int(parts[0]), model=parts[1],
            width=int(parts[2]), height=int(parts[3]),
            params=[float(p) for p in parts[4:]],
        ))
    return cams


def _write_cameras_txt(path: Path, cams: list[dict]) -> None:
    lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"# Number of cameras: {len(cams)}",
    ]
    for c in cams:
        params_s = " ".join(f"{p:.10g}" for p in c["params"])
        lines.append(f"{c['id']} {c['model']} {c['width']} {c['height']} {params_s}")
    path.write_text("\n".join(lines) + "\n")


def _autorescale_cameras_to_actual_image_dim(source_path: Path) -> None:
    """Detect actual image dim and rewrite cameras with matching dim."""
    images_dir = source_path / "images"
    sparse_dir = source_path / "sparse" / "0"
    if not images_dir.exists() or not sparse_dir.is_dir():
        return

    img_paths = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not img_paths:
        return
    try:
        from PIL import Image
    except ImportError as e:  # pragma: no cover
        raise ImportError("Pillow required; `pip install Pillow`") from e
    with Image.open(img_paths[0]) as im:
        actual_w, actual_h = im.size

    cameras_txt = sparse_dir / "cameras.txt"
    cameras_bin = sparse_dir / "cameras.bin"
    if cameras_txt.is_file():
        cams = _read_cameras_txt(cameras_txt)
    elif cameras_bin.is_file():
        cams = _read_cameras_bin(cameras_bin)
    else:
        return

    if all(c["width"] == actual_w and c["height"] == actual_h for c in cams):
        return  # already matches

    declared = (cams[0]["width"], cams[0]["height"])
    new_cams: list[dict] = []
    for c in cams:
        sx = actual_w / c["width"]
        sy = actual_h / c["height"]
        n_pix = _PIXEL_PARAM_COUNT.get(c["model"], 4)
        params = list(c["params"])
        if n_pix == 3:
            params[0] *= sx          # f
            params[1] *= sx          # cx
            params[2] *= sy          # cy
        else:
            params[0] *= sx          # fx
            params[1] *= sy          # fy
            params[2] *= sx          # cx
            params[3] *= sy          # cy
        new_cams.append(dict(c, width=actual_w, height=actual_h, params=params))

    _write_cameras_txt(cameras_txt, new_cams)
    print(
        f"[download] rescaled cameras: {declared[0]}x{declared[1]} → "
        f"{actual_w}x{actual_h} (model={cams[0]['model']})"
    )


# ---- public API ------------------------------------------------------------

def download_scene(
    scene_hash: str,
    cache_dir: Path,
    *,
    images_variant: str = "images_4",
    subset: str = "gaussian_splat",
) -> DL3DVSceneRef:
    """Download just ``sparse/`` and one image variant for a single scene.

    Parameters
    ----------
    scene_hash
        64-char hex hash from DL3DV-Benchmark.
    cache_dir
        Local download root. Scene lands at ``<cache_dir>/<scene_hash>/``.
    images_variant
        One of ``images``, ``images_2``, ``images_4`` (default), ``images_8``.
        Cameras in ``sparse/`` are auto-rescaled to match.
    subset
        Subtree name under the scene; default ``gaussian_splat`` (the only
        one with the post-undistortion sparse).
    """
    if images_variant not in VALID_IMAGE_VARIANTS:
        raise ValueError(
            f"images_variant must be one of {VALID_IMAGE_VARIANTS}, got {images_variant!r}"
        )

    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    except ImportError as e:  # pragma: no cover
        raise ImportError("huggingface_hub required; `pip install huggingface_hub`") from e

    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Tight allow_patterns: only sparse/ and the chosen variant. transforms
    # and run-colmap shells are tiny and useful for debugging — keep them.
    base = f"{scene_hash}/{subset}"
    patterns = [
        f"{base}/sparse/**",
        f"{base}/sparse/*",
        f"{base}/{images_variant}/**",
        f"{base}/{images_variant}/*",
        f"{base}/transforms.json",
        f"{base}/run-colmap-*.sh",
    ]
    print(f"[download] {DL3DV_REPO_ID} :: {scene_hash[:8]}…/{subset}/")
    print(f"[download] variants: sparse/ + {images_variant}/")

    try:
        local_dir = snapshot_download(
            repo_id=DL3DV_REPO_ID,
            repo_type="dataset",
            allow_patterns=patterns,
            local_dir=str(cache_dir),
        )
    except GatedRepoError as e:
        raise RuntimeError(
            f"DL3DV-Benchmark is gated. Run `huggingface-cli login` and accept "
            f"terms at https://huggingface.co/datasets/{DL3DV_REPO_ID}"
        ) from e
    except RepositoryNotFoundError as e:
        raise RuntimeError(f"could not find {DL3DV_REPO_ID} on the Hub") from e

    scene_root = Path(local_dir) / scene_hash
    if not scene_root.is_dir():
        raise FileNotFoundError(
            f"download finished but {scene_root} is missing — check that hash "
            f"{scene_hash!r} exists in {DL3DV_REPO_ID}"
        )

    gs = scene_root / subset
    pulled = sorted(gs.glob("*"))
    print(f"[download] under {gs}: {[p.name for p in pulled]}")

    # FastGS expects <source_path>/images/. Symlink to the chosen variant.
    if images_variant != "images":
        target = gs / "images"
        # If there's a stale dir from a prior partial download (not a symlink),
        # leave it alone — the user may want it. Use a sibling .images_active
        # name only when there's no images/ at all.
        if not target.exists():
            target.symlink_to(images_variant)
            print(f"[download] symlinked {target} -> {images_variant}")
        elif target.is_symlink() and target.readlink() != Path(images_variant):
            target.unlink()
            target.symlink_to(images_variant)
            print(f"[download] re-pointed {target} -> {images_variant}")

    _autorescale_cameras_to_actual_image_dim(gs)

    colmap_source_path = _resolve_colmap_source(scene_root)
    ref = DL3DVSceneRef(
        scene_hash=scene_hash,
        root=scene_root,
        colmap_source_path=colmap_source_path,
    )
    ref.assert_colmap_layout()
    return ref
