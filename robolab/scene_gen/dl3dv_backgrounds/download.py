# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Single-scene downloader for DL3DV-Benchmark.

Mirrors the workflow of DL3DV's official ``download.py`` (which is the
only thing that works against this dataset's layout):

    1. Fetch ``benchmark-meta.csv`` to validate the scene hash.
    2. Fetch ``.cache/filelist.bin`` — a pickled ``{hash: [file_path, ...]}``
       dict — to enumerate the files belonging to this scene.
    3. Download each file via ``hf_hub_download`` with retries.

Why not ``snapshot_download(allow_patterns=...)``?  DL3DV scenes don't
sit at predictable glob paths; the canonical truth is the filelist.

Auth: gated. Run ``huggingface-cli login`` and accept terms at
https://huggingface.co/datasets/DL3DV/DL3DV-10K-Benchmark.

Resolution: scenes ship at full-res ``images/`` plus downsampled
``images_4/`` (960x540) and ``images_8/``. ``low_res=True`` pulls only
``images_4/`` (a few hundred MB vs several GB) — a strict win for
prototyping. Note: COLMAP intrinsics are calibrated against full-res,
so when using ``images_4/`` we rescale ``cameras.txt`` by 1/4.
"""

from __future__ import annotations

import os
import pickle
import shutil
import traceback
from dataclasses import dataclass
from pathlib import Path

DL3DV_REPO_ID = "DL3DV/DL3DV-10K-Benchmark"


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
    """Find the COLMAP source path inside a downloaded DL3DV scene.

    DL3DV-10K layout: ``<hash>/colmap/sparse/0/`` with images at
    ``<hash>/images/`` (sibling of ``colmap/``). FastGS expects
    images and sparse to be siblings inside the same source_path, so
    we treat the scene root itself as the source path and the sparse/
    indirection lands one level deeper.

    Some mirrors use ``colmaps/`` (with an s) — handled too.
    """
    candidates = [
        # FastGS-friendly: scene_root has images/ at top level and sparse/ via colmap symlink
        scene_root,
        scene_root / "colmap",
        scene_root / "colmaps",
    ]
    for c in candidates:
        if (c / "images").is_dir() and (c / "sparse").is_dir():
            return c
    raise FileNotFoundError(
        f"no COLMAP layout (images/+sparse/) found under {scene_root}; "
        f"tried: {[str(c) for c in candidates]}. "
        f"Run with --no-skip-download or inspect the directory."
    )


# ---- HF API helpers --------------------------------------------------------

def _hf_download_one(repo_path: str, local_dir: Path, *, max_try: int = 5) -> bool:
    """Download a single file from the dataset repo with retries.

    ``repo_path`` is relative to the repo root (e.g. ``<hash>/colmap/...``).
    """
    from huggingface_hub import HfApi

    api = HfApi()
    rel_path = os.path.relpath(repo_path, DL3DV_REPO_ID)

    for attempt in range(1, max_try + 1):
        try:
            api.hf_hub_download(
                repo_id=DL3DV_REPO_ID,
                filename=rel_path,
                repo_type="dataset",
                local_dir=str(local_dir),
                cache_dir=str(local_dir / ".cache"),
            )
            return True
        except Exception:
            traceback.print_exc()
            print(f"[download] retry {attempt}/{max_try} for {rel_path}")
    print(f"[download] ERROR giving up on {rel_path}")
    return False


def _filter_files_for_scene(
    all_files: list[str], *, low_res: bool
) -> list[str]:
    """Pick the files we actually want from a scene's filelist.

    Structure-agnostic filter (works whether paths are
    ``<hash>/images_4/...`` or ``images_4/...``):

      - skip files inside an ``input`` directory (raw camera frames)
      - if ``low_res``: skip files inside ``images/`` or ``images_8/``
        directories — keep ``images_4/`` and everything else (colmap
        data, transforms.json, etc).

    Note: we check directory *parts*, not substrings, so the COLMAP file
    ``colmap/sparse/0/images.bin`` is correctly kept (the substring
    check used by DL3DV's official downloader is buggy here).
    """
    chosen: list[str] = []
    for f in all_files:
        parts = Path(f).parts
        if "input" in parts:
            continue
        if low_res and ("images" in parts or "images_8" in parts):
            continue
        chosen.append(f)
    return chosen


# ---- low-res intrinsic rescale --------------------------------------------

def _rescale_cameras_txt_for_low_res(cameras_txt: Path, factor: float = 0.25) -> None:
    """Scale focal lengths and principal points in cameras.txt by ``factor``.

    Only touches lines that look like camera entries:

        CAM_ID MODEL WIDTH HEIGHT fx fy cx cy [k1 k2 p1 p2 ...]

    Idempotency: writes a sibling ``cameras.txt.fullres.bak`` once and
    refuses to scale again if the backup already exists.
    """
    backup = cameras_txt.with_suffix(".txt.fullres.bak")
    if backup.exists():
        print(f"[download] cameras.txt already rescaled (backup at {backup}), skipping")
        return
    if not cameras_txt.is_file():
        return
    shutil.copy(cameras_txt, backup)

    out_lines: list[str] = []
    for raw in cameras_txt.read_text().splitlines():
        if raw.startswith("#") or not raw.strip():
            out_lines.append(raw)
            continue
        parts = raw.split()
        # parts: [id, model, w, h, fx, fy, cx, cy, ...distortion]
        try:
            parts[2] = str(int(round(int(parts[2]) * factor)))
            parts[3] = str(int(round(int(parts[3]) * factor)))
            for i in range(4, len(parts)):
                # Distortion params are dimensionless; only fx/fy/cx/cy scale.
                # Heuristic: scale the first 4 numeric params after w,h.
                if i < 8:
                    parts[i] = f"{float(parts[i]) * factor:.6f}"
        except (ValueError, IndexError):
            pass
        out_lines.append(" ".join(parts))
    cameras_txt.write_text("\n".join(out_lines) + "\n")
    print(f"[download] rescaled {cameras_txt} for low-res images (factor={factor})")


def _ensure_images_dir_alias(scene_root: Path, low_res: bool) -> None:
    """If we downloaded only ``images_4/``, make ``images/`` point at it.

    FastGS's COLMAP loader joins ``source_path/images`` with the basename
    in ``images.txt``. The names are the same across resolutions; only
    the parent dir differs. A symlink keeps the loader happy without
    rewriting images.txt.
    """
    if not low_res:
        return
    images_4 = scene_root / "images_4"
    images = scene_root / "images"
    if not images_4.is_dir() or images.exists():
        return
    images.symlink_to(images_4.name)
    print(f"[download] symlinked {images} -> {images_4.name}")


def _ensure_sparse_alias(scene_root: Path) -> None:
    """Symlink ``<scene_root>/sparse`` to ``colmap/sparse`` (or ``colmaps/sparse``).

    DL3DV places sparse data inside ``colmap/`` while images live at the
    scene root. FastGS expects ``<source_path>/{images, sparse}`` as
    siblings. Symlinking lets us use the scene root as the source path
    without copying gigabytes of images.
    """
    target_sparse = scene_root / "sparse"
    if target_sparse.exists():
        return
    for candidate in ("colmap", "colmaps"):
        sparse_inner = scene_root / candidate / "sparse"
        if sparse_inner.is_dir():
            target_sparse.symlink_to(Path(candidate) / "sparse")
            print(f"[download] symlinked {target_sparse} -> {candidate}/sparse")
            return


# ---- public API ------------------------------------------------------------

def download_scene(
    scene_hash: str,
    cache_dir: Path,
    *,
    low_res: bool = False,
    clean_hf_cache: bool = True,
) -> DL3DVSceneRef:
    """Download a single DL3DV scene into ``cache_dir/<scene_hash>``.

    Uses the official meta-csv + filelist.bin workflow.

    Parameters
    ----------
    scene_hash
        64-char hex hash from DL3DV-Benchmark.
    cache_dir
        Local download root. The scene lands at ``<cache_dir>/<scene_hash>/``;
        meta files at ``<cache_dir>/{benchmark-meta.csv, .cache/filelist.bin}``.
    low_res
        If True, only fetch ``images_4/`` (960x540, ~few hundred MB) and
        rescale ``cameras.txt`` accordingly. Recommended for prototyping.
    clean_hf_cache
        If True, remove ``<cache_dir>/.cache/datasets--*`` after download
        to reclaim space (HF's per-blob cache duplicates files).
    """
    try:
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "huggingface_hub is required; `pip install huggingface_hub pandas tqdm`"
        ) from e
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise ImportError("pandas is required; `pip install pandas`") from e

    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. meta-csv to validate hash
    meta_relpath = "benchmark-meta.csv"
    if not (cache_dir / meta_relpath).is_file():
        if not _hf_download_one(f"{DL3DV_REPO_ID}/{meta_relpath}", cache_dir):
            raise RuntimeError(
                f"failed to download {meta_relpath}. "
                f"If gated: run `huggingface-cli login` and accept terms at "
                f"https://huggingface.co/datasets/{DL3DV_REPO_ID}"
            )
    df = pd.read_csv(cache_dir / meta_relpath)
    valid_hashes = set(df["hash"].tolist())
    if scene_hash not in valid_hashes:
        raise ValueError(
            f"scene hash {scene_hash!r} not in DL3DV-10K-Benchmark "
            f"({len(valid_hashes)} scenes available)"
        )

    # 2. filelist.bin to enumerate scene files
    filelist_relpath = ".cache/filelist.bin"
    if not (cache_dir / filelist_relpath).is_file():
        if not _hf_download_one(f"{DL3DV_REPO_ID}/{filelist_relpath}", cache_dir):
            raise RuntimeError(f"failed to download {filelist_relpath}")
    with (cache_dir / filelist_relpath).open("rb") as f:
        filelist = pickle.load(f)

    if scene_hash not in filelist:
        raise RuntimeError(
            f"scene hash {scene_hash} present in meta-csv but not in filelist.bin"
        )
    all_files = filelist[scene_hash]
    print(f"[download] filelist sample (first 5 of {len(all_files)}):")
    for f in all_files[:5]:
        print(f"           {f}")
    chosen = _filter_files_for_scene(all_files, low_res=low_res)
    if not chosen:
        sample = "\n  ".join(all_files[:10])
        raise RuntimeError(
            f"no files selected for scene {scene_hash} (low_res={low_res}); "
            f"filelist had {len(all_files)} entries. Sample paths:\n  {sample}"
        )

    print(
        f"[download] scene {scene_hash[:8]}…: {len(chosen)} files "
        f"({'low-res' if low_res else 'full-res'})"
    )

    # 3. per-file download with retry
    for relpath in chosen:
        if not _hf_download_one(f"{DL3DV_REPO_ID}/{relpath}", cache_dir):
            raise RuntimeError(f"download of {relpath} failed after retries")

    if clean_hf_cache:
        hf_blob_cache = cache_dir / ".cache" / "datasets--DL3DV--DL3DV-10K-Benchmark"
        if hf_blob_cache.is_dir():
            shutil.rmtree(hf_blob_cache)

    # 4. fix up layout for FastGS
    scene_root = cache_dir / scene_hash
    if not scene_root.is_dir():
        raise FileNotFoundError(
            f"download finished but {scene_root} is missing — check filelist filtering"
        )

    _ensure_images_dir_alias(scene_root, low_res=low_res)
    _ensure_sparse_alias(scene_root)
    if low_res:
        # cameras.txt is at <hash>/colmap/sparse/0/cameras.txt typically
        for cameras_txt in scene_root.rglob("cameras.txt"):
            _rescale_cameras_txt_for_low_res(cameras_txt, factor=0.25)

    colmap_source_path = _resolve_colmap_source(scene_root)
    ref = DL3DVSceneRef(
        scene_hash=scene_hash,
        root=scene_root,
        colmap_source_path=colmap_source_path,
    )
    ref.assert_colmap_layout()
    return ref
