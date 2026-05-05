# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Single-scene downloader for DL3DV/DL3DV-Benchmark.

Each scene under ``DL3DV/DL3DV-Benchmark`` has two sibling subtrees::

    <hash>/
      gaussian_splat/      <- COLMAP-ready: images/ + sparse/0/{*.bin}
      nerfstudio/          <- transforms.json + images_2/, images_4/

We pull the ``gaussian_splat/`` subtree because fast-pgsr expects
exactly that layout (``<source_path>/{images, sparse}``). No symlinks,
no intrinsic rescale needed — the cameras are calibrated against the
images shipped in the same folder.

Auth: the dataset is gated. Run ``huggingface-cli login`` and accept
terms at https://huggingface.co/datasets/DL3DV/DL3DV-Benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DL3DV_REPO_ID = "DL3DV/DL3DV-Benchmark"


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
    """Find the COLMAP source path inside a downloaded scene.

    Preferred: ``<scene_root>/gaussian_splat/`` (DL3DV-Benchmark).
    Fallback: any directory with ``images/`` + ``sparse/`` siblings.
    """
    candidates = [
        scene_root / "gaussian_splat",
        scene_root,
        scene_root / "colmap",
        scene_root / "colmaps",
    ]
    for c in candidates:
        if (c / "images").is_dir() and (c / "sparse").is_dir():
            return c
    raise FileNotFoundError(
        f"no COLMAP layout (images/+sparse/) under {scene_root}; "
        f"tried: {[str(c) for c in candidates]}"
    )


def download_scene(
    scene_hash: str,
    cache_dir: Path,
    *,
    subset: str = "gaussian_splat",
) -> DL3DVSceneRef:
    """Download one scene's ``gaussian_splat/`` (or ``nerfstudio/``) subtree.

    Parameters
    ----------
    scene_hash
        64-char hex hash from DL3DV-Benchmark.
    cache_dir
        Local download root. The scene lands at ``<cache_dir>/<scene_hash>/``.
    subset
        Either ``"gaussian_splat"`` (COLMAP-ready, default) or
        ``"nerfstudio"`` (transforms.json + downsampled images). Use
        gaussian_splat for fast-pgsr; nerfstudio is exposed for users
        who want to drive nerfstudio/splatfacto from the same data.
    """
    if subset not in {"gaussian_splat", "nerfstudio"}:
        raise ValueError(f"subset must be gaussian_splat or nerfstudio; got {subset!r}")

    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "huggingface_hub is required; `pip install huggingface_hub`"
        ) from e

    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    patterns = [f"{scene_hash}/{subset}/**", f"{scene_hash}/{subset}/*"]
    print(f"[download] {DL3DV_REPO_ID} :: {scene_hash[:8]}…/{subset}/")

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
            f"download finished but {scene_root} is missing — check that "
            f"hash {scene_hash!r} exists in {DL3DV_REPO_ID}"
        )

    # Show what landed so layout surprises are immediately visible.
    pulled = sorted(p.relative_to(scene_root) for p in scene_root.rglob("*") if p.is_file())[:5]
    print(f"[download] sample files: {[str(p) for p in pulled]}")

    colmap_source_path = _resolve_colmap_source(scene_root)
    ref = DL3DVSceneRef(
        scene_hash=scene_hash,
        root=scene_root,
        colmap_source_path=colmap_source_path,
    )
    ref.assert_colmap_layout()
    return ref
