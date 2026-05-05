# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Single-scene downloader for DL3DV-Benchmark.

DL3DV-Benchmark stores 140 scenes on the HuggingFace Hub at
``DL3DV/DL3DV-Benchmark``, one folder per scene-hash. Each scene ships
with a COLMAP reconstruction under ``<hash>/colmaps/`` (images + sparse).

The full dataset is >1 TB; this helper pulls one scene at a time using
``huggingface_hub.snapshot_download`` with allow_patterns.

Auth: the dataset is gated. The user must run ``huggingface-cli login``
once and have accepted the dataset terms on the HF site. We surface a
clear error if that hasn't happened.
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

    DL3DV historically nests under ``colmaps/`` (with images + sparse as
    siblings) but some mirrors place them at scene root. Try the common
    layouts and return the first match.
    """
    candidates = [
        scene_root / "colmaps",
        scene_root,
    ]
    for c in candidates:
        if (c / "images").is_dir() and (c / "sparse").is_dir():
            return c
    raise FileNotFoundError(
        f"no COLMAP layout (images/+sparse/) found under {scene_root}; "
        f"tried: {[str(c) for c in candidates]}"
    )


def download_scene(
    scene_hash: str,
    cache_dir: Path,
    *,
    revision: str = "main",
    allow_patterns: tuple[str, ...] | None = None,
) -> DL3DVSceneRef:
    """Download a single DL3DV scene into ``cache_dir/<scene_hash>``.

    By default we fetch only the COLMAP subtree (images + sparse), which
    is what FastGS / fast-pgsr needs. Pass ``allow_patterns`` to override
    (e.g. to also pull the nerfstudio variant).
    """
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    except ImportError as e:  # pragma: no cover - install hint
        raise ImportError(
            "huggingface_hub is required for DL3DV download; "
            "install with `pip install huggingface_hub`"
        ) from e

    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    if allow_patterns is None:
        allow_patterns = (f"{scene_hash}/colmaps/*", f"{scene_hash}/colmaps/**")

    try:
        local_dir = snapshot_download(
            repo_id=DL3DV_REPO_ID,
            repo_type="dataset",
            revision=revision,
            allow_patterns=list(allow_patterns),
            local_dir=str(cache_dir),
        )
    except GatedRepoError as e:
        raise RuntimeError(
            f"DL3DV-Benchmark is a gated dataset. Run `huggingface-cli login` and "
            f"accept terms at https://huggingface.co/datasets/{DL3DV_REPO_ID}"
        ) from e
    except RepositoryNotFoundError as e:
        raise RuntimeError(f"could not find {DL3DV_REPO_ID} on the Hub") from e

    scene_root = Path(local_dir) / scene_hash
    if not scene_root.is_dir():
        raise FileNotFoundError(
            f"download succeeded but {scene_root} is missing — "
            f"check that scene hash {scene_hash!r} exists in the dataset"
        )

    colmap_source_path = _resolve_colmap_source(scene_root)
    ref = DL3DVSceneRef(
        scene_hash=scene_hash,
        root=scene_root,
        colmap_source_path=colmap_source_path,
    )
    ref.assert_colmap_layout()
    return ref
