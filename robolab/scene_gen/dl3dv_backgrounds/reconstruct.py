# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Drive FastGS `fast-pgsr` branch on a DL3DV COLMAP scene.

Why a separate driver from ``diffusion_harmonizer/fastgs_export.py``?
----------------------------------------------------------------------
That export converts hemispheric Isaac captures (OpenGL poses) into a
COLMAP-style FastGS dataset. DL3DV scenes are *already* COLMAP — no
pose conversion needed. We only need to invoke fast-pgsr's train.py +
mesh-extraction step against the existing source path.

Conda env: fast-pgsr ships its own ``environment.yaml`` that creates a
conda env named ``fast-pgsr`` (separate from the vanilla ``fastgs`` env
used elsewhere in this repo, since hyperparams + deps differ). We
activate it via ``conda run`` so we don't need an interactive shell.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FastPgsrConfig:
    """Knobs for one fast-pgsr training + mesh-extraction run."""

    repo_path: Path
    """Local clone of FastGS checked out to the ``fast-pgsr`` branch."""

    conda_env: str = "fast-pgsr"
    """Conda env created from the branch's environment.yaml."""

    iterations: int = 30000

    extra_train_args: tuple[str, ...] = ()
    """Pass-through for branch-specific flags (e.g. --loss_thresh, --dense)."""

    mesh_script: str = "extract_mesh.py"
    """Name of the mesh-extraction entry point inside the FastGS repo.
    fast-pgsr's TSDF-fusion mesh extractor; if upstream renames it, override here."""


def _conda_run(env: str, cwd: Path, cmd: list[str]) -> None:
    """Invoke a command inside a named conda env without an interactive shell.

    Uses ``conda run --no-capture-output`` so stdout/stderr stream live.
    """
    if shutil.which("conda") is None:
        raise RuntimeError("conda not on PATH; cannot activate fast-pgsr env")
    full = ["conda", "run", "--no-capture-output", "-n", env, *cmd]
    print(f"[fast-pgsr] $ (cwd={cwd}) {' '.join(full)}", flush=True)
    subprocess.run(full, cwd=str(cwd), check=True)


def run_fast_pgsr(
    cfg: FastPgsrConfig,
    *,
    source_path: Path,
    model_path: Path,
    train: bool = True,
    extract_mesh: bool = True,
) -> Path:
    """Train fast-pgsr on a COLMAP source dir and extract a mesh.

    Returns the path to the produced mesh (``<model_path>/mesh.ply`` by
    convention; we discover the actual path by globbing post-extraction
    so we tolerate upstream naming changes).
    """
    repo = cfg.repo_path.resolve()
    if not (repo / "train.py").is_file():
        raise FileNotFoundError(
            f"{repo} does not look like the FastGS repo (no train.py). "
            f"Clone https://github.com/fastgs/FastGS and `git checkout fast-pgsr`."
        )

    source_path = source_path.resolve()
    model_path = model_path.resolve()
    model_path.mkdir(parents=True, exist_ok=True)

    if train:
        train_cmd = [
            "python", "train.py",
            "--source_path", str(source_path),
            "--model_path", str(model_path),
            "--iterations", str(cfg.iterations),
            *cfg.extra_train_args,
        ]
        _conda_run(cfg.conda_env, repo, train_cmd)

    if extract_mesh:
        mesh_entry = repo / cfg.mesh_script
        if not mesh_entry.is_file():
            raise FileNotFoundError(
                f"mesh extractor not found at {mesh_entry} — fast-pgsr must "
                f"expose one (default name: extract_mesh.py). Override via "
                f"FastPgsrConfig.mesh_script."
            )
        mesh_cmd = [
            "python", cfg.mesh_script,
            "--source_path", str(source_path),
            "--model_path", str(model_path),
            "--iteration", str(cfg.iterations),
        ]
        _conda_run(cfg.conda_env, repo, mesh_cmd)

    # Discover the produced mesh. PGSR-style fusion typically writes
    # something like <model_path>/train/ours_<iter>/fuse_post.ply or
    # <model_path>/mesh.ply. We pick the most-recently-written .ply
    # under model_path that isn't the input point cloud.
    candidates = sorted(
        (
            p for p in model_path.rglob("*.ply")
            if "input.ply" not in p.name and "point_cloud" not in p.parts
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"no mesh .ply produced under {model_path}; check fast-pgsr logs"
        )
    mesh_path = candidates[0]
    print(f"[fast-pgsr] mesh -> {mesh_path}")
    return mesh_path
