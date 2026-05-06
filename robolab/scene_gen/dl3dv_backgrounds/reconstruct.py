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

    # TSDF-fusion knobs (fed to render.py). PGSR defaults are voxel=2mm /
    # max_depth=5m, which produce huge meshes for room-scale scenes — we
    # default to 1cm voxels and 10m cutoff to keep meshes manageable while
    # still good enough for robot-scene collision.
    tsdf_voxel_size: float = 0.01
    tsdf_max_depth: float = 10.0
    tsdf_num_cluster: int = 1
    use_depth_filter: bool = True


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
        # fast-pgsr does TSDF fusion inside render.py: it renders depth
        # for each training view, then fuses → <model_path>/mesh/*.ply.
        render_cmd = [
            "python", "render.py",
            "--source_path", str(source_path),
            "--model_path", str(model_path),
            "--iteration", str(cfg.iterations),
            "--skip_test",
            "--max_depth", str(cfg.tsdf_max_depth),
            "--voxel_size", str(cfg.tsdf_voxel_size),
            "--num_cluster", str(cfg.tsdf_num_cluster),
        ]
        if cfg.use_depth_filter:
            render_cmd.append("--use_depth_filter")
        _conda_run(cfg.conda_env, repo, render_cmd)

    # Prefer the post-processed (cluster-filtered) mesh; fall back to raw.
    mesh_dir = model_path / "mesh"
    for name in ("tsdf_fusion_post.ply", "tsdf_fusion.ply"):
        candidate = mesh_dir / name
        if candidate.is_file():
            print(f"[fast-pgsr] mesh -> {candidate}")
            return candidate

    # Last-resort glob in case upstream renames things again.
    fallback = sorted(
        (p for p in model_path.rglob("*.ply") if "point_cloud" not in p.parts),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if fallback:
        print(f"[fast-pgsr] mesh (fallback discovery) -> {fallback[0]}")
        return fallback[0]
    raise FileNotFoundError(
        f"no mesh .ply produced under {mesh_dir} — check render.py logs"
    )
