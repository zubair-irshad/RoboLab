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

    # TSDF-fusion knobs (fed to render.py). PGSR defaults (voxel=2mm,
    # max_depth=5m) target object-scale; for room-scale collision use
    # 2cm voxels + 6m cutoff is plenty and ~10x faster than 1cm/10m.
    tsdf_voxel_size: float = 0.02
    tsdf_max_depth: float = 6.0
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
        # render.py has an upstream bug where it crashes after writing
        # tsdf_fusion_post.ply but before saving an optional colored
        # variant (NameError on `mesh_path`). The fusion mesh we want
        # is already on disk by then, so we tolerate non-zero exit and
        # let the post-condition check below decide success.
        try:
            _conda_run(cfg.conda_env, repo, render_cmd)
        except subprocess.CalledProcessError as e:
            print(
                f"[fast-pgsr] render.py exited non-zero ({e.returncode}); "
                f"checking for the fusion mesh on disk anyway..."
            )

    # Discover the produced mesh. fast-pgsr writes any of (depending on
    # code path / version):
    #   <model_path>/mesh/{tsdf_fusion,tsdf_fusion_post}.ply
    #   <model_path>/train/ours_<iter>/mesh/{mesh_color,mesh_nocolor}.ply
    #
    # Prefer cluster-filtered variants — _post drops disconnected TSDF
    # "shadow" floaters that otherwise contaminate floor detection.
    preferred_names = (
        "tsdf_fusion_post.ply",  # cluster-filtered TSDF (cleanest)
        "mesh_color.ply",         # alt naming, also cluster-filtered if --num_cluster 1
        "tsdf_fusion.ply",        # raw TSDF (has floaters)
        "mesh_nocolor.ply",       # geometry only
    )
    mesh_dir_candidates = list(model_path.rglob("mesh"))
    for mesh_dir in sorted(mesh_dir_candidates, key=lambda p: p.stat().st_mtime, reverse=True):
        if not mesh_dir.is_dir():
            continue
        for name in preferred_names:
            candidate = mesh_dir / name
            if candidate.is_file() and candidate.stat().st_size > 0:
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
        f"no mesh .ply produced under {model_path} — check render.py logs. "
        f"If you see a NameError on `mesh_path`, run "
        f"`python scripts/patch_fastpgsr_render.py --repo {cfg.repo_path}` first."
    )
