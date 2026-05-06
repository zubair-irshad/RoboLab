# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Convert fast-pgsr's Gaussian PLY to a NuRec USDZ via 3DGUT.

3DGUT (https://github.com/nv-tlabs/3dgrut) ships a converter that
authors a Gaussian-splat USDZ Isaac Sim's NuRec renderer can load
directly. Per the upstream README::

    python -m threedgrut.export.scripts.ply_to_usd path/to/model.ply \
        --output_file path/to/output.usdz

3DGUT has its own conda env (separate from ``fastgs`` and ``fast-pgsr``).
This wrapper invokes the converter via ``conda run`` so we don't need
an interactive shell, and it auto-discovers the latest
``point_cloud.ply`` under the prepared scene's fast-pgsr output.

The output USDZ is in **COLMAP frame** — same as the input PLY. The
render script applies the alignment + placement transform at runtime,
so we don't pre-bake anything here.

Usage::

    python scripts/dl3dv_gs_to_usdz.py \
        --scene-dir data/dl3dv_backgrounds/scenes/<hash>

    # Or override paths:
    python scripts/dl3dv_gs_to_usdz.py \
        --scene-dir data/dl3dv_backgrounds/scenes/<hash> \
        --threedgrut-repo third_party/3dgrut \
        --conda-env 3dgrut

Output:
    <scene-dir>/gaussians.usdz
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _discover_latest_gs_ply(scene_dir: Path) -> Path:
    """Locate the most-recent point_cloud.ply under fastpgsr/."""
    pc_root = scene_dir / "fastpgsr" / "point_cloud"
    if not pc_root.is_dir():
        raise FileNotFoundError(
            f"no fastpgsr/point_cloud/ under {scene_dir}; was fast-pgsr trained?"
        )
    candidates = sorted(pc_root.rglob("point_cloud.ply"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"no point_cloud.ply under {pc_root}")
    return candidates[0]


def _conda_run(env: str, cwd: Path, cmd: list[str]) -> None:
    if shutil.which("conda") is None:
        raise RuntimeError("conda not on PATH; cannot activate 3DGUT env")
    full = ["conda", "run", "--no-capture-output", "-n", env, *cmd]
    print(f"[gs→usdz] $ (cwd={cwd}) {' '.join(full)}", flush=True)
    subprocess.run(full, cwd=str(cwd), check=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene-dir", type=Path, required=True,
                   help="prepared DL3DV scene dir (output of prepare_dl3dv_scene.py)")
    p.add_argument("--threedgrut-repo", type=Path, default=Path("third_party/3dgrut"),
                   help="local clone of nv-tlabs/3dgrut. We `cd` here before "
                        "invoking the python module so it picks up the right "
                        "PYTHONPATH.")
    p.add_argument("--conda-env", default="3dgrut",
                   help="conda env name where 3DGUT + its deps are installed")
    p.add_argument("--out-name", default="gaussians.usdz",
                   help="filename written next to mesh_aligned.* under scene-dir")
    p.add_argument("--ply", type=Path, default=None,
                   help="explicit PLY path to convert; default auto-discovers "
                        "the latest fast-pgsr point_cloud.ply")
    args = p.parse_args()

    scene_dir = args.scene_dir.resolve()
    if not scene_dir.is_dir():
        raise FileNotFoundError(scene_dir)

    repo = args.threedgrut_repo.resolve()
    if not repo.is_dir():
        raise FileNotFoundError(
            f"3DGUT repo not at {repo}. Clone https://github.com/nv-tlabs/3dgrut"
        )

    ply = (args.ply.resolve() if args.ply is not None
           else _discover_latest_gs_ply(scene_dir))
    if not ply.is_file():
        raise FileNotFoundError(ply)

    out_usdz = scene_dir / args.out_name
    out_usdz.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "threedgrut.export.scripts.ply_to_usd",
        str(ply),
        "--output_file", str(out_usdz),
    ]
    print(f"[gs→usdz] input  : {ply}")
    print(f"[gs→usdz] output : {out_usdz}")
    _conda_run(args.conda_env, repo, cmd)

    if not out_usdz.is_file() or out_usdz.stat().st_size == 0:
        raise RuntimeError(
            f"3DGUT exited cleanly but {out_usdz} is missing/empty — "
            f"check upstream logs for silent failures"
        )
    print(f"[gs→usdz] OK ({out_usdz.stat().st_size / 1024 / 1024:.1f} MB)")

    print(
        f"\n[gs→usdz] next: render with the GS as visual:\n"
        f"  PYTHONPATH=. python scripts/render_dl3dv_with_robot.py \\\n"
        f"      --task UtensilsInMugTask \\\n"
        f"      --scene-dir {scene_dir} \\\n"
        f"      --placement-idx 0 --gs-usdz {out_usdz}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
