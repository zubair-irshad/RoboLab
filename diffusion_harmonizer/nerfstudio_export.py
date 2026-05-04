"""Export hemispheric captures to nerfstudio dataparser format.

Reads ``<env>/01_artifacts_correction/views/<NNNN>/{rgb.png, intrinsics.json,
extrinsics.json}`` and writes a tree consumable by ``ns-train splatfacto``:

  ``<env>/01_artifacts_correction/nerfstudio/``
    ├── images/                     ← shared rgb pool (frame_<NNNN>.png)
    ├── full/transforms.json        all 120 frames        (reference)
    ├── sparse_k/transforms.json    K of N frames         (sparse)
    ├── underfit/transforms.json    all frames            (early-stop variant)
    ├── cycle/transforms.json       even=train odd=eval   (cycle)
    └── cross_ref/transforms.json   1st half=train, 2nd=eval (cross-ref)

Each strategy directory symlinks its ``images`` to the shared pool so you only
keep one copy of the rgb on disk. ``transforms.json`` follows nerfstudio's
schema (camera_model="OPENCV", per-frame ``transform_matrix`` in OpenGL
convention which matches Isaac Sim's Replicator OpenGL pose).

For the strategies that hold out frames as eval (cycle / cross_ref) we write
``train_filenames`` and ``eval_filenames`` arrays into transforms.json so
nerfstudio's nerfstudio_dataparser routes the splits accordingly. For sparse
and underfit there's no held-out split — train_split_fraction stays default
and the strategy variation is in the iteration count or train-frame count
controlled at ``ns-train`` time.

No torch / no Isaac Sim — pure stdlib + numpy + imageio.
"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class StrategyExport:
    name: str
    transforms_path: Path
    num_train: int
    num_eval: int
    note: str
    suggested_ns_train_args: str


def _read_view(view_dir: Path) -> dict | None:
    rgb_path = view_dir / "rgb.png"
    K_path = view_dir / "intrinsics.json"
    E_path = view_dir / "extrinsics.json"
    if not (rgb_path.exists() and K_path.exists() and E_path.exists()):
        return None
    K = np.asarray(json.loads(K_path.read_text())["K"], dtype=np.float64)
    T = np.asarray(json.loads(E_path.read_text())["world_T_cam_gl"], dtype=np.float64)
    return {"view_id": int(view_dir.name), "rgb_path": rgb_path, "K": K, "T": T}


def _stage_image_pool(views: list[dict], images_dir: Path) -> None:
    """Symlink (or copy if symlinks aren't allowed) each rgb into a shared pool."""

    images_dir.mkdir(parents=True, exist_ok=True)
    for v in views:
        dst = images_dir / f"frame_{v['view_id']:04d}.png"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            dst.symlink_to(v["rgb_path"].resolve())
        except (OSError, NotImplementedError):
            shutil.copy2(v["rgb_path"], dst)


def _build_transforms(
    views: list[dict],
    *,
    width: int,
    height: int,
    train_ids: list[int] | None,
    eval_ids: list[int] | None,
) -> dict:
    """Compose a nerfstudio transforms.json dict.

    All views go into ``frames``; if ``train_ids`` / ``eval_ids`` are
    provided we also emit ``train_filenames`` and ``eval_filenames``
    so nerfstudio's nerfstudio_dataparser respects the split.
    """

    K = views[0]["K"]
    transforms = {
        "camera_model": "OPENCV",
        "fl_x": float(K[0, 0]),
        "fl_y": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "w": int(width),
        "h": int(height),
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "frames": [
            {
                "file_path": f"images/frame_{v['view_id']:04d}.png",
                "transform_matrix": v["T"].tolist(),
            }
            for v in views
        ],
    }
    if train_ids is not None:
        transforms["train_filenames"] = [
            f"images/frame_{vid:04d}.png" for vid in sorted(train_ids)
        ]
    if eval_ids is not None:
        transforms["eval_filenames"] = [
            f"images/frame_{vid:04d}.png" for vid in sorted(eval_ids)
        ]
    return transforms


def _ensure_images_link(strategy_dir: Path, images_pool: Path) -> None:
    link = strategy_dir / "images"
    if link.exists() or link.is_symlink():
        link.unlink()
    try:
        link.symlink_to(Path("..") / images_pool.name, target_is_directory=True)
    except (OSError, NotImplementedError):
        shutil.copytree(images_pool, link, dirs_exist_ok=True)


def export_env(
    env_artifacts_dir: Path,
    output_dir: Path | None = None,
    sparse_k: int = 24,
    seed: int = 42,
) -> tuple[Path, list[StrategyExport]]:
    """Export one env's captures into the nerfstudio strategy tree.

    Returns ``(nerfstudio_root, [StrategyExport, ...])``.
    """

    import imageio.v3 as iio

    views_dir = Path(env_artifacts_dir) / "views"
    if not views_dir.exists():
        raise FileNotFoundError(f"No views dir under {env_artifacts_dir}.")
    if output_dir is None:
        output_dir = Path(env_artifacts_dir) / "nerfstudio"
    output_dir.mkdir(parents=True, exist_ok=True)

    views = sorted(
        (v for v in (_read_view(d) for d in views_dir.iterdir() if d.is_dir()) if v is not None),
        key=lambda v: v["view_id"],
    )
    if not views:
        raise FileNotFoundError(f"No usable view subdirs under {views_dir}.")

    img = np.asarray(iio.imread(views[0]["rgb_path"]))
    height, width = int(img.shape[0]), int(img.shape[1])
    if img.shape[-1] == 4:
        # nerfstudio handles RGBA but we drop alpha for cleanness.
        for v in views:
            arr = np.asarray(iio.imread(v["rgb_path"]))[..., :3]
            iio.imwrite(v["rgb_path"], arr)

    images_pool = output_dir / "images"
    _stage_image_pool(views, images_pool)

    del seed  # kept for backward compat; we now pick spatially deterministic subsets
    all_ids = sorted(v["view_id"] for v in views)
    n = len(all_ids)
    k = min(max(1, sparse_k), n)
    # Equally-spaced sparse: every (n/k)-th view of the Fibonacci spiral.
    # Deterministic and gives the same hemispheric coverage as random sampling
    # but with predictable spacing.
    spaced_indices = [int(round(i * (n - 1) / max(k - 1, 1))) for i in range(k)]
    sparse_ids = sorted({all_ids[i] for i in spaced_indices})
    sparse_eval_ids = [vid for vid in all_ids if vid not in set(sparse_ids)]
    # Contiguous-arc holdout: training views form a continuous block of the
    # Fibonacci spiral (which traces a connected hemispheric path), so the
    # held-out indices sit on a completely different angular sector and
    # splatfacto genuinely has to extrapolate — gives much stronger
    # paired-data signal on overlapping-view datasets per the DIFIX3D+
    # paper's note about sparse reconstruction.
    arc_ids = all_ids[: max(1, n // 3)]  # first third of the spiral as train
    arc_eval_ids = all_ids[n // 3 :]

    strategies: list[StrategyExport] = []

    def write_variant(name: str, frames: list[dict], train_ids, eval_ids, note, ns_train_args) -> None:
        strat_dir = output_dir / name
        strat_dir.mkdir(parents=True, exist_ok=True)
        _ensure_images_link(strat_dir, images_pool)
        transforms = _build_transforms(
            frames, width=width, height=height, train_ids=train_ids, eval_ids=eval_ids,
        )
        (strat_dir / "transforms.json").write_text(json.dumps(transforms, indent=2))
        strategies.append(
            StrategyExport(
                name=name,
                transforms_path=strat_dir / "transforms.json",
                num_train=len(train_ids) if train_ids is not None else len(frames),
                num_eval=len(eval_ids) if eval_ids is not None else 0,
                note=note,
                suggested_ns_train_args=ns_train_args,
            )
        )

    write_variant(
        "full",
        views,
        train_ids=None,
        eval_ids=None,
        note="All views as train. Clean reference splatfacto run; pairs as the target.",
        ns_train_args="splatfacto --max-num-iterations 30000",
    )
    write_variant(
        "sparse_k",
        views,  # all frames in transforms.json; train_filenames picks the subset
        train_ids=sparse_ids,
        eval_ids=sparse_eval_ids,
        note=(
            f"{len(sparse_ids)} equally-spaced views from the Fibonacci spiral as "
            f"train_filenames; remaining {len(sparse_eval_ids)} as eval_filenames. "
            "WARNING: per DIFIX3D+ §3.2, sparse reconstruction is suboptimal when "
            "held-out views observe the same region as training views — which is the "
            "case here on a tightly-sampled hemisphere. Use sparse_arc instead for "
            "stronger paired-data signal."
        ),
        ns_train_args="splatfacto --max-num-iterations 30000",
    )
    write_variant(
        "sparse_arc",
        views,
        train_ids=arc_ids,
        eval_ids=arc_eval_ids,
        note=(
            f"Train on the first {len(arc_ids)} contiguous views of the Fibonacci "
            f"spiral; render the remaining {len(arc_eval_ids)} as held-out. The held-"
            "out views sit on a different angular sector of the hemisphere, so "
            "splatfacto cannot interpolate them well — strong DIFIX3D+ sparse-"
            "reconstruction supervision signal even on overlapping-view captures."
        ),
        ns_train_args="splatfacto --max-num-iterations 30000",
    )
    write_variant(
        "underfit",
        views,
        train_ids=None,
        eval_ids=None,
        note=(
            "All views as train, deliberately undertrained. Pass --max-num-iterations "
            "1500-3000 so densification stops before the model converges; renders show "
            "spurious geometry and missing details across the whole hemisphere."
        ),
        ns_train_args="splatfacto --max-num-iterations 1500",
    )

    return output_dir, strategies


def export_all(
    output_root: Path,
    sparse_k: int = 24,
    seed: int = 42,
) -> dict[str, list[StrategyExport]]:
    """Run :func:`export_env` over every env that has a hemispheric snapshot."""

    output_root = Path(output_root)
    summary: dict[str, list[StrategyExport]] = {}
    for env_dir in sorted(p for p in output_root.iterdir() if p.is_dir()):
        artifacts_dir = env_dir / "01_artifacts_correction"
        if not (artifacts_dir / "views").exists():
            continue
        _, strats = export_env(artifacts_dir, sparse_k=sparse_k, seed=seed)
        summary[env_dir.name] = strats
    return summary
