"""In-process 3DGS / 2DGS trainer for DiffusionHarmonizer paired-data generation.

Faithful to the paper: degraded renderings are produced by deliberately handicapping
the same trainer that produces the clean reference. Four standard handicaps from
DIFIX3D+ / DiffusionHarmonizer §3.2 are exposed as named strategies:

  * sparse_k     - train on only K of the N privileged sphere views (K << N)
  * underfit     - train on all views but stop after a fraction of full iterations
  * cycle        - train on the full set, then re-render at views the trainer never
                   optimised against (held-out bin) producing reconstruction holes
  * cross_ref    - train on view-set A and render view-set B (camera-domain shift)

A reference run with all 100 views and full iterations supplies the matching
``target.png`` for every degraded ``input.png``.

The trainer is built on the public ``gsplat`` library (Strategy + rasterization).
It runs entirely in-process so the orchestrator does not need a CUDA sidecar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


@dataclass
class GSplatConfig:
    iterations: int = 30000
    sh_degree: int = 3
    init_num_points: int = 100_000
    init_scale: float = 0.01
    init_opacity: float = 0.1
    learning_rate_means: float = 1.6e-4
    learning_rate_scales: float = 5e-3
    learning_rate_quats: float = 1e-3
    learning_rate_opacities: float = 5e-2
    learning_rate_sh: float = 2.5e-3
    background: tuple[float, float, float] = (0.0, 0.0, 0.0)
    splat_kind: str = "3dgs"  # or "2dgs"
    refine_every: int = 100
    refine_start_iter: int = 500
    refine_stop_iter: int = 15000
    densification_threshold: float = 2e-4
    pruning_threshold: float = 5e-3
    seed: int = 0


@dataclass
class DegradedStrategy:
    """One degraded-rendering recipe paired with the same scene's clean reference.

    The named strategies match DIFIX3D+ §3.2 / DiffusionHarmonizer §3.2:
    sparse-reconstruction, deliberate underfitting, cycle-reconstruction, and
    cross-referencing.
    """

    name: str
    train_view_indices: list[int]
    render_view_indices: list[int]
    iterations: int
    note: str = ""


@dataclass
class CapturedView:
    rgb: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray | None  # (H, W) float32 or None
    intrinsics: np.ndarray  # (3, 3)
    world_T_cam: np.ndarray  # (4, 4) — OpenGL convention from Replicator
    image_name: str


def opengl_to_opencv(world_T_cam_gl: np.ndarray) -> np.ndarray:
    """Convert Replicator's OpenGL camera-to-world to OpenCV camera-to-world."""

    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    return world_T_cam_gl @ flip


def initialize_gaussians_from_views(views: list[CapturedView], cfg: GSplatConfig):
    """Seed the splat with depth-back-projected points so optimisation converges.

    Falls back to a uniform cube if no depth is available.
    """

    import torch

    points: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    for view in views:
        if view.depth is None:
            continue
        h, w = view.depth.shape
        ys, xs = np.mgrid[0:h:8, 0:w:8]
        z = view.depth[ys, xs]
        valid = np.isfinite(z) & (z > 0.05) & (z < 50.0)
        if not np.any(valid):
            continue
        fx, fy = view.intrinsics[0, 0], view.intrinsics[1, 1]
        cx, cy = view.intrinsics[0, 2], view.intrinsics[1, 2]
        x = (xs[valid] - cx) * z[valid] / fx
        y = (ys[valid] - cy) * z[valid] / fy
        cam = np.stack([x, y, -z[valid], np.ones_like(z[valid])], axis=-1)
        world = (opengl_to_opencv(view.world_T_cam) @ cam.T).T[:, :3]
        points.append(world.astype(np.float32))
        colors.append(view.rgb[ys[valid], xs[valid]].astype(np.float32) / 255.0)
    if points:
        means = np.concatenate(points, axis=0)
        rgb = np.concatenate(colors, axis=0)
    else:
        means = (np.random.RandomState(cfg.seed).rand(cfg.init_num_points, 3).astype(np.float32) - 0.5) * 2.0
        rgb = np.full((means.shape[0], 3), 0.5, dtype=np.float32)
    if means.shape[0] > cfg.init_num_points:
        idx = np.random.RandomState(cfg.seed).choice(means.shape[0], cfg.init_num_points, replace=False)
        means = means[idx]
        rgb = rgb[idx]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    means_t = torch.tensor(means, dtype=torch.float32, device=device)
    quats_t = torch.zeros((means.shape[0], 4), dtype=torch.float32, device=device)
    quats_t[:, 0] = 1.0
    scales_t = torch.full((means.shape[0], 3), math.log(cfg.init_scale), dtype=torch.float32, device=device)
    opacities_t = torch.full((means.shape[0],), _logit(cfg.init_opacity), dtype=torch.float32, device=device)
    sh0_t = torch.tensor(_rgb_to_sh0(rgb), dtype=torch.float32, device=device).unsqueeze(1)
    shN_t = torch.zeros((means.shape[0], (cfg.sh_degree + 1) ** 2 - 1, 3), dtype=torch.float32, device=device)
    return {"means": means_t, "quats": quats_t, "scales": scales_t, "opacities": opacities_t, "sh0": sh0_t, "shN": shN_t}


def _logit(p: float) -> float:
    p = float(min(max(p, 1e-4), 1.0 - 1e-4))
    return math.log(p / (1.0 - p))


def _rgb_to_sh0(rgb: np.ndarray) -> np.ndarray:
    return (rgb - 0.5) / 0.28209479177387814


def train_and_render(
    views: list[CapturedView],
    train_indices: Iterable[int],
    render_indices: Iterable[int],
    cfg: GSplatConfig,
    progress_cb: Callable[[int, float], None] | None = None,
) -> dict[int, np.ndarray]:
    """Train splats on ``train_indices`` views and render ``render_indices`` views.

    Returns ``{render_idx: rgb_uint8}``. The implementation uses ``gsplat``'s
    rasterisation kernel and a simple Adam loop. For both 3DGS and 2DGS the same
    kernel is invoked with a different ``rasterize_mode``.
    """

    import torch
    from gsplat import rasterization
    try:
        from gsplat.strategy import DefaultStrategy
    except Exception:
        DefaultStrategy = None  # type: ignore

    train_indices = list(train_indices)
    render_indices = list(render_indices)
    if not train_indices:
        raise ValueError("train_and_render requires at least one training view")

    torch.manual_seed(cfg.seed)
    params = initialize_gaussians_from_views([views[i] for i in train_indices], cfg)
    for tensor in params.values():
        tensor.requires_grad_(True)

    optimizer = torch.optim.Adam(
        [
            {"params": [params["means"]], "lr": cfg.learning_rate_means, "name": "means"},
            {"params": [params["quats"]], "lr": cfg.learning_rate_quats, "name": "quats"},
            {"params": [params["scales"]], "lr": cfg.learning_rate_scales, "name": "scales"},
            {"params": [params["opacities"]], "lr": cfg.learning_rate_opacities, "name": "opacities"},
            {"params": [params["sh0"]], "lr": cfg.learning_rate_sh, "name": "sh0"},
            {"params": [params["shN"]], "lr": cfg.learning_rate_sh / 20.0, "name": "shN"},
        ]
    )

    strategy = None
    strategy_state = None
    if DefaultStrategy is not None and cfg.iterations >= cfg.refine_start_iter:
        strategy = DefaultStrategy(
            refine_start_iter=cfg.refine_start_iter,
            refine_stop_iter=min(cfg.refine_stop_iter, cfg.iterations - 100),
            refine_every=cfg.refine_every,
            grow_grad2d=cfg.densification_threshold,
            prune_opa=cfg.pruning_threshold,
        )
        strategy.check_sanity(params, optimizer)
        strategy_state = strategy.initialize_state()

    rasterize_mode = "antialiased" if cfg.splat_kind == "3dgs" else "RGB+ED"
    bg_color = torch.tensor(cfg.background, dtype=torch.float32, device=params["means"].device)

    for step in range(cfg.iterations):
        view = views[train_indices[step % len(train_indices)]]
        target = torch.tensor(view.rgb, dtype=torch.float32, device=params["means"].device) / 255.0
        rendered, alpha, info = _render_single(view, params, cfg, bg_color, rasterize_mode)
        loss = (rendered - target).abs().mean() + 0.1 * (1.0 - _ssim(rendered, target))
        loss.backward()
        if strategy is not None:
            strategy.step_pre_backward(params, optimizer, strategy_state, step, info)
        optimizer.step()
        if strategy is not None:
            strategy.step_post_backward(params, optimizer, strategy_state, step, info)
        optimizer.zero_grad(set_to_none=True)
        if progress_cb and step % 200 == 0:
            progress_cb(step, float(loss.detach().cpu()))

    rendered_per_view: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for idx in render_indices:
            rgb, _alpha, _info = _render_single(views[idx], params, cfg, bg_color, rasterize_mode)
            rendered_per_view[idx] = (rgb.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
    return rendered_per_view


def _render_single(view: CapturedView, params: dict, cfg: GSplatConfig, bg_color, rasterize_mode):
    import torch
    from gsplat import rasterization

    width = int(view.rgb.shape[1])
    height = int(view.rgb.shape[0])
    K = torch.tensor(view.intrinsics, dtype=torch.float32, device=params["means"].device).unsqueeze(0)
    cam_T_world = torch.tensor(np.linalg.inv(opengl_to_opencv(view.world_T_cam)), dtype=torch.float32, device=params["means"].device).unsqueeze(0)

    colors = torch.cat([params["sh0"], params["shN"]], dim=1)
    rendered, alpha, info = rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=colors,
        viewmats=cam_T_world,
        Ks=K,
        width=width,
        height=height,
        sh_degree=cfg.sh_degree,
        backgrounds=bg_color.unsqueeze(0),
        rasterize_mode=rasterize_mode,
    )
    return rendered[0], alpha[0], info


def _ssim(pred, target) -> "torch.Tensor":  # type: ignore  # noqa: F821
    import torch
    import torch.nn.functional as F

    pred = pred.permute(2, 0, 1).unsqueeze(0)
    target = target.permute(2, 0, 1).unsqueeze(0)
    mu1 = F.avg_pool2d(pred, 11, 1, 5)
    mu2 = F.avg_pool2d(target, 11, 1, 5)
    sigma1 = F.avg_pool2d(pred * pred, 11, 1, 5) - mu1 * mu1
    sigma2 = F.avg_pool2d(target * target, 11, 1, 5) - mu2 * mu2
    sigma12 = F.avg_pool2d(pred * target, 11, 1, 5) - mu1 * mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1 * mu1 + mu2 * mu2 + c1) * (sigma1 + sigma2 + c2))
    return ssim_map.mean()


def default_degraded_strategies(num_views: int, full_iters: int = 30000) -> list[DegradedStrategy]:
    """Four DIFIX3D+/DiffusionHarmonizer-faithful handicapped recipes."""

    rng = np.random.RandomState(0)
    all_idx = list(range(num_views))
    sparse_k = sorted(rng.choice(all_idx, size=max(8, num_views // 10), replace=False).tolist())
    held_out = [i for i in all_idx if i not in set(sparse_k)]
    even = all_idx[::2]
    odd = all_idx[1::2]
    return [
        DegradedStrategy(
            name="sparse_k",
            train_view_indices=sparse_k,
            render_view_indices=held_out[: min(40, len(held_out))],
            iterations=full_iters,
            note=f"Train on {len(sparse_k)}/{num_views} views; render the held-out views.",
        ),
        DegradedStrategy(
            name="underfit",
            train_view_indices=all_idx,
            render_view_indices=all_idx[: min(40, len(all_idx))],
            iterations=max(2000, full_iters // 8),
            note="Train on all views but stop early; reconstruction is incomplete.",
        ),
        DegradedStrategy(
            name="cycle",
            train_view_indices=even,
            render_view_indices=odd[: min(40, len(odd))],
            iterations=full_iters,
            note="Train on even views, render odd views (held-out interleave).",
        ),
        DegradedStrategy(
            name="cross_ref",
            train_view_indices=all_idx[: num_views // 2],
            render_view_indices=all_idx[num_views // 2 :][: min(40, num_views // 2)],
            iterations=full_iters,
            note="Train on first half, render second half (camera-domain shift).",
        ),
    ]


def build_full_reference_strategy(num_views: int, full_iters: int = 30000) -> DegradedStrategy:
    """The reference (clean) run that supplies ``target.png`` for every pair."""

    return DegradedStrategy(
        name="reference_full",
        train_view_indices=list(range(num_views)),
        render_view_indices=list(range(num_views)),
        iterations=full_iters,
        note="All views, full iterations — clean reference.",
    )


@dataclass
class StrategyArtifacts:
    name: str
    iterations: int
    train_view_indices: list[int]
    render_view_indices: list[int]
    renders: dict[int, np.ndarray] = field(default_factory=dict)


def run_strategy_suite(
    views: list[CapturedView],
    cfg: GSplatConfig,
    strategies: list[DegradedStrategy] | None = None,
    output_root: Path | None = None,
    progress_cb: Callable[[str, int, float], None] | None = None,
) -> dict[str, StrategyArtifacts]:
    """Run the reference + every degraded strategy and (optionally) dump renders."""

    if strategies is None:
        strategies = [build_full_reference_strategy(len(views), cfg.iterations)] + default_degraded_strategies(len(views), cfg.iterations)

    artifacts: dict[str, StrategyArtifacts] = {}
    from diffusion_harmonizer.data.image_io import save_png

    for strategy in strategies:
        local_cfg = GSplatConfig(**{**cfg.__dict__, "iterations": strategy.iterations})
        cb = (lambda step, loss, _name=strategy.name: progress_cb(_name, step, loss)) if progress_cb else None
        renders = train_and_render(views, strategy.train_view_indices, strategy.render_view_indices, local_cfg, cb)
        artifacts[strategy.name] = StrategyArtifacts(
            name=strategy.name,
            iterations=strategy.iterations,
            train_view_indices=list(strategy.train_view_indices),
            render_view_indices=list(strategy.render_view_indices),
            renders=renders,
        )
        if output_root is not None:
            strategy_dir = Path(output_root) / strategy.name
            strategy_dir.mkdir(parents=True, exist_ok=True)
            for view_idx, image in renders.items():
                save_png(strategy_dir / f"{view_idx:04d}.png", image)
    return artifacts
