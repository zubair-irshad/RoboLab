from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from diffusion_harmonizer.components.robot_pose_sampler import RobotPoseSampler
from diffusion_harmonizer.scene_templates import ObjectPoolSampler, SceneTemplate


@dataclass
class SceneState:
    scene_id: str | None
    robot_pose: dict[str, object]
    object_placements: list[dict[str, object]]
    hdri: str | None = None


class Phase1SceneRandomizer:
    """Small indoor tabletop randomizer for ISP and shadow demo pairs."""

    def __init__(
        self,
        renderer,
        robot_prim_path: str = "/World/Robot",
        object_prim_paths: list[str] | None = None,
        object_z: float = 0.722,
        object_scale: float = 0.8,
        object_surface_sink: float = 0.006,
        templates: list[SceneTemplate] | None = None,
        object_sampler: ObjectPoolSampler | None = None,
        prefer_pyroki: bool = True,
        seed: int = 42,
    ):
        self.renderer = renderer
        self.robot_prim_path = robot_prim_path
        self.object_prim_paths = object_prim_paths or []
        self.object_z = object_z
        self.object_scale = object_scale
        self.object_surface_sink = object_surface_sink
        self.templates = templates or []
        self.object_sampler = object_sampler
        self.pose_sampler = RobotPoseSampler(seed=seed, prefer_pyroki=prefer_pyroki)
        self.rng = random.Random(seed)

    def __call__(self, pair_index: int, camera_name: str) -> dict[str, object]:
        del camera_name
        template = self.templates[pair_index % len(self.templates)] if self.templates else None
        if template:
            self.renderer.set_preview_surface_material("/World/TableMat/Shader", template.table_color, template.table_roughness)
            self.renderer.set_prim_display_color("/World/Table", template.table_color)
            hdri = self.object_sampler.sample_hdri(template) if self.object_sampler else None
            if hdri:
                self.renderer.set_dome_light(str(hdri), intensity=1200.0, rotation_deg=self.rng.uniform(0.0, 360.0))
            self.renderer.set_prim_transform(
                self.robot_prim_path,
                translate=template.robot_mount_xyz,
                rotate=(0.0, 0.0, template.robot_yaw_deg),
                scale=(1.0, 1.0, 1.0),
            )
            self.renderer.align_prim_bottom_to_z(self.robot_prim_path, template.table_height - 0.001)

        placements = []
        active_paths = list(self.object_prim_paths)
        if template and self.object_sampler:
            lo, hi = template.object_count
            count = min(len(active_paths), self.rng.randint(lo, hi))
            assets = self.object_sampler.sample(template, count=count)
            active_paths = active_paths[: len(assets)]
        else:
            assets = [None] * len(active_paths)

        for idx, prim_path in enumerate(active_paths):
            asset = assets[idx] if idx < len(assets) else None
            if asset:
                self.renderer.reference_asset(str(asset), prim_path)
            if template:
                x, y = self._sample_non_overlapping_xy(template.object_region_xy, placements)
                z = template.table_height - self.object_surface_sink
                target_extent = self.rng.uniform(*template.object_size_range)
            else:
                x = -0.25 + 0.25 * idx + self.rng.uniform(-0.06, 0.06)
                y = self.rng.uniform(-0.16, 0.16)
                z = self.object_z
                target_extent = None
            yaw = self.rng.uniform(-35.0, 35.0)
            self.renderer.set_prim_transform(
                prim_path,
                translate=(x, y, z),
                rotate=(0.0, 0.0, yaw),
                scale=(self.object_scale, self.object_scale, self.object_scale),
            )
            fit_scale = self.renderer.fit_prim_max_extent(prim_path, target_extent) if target_extent else None
            align_delta = self.renderer.align_prim_bottom_to_z(prim_path, z)
            placements.append(
                {
                    "prim_path": prim_path,
                    "asset": str(Path(asset)) if asset else None,
                    "translate": [x, y, z],
                    "rotate_deg": [0.0, 0.0, yaw],
                    "base_scale": self.object_scale,
                    "target_max_extent": target_extent,
                    "fit_scale_multiplier": fit_scale,
                    "bottom_alignment_dz": align_delta,
                }
            )
        for prim_path in self.object_prim_paths[len(active_paths):]:
            self.renderer.set_prim_visibility(prim_path, False)
        for prim_path in active_paths:
            self.renderer.set_prim_visibility(prim_path, True)

        robot_pose = self.pose_sampler.sample_and_apply(self.renderer, self.robot_prim_path, placements, pair_index)

        return SceneState(
            scene_id=template.scene_id if template else None,
            robot_pose=robot_pose.__dict__,
            object_placements=placements,
            hdri=str(hdri) if template and hdri else None,
        ).__dict__

    def _sample_non_overlapping_xy(self, region: tuple[float, float, float, float], placements: list[dict[str, object]]) -> tuple[float, float]:
        x0, x1, y0, y1 = region
        for _ in range(30):
            x = self.rng.uniform(x0, x1)
            y = self.rng.uniform(y0, y1)
            if all((x - float(item["translate"][0])) ** 2 + (y - float(item["translate"][1])) ** 2 > 0.22 ** 2 for item in placements):
                return x, y
        return self.rng.uniform(x0, x1), self.rng.uniform(y0, y1)
