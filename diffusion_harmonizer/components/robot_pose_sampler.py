from __future__ import annotations

import math
import random
from dataclasses import dataclass


FRANKA_JOINT_NAME_SETS = [
    [f"panda_joint{i}" for i in range(1, 8)],
    [f"franka_joint{i}" for i in range(1, 8)],
    [f"joint{i}" for i in range(1, 8)],
]


CURATED_REACH_POSES = [
    ("center_low", [0.0, -0.55, 0.0, -2.15, 0.0, 1.65, 0.78]),
    ("left_reach", [0.42, -0.72, 0.25, -2.35, -0.10, 1.90, 1.05]),
    ("right_reach", [-0.42, -0.70, -0.25, -2.28, 0.12, 1.82, 0.48]),
    ("far_reach", [0.18, -0.98, 0.48, -2.45, -0.30, 2.02, 0.18]),
    ("near_fold", [-0.20, -0.38, -0.55, -1.78, 0.45, 1.28, 0.95]),
    ("over_table", [0.62, -0.48, -0.32, -1.88, 0.35, 1.38, 1.25]),
    ("inspect_left", [0.75, -0.62, 0.18, -2.05, -0.42, 1.56, 1.48]),
    ("inspect_right", [-0.75, -0.62, -0.18, -2.05, 0.42, 1.56, -0.12]),
]


@dataclass
class RobotPoseSample:
    pose_name: str
    joint_positions: dict[str, float]
    applied_joints: dict[str, float]
    target_object: dict[str, object] | None
    sampler: str


class RobotPoseSampler:
    """Object-conditioned Franka pose sampler with optional PyRoki hook.

    PyRoki is intentionally optional. If it is not installed, or if the robot USD
    lacks the model metadata needed by a local IK solve, curated reach poses are
    selected based on the tabletop object nearest the desired interaction region.
    """

    def __init__(self, seed: int = 42, prefer_pyroki: bool = True):
        self.rng = random.Random(seed)
        self.prefer_pyroki = prefer_pyroki
        self.pyroki_available = self._has_pyroki()

    def sample_and_apply(
        self,
        renderer,
        robot_prim_path: str,
        object_placements: list[dict[str, object]],
        pair_index: int,
    ) -> RobotPoseSample:
        target = self._target_object(object_placements)
        if self.prefer_pyroki and self.pyroki_available:
            solved = self._try_pyroki(target)
            if solved:
                applied = renderer.set_articulation_joint_positions(robot_prim_path, solved)
                return RobotPoseSample("pyroki_ik", solved, applied, target, "pyroki")

        pose_name, values = self._curated_pose(target, pair_index)
        authored = dict(zip(FRANKA_JOINT_NAME_SETS[0], values))
        applied = {}
        for names in FRANKA_JOINT_NAME_SETS:
            candidate = dict(zip(names, values))
            applied = renderer.set_articulation_joint_positions(robot_prim_path, candidate)
            if applied:
                authored = candidate
                break
        return RobotPoseSample(pose_name, authored, applied, target, "curated_fallback")

    def _curated_pose(self, target: dict[str, object] | None, pair_index: int) -> tuple[str, list[float]]:
        if not target:
            return CURATED_REACH_POSES[pair_index % len(CURATED_REACH_POSES)]
        x, y, _ = target["translate"]
        angle = math.atan2(float(y), max(abs(float(x)), 1e-3))
        if float(x) > 0.18:
            preferred = ["far_reach", "over_table"]
        elif angle > 0.25:
            preferred = ["left_reach", "inspect_left"]
        elif angle < -0.25:
            preferred = ["right_reach", "inspect_right"]
        else:
            preferred = ["center_low", "near_fold"]
        candidates = [pose for pose in CURATED_REACH_POSES if pose[0] in preferred]
        return candidates[pair_index % len(candidates)]

    def _target_object(self, object_placements: list[dict[str, object]]) -> dict[str, object] | None:
        if not object_placements:
            return None
        return min(object_placements, key=lambda item: abs(float(item["translate"][0])) + abs(float(item["translate"][1])))

    def _try_pyroki(self, target: dict[str, object] | None) -> dict[str, float] | None:
        del target
        # Hook point for a real PyRoki solve. We keep this non-invasive because
        # robot USD/URDF model paths differ between RoboVerse and Genie assets.
        return None

    @staticmethod
    def _has_pyroki() -> bool:
        try:
            import pyroki  # noqa: F401
        except Exception:
            return False
        return True
