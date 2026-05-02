from __future__ import annotations

import json
from pathlib import Path

from diffusion_harmonizer.scene_templates.templates import (
    SceneTemplate,
    normalize_object_size_range,
    normalize_table_color,
    validate_templates,
)


def load_genie_scene_templates(path: str | Path) -> list[SceneTemplate]:
    """Load validated templates exported by Genie Sim's scene generator.

    The LLM generator should remain an optional producer. For this pipeline it
    only needs to emit this compact JSON schema; validation catches unsafe table
    heights, unreachable object regions, and malformed object-count ranges
    before Isaac rendering begins.
    """

    payload = json.loads(Path(path).read_text())
    raw_templates = payload.get("templates", payload) if isinstance(payload, dict) else payload
    templates = []
    for idx, item in enumerate(raw_templates):
        templates.append(
            SceneTemplate(
                scene_id=str(item.get("scene_id", f"genie_{idx:02d}")),
                description=str(item.get("description", "generated tabletop scene")),
                table_height=float(item.get("table_height", 0.74)),
                table_size=tuple(item.get("table_size", (1.4, 0.9))),
                robot_mount_xyz=tuple(item.get("robot_mount_xyz", (-0.9, 0.0, 0.742))),
                robot_yaw_deg=float(item.get("robot_yaw_deg", 0.0)),
                object_region_xy=tuple(item.get("object_region_xy", (-0.45, 0.35, -0.25, 0.25))),
                object_count=tuple(item.get("object_count", (2, 4))),
                object_queries=tuple(item.get("object_queries", ("bottle", "cup", "box"))),
                object_size_range=normalize_object_size_range(tuple(item.get("object_size_range", (0.11, 0.24)))),
                table_color=normalize_table_color(tuple(item.get("table_color", (0.34, 0.29, 0.22)))),
                table_roughness=float(item.get("table_roughness", 0.62)),
                hdri_query=str(item.get("hdri_query", "indoor,studio,room")),
                background_query=str(item.get("background_query", item.get("description", ""))),
            )
        )
    validate_templates(templates)
    return templates
