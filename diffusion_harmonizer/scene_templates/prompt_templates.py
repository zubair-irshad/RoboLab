from __future__ import annotations

import random
import re
from pathlib import Path

from diffusion_harmonizer.scene_templates.templates import SceneTemplate, TABLE_MATERIALS, validate_templates


DEFAULT_SCENE_PROMPTS = [
    "Retail checkout counter: place 3-4 packaged goods, drink cans, cartons, and bottles near the robot reachable zone.",
    "Grocery sorting table: arrange fruit, bottles, cans, and snack packages in a loose cluster for pick-and-place.",
    "Home kitchen counter: place mugs, bowls, utensils, bottles, and food containers with natural spacing.",
    "Office desk manipulation: place a phone-sized object, a cup, a small box, and a book near the front half of the table.",
    "Laboratory bench: place tubes, medicine bottles, small boxes, and containers as a precise sorting task.",
    "Catering prep table: place cups, plates, bowls, and food packages in a service-counter layout.",
    "Warehouse packing station: place parcels, boxes, tools, and containers in a sparse packing-task arrangement.",
    "Coffee bar counter: place mugs, cups, cans, bottles, and small boxes near the operator side.",
    "Clinic toiletry table: place medicine bottles, tubes, boxes, and small containers with clean spacing.",
    "Industrial workbench: place blocks, tools, parts, and containers in a reachable inspection layout.",
]


KEYWORD_OBJECTS = {
    "retail": ("bottle", "can", "carton", "package", "snack", "box"),
    "checkout": ("bottle", "can", "carton", "package", "snack", "box"),
    "grocery": ("fruit", "bottle", "can", "carton", "package", "vegetable"),
    "kitchen": ("cup", "mug", "bowl", "bottle", "utensil", "food"),
    "home": ("cup", "mug", "bowl", "bottle", "book", "remote"),
    "office": ("phone", "mouse", "cup", "book", "box", "container"),
    "desk": ("phone", "mouse", "cup", "book", "box", "container"),
    "laboratory": ("tube", "medicine", "bottle", "box", "container", "cup"),
    "lab": ("tube", "medicine", "bottle", "box", "container", "cup"),
    "catering": ("cup", "plate", "bowl", "bottle", "food", "box"),
    "warehouse": ("box", "package", "parcel", "tool", "container", "bottle"),
    "coffee": ("mug", "cup", "bottle", "can", "box", "package"),
    "clinic": ("medicine", "tube", "bottle", "box", "container", "cup"),
    "industrial": ("tool", "part", "block", "box", "container", "bottle"),
    "workbench": ("tool", "part", "block", "box", "container", "bottle"),
}


KEYWORD_HDRI = {
    "retail": "supermarket,retail,store,indoor",
    "checkout": "supermarket,retail,store,indoor",
    "grocery": "supermarket,grocery,indoor",
    "kitchen": "kitchen,home,room,indoor",
    "home": "home,room,studio,indoor",
    "office": "office,room,studio,indoor",
    "desk": "office,room,studio,indoor",
    "laboratory": "laboratory,studio,indoor",
    "lab": "laboratory,studio,indoor",
    "catering": "catering,kitchen,indoor",
    "warehouse": "warehouse,industrial,indoor",
    "coffee": "cafe,catering,kitchen,indoor",
    "clinic": "clinic,laboratory,indoor",
    "industrial": "industrial,warehouse,studio",
    "workbench": "industrial,warehouse,studio",
}


def load_prompt_file(path: str | Path) -> list[str]:
    """Load plain-text or markdown prompts, ignoring empty/comment lines."""

    prompts = []
    for line in Path(path).read_text().splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        text = re.sub(r"^[-*]\s+", "", text)
        if text:
            prompts.append(text)
    return prompts


def generate_templates_from_text_prompts(
    prompts: list[str] | None = None,
    count: int = 20,
    seed: int = 42,
) -> list[SceneTemplate]:
    """Generate validated tabletop templates from natural-language commands.

    This is intentionally local and deterministic. Genie Sim's LLM scene
    generator, Gemini, or any other model can be used upstream later, but the
    paired renderer consumes this compact validated schema rather than GUI state.
    """

    source_prompts = [prompt.strip() for prompt in (prompts or DEFAULT_SCENE_PROMPTS) if prompt.strip()]
    if not source_prompts:
        source_prompts = DEFAULT_SCENE_PROMPTS

    rng = random.Random(seed)
    templates: list[SceneTemplate] = []
    for idx in range(max(1, count)):
        prompt = source_prompts[idx % len(source_prompts)]
        tokens = set(re.findall(r"[a-zA-Z0-9_]+", prompt.lower()))
        object_queries = _queries_for_tokens(tokens)
        hdri_query = _hdri_for_tokens(tokens)
        table_height = rng.uniform(0.70, 0.82)
        table_x = rng.uniform(1.35, 1.85)
        table_y = rng.uniform(0.85, 1.15)
        margin = 0.22
        robot_side = rng.choice([-1.0, 1.0])
        robot_x = robot_side * rng.uniform(0.72, 0.98)
        robot_y = rng.uniform(-0.12, 0.12)
        robot_yaw = 180.0 if robot_x > 0 else 0.0
        count_range = _object_count_for_tokens(tokens)
        size_range = _size_range_for_tokens(tokens)
        table_color, table_roughness = rng.choice(TABLE_MATERIALS)
        scene_key = _slug(prompt)[:42] or "prompt_scene"
        templates.append(
            SceneTemplate(
                scene_id=f"{idx:02d}_{scene_key}",
                description=prompt,
                table_height=table_height,
                table_size=(table_x, table_y),
                robot_mount_xyz=(robot_x, robot_y, table_height),
                robot_yaw_deg=robot_yaw,
                object_region_xy=(
                    -table_x * 0.5 + margin,
                    table_x * 0.28,
                    -table_y * 0.5 + margin,
                    table_y * 0.5 - margin,
                ),
                object_count=count_range,
                object_queries=object_queries,
                object_size_range=size_range,
                table_color=table_color,
                table_roughness=table_roughness,
                hdri_query=hdri_query,
                background_query=prompt,
            )
        )
    validate_templates(templates)
    return templates


def _queries_for_tokens(tokens: set[str]) -> tuple[str, ...]:
    queries: list[str] = []
    for token, values in KEYWORD_OBJECTS.items():
        if token in tokens:
            queries.extend(values)
    if not queries:
        queries.extend(("bottle", "cup", "box", "can", "container", "block"))
    deduped = []
    for query in queries:
        if query not in deduped:
            deduped.append(query)
    return tuple(deduped[:8])


def _hdri_for_tokens(tokens: set[str]) -> str:
    for token, value in KEYWORD_HDRI.items():
        if token in tokens:
            return value
    return "indoor,studio,room"


def _object_count_for_tokens(tokens: set[str]) -> tuple[int, int]:
    if {"grid", "many", "dense"} & tokens:
        return (4, 6)
    if {"sparse", "inspection", "precise"} & tokens:
        return (2, 3)
    return (3, 5)


def _size_range_for_tokens(tokens: set[str]) -> tuple[float, float]:
    if {"industrial", "warehouse", "packing", "parcel"} & tokens:
        return (0.13, 0.28)
    if {"laboratory", "lab", "clinic", "medicine", "tube"} & tokens:
        return (0.08, 0.18)
    return (0.11, 0.24)


def _slug(text: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-zA-Z0-9]+", "_", text.lower())).strip("_")
