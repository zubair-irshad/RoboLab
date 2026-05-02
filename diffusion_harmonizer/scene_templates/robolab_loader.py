"""Load a RoboLab task scene into the standalone GenieSimRenderer stage.

The integrated RoboLab environment (see ``robolab/core/environments/factory.py``
and the recorded ``env_cfg.json`` examples) wires three USD payloads onto a
shared stage:

  * ``/World/envs/env_0/scene``     - ``assets/scenes/<name>.usda`` (objects + table)
  * ``/World/envs/env_0/robot``     - ``assets/robots/franka_robotiq_2f_85_flattened.usd``
  * ``/World/background``           - DomeLight + HDRI from ``assets/backgrounds/...``

For paired data generation we don't need physics or the gym wrapper — we just
need the same three payloads on a stage so the existing components can address
them via ``foreground_paths``. The Franka is the *robot* foreground; rigid prims
under ``scene/`` whose payloads point into ``assets/objects/`` are *object*
foregrounds; prims pointing into ``assets/fixtures/`` are receivers.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


_OBJECT_PAYLOAD_RE = re.compile(r"\.\./objects/", re.IGNORECASE)
_FIXTURE_PAYLOAD_RE = re.compile(r"\.\./fixtures/", re.IGNORECASE)
_INTERNAL_PRIM_NAMES = {"looks", "physicsscene", "physicsmaterial", "lights", "light", "defaultlight"}


@dataclass
class RoboLabSceneSpec:
    scene_id: str
    scene_usda: Path
    robot_usd: Path
    hdri_path: Path
    object_prim_paths: list[str] = field(default_factory=list)
    receiver_prim_paths: list[str] = field(default_factory=list)
    robot_prim_path: str = "/World/envs/env_0/robot"
    scene_root_prim_path: str = "/World/envs/env_0/scene"
    table_height: float = 0.0
    object_count: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def foreground_prim_paths(self) -> list[str]:
        """Robot + manipulable objects, used by ISP/relight/shadow components."""

        return [self.robot_prim_path] + list(self.object_prim_paths)


def discover_scenes(
    scenes_root: Path | str = "assets/scenes",
    backgrounds_root: Path | str = "assets/backgrounds",
    robot_usd: Path | str = "assets/robots/franka_robotiq_2f_85_flattened.usd",
    metadata_json: Path | str | None = None,
    limit: int | None = None,
) -> list[RoboLabSceneSpec]:
    """Build a RoboLabSceneSpec for every USDA in ``scenes_root``.

    Foreground/receiver classification is read from ``scene_metadata.json`` when
    available; otherwise we fall back to parsing each USDA for relative payload
    references (``../objects/*`` vs ``../fixtures/*``). Missing data does not
    raise — we just skip the scene with a warning.
    """

    scenes_dir = Path(scenes_root)
    backgrounds_dir = Path(backgrounds_root)
    robot_path = Path(robot_usd).resolve()
    if not robot_path.exists():
        raise FileNotFoundError(f"Franka USD not found at {robot_path}")
    metadata = _load_metadata(metadata_json or scenes_dir / "_metadata" / "scene_metadata.json")
    hdris = sorted(_collect_hdris(backgrounds_dir))
    if not hdris:
        raise FileNotFoundError(f"No HDRI files under {backgrounds_dir}")

    specs: list[RoboLabSceneSpec] = []
    for scene_path in sorted(scenes_dir.glob("*.usda")):
        scene_path_resolved = scene_path.resolve()
        if scene_path.name.startswith(("base_", "_")):
            continue
        try:
            objects, receivers, object_count = _classify_scene(scene_path, metadata)
        except Exception:
            continue
        if not objects:
            continue
        hdri = hdris[len(specs) % len(hdris)]
        specs.append(
            RoboLabSceneSpec(
                scene_id=scene_path.stem,
                scene_usda=scene_path_resolved,
                robot_usd=robot_path,
                hdri_path=hdri,
                object_prim_paths=objects,
                receiver_prim_paths=receivers,
                object_count=object_count,
                metadata={
                    "source_metadata": str((scenes_dir / "_metadata" / "scene_metadata.json").resolve()) if metadata else None,
                    "hdri_index": len(specs) % len(hdris),
                },
            )
        )
        if limit is not None and len(specs) >= limit:
            break
    return specs


def reference_scene_into_renderer(renderer, spec: RoboLabSceneSpec, dome_intensity: float = 800.0) -> None:
    """Reference the RoboLab payloads onto the standalone Isaac Sim stage.

    Mirrors the prim layout the RoboLab env factory produces so the existing
    foreground-path needles continue to match. Does not rely on Isaac Lab gym.
    """

    renderer.reference_asset(str(spec.scene_usda), spec.scene_root_prim_path)
    renderer.reference_asset(str(spec.robot_usd), spec.robot_prim_path)
    renderer.set_dome_light(str(spec.hdri_path), intensity=dome_intensity, rotation_deg=0.0)


def _load_metadata(metadata_json: Path | str) -> dict:
    path = Path(metadata_json)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _classify_scene(scene_path: Path, metadata: dict) -> tuple[list[str], list[str], int]:
    """Return (object_prim_paths, receiver_prim_paths, object_count) for a scene.

    Uses ``scene_metadata.json`` when entries exist for this scene; falls back
    to parsing ``payload`` strings out of the USDA text otherwise.
    """

    objects: list[str] = []
    receivers: list[str] = []
    entries = metadata.get(scene_path.name)
    if entries:
        for entry in entries:
            name = str(entry.get("name") or "")
            if not name or name.lower() in _INTERNAL_PRIM_NAMES:
                continue
            payload = entry.get("payload") or []
            if not isinstance(payload, list) or not payload:
                continue
            ref = str(payload[0])
            prim = f"/World/envs/env_0/scene/{name}"
            if _OBJECT_PAYLOAD_RE.search(ref):
                objects.append(prim)
            elif _FIXTURE_PAYLOAD_RE.search(ref):
                receivers.append(prim)
        return objects, receivers, len(objects)

    text = scene_path.read_text(errors="ignore")
    for match in re.finditer(r"def\s+\"?(\w+)\"?[^{]*?prepend\s+references?\s*=\s*@([^@]+)@", text):
        name, ref = match.group(1), match.group(2)
        if name.lower() in _INTERNAL_PRIM_NAMES:
            continue
        prim = f"/World/envs/env_0/scene/{name}"
        if _OBJECT_PAYLOAD_RE.search(ref):
            objects.append(prim)
        elif _FIXTURE_PAYLOAD_RE.search(ref):
            receivers.append(prim)
    return objects, receivers, len(objects)


def _collect_hdris(backgrounds_dir: Path) -> list[Path]:
    suffixes = {".hdr", ".exr"}
    hdris: list[Path] = []
    for path in backgrounds_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            hdris.append(path.resolve())
    return hdris
