"""RoboLab-native runtime for DiffusionHarmonizer paired-data generation.

Wraps a ``RobolabEnv`` produced by ``robolab.core.environments.runtime.create_env``
and exposes the operations the five paired-data components need:

  * resolve foreground vs receiver prim paths from ``env_cfg.contact_object_list``
  * add N privileged spherical Replicator cameras on the live stage
  * capture rgb / depth / instance-segmentation per camera
  * swap HDRI on the existing ``/World/background`` dome light
  * toggle shadows / set per-light shadow-link excludes
  * toggle USD prim visibility
  * advance physics via ``env.step`` (so cameras settle, dome reloads, etc.)

The runtime does NOT create its own ``SimulationApp``. Callers must launch
Isaac Sim via ``isaaclab.app.AppLauncher`` first (see ``run_empty.py``); the
runtime then lives entirely inside the existing app and consumes the env
that ``create_env`` returns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


_RECEIVER_NAME_HINTS = (
    "table", "tabletop", "desk", "counter", "shelf", "shelving",
    "rack", "bin", "tray", "pallet", "platform", "surface", "drawer",
    "cabinet", "stand", "ground", "floor",
)


@dataclass
class CameraSpec:
    name: str
    prim_path: str
    resolution: tuple[int, int]
    focal_length: float
    horizontal_aperture: float
    vertical_aperture: float
    position: tuple[float, float, float]
    look_at: tuple[float, float, float]


class HarmonizerRuntime:
    """Live wrapper over a RoboLab env tailored to paired-data generation."""

    def __init__(
        self,
        env_name: str,
        *,
        device: str = "cuda:0",
        seed: int = 0,
        instruction_type: str = "default",
        env_index: int = 0,
        num_envs: int = 1,
        physx_buffer_scale: float = 0.1,
        enable_depth: bool = False,
        enable_normals: bool = False,
    ):
        # Imports are deferred so this module is importable without Isaac Sim.
        from robolab.core.environments.config import parse_env_cfg
        from robolab.core.environments.runtime import create_env, end_episode  # noqa
        import omni.replicator.core as rep
        import omni.usd

        self._end_episode = end_episode
        self._rep = rep
        self._omni_usd = omni.usd

        self.env_name = env_name
        self.env_index = env_index

        # Build cfg first so we can shrink the GPU PhysX scratch buffers before
        # the env is constructed. Isaac Lab's defaults (1 GB collision stack,
        # 1 GB heap, 1 GB temp, 8M contacts) are tuned for many-env parallel
        # training and OOM small-VRAM cards on a single env. For data-gen we
        # only need a fraction of that.
        env_cfg = parse_env_cfg(
            env_name, device=device, seed=seed, num_envs=num_envs, use_fabric=True,
        )
        if device.startswith("cuda") and physx_buffer_scale > 0 and physx_buffer_scale < 1.0:
            _shrink_gpu_physx_buffers(env_cfg, physx_buffer_scale)

        # RoboLab task cfgs ship cameras with data_types=["rgb"]. To capture
        # depth / normals (needed for the artifacts gsplat init PLY) we have
        # to extend each TiledCamera's data_types BEFORE env construction;
        # the renderer wires its output buffers from that list at startup
        # and ignores additions afterwards.
        if enable_depth or enable_normals:
            updated = _ensure_camera_data_types(env_cfg, depth=enable_depth, normals=enable_normals)
            if updated:
                print(
                    f"[runtime:{env_name}] extended camera data_types on {updated}: "
                    f"depth={enable_depth} normals={enable_normals}",
                    flush=True,
                )

        self.env, self.env_cfg = create_env(
            scene=env_cfg,
            device=device,
            seed=seed,
            num_envs=num_envs,
            use_fabric=True,
            instruction_type=instruction_type,
        )
        self.env.reset()

        self.stage = self._omni_usd.get_context().get_stage()
        self.cameras: dict[str, CameraSpec] = {}
        self.render_products: dict[str, Any] = {}
        self.annotators: dict[str, dict[str, Any]] = {}

        self._env_ns = f"/World/envs/env_{self.env_index}"
        self._dome_prim_path = "/World/background"
        self._harmonizer_root = "/World/HarmonizerCameras"

    # ---- prim-path resolution ----------------------------------------------

    @property
    def robot_prim_path(self) -> str:
        return f"{self._env_ns}/robot"

    @property
    def scene_root_prim_path(self) -> str:
        return f"{self._env_ns}/scene"

    def contact_object_names(self) -> list[str]:
        names = getattr(self.env_cfg, "contact_object_list", None) or []
        return [n for n in names if isinstance(n, str)]

    def foreground_prim_paths(self) -> list[str]:
        """Robot + every non-receiver entry from ``contact_object_list``.

        Used by shadow simulation (the robot also casts shadows we want to
        learn) and asset re-insertion (the robot is foreground that gets
        composited over the gsplat background).
        """

        return [self.robot_prim_path] + self.object_prim_paths()

    def object_prim_paths(self) -> list[str]:
        """Manipulable objects only - excludes the robot and receivers.

        Used by ISP modification: paper applies ISP to *inserted* objects so
        they look tonally mismatched against the background; the robot is part
        of the embodiment / native scene and should not be ISP-perturbed.
        """

        return [
            f"{self.scene_root_prim_path}/{name}"
            for name in self.contact_object_names()
            if not _looks_like_receiver(name)
        ]

    def receiver_prim_paths(self) -> list[str]:
        return [
            f"{self.scene_root_prim_path}/{name}"
            for name in self.contact_object_names()
            if _looks_like_receiver(name)
        ]

    def all_scene_object_prim_paths(self) -> list[str]:
        return [f"{self.scene_root_prim_path}/{name}" for name in self.contact_object_names()]

    # ---- physics / sim stepping --------------------------------------------

    def step(self, n: int = 1) -> None:
        """Advance the env so renderers see the latest stage state.

        Uses ``isaaclab.envs.utils.spaces.sample_space`` to draw an action from
        the env's action space — matching ``examples/demo/run_empty.py``.
        Sending zero-actions tends to command the Franka into self-collision
        which crashes GPU PhysX on memory-constrained GPUs.
        """

        if n <= 0:
            return
        from isaaclab.envs.utils.spaces import sample_space

        for _ in range(n):
            actions = sample_space(self.env.single_action_space, device=self.env.device, batch_size=self.env.num_envs)
            self.env.step(actions)

    # ---- camera management on the live stage --------------------------------

    def add_sphere_cameras(
        self,
        center: tuple[float, float, float],
        radius: float = 1.6,
        num_cameras: int = 100,
        resolution: tuple[int, int] = (512, 512),
        name_prefix: str = "harmonizer_sphere",
        focal_length: float = 24.0,
        horizontal_aperture: float = 20.955,
    ) -> list[str]:
        """Fibonacci-sphere capture rig added to the live env stage."""

        names: list[str] = []
        cx, cy, cz = center
        golden = math.pi * (3.0 - math.sqrt(5.0))
        for idx in range(num_cameras):
            y = 1.0 - (idx / max(num_cameras - 1, 1)) * 2.0
            r = math.sqrt(max(0.0, 1.0 - y * y))
            theta = golden * idx
            position = (
                cx + radius * math.cos(theta) * r,
                cy + radius * math.sin(theta) * r,
                cz + radius * y,
            )
            name = f"{name_prefix}_{idx:03d}"
            self._add_camera(
                name=name,
                position=position,
                look_at=center,
                resolution=resolution,
                focal_length=focal_length,
                horizontal_aperture=horizontal_aperture,
            )
            names.append(name)
        return names

    def add_orbit_cameras(
        self,
        center: tuple[float, float, float],
        radius: float = 1.4,
        height: float = 0.6,
        num_cameras: int = 8,
        resolution: tuple[int, int] = (640, 480),
        name_prefix: str = "harmonizer_orbit",
    ) -> list[str]:
        names: list[str] = []
        cx, cy, cz = center
        for idx in range(num_cameras):
            theta = 2.0 * math.pi * idx / max(num_cameras, 1)
            position = (cx + radius * math.cos(theta), cy + radius * math.sin(theta), cz + height)
            name = f"{name_prefix}_{idx:03d}"
            self._add_camera(name=name, position=position, look_at=center, resolution=resolution)
            names.append(name)
        return names

    def _add_camera(
        self,
        name: str,
        position: tuple[float, float, float],
        look_at: tuple[float, float, float],
        resolution: tuple[int, int],
        focal_length: float = 24.0,
        horizontal_aperture: float = 20.955,
    ) -> None:
        from pxr import Gf, UsdGeom

        UsdGeom.Xform.Define(self.stage, self._harmonizer_root)
        prim_path = name if name.startswith("/") else f"{self._harmonizer_root}/{name}"
        width, height = int(resolution[0]), int(resolution[1])
        vertical_aperture = horizontal_aperture * height / width

        camera = UsdGeom.Camera.Define(self.stage, prim_path)
        camera.CreateFocalLengthAttr(float(focal_length))
        camera.CreateHorizontalApertureAttr(float(horizontal_aperture))
        camera.CreateVerticalApertureAttr(float(vertical_aperture))
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))

        quat = _lookat_quaternion(position, look_at)
        xf = UsdGeom.Xformable(camera.GetPrim())
        _set_xform_op(xf, UsdGeom.XformOp.TypeTranslate, position, double=True)
        _set_orient_quat(xf, quat)

        render_product = self._rep.create.render_product(prim_path, (width, height))
        annotators: dict[str, Any] = {}
        for modality, name_in_replicator in (
            ("rgb", "rgb"),
            ("depth", "distance_to_camera"),
            ("segmentation", "instance_segmentation"),
            ("normal", "normals"),
        ):
            annotator = self._rep.AnnotatorRegistry.get_annotator(name_in_replicator)
            annotator.attach([render_product])
            annotators[modality] = annotator
        self.render_products[name] = render_product
        self.annotators[name] = annotators
        # Replicator products need at least one app update to wire up before
        # capture_frame can pull data without stalling indefinitely.
        try:
            from isaacsim import SimulationApp  # noqa: F401
            import omni.kit.app

            omni.kit.app.get_app().update()
        except Exception:
            pass
        self.cameras[name] = CameraSpec(
            name=name,
            prim_path=prim_path,
            resolution=(width, height),
            focal_length=float(focal_length),
            horizontal_aperture=float(horizontal_aperture),
            vertical_aperture=float(vertical_aperture),
            position=position,
            look_at=look_at,
        )

    def capture_frame(
        self,
        camera_name: str,
        rgb: bool = True,
        depth: bool = False,
        segmentation: bool = False,
        normal: bool = False,
    ) -> dict[str, Any]:
        self._rep.orchestrator.step()
        ann = self.annotators[camera_name]
        out: dict[str, Any] = {
            "rgb": None,
            "depth": None,
            "segmentation": None,
            "segmentation_mapping": None,
            "normal": None,
            "camera_intrinsics": self.camera_intrinsics(camera_name),
            "camera_extrinsics": self.camera_extrinsics(camera_name),
        }
        if rgb:
            out["rgb"] = _rgb_to_uint8(ann["rgb"].get_data())
        if depth:
            out["depth"] = np.asarray(ann["depth"].get_data(), dtype=np.float32)
        if segmentation:
            data = ann["segmentation"].get_data()
            out["segmentation"], out["segmentation_mapping"] = _segmentation_arrays(data)
        if normal:
            out["normal"] = np.asarray(ann["normal"].get_data(), dtype=np.float32)[..., :3]
        return out

    def capture_multiview(self, camera_names: Iterable[str], **kw) -> list[dict[str, Any]]:
        return [self.capture_frame(name, **kw) for name in camera_names]

    def camera_intrinsics(self, camera_name: str) -> np.ndarray:
        spec = self.cameras[camera_name]
        width, height = spec.resolution
        fx = width * spec.focal_length / spec.horizontal_aperture
        fy = height * spec.focal_length / spec.vertical_aperture
        return np.array([[fx, 0.0, width * 0.5], [0.0, fy, height * 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)

    def camera_extrinsics(self, camera_name: str) -> np.ndarray:
        from pxr import Usd, UsdGeom

        spec = self.cameras[camera_name]
        prim = self.stage.GetPrimAtPath(spec.prim_path)
        mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return np.array(mat, dtype=np.float64).reshape(4, 4)

    # ---- lighting / shadows / visibility -----------------------------------

    def set_dome_hdri(self, hdri_path: str | Path | None, intensity: float | None = None, rotation_deg: float | None = None) -> None:
        """Swap the texture / intensity / yaw of the existing ``/World/background``."""

        from pxr import Gf, Sdf, UsdGeom, UsdLux

        prim = self.stage.GetPrimAtPath(self._dome_prim_path)
        if not prim.IsValid():
            # Background cfg wasn't supplied for this task — create one on demand.
            UsdLux.DomeLight.Define(self.stage, self._dome_prim_path)
            prim = self.stage.GetPrimAtPath(self._dome_prim_path)
        light = UsdLux.DomeLight(prim)
        if hdri_path is not None:
            tex_attr = prim.GetAttribute("inputs:texture:file")
            if not tex_attr.IsValid():
                tex_attr = prim.CreateAttribute("inputs:texture:file", Sdf.ValueTypeNames.Asset)
            tex_attr.Set(str(Path(hdri_path).expanduser().resolve()))
        if intensity is not None:
            light.CreateIntensityAttr(float(intensity))
        if rotation_deg is not None:
            xf = UsdGeom.Xformable(prim)
            _set_xform_op(xf, UsdGeom.XformOp.TypeRotateXYZ, (0.0, 0.0, float(rotation_deg)), double=False)

    def set_distant_light(
        self,
        prim_path: str = "/World/HarmonizerSun",
        intensity: float | None = 3000.0,
        angle_deg: float = 2.0,
        direction: tuple[float, float, float] = (0.3, 0.4, -1.0),
    ) -> None:
        """Create or update a directional sun light for visible cast shadows.

        Dome HDRIs alone produce soft ambient illumination, so shadowLink
        excludes barely change the receiver. A directional sun gives crisp
        cast shadows that disappear sharply when foreground is excluded from
        the shadow-caster collection — which is what the shadow-simulation
        component needs.

        Pass ``intensity=None`` to delete the prim.
        """

        from pxr import Gf, UsdGeom, UsdLux

        if intensity is None:
            self.stage.RemovePrim(prim_path)
            return
        light = UsdLux.DistantLight.Define(self.stage, prim_path)
        light.CreateIntensityAttr(float(intensity))
        light.CreateAngleAttr(float(angle_deg))
        quat = _lookat_quaternion((0.0, 0.0, 0.0), tuple(direction))
        xf = UsdGeom.Xformable(light.GetPrim())
        _set_orient_quat(xf, quat)

    def set_path_tracing(self, enabled: bool, spp: int = 64) -> None:
        import carb

        settings = carb.settings.get_settings()
        if enabled:
            settings.set("/rtx/rendermode", "PathTracing")
            settings.set("/rtx/pathtracing/spp", int(spp))
            settings.set("/rtx/pathtracing/totalSpp", int(spp))
        else:
            settings.set("/rtx/rendermode", "RayTracedLighting")
            settings.set("/rtx/pathtracing/spp", 1)
            settings.set("/rtx/pathtracing/totalSpp", 1)

    def set_shadows_enabled(self, enabled: bool) -> None:
        import carb
        from pxr import Sdf, UsdLux

        settings = carb.settings.get_settings()
        for key in (
            "/rtx/shadows/enabled",
            "/rtx/directLighting/shadows/enabled",
            "/rtx/raytracing/shadows/enabled",
            "/persistent/rtx/shadows/enabled",
        ):
            try:
                settings.set(key, bool(enabled))
            except Exception:
                pass
        light_types = (UsdLux.DomeLight, UsdLux.DistantLight, UsdLux.SphereLight, UsdLux.RectLight, UsdLux.DiskLight)
        for prim in self.stage.Traverse():
            if not prim.HasAPI(UsdLux.LightAPI) and not any(prim.IsA(t) for t in light_types):
                continue
            for attr_name in ("inputs:shadow:enable", "inputs:shadow:enabled", "inputs:enableShadows"):
                attr = prim.GetAttribute(attr_name)
                if not attr.IsValid():
                    attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Bool)
                attr.Set(bool(enabled))

    def set_shadow_link_excludes(self, prim_paths: list[str], enabled: bool = True) -> list[str]:
        """Exclude prims from cast-shadow generation while keeping them lit."""

        from pxr import Sdf, UsdLux

        resolved = self._resolve_prim_prefixes(prim_paths)
        light_types = (UsdLux.DomeLight, UsdLux.DistantLight, UsdLux.SphereLight, UsdLux.RectLight, UsdLux.DiskLight)
        for prim in self.stage.Traverse():
            if not prim.HasAPI(UsdLux.LightAPI) and not any(prim.IsA(t) for t in light_types):
                continue
            prim.CreateAttribute("collection:shadowLink:includeRoot", Sdf.ValueTypeNames.Bool, custom=False).Set(True)
            prim.CreateAttribute("collection:shadowLink:expansionRule", Sdf.ValueTypeNames.Token, custom=False).Set("expandPrims")
            rel = prim.CreateRelationship("collection:shadowLink:excludes", custom=False)
            if enabled and resolved:
                rel.SetTargets([Sdf.Path(p) for p in resolved])
            else:
                rel.ClearTargets(True)
        return resolved

    def set_prims_visibility(self, prim_paths: list[str], visible: bool) -> None:
        from pxr import UsdGeom

        for prim_path in self._resolve_prim_prefixes(prim_paths):
            prim = self.stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue
            imageable = UsdGeom.Imageable(prim)
            if visible:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()

    def _resolve_prim_prefixes(self, prim_paths: list[str]) -> list[str]:
        resolved: list[str] = []
        for prim_path in prim_paths:
            prim = self.stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                resolved.append(str(prim.GetPath()))
                continue
            for candidate in self.stage.Traverse():
                path = str(candidate.GetPath())
                if path.startswith(prim_path.rstrip("/") + "/"):
                    resolved.append(path)
        return sorted(set(resolved))

    # ---- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        try:
            self._end_episode(self.env)
        except Exception:
            pass
        try:
            self.env.close()
        except Exception:
            pass


def _looks_like_receiver(name: str) -> bool:
    low = name.lower()
    return any(hint in low for hint in _RECEIVER_NAME_HINTS)


def _ensure_camera_data_types(env_cfg, *, depth: bool = False, normals: bool = False) -> list[str]:
    """Add ``distance_to_image_plane`` / ``normals`` to every Camera in env_cfg.scene.

    Returns the list of camera attribute names that were updated. Walks
    ``env_cfg.scene`` looking for any cfg with a ``data_types`` field
    (matches both ``CameraCfg`` and ``TiledCameraCfg``). Idempotent —
    skips cameras that already include the requested types.
    """

    extra: list[str] = []
    if depth:
        extra.append("distance_to_image_plane")
    if normals:
        extra.append("normals")
    if not extra:
        return []

    scene = getattr(env_cfg, "scene", None)
    if scene is None:
        return []

    updated: list[str] = []
    for attr_name in dir(scene):
        if attr_name.startswith("_"):
            continue
        try:
            attr = getattr(scene, attr_name)
        except AttributeError:
            continue
        data_types = getattr(attr, "data_types", None)
        if not isinstance(data_types, (list, tuple)):
            continue
        new_types = list(data_types)
        for t in extra:
            if t not in new_types:
                new_types.append(t)
        if new_types != list(data_types):
            try:
                attr.data_types = new_types
                updated.append(attr_name)
            except (AttributeError, TypeError):
                pass
    return updated


def _shrink_gpu_physx_buffers(env_cfg, scale: float) -> None:
    """Reduce Isaac Lab's GPU PhysX buffer sizes for single-env data gen.

    Isaac Lab's defaults assume parallel training with hundreds of envs.
    On an 8 GB card running a single env those buffers — 1 GB collision
    stack, 1 GB heap, 1 GB temp, 8 M contacts — push PhysX into OOM long
    before rendering even starts. We scale them down (default 0.1×) but
    floor each to a value that still works for typical manipulation scenes.
    """

    physx = getattr(getattr(env_cfg, "sim", None), "physx", None)
    if physx is None:
        return
    floors_and_defaults = {
        "gpu_max_rigid_contact_count":            ( 524_288, 8_388_608),
        "gpu_max_rigid_patch_count":              (  32_768,   163_840),
        "gpu_found_lost_pairs_capacity":          ( 131_072, 2_097_152),
        "gpu_found_lost_aggregate_pairs_capacity":( 524_288, 33_554_432),
        "gpu_total_aggregate_pairs_capacity":     ( 131_072, 2_097_152),
        "gpu_collision_stack_size":               (67_108_864, 1_073_741_824),  # 64 MB floor, 1 GB default
        "gpu_heap_capacity":                      (67_108_864, 1_073_741_824),
        "gpu_temp_buffer_capacity":               (67_108_864, 1_073_741_824),
        "gpu_max_soft_body_contacts":             (  65_536, 1_048_576),
        "gpu_max_particle_contacts":              (  65_536, 1_048_576),
    }
    for field, (floor, default) in floors_and_defaults.items():
        current = getattr(physx, field, default)
        scaled = max(int(floor), int(current * scale))
        try:
            setattr(physx, field, scaled)
        except Exception:
            pass


def _set_xform_op(xf, op_type, value, *, double: bool) -> None:
    from pxr import Gf, UsdGeom

    existing = next((op for op in xf.GetOrderedXformOps() if op.GetOpType() == op_type), None)
    if existing is not None:
        if existing.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
            existing.Set(Gf.Vec3d(*[float(c) for c in value]))
        else:
            existing.Set(Gf.Vec3f(*[float(c) for c in value]))
        return
    if op_type == UsdGeom.XformOp.TypeTranslate:
        precision = UsdGeom.XformOp.PrecisionDouble if double else UsdGeom.XformOp.PrecisionFloat
        op = xf.AddTranslateOp(precision)
    elif op_type == UsdGeom.XformOp.TypeRotateXYZ:
        op = xf.AddRotateXYZOp(UsdGeom.XformOp.PrecisionFloat)
    elif op_type == UsdGeom.XformOp.TypeScale:
        op = xf.AddScaleOp(UsdGeom.XformOp.PrecisionFloat)
    else:
        return
    if double or op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
        op.Set(Gf.Vec3d(*[float(c) for c in value]))
    else:
        op.Set(Gf.Vec3f(*[float(c) for c in value]))


def _set_orient_quat(xf, quat: tuple[float, float, float, float]) -> None:
    from pxr import Gf, UsdGeom

    existing = next((op for op in xf.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeOrient), None)
    if existing is None:
        existing = xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat)
    existing.Set(Gf.Quatf(*quat))


def _lookat_quaternion(eye, target, world_up=(0.0, 0.0, 1.0)) -> tuple[float, float, float, float]:
    def normalize(v):
        n = math.sqrt(sum(c * c for c in v))
        return tuple(c / n for c in v) if n > 1e-9 else v

    def cross(a, b):
        return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

    fwd = normalize(tuple(t - e for t, e in zip(target, eye)))
    right = normalize(cross(fwd, world_up))
    if sum(c * c for c in right) < 1e-9:
        right = (1.0, 0.0, 0.0)
    up = cross(right, fwd)
    lz = tuple(-f for f in fwd)
    m00, m01, m02 = right[0], up[0], lz[0]
    m10, m11, m12 = right[1], up[1], lz[1]
    m20, m21, m22 = right[2], up[2], lz[2]
    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w, x, y, z = 0.25 / s, (m21 - m12) * s, (m02 - m20) * s, (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * math.sqrt(max(0.0, 1.0 + m00 - m11 - m22))
        w, x, y, z = (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * math.sqrt(max(0.0, 1.0 + m11 - m00 - m22))
        w, x, y, z = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
    else:
        s = 2.0 * math.sqrt(max(0.0, 1.0 + m22 - m00 - m11))
        w, x, y, z = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return (w / n, x / n, y / n, z / n)


def _rgb_to_uint8(data: Any) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr.astype(np.uint8, copy=False)


def _segmentation_arrays(data: Any) -> tuple[np.ndarray, dict[int, str]]:
    if isinstance(data, dict):
        mask = np.asarray(data.get("data"), dtype=np.int32)
        info = data.get("info", {})
    else:
        mask = np.asarray(data, dtype=np.int32)
        info = {}
    raw_mapping = info.get("idToLabels") or info.get("idToSemantics") or info.get("idToPrimPaths") or {}
    mapping: dict[int, str] = {}
    for key, value in raw_mapping.items():
        try:
            int_key = int(key)
        except Exception:
            continue
        if isinstance(value, dict):
            mapping[int_key] = str(value.get("primPath") or value.get("class") or value)
        else:
            mapping[int_key] = str(value)
    return mask, mapping
