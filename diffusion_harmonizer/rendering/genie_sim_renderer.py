from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class CameraSpec:
    prim_path: str
    resolution: tuple[int, int]
    focal_length: float
    horizontal_aperture: float
    vertical_aperture: float


def launch_renderer(headless: bool = True, renderer_type: str = "raytraced") -> "GenieSimRenderer":
    """Launch Isaac Sim first, then construct the renderer wrapper.

    All `isaacsim`, `omni`, and `pxr` imports used by the renderer happen after
    this function creates `SimulationApp`.
    """

    from isaacsim import SimulationApp

    kit_renderer = "RayTracedLighting"
    if renderer_type.lower() in {"pathtraced", "path_traced", "realtimepathtracing"}:
        kit_renderer = "RealTimePathTracing"

    simulation_app = SimulationApp({"headless": headless, "renderer": kit_renderer})
    simulation_app._carb_settings.set("/omni/replicator/asyncRendering", False)
    simulation_app._carb_settings.set("/app/asyncRendering", False)
    return GenieSimRenderer(simulation_app, renderer_type=renderer_type)


class GenieSimRenderer:
    """Isaac Sim + Replicator renderer for paired DiffusionHarmonizer data."""

    def __init__(self, simulation_app: Any, renderer_type: str = "raytraced"):
        from isaacsim.core.api import World
        import omni.replicator.core as rep
        import omni.usd

        self._app = simulation_app
        self._rep = rep
        self._omni_usd = omni.usd
        self.world = World(stage_units_in_meters=1.0)
        self.stage = self.world.stage
        self.renderer_type = renderer_type
        self.render_products: dict[str, Any] = {}
        self.annotators: dict[str, dict[str, Any]] = {}
        self.cameras: dict[str, CameraSpec] = {}
        self._configure_rtx(renderer_type)

    def _settings(self):
        import carb

        return carb.settings.get_settings()

    def _configure_rtx(self, renderer_type: str) -> None:
        renderer_type = renderer_type.lower()
        if renderer_type in {"pathtraced", "path_traced", "pt"}:
            self.set_path_tracing(True, spp=64)
        else:
            self.set_path_tracing(False)

    def open_scene(self, usd_path: str) -> None:
        """Open an existing USD scene file."""

        from isaacsim.core.api import World

        usd = str(Path(usd_path).expanduser().resolve())
        self._omni_usd.get_context().open_stage(usd)
        self.stage = self._omni_usd.get_context().get_stage()
        self.world = World(stage_units_in_meters=1.0)
        self.world.reset()

    def reference_asset(
        self,
        usd_path: str,
        prim_path: str,
        translate=(0, 0, 0),
        rotate=(0, 0, 0),
        scale=(1, 1, 1),
    ) -> None:
        """Add a USD reference at `prim_path` with a transform."""

        from pxr import Gf, UsdGeom

        self._ensure_world()
        asset_path = Path(usd_path).expanduser().resolve()
        if not asset_path.exists():
            raise FileNotFoundError(f"USD asset does not exist: {asset_path}")

        prim = UsdGeom.Xform.Define(self.stage, prim_path).GetPrim()
        prim.GetReferences().ClearReferences()
        if not prim.GetReferences().AddReference(str(asset_path)):
            raise RuntimeError(f"Failed to add USD reference {asset_path} at {prim_path}")
        xf = UsdGeom.Xformable(prim)
        self._clear_xform_ops(xf)
        xf.AddTranslateOp().Set(Gf.Vec3d(*translate))
        rx, ry, rz = rotate
        xf.AddRotateXYZOp().Set(Gf.Vec3f(rx, ry, rz))
        xf.AddScaleOp().Set(Gf.Vec3f(*scale))

    def set_prim_transform(
        self,
        prim_path: str,
        translate=(0, 0, 0),
        rotate=(0, 0, 0),
        scale=(1, 1, 1),
    ) -> None:
        from pxr import Gf, UsdGeom

        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return
        xf = UsdGeom.Xformable(prim)
        self._clear_xform_ops(xf)
        xf.AddTranslateOp().Set(Gf.Vec3d(*translate))
        xf.AddRotateXYZOp().Set(Gf.Vec3f(*rotate))
        xf.AddScaleOp().Set(Gf.Vec3f(*scale))

    def align_prim_bottom_to_z(self, prim_path: str, target_z: float) -> float | None:
        """Shift a prim so its world-space bounding-box bottom sits on `target_z`."""

        from pxr import Gf, Usd, UsdGeom

        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return None
        try:
            self._app.update()
        except Exception:
            pass
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=True,
        )
        aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        min_z = float(aligned.GetMin()[2])
        delta_z = float(target_z) - min_z
        if abs(delta_z) < 1e-5:
            return delta_z

        xf = UsdGeom.Xformable(prim)
        translate_op = None
        for op in xf.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break
        if translate_op is None:
            translate_op = xf.AddTranslateOp()
            translate = Gf.Vec3d(0.0, 0.0, delta_z)
        else:
            current = translate_op.Get() or Gf.Vec3d(0.0, 0.0, 0.0)
            translate = Gf.Vec3d(float(current[0]), float(current[1]), float(current[2]) + delta_z)
        translate_op.Set(translate)
        return delta_z

    def prim_bbox_extent(self, prim_path: str) -> tuple[float, float, float] | None:
        """Return world-space aligned bounding-box extent for a prim."""

        from pxr import Usd, UsdGeom

        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return None
        try:
            self._app.update()
        except Exception:
            pass
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=True,
        )
        aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        extent = aligned.GetMax() - aligned.GetMin()
        return (float(extent[0]), float(extent[1]), float(extent[2]))

    def fit_prim_max_extent(self, prim_path: str, target_max_extent: float, min_scale: float = 0.05, max_scale: float = 20.0) -> float | None:
        """Uniformly scale a prim so its max bbox dimension reaches target size."""

        from pxr import Gf, UsdGeom

        extent = self.prim_bbox_extent(prim_path)
        if not extent:
            return None
        current = max(extent)
        if current <= 1e-6:
            return None
        factor = max(min(float(target_max_extent) / current, max_scale), min_scale)
        prim = self.stage.GetPrimAtPath(prim_path)
        xf = UsdGeom.Xformable(prim)
        scale_op = None
        for op in xf.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                scale_op = op
                break
        if scale_op is None:
            scale_op = xf.AddScaleOp()
            scale = Gf.Vec3f(factor, factor, factor)
        else:
            current_scale = scale_op.Get() or Gf.Vec3f(1.0, 1.0, 1.0)
            scale = Gf.Vec3f(float(current_scale[0]) * factor, float(current_scale[1]) * factor, float(current_scale[2]) * factor)
        scale_op.Set(scale)
        return factor

    def set_preview_surface_material(
        self,
        shader_path: str,
        diffuse_color: tuple[float, float, float],
        roughness: float | None = None,
    ) -> None:
        """Update a UsdPreviewSurface shader used by procedural scene geometry."""

        from pxr import Gf, Sdf, UsdShade

        prim = self.stage.GetPrimAtPath(shader_path)
        shader = UsdShade.Shader(prim) if prim.IsValid() else UsdShade.Shader.Define(self.stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(c) for c in diffuse_color]))
        if roughness is not None:
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))

    def set_prim_display_color(self, prim_path: str, color: tuple[float, float, float]) -> None:
        """Set displayColor as a fallback for renderers/materials that ignore PreviewSurface."""

        from pxr import Gf, UsdGeom

        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return
        gprim = UsdGeom.Gprim(prim)
        if not gprim:
            return
        gprim.CreateDisplayColorAttr([Gf.Vec3f(*[float(c) for c in color])])

    def set_articulation_joint_positions(self, prim_path: str, joint_positions: dict[str, float]) -> dict[str, float]:
        """Apply joint positions in radians, returning the joints that were applied.

        Isaac's articulation API is preferred. The USD DriveAPI fallback exists so
        visual-only robot USDs can still receive target poses when possible.
        """

        applied = self._set_articulation_positions(prim_path, joint_positions)
        if applied:
            self.step(2)
            return applied
        applied = self._set_joint_drive_targets(prim_path, joint_positions)
        if applied:
            self.step(2)
        return applied

    def set_dome_light(self, hdri_path: str | None, intensity: float = 1000.0, rotation_deg: float = 0.0) -> None:
        """Create or update `/World/DomeLight`."""

        from pxr import Gf, Sdf, UsdLux, UsdGeom

        self._ensure_world()
        path = "/World/DomeLight"
        if hdri_path is None:
            self.stage.RemovePrim(path)
            return
        light = UsdLux.DomeLight.Define(self.stage, path)
        light.CreateIntensityAttr(float(intensity))
        texture_attr = light.GetPrim().CreateAttribute("inputs:texture:file", Sdf.ValueTypeNames.Asset)
        texture_attr.Set(str(Path(hdri_path).expanduser().resolve()))
        xf = UsdGeom.Xformable(light.GetPrim())
        self._clear_xform_ops(xf)
        xf.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, float(rotation_deg)))

    def set_distant_light(
        self,
        prim_path: str = "/World/Sun",
        intensity: float | None = 3000.0,
        angle_deg: float = 0.5,
        direction=(0, 0, -1),
    ) -> None:
        """Create/update a DistantLight; pass `intensity=None` to delete it."""

        from pxr import Gf, UsdGeom, UsdLux

        self._ensure_world()
        if intensity is None:
            self.stage.RemovePrim(prim_path)
            return
        light = UsdLux.DistantLight.Define(self.stage, prim_path)
        light.CreateIntensityAttr(float(intensity))
        light.CreateAngleAttr(float(angle_deg))
        quat = self._lookat_quaternion((0.0, 0.0, 0.0), tuple(direction))
        xf = UsdGeom.Xformable(light.GetPrim())
        self._clear_xform_ops(xf)
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(*quat))

    def set_path_tracing(self, enabled: bool, spp: int = 64) -> None:
        """Toggle RTX path tracing and samples-per-pixel."""

        settings = self._settings()
        if enabled:
            settings.set("/rtx/rendermode", "PathTracing")
            settings.set("/rtx/pathtracing/spp", int(spp))
            settings.set("/rtx/pathtracing/totalSpp", int(spp))
        else:
            settings.set("/rtx/rendermode", "RayTracedLighting")
            settings.set("/rtx/pathtracing/spp", 1)
            settings.set("/rtx/pathtracing/totalSpp", 1)

    def set_shadows_enabled(self, enabled: bool) -> None:
        """Best-effort shadow toggle for Isaac Sim / Kit builds."""

        from pxr import Sdf, UsdLux

        settings = self._settings()
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
            if not prim.HasAPI(UsdLux.LightAPI) and not any(prim.IsA(light_type) for light_type in light_types):
                continue
            for attr_name in (
                "inputs:shadow:enable",
                "inputs:shadow:enabled",
                "inputs:enableShadows",
            ):
                attr = prim.GetAttribute(attr_name)
                if not attr.IsValid():
                    attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Bool)
                attr.Set(bool(enabled))

    def set_shadow_link_excludes(self, prim_paths: list[str], enabled: bool = True) -> list[str]:
        """Exclude prims from casting shadows while keeping their illumination.

        This uses USD/UsdLux shadow-linking. With includeRoot=True, all scene
        geometry casts shadows by default; collection:shadowLink:excludes
        subtracts foreground robot/objects from the shadow-caster collection for
        every light. The foreground remains visible and lit.
        """

        from pxr import Sdf, UsdLux

        resolved = self._resolve_prim_paths_or_prefixes(prim_paths)
        light_types = (UsdLux.DomeLight, UsdLux.DistantLight, UsdLux.SphereLight, UsdLux.RectLight, UsdLux.DiskLight)
        for prim in self.stage.Traverse():
            if not prim.HasAPI(UsdLux.LightAPI) and not any(prim.IsA(light_type) for light_type in light_types):
                continue
            prim.CreateAttribute("collection:shadowLink:includeRoot", Sdf.ValueTypeNames.Bool, custom=False).Set(True)
            prim.CreateAttribute("collection:shadowLink:expansionRule", Sdf.ValueTypeNames.Token, custom=False).Set("expandPrims")
            rel = prim.CreateRelationship("collection:shadowLink:excludes", custom=False)
            if enabled and resolved:
                rel.SetTargets([Sdf.Path(path) for path in resolved])
            else:
                rel.ClearTargets(True)
        return resolved

    def set_prim_visibility(self, prim_path: str, visible: bool) -> None:
        """Set USD visibility on a prim root.

        Used by paired-data components to render receiver-only passes without
        robot/object cast shadows while preserving the same camera and lighting.
        """

        from pxr import UsdGeom

        prims = [self.stage.GetPrimAtPath(path) for path in self._resolve_prim_paths_or_prefixes([prim_path])]
        for candidate in prims:
            imageable = UsdGeom.Imageable(candidate)
            if visible:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()

    def set_prims_visibility(self, prim_paths: list[str], visible: bool) -> None:
        for prim_path in prim_paths:
            self.set_prim_visibility(prim_path, visible)

    def add_camera(
        self,
        name: str,
        position,
        look_at,
        resolution=(640, 480),
        focal_length: float = 24.0,
        horizontal_aperture: float = 20.955,
    ) -> None:
        """Create render product and rgb/depth/instance/normals annotators."""

        from pxr import Gf, UsdGeom

        self._ensure_world()
        UsdGeom.Xform.Define(self.stage, "/World/Cameras")
        width, height = int(resolution[0]), int(resolution[1])
        vertical_aperture = horizontal_aperture * height / width
        prim_path = name if name.startswith("/") else f"/World/Cameras/{name}"
        camera = UsdGeom.Camera.Define(self.stage, prim_path)
        camera.CreateFocalLengthAttr(float(focal_length))
        camera.CreateHorizontalApertureAttr(float(horizontal_aperture))
        camera.CreateVerticalApertureAttr(float(vertical_aperture))
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
        quat = self._lookat_quaternion(tuple(position), tuple(look_at))
        xf = UsdGeom.Xformable(camera.GetPrim())
        self._clear_xform_ops(xf)
        xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*position))
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(*quat))

        render_product = self._rep.create.render_product(prim_path, (width, height))
        annotators = {}
        for modality, annotator_name in {
            "rgb": "rgb",
            "depth": "distance_to_camera",
            "segmentation": "instance_segmentation",
            "normal": "normals",
        }.items():
            annotator = self._rep.AnnotatorRegistry.get_annotator(annotator_name)
            annotator.attach([render_product])
            annotators[modality] = annotator

        self.render_products[name] = render_product
        self.annotators[name] = annotators
        self.cameras[name] = CameraSpec(
            prim_path=prim_path,
            resolution=(width, height),
            focal_length=float(focal_length),
            horizontal_aperture=float(horizontal_aperture),
            vertical_aperture=float(vertical_aperture),
        )

    def add_orbit_cameras(
        self,
        name_prefix,
        center,
        radius,
        height,
        num_cameras,
        resolution=(512, 512),
    ) -> list[str]:
        names = []
        cx, cy, cz = center
        for idx in range(num_cameras):
            theta = 2.0 * math.pi * idx / max(num_cameras, 1)
            pos = (cx + radius * math.cos(theta), cy + radius * math.sin(theta), cz + height)
            name = f"{name_prefix}_{idx:03d}"
            self.add_camera(name, pos, center, resolution=resolution)
            names.append(name)
        return names

    def add_sphere_cameras(
        self,
        name_prefix,
        center,
        radius,
        num_cameras: int = 100,
        resolution=(512, 512),
    ) -> list[str]:
        """Fibonacci sphere cameras for privileged static 3DGS capture."""

        names = []
        cx, cy, cz = center
        golden = math.pi * (3.0 - math.sqrt(5.0))
        for idx in range(num_cameras):
            y = 1.0 - (idx / max(num_cameras - 1, 1)) * 2.0
            r = math.sqrt(max(0.0, 1.0 - y * y))
            theta = golden * idx
            pos = (
                cx + radius * math.cos(theta) * r,
                cy + radius * math.sin(theta) * r,
                cz + radius * y,
            )
            name = f"{name_prefix}_{idx:03d}"
            self.add_camera(name, pos, center, resolution=resolution)
            names.append(name)
        return names

    def capture_frame(
        self,
        camera_name: str,
        rgb: bool = True,
        depth: bool = False,
        segmentation: bool = False,
        normal: bool = False,
    ) -> dict[str, Any]:
        """Step Replicator once and return requested modalities plus camera matrices."""

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
            out["rgb"] = self._rgb_array(ann["rgb"].get_data())
        if depth:
            out["depth"] = np.asarray(ann["depth"].get_data(), dtype=np.float32)
        if segmentation:
            data = ann["segmentation"].get_data()
            out["segmentation"], out["segmentation_mapping"] = self._segmentation_arrays(data)
        if normal:
            out["normal"] = np.asarray(ann["normal"].get_data(), dtype=np.float32)[..., :3]
        return out

    def capture_multiview(self, camera_names, **kw) -> list[dict[str, Any]]:
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

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self.world.step(render=True)

    def shutdown(self) -> None:
        self._app.close()

    def _set_articulation_positions(self, prim_path: str, joint_positions: dict[str, float]) -> dict[str, float]:
        try:
            from isaacsim.core.prims import SingleArticulation
        except Exception:
            return {}
        try:
            articulation = SingleArticulation(prim_path=prim_path, name=prim_path.strip("/").replace("/", "_"))
            articulation.initialize()
            dof_names = list(articulation.dof_names)
            positions = articulation.get_joint_positions()
            if positions is None:
                positions = np.zeros(len(dof_names), dtype=np.float32)
            else:
                positions = np.asarray(positions, dtype=np.float32)
            applied = {}
            for name, value in joint_positions.items():
                if name in dof_names:
                    positions[dof_names.index(name)] = float(value)
                    applied[name] = float(value)
            if applied:
                articulation.set_joint_positions(positions)
            return applied
        except Exception:
            return {}

    def _set_joint_drive_targets(self, prim_path: str, joint_positions: dict[str, float]) -> dict[str, float]:
        from pxr import UsdPhysics

        root = self.stage.GetPrimAtPath(prim_path)
        if not root.IsValid():
            return {}
        wanted = {name.lower(): value for name, value in joint_positions.items()}
        applied = {}
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            if not path.startswith(prim_path.rstrip("/") + "/"):
                continue
            name = prim.GetName().lower()
            if name not in wanted:
                continue
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if not drive:
                drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            # USD angular drive target positions are authored in degrees.
            drive.CreateTargetPositionAttr(float(np.degrees(wanted[name])))
            applied[prim.GetName()] = float(wanted[name])
        return applied

    @staticmethod
    def _clear_xform_ops(xf) -> None:
        try:
            xf.ClearXformOpOrder()
        except Exception:
            pass

    def _ensure_world(self) -> None:
        from pxr import UsdGeom

        UsdGeom.Xform.Define(self.stage, "/World")

    def _resolve_prim_paths_or_prefixes(self, prim_paths: list[str]) -> list[str]:
        resolved: list[str] = []
        for prim_path in prim_paths:
            prim = self.stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                resolved.append(str(prim.GetPath()))
                continue
            matches = [str(candidate.GetPath()) for candidate in self.stage.Traverse() if str(candidate.GetPath()).startswith(prim_path)]
            resolved.extend(matches)
        return sorted(set(resolved))

    @staticmethod
    def _rgb_array(data: Any) -> np.ndarray:
        arr = np.asarray(data)
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        return arr.astype(np.uint8, copy=False)

    @staticmethod
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

    @staticmethod
    def _lookat_quaternion(eye, target, world_up=(0.0, 0.0, 1.0)):
        def normalize(v):
            n = math.sqrt(sum(c * c for c in v))
            return tuple(c / n for c in v) if n > 1e-9 else v

        def cross(a, b):
            return (
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            )

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
            w = 0.25 / s
            x = (m21 - m12) * s
            y = (m02 - m20) * s
            z = (m10 - m01) * s
        elif m00 > m11 and m00 > m22:
            s = 2.0 * math.sqrt(max(0.0, 1.0 + m00 - m11 - m22))
            w = (m21 - m12) / s
            x = 0.25 * s
            y = (m01 + m10) / s
            z = (m02 + m20) / s
        elif m11 > m22:
            s = 2.0 * math.sqrt(max(0.0, 1.0 + m11 - m00 - m22))
            w = (m02 - m20) / s
            x = (m01 + m10) / s
            y = 0.25 * s
            z = (m12 + m21) / s
        else:
            s = 2.0 * math.sqrt(max(0.0, 1.0 + m22 - m00 - m11))
            w = (m10 - m01) / s
            x = (m02 + m20) / s
            y = (m12 + m21) / s
            z = 0.25 * s
        n = math.sqrt(w * w + x * x + y * y + z * z)
        return (w / n, x / n, y / n, z / n)
