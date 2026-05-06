# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Convert ``mesh_aligned.ply`` to a USD that Isaac Sim can reference.

Authors a single ``UsdGeom.Mesh`` prim under ``/World/DL3DVScene`` with
the mesh's vertex positions, vertex colors (if present), and triangle
faces. Writes ``mesh_aligned.usd`` next to the input PLY.

This is the bridge between the prep pipeline (PLY-based) and the Isaac
compositing step which uses ``runtime.set_background_scene(usd_path)``.
The mesh is gravity-aligned + metric + floor-at-z=0 already, so it can
be referenced at the world origin directly.

We do NOT yet convert the GS .ply (Gaussians) to USDZ — that requires
3DGUT and is the next step. The mesh-only USD lets us validate sizing
and placement before bringing the photoreal GS into the loop.

Usage::

    python scripts/dl3dv_mesh_to_usd.py \
        --scene-dir data/dl3dv_backgrounds/scenes/<hash>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load_mesh(mesh_path: Path):
    import trimesh
    mesh = trimesh.load(str(mesh_path), process=False)
    if hasattr(mesh, "dump"):
        try:
            mesh = mesh.dump(concatenate=True)
        except Exception:
            pass
    return mesh


def convert_ply_to_usd(
    ply_path: Path,
    usd_path: Path,
    *,
    prim_path: str = "/World/DL3DVScene",
    crop_below_z_m: float | None = -0.1,
) -> None:
    """Author a USD with one Mesh prim from the PLY's geometry.

    ``crop_below_z_m`` (default −10 cm): drops vertices with z below
    this threshold. Removes the lingering TSDF "shadow" floaters under
    the floor without affecting the actual scene. Set to ``None`` to
    skip cropping.
    """
    try:
        from pxr import Usd, UsdGeom, Vt, Sdf, Gf
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "pxr (USD) required — run inside an Isaac Sim env or install "
            "`pip install usd-core`"
        ) from e

    mesh = _load_mesh(ply_path)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    has_colors = (
        getattr(mesh.visual, "vertex_colors", None) is not None
        and len(mesh.visual.vertex_colors) == len(verts)
    )
    colors = (
        np.asarray(mesh.visual.vertex_colors, dtype=np.float32)[:, :3] / 255.0
        if has_colors else None
    )

    if crop_below_z_m is not None:
        keep = verts[:, 2] >= crop_below_z_m
        if not keep.all():
            # Re-index faces: drop any face that touches a removed vertex
            old_to_new = -np.ones(len(verts), dtype=np.int64)
            old_to_new[keep] = np.arange(int(keep.sum()))
            face_keep = keep[faces].all(axis=1)
            verts = verts[keep]
            faces = old_to_new[faces[face_keep]].astype(np.int32)
            if colors is not None:
                colors = colors[keep]
            print(
                f"[mesh→usd] cropped {int((~keep).sum())} verts below "
                f"z={crop_below_z_m} m (TSDF floaters)"
            )

    print(f"[mesh→usd] writing {len(verts):,} verts, {len(faces):,} faces -> {usd_path}")
    usd_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    UsdGeom.Xform.Define(stage, "/World")
    mesh_prim = UsdGeom.Mesh.Define(stage, prim_path)
    mesh_prim.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts.astype(np.float32)))
    mesh_prim.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(
        np.full(len(faces), 3, dtype=np.int32)
    ))
    mesh_prim.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.flatten()))

    if colors is not None:
        primvars_api = UsdGeom.PrimvarsAPI(mesh_prim)
        cv = primvars_api.CreatePrimvar(
            "displayColor",
            Sdf.ValueTypeNames.Color3fArray,
            UsdGeom.Tokens.vertex,
        )
        cv.Set(Vt.Vec3fArray.FromNumpy(colors.astype(np.float32)))

    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    stage.GetRootLayer().Save()
    print(f"[mesh→usd] OK")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene-dir", type=Path, required=True)
    p.add_argument("--out-name", default="mesh_aligned.usd",
                   help="output filename (placed next to mesh_aligned.ply)")
    p.add_argument("--no-crop", action="store_true",
                   help="don't drop vertices below z=-0.1 (keeps TSDF floaters)")
    args = p.parse_args()

    scene_dir = args.scene_dir.resolve()
    ply = scene_dir / "mesh_aligned.ply"
    if not ply.is_file():
        raise FileNotFoundError(f"missing {ply}")
    out = scene_dir / args.out_name
    convert_ply_to_usd(
        ply, out,
        crop_below_z_m=None if args.no_crop else -0.1,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
