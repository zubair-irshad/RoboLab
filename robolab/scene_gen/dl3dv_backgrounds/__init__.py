# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""DL3DV-Benchmark scenes as photorealistic backgrounds for RoboLab tasks.

Pipeline (Phase 1 = scene prep, before any task wiring):

    download   -> pull a single DL3DV scene (COLMAP layout) from HF Hub
    reconstruct -> run FastGS `fast-pgsr` branch to produce GS + mesh
    align      -> rotate/translate/scale the scene so gravity is -Z and
                  units are roughly metric (floor at z=0)

Phase 2/3 (placement sampling, USD compositing, task integration) live
in sibling modules added later.
"""

from .download import DL3DVSceneRef, download_scene
from .reconstruct import FastPgsrConfig, run_fast_pgsr
from .align import AlignedScene, align_scene_from_mesh

__all__ = [
    "DL3DVSceneRef",
    "download_scene",
    "FastPgsrConfig",
    "run_fast_pgsr",
    "AlignedScene",
    "align_scene_from_mesh",
]
