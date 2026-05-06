# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Patch fastgs/FastGS @ fast-pgsr branch render.py for the missing mesh_path.

Upstream's ``render_set`` references ``mesh_path`` without defining it,
crashing right before the mesh save (NameError on the line
``mesh_path_color = os.path.join(mesh_path, "mesh_color.ply")``).

This patcher adds the missing definition derived from ``render_path``:

    mesh_path = os.path.join(os.path.dirname(render_path), "mesh")
    os.makedirs(mesh_path, exist_ok=True)

inserted just above the TSDF fusion block. Idempotent: re-running is a
no-op once patched. Writes a ``.bak`` next to the patched file.

Usage::

    python scripts/patch_fastpgsr_render.py --repo third_party/FastGS-pgsr
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ANCHOR = "if volume is not None:"
PATCH_LINES = (
    "    mesh_path = os.path.join(os.path.dirname(render_path), \"mesh\")\n"
    "    os.makedirs(mesh_path, exist_ok=True)\n"
)
PATCH_MARKER = "mesh_path = os.path.join(os.path.dirname(render_path), \"mesh\")"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, default=Path("third_party/FastGS-pgsr"))
    args = p.parse_args()

    target = args.repo / "render.py"
    if not target.is_file():
        print(f"ERROR: {target} not found", file=sys.stderr)
        return 1

    src = target.read_text()
    if PATCH_MARKER in src:
        print(f"[patch] {target}: already patched, no-op")
        return 0

    if ANCHOR not in src:
        print(
            f"ERROR: anchor {ANCHOR!r} not found in {target}; "
            f"upstream layout may have changed",
            file=sys.stderr,
        )
        return 2

    backup = target.with_suffix(".py.bak")
    if not backup.exists():
        shutil.copy(target, backup)
        print(f"[patch] backed up {target} -> {backup}")

    # Insert the patch lines on the line BEFORE the anchor so the make-dirs
    # runs whether or not we enter the if-block. Find the start of the line
    # containing the anchor and inject.
    idx = src.index(ANCHOR)
    line_start = src.rfind("\n", 0, idx) + 1  # start of anchor line
    patched = src[:line_start] + PATCH_LINES + src[line_start:]

    target.write_text(patched)
    print(f"[patch] inserted mesh_path definition into {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
