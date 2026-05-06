#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# End-to-end pipeline for a single DL3DV-Benchmark scene:
#
#   1. Download <hash>'s sparse + chosen image variant
#   2. Train fast-pgsr → cluster-filtered TSDF mesh + point_cloud.ply
#   3. Gravity-align + metric-scale the mesh
#   4. mesh_aligned.ply → mesh_aligned.usd  (Isaac collider)
#   5. point_cloud.ply  → gaussians.usdz   (Isaac visual via 3DGUT)
#   6. Sample placements (camera-aware)
#
# After this lands, render with:
#
#   PYTHONPATH=. python scripts/render_dl3dv_with_robot.py \
#       --task UtensilsInMugTask \
#       --scene-dir <output-dir>/<hash> \
#       --placement-idx 0 --num-views 12
#
# Usage:
#   bash scripts/build_dl3dv_scene.sh <SCENE_HASH>
#
# Common overrides via env vars:
#   CACHE_DIR        (default: data/dl3dv_backgrounds/cache)
#   OUTPUT_DIR       (default: data/dl3dv_backgrounds/scenes)
#   FASTPGSR_REPO    (default: third_party/FastGS-pgsr)
#   THREEDGRUT_REPO  (default: third_party/3dgrut)
#   ITERATIONS       (default: 30000)
#   IMAGES_VARIANT   (default: images_4)
#   N_PLACEMENTS     (default: 12)
#   MAX_CAM_DIST_M   (default: 1.5)  # camera-distance bias for placements
#   SKIP_DOWNLOAD    (default: 0)    # set to 1 to skip the HF download step
#   SKIP_TRAIN       (default: 0)    # set to 1 to skip fast-pgsr training
#   SKIP_MESH        (default: 0)    # set to 1 to skip mesh extraction (render.py)
#   SKIP_GS_USDZ     (default: 0)    # set to 1 to skip 3DGUT conversion

set -euo pipefail

SCENE_HASH="${1:?Usage: $0 <scene_hash>}"

CACHE_DIR="${CACHE_DIR:-data/dl3dv_backgrounds/cache}"
OUTPUT_DIR="${OUTPUT_DIR:-data/dl3dv_backgrounds/scenes}"
FASTPGSR_REPO="${FASTPGSR_REPO:-third_party/FastGS-pgsr}"
THREEDGRUT_REPO="${THREEDGRUT_REPO:-third_party/3dgrut}"
ITERATIONS="${ITERATIONS:-30000}"
IMAGES_VARIANT="${IMAGES_VARIANT:-images_4}"
N_PLACEMENTS="${N_PLACEMENTS:-12}"
MAX_CAM_DIST_M="${MAX_CAM_DIST_M:-1.5}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_MESH="${SKIP_MESH:-0}"
SKIP_GS_USDZ="${SKIP_GS_USDZ:-0}"

SCENE_DIR="${OUTPUT_DIR}/${SCENE_HASH}"

echo "============================================================"
echo "DL3DV scene pipeline"
echo "  hash         : $SCENE_HASH"
echo "  cache_dir    : $CACHE_DIR"
echo "  output_dir   : $OUTPUT_DIR"
echo "  scene_dir    : $SCENE_DIR"
echo "  iterations   : $ITERATIONS"
echo "  images_var   : $IMAGES_VARIANT"
echo "  fastpgsr     : $FASTPGSR_REPO"
echo "  3dgrut       : $THREEDGRUT_REPO"
echo "============================================================"

# ------ 1-3: download + train + mesh + align ------
PREP_FLAGS=(
    --scene-hash "$SCENE_HASH"
    --cache-dir "$CACHE_DIR"
    --output-dir "$OUTPUT_DIR"
    --fastpgsr-repo "$FASTPGSR_REPO"
    --iterations "$ITERATIONS"
    --images-variant "$IMAGES_VARIANT"
)
[[ "$SKIP_DOWNLOAD" == "1" ]] && PREP_FLAGS+=(--skip-download)
[[ "$SKIP_TRAIN"    == "1" ]] && PREP_FLAGS+=(--skip-train)
[[ "$SKIP_MESH"     == "1" ]] && PREP_FLAGS+=(--skip-mesh)

echo
echo "[1/5] === download + train + mesh + align ==="
python scripts/prepare_dl3dv_scene.py "${PREP_FLAGS[@]}"

# ------ 4: alignment viz (top-down + side PNGs) ------
echo
echo "[2/6] === alignment viz (viz_topdown.png + viz_side.png) ==="
python scripts/visualize_dl3dv_alignment.py --scene-dir "$SCENE_DIR" || \
    echo "[build] viz step failed (non-fatal); continuing"

# ------ 5: mesh_aligned.ply -> mesh_aligned.usd ------
echo
echo "[3/6] === mesh_aligned.ply -> mesh_aligned.usd ==="
python scripts/dl3dv_mesh_to_usd.py --scene-dir "$SCENE_DIR"

# ------ 6: point_cloud.ply -> gaussians.usdz (3DGUT) ------
if [[ "$SKIP_GS_USDZ" == "1" ]]; then
    echo
    echo "[4/6] === SKIP_GS_USDZ=1; not running 3DGUT ==="
else
    echo
    echo "[4/6] === point_cloud.ply -> gaussians.usdz (3DGUT) ==="
    python scripts/dl3dv_gs_to_usdz.py \
        --scene-dir "$SCENE_DIR" \
        --threedgrut-repo "$THREEDGRUT_REPO"
fi

# ------ 7: sample placements ------
echo
echo "[5/6] === sample placements (camera-aware) ==="
python scripts/sample_dl3dv_placements.py \
    --scene-dir "$SCENE_DIR" \
    --n "$N_PLACEMENTS" \
    --max-distance-to-camera "$MAX_CAM_DIST_M"

echo
echo "[6/6] === DONE ==="
echo "Scene artifacts under: $SCENE_DIR"
ls -lh "$SCENE_DIR"/*.{ply,usd,usdz,json,png} 2>/dev/null || true

cat <<EOF

Render a robot view at placement 0:

    PYTHONPATH=. python scripts/render_dl3dv_with_robot.py \\
        --task UtensilsInMugTask \\
        --scene-dir $SCENE_DIR \\
        --placement-idx 0 --num-views 12 --spp 32

(use --spp 64+ for cleaner GS output; --spp 8 is fast but noisy)
EOF
