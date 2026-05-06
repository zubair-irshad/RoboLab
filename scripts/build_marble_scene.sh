#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# End-to-end pipeline for a single Marble / Echo2 Gaussian-splat PLY
# (no COLMAP, no images, no fast-pgsr — just the .ply):
#
#   1. prepare_marble_scene.py        → mesh_aligned.ply + metadata.json
#                                        + source.ply (3DGUT-normalized) + source.ply.original
#   2. visualize_dl3dv_alignment.py   → viz_topdown.png + viz_side.png
#   3. dl3dv_mesh_to_usd.py           → mesh_aligned.usd (depth collider)
#   4. dl3dv_gs_to_usdz.py            → gaussians.usdz   (rgb visual via 3DGUT)
#   5. sample_dl3dv_placements.py     → placements.json + viz_placements.png
#
# After this lands, render with:
#
#   PYTHONPATH=. python scripts/render_dl3dv_with_robot.py \
#       --task UtensilsInMugTask \
#       --scene-dir <output-dir> \
#       --placement-idx 0 --num-views 12 --spp 64
#
# Usage:
#   bash scripts/build_marble_scene.sh <PLY_PATH> [OUTPUT_DIR]
#
# Examples:
#   bash scripts/build_marble_scene.sh data/echo-2/Designer_Bath_Laundry_Nook.ply
#   UP_AXIS=+z CEILING_M=2.4 bash scripts/build_marble_scene.sh foo.ply out/foo
#
# Env-var overrides (with defaults):
#   UP_AXIS          (auto)                 PLY's up axis. 'auto' scans all 6 candidates via RANSAC; or +x/-x/+y/-y/+z/-z to override.
#   GS_SH_MODE       (pad-to-3)             3DGUT requires SH degree 0 or 3. pad-to-3 keeps source's degree-1/2 colour bands; drop = DC only; preserve = passthrough (only valid when source already has 0 or 45 f_rest_*).
#   METRIC_SCALE     (unset → ceiling)      explicit PLY-units → metres scalar
#   CEILING_M        (2.7)                  ceiling-height heuristic target (only when METRIC_SCALE unset)
#   PROXY_METHOD     (poisson)              poisson | alpha | voxel-points
#   VOXEL_M          (0.05)                 voxel downsample size (m) for proxy mesh
#   POISSON_DEPTH    (9)                    octree depth for Poisson reconstruction
#   ALPHA_M          (0.20)                 alpha radius (m) for alpha-shape
#   OPACITY_THRESH   (0.05)                 sigmoid(opacity) cutoff for 3DGS PLYs (0 = no filter)
#   N_PLACEMENTS     (12)                   how many task placements to sample
#   FOOTPRINT_LEN    (1.0)                  task footprint length (m) along +X. Bump to 1.5 for full-size kitchen scenes.
#   FOOTPRINT_WID    (0.7)                  task footprint width  (m) along +Y
#   ROBOT_REACH      (0.55)                 lateral arm reach beyond footprint (m). 0.85 (Franka full reach) usually fails inside small rooms; the placement check is conservative.
#   CLEARANCE_M      (1.8)                  top of robot working volume (m). Walls in [TABLE_HEIGHT_M, CLEARANCE_M] are "high obstacles"; lower this to ignore upper walls/cabinets that the robot won't actually hit.
#   TABLE_HEIGHT_M   (0.75)                 split between low-obstacle and high-obstacle bands (m).
#   FLOOR_CLOSE_M    (1.0)                  morph-closing radius on the floor mask (m). Higher bridges floor gaps where furniture occluded capture.
#   MIN_FLOOR_AREA_M2 (3.0)                  drop floor connected-components smaller than this (m²). Filters RANSAC false-positives in adjacent rooms / on furniture. Set 0 to disable.
#   MIN_FREE_AREA_M2  (1.0)                  drop tiny components from the FINAL free-placement mask. Catches the 'doorway peninsula' failure mode where a passage survives the floor-area filter but yields only a sliver of valid cells. Set 0 to disable.
#   FLOOR_DEFINITION  (occupied-column)      'occupied-column' = floor wherever the column's lowest mesh point is near floor_z (best for synthesis-based PLYs where gaussians cluster on furniture). 'ransac' = only RANSAC inliers count (best when capture has dense bare-floor coverage).
#   FLOOR_Z_TOL_M     (0.15)                 occupied-column tolerance: how far above floor_z the lowest mesh point may sit and still count as floor.
#   THREEDGRUT_REPO  (third_party/3dgrut)   local clone of nv-tlabs/3dgrut
#   THREEDGRUT_ENV   ("")                   conda env for 3DGUT (empty = current env)
#   SKIP_PREP        (0)                    set 1 to skip prepare_marble_scene
#   SKIP_VIZ         (0)                    set 1 to skip viz_topdown / viz_side PNGs
#   SKIP_MESH_USD    (0)                    set 1 to skip mesh_aligned.usd
#   SKIP_GS_USDZ     (0)                    set 1 to skip 3DGUT (gaussians.usdz)
#   SKIP_PLACEMENTS  (0)                    set 1 to skip placement sampling

set -euo pipefail

PLY="${1:?Usage: $0 <ply-file> [output-dir]}"
if [[ ! -f "$PLY" ]]; then
    echo "error: PLY not found: $PLY" >&2
    exit 1
fi

# Default output directory: data/marble_backgrounds/scenes/<basename-no-ext>
DEFAULT_OUT="data/marble_backgrounds/scenes/$(basename "${PLY%.ply}")"
OUT="${2:-$DEFAULT_OUT}"

UP_AXIS="${UP_AXIS:-auto}"
GS_SH_MODE="${GS_SH_MODE:-pad-to-3}"
METRIC_SCALE="${METRIC_SCALE:-}"
CEILING_M="${CEILING_M:-2.7}"
PROXY_METHOD="${PROXY_METHOD:-poisson}"
VOXEL_M="${VOXEL_M:-0.05}"
POISSON_DEPTH="${POISSON_DEPTH:-9}"
ALPHA_M="${ALPHA_M:-0.20}"
OPACITY_THRESH="${OPACITY_THRESH:-0.05}"
N_PLACEMENTS="${N_PLACEMENTS:-12}"
FOOTPRINT_LEN="${FOOTPRINT_LEN:-1.0}"
FOOTPRINT_WID="${FOOTPRINT_WID:-0.7}"
ROBOT_REACH="${ROBOT_REACH:-0.55}"
CLEARANCE_M="${CLEARANCE_M:-1.8}"
TABLE_HEIGHT_M="${TABLE_HEIGHT_M:-0.75}"
FLOOR_CLOSE_M="${FLOOR_CLOSE_M:-1.0}"
MIN_FLOOR_AREA_M2="${MIN_FLOOR_AREA_M2:-3.0}"
MIN_FREE_AREA_M2="${MIN_FREE_AREA_M2:-1.0}"
FLOOR_DEFINITION="${FLOOR_DEFINITION:-occupied-column}"
FLOOR_Z_TOL_M="${FLOOR_Z_TOL_M:-0.15}"
THREEDGRUT_REPO="${THREEDGRUT_REPO:-third_party/3dgrut}"
THREEDGRUT_ENV="${THREEDGRUT_ENV:-}"
SKIP_PREP="${SKIP_PREP:-0}"
SKIP_VIZ="${SKIP_VIZ:-0}"
SKIP_MESH_USD="${SKIP_MESH_USD:-0}"
SKIP_GS_USDZ="${SKIP_GS_USDZ:-0}"
SKIP_PLACEMENTS="${SKIP_PLACEMENTS:-0}"

echo "============================================================"
echo "Marble scene pipeline"
echo "  ply           : $PLY"
echo "  output dir    : $OUT"
echo "  up axis       : $UP_AXIS"
if [[ -n "$METRIC_SCALE" ]]; then
    echo "  metric scale  : $METRIC_SCALE (user)"
else
    echo "  metric scale  : (heuristic, target ceiling = $CEILING_M m)"
fi
echo "  proxy method  : $PROXY_METHOD"
echo "  3dgrut repo   : $THREEDGRUT_REPO"
echo "  placements    : ${N_PLACEMENTS} x footprint ${FOOTPRINT_LEN}x${FOOTPRINT_WID} m, reach ${ROBOT_REACH} m, clearance ${CLEARANCE_M} m"
echo "============================================================"

mkdir -p "$OUT"

# ------ 1: prepare ----------------------------------------------------------
if [[ "$SKIP_PREP" == "1" ]]; then
    echo
    echo "[1/4] === SKIP_PREP=1; not running prepare_marble_scene ==="
else
    echo
    echo "[1/5] === prepare_marble_scene (PLY -> aligned mesh + metadata) ==="
    PREP_FLAGS=(
        --ply "$PLY"
        --out-dir "$OUT"
        --up-axis "$UP_AXIS"
        --gs-sh-mode "$GS_SH_MODE"
        --opacity-threshold "$OPACITY_THRESH"
        --proxy-method "$PROXY_METHOD"
        --voxel-size-m "$VOXEL_M"
        --poisson-depth "$POISSON_DEPTH"
        --alpha-m "$ALPHA_M"
    )
    if [[ -n "$METRIC_SCALE" ]]; then
        PREP_FLAGS+=(--metric-scale "$METRIC_SCALE")
    else
        PREP_FLAGS+=(--ceiling-height-m "$CEILING_M")
    fi
    python scripts/prepare_marble_scene.py "${PREP_FLAGS[@]}"
fi

# ------ 2: alignment viz (top-down + side PNGs) ----------------------------
if [[ "$SKIP_VIZ" == "1" ]]; then
    echo
    echo "[2/5] === SKIP_VIZ=1; not rendering alignment viz ==="
else
    echo
    echo "[2/5] === alignment viz (viz_topdown.png + viz_side.png) ==="
    python scripts/visualize_dl3dv_alignment.py --scene-dir "$OUT"
fi

# ------ 3: aligned mesh -> USD collider ------------------------------------
if [[ "$SKIP_MESH_USD" == "1" ]]; then
    echo
    echo "[3/5] === SKIP_MESH_USD=1; not authoring mesh_aligned.usd ==="
else
    echo
    echo "[3/5] === mesh_aligned.ply -> mesh_aligned.usd ==="
    python scripts/dl3dv_mesh_to_usd.py --scene-dir "$OUT"
fi

# ------ 4: source PLY -> gaussians.usdz (3DGUT) ----------------------------
if [[ "$SKIP_GS_USDZ" == "1" ]]; then
    echo
    echo "[4/5] === SKIP_GS_USDZ=1; not running 3DGUT ==="
else
    echo
    echo "[4/5] === source.ply -> gaussians.usdz (3DGUT) ==="
    # Pass the *normalized* PLY (degree-3 SH) — the original may have a
    # non-canonical f_rest_* count that 3DGUT rejects.
    python scripts/dl3dv_gs_to_usdz.py \
        --scene-dir "$OUT" \
        --threedgrut-repo "$THREEDGRUT_REPO" \
        --conda-env "$THREEDGRUT_ENV" \
        --ply "$OUT/source.ply"
fi

# ------ 5: placement sampling ----------------------------------------------
if [[ "$SKIP_PLACEMENTS" == "1" ]]; then
    echo
    echo "[5/5] === SKIP_PLACEMENTS=1; not sampling placements ==="
else
    echo
    echo "[5/5] === sample placements ==="
    # No COLMAP cameras → disable camera-aware floor evidence and the
    # camera-distance filter. The placement script tolerates this and
    # falls back to RANSAC-only floor coverage.
    python scripts/sample_dl3dv_placements.py \
        --scene-dir "$OUT" \
        --n "$N_PLACEMENTS" \
        --footprint-len "$FOOTPRINT_LEN" \
        --footprint-wid "$FOOTPRINT_WID" \
        --robot-reach "$ROBOT_REACH" \
        --clearance "$CLEARANCE_M" \
        --table-height "$TABLE_HEIGHT_M" \
        --floor-close-radius "$FLOOR_CLOSE_M" \
        --min-floor-area-m2 "$MIN_FLOOR_AREA_M2" \
        --min-free-area-m2 "$MIN_FREE_AREA_M2" \
        --floor-definition "$FLOOR_DEFINITION" \
        --floor-z-tolerance-m "$FLOOR_Z_TOL_M" \
        --camera-floor-radius 0
fi

echo
echo "=== DONE ==="
echo "Scene artifacts under: $OUT"
ls -lh "$OUT"/*.{ply,usd,usdz,json,png} 2>/dev/null || true

cat <<EOF

Render a robot view at placement 0:

    PYTHONPATH=. python scripts/render_dl3dv_with_robot.py \\
        --task UtensilsInMugTask \\
        --scene-dir $OUT \\
        --placement-idx 0 --num-views 12 --spp 64

(The render script is shared with the DL3DV pipeline — it reads the
same metadata.json + mesh_aligned.usd + gaussians.usdz layout we just
produced. Use --no-gs to render mesh-only when 3DGUT was skipped.)
EOF
