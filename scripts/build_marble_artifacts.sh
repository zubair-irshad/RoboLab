#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# End-to-end marble + robot → FastGS-trained artifact-correction pairs.
#
# Pipeline:
#
#   1. render_dl3dv_with_robot.py --artifact-output-root data/diffusion_harmonizer
#      → 120 RGB+depth+intrinsics+extrinsics views around the placement,
#        in the SAME per-view layout capture_artifact_views.py emits.
#   2. export_to_fastgs.py
#      → repackages the views as 3 COLMAP datasets:
#        full (all views, 30k iters), underfit (all views, 1.5k iters),
#        sparse_arc (40-view contiguous-arc holdout, 30k iters).
#   3. build_artifacts_via_fastgs.sh
#      → trains all 3 strategies, renders each at every viewpoint, and
#        pairs the degraded (sparse_arc, underfit) renders against the GT
#        captured RGB to make {input.png, target.png, comparison.png}
#        triples for the diffusion model.
#
# All FastGS train logs land in runs_fastgs/<env-name>/{full,sparse_arc,underfit}/,
# and the final paired data ends up under
# data/diffusion_harmonizer/<env-name>/01_artifacts_correction/<NNNN>/.
#
# Usage:
#   bash scripts/build_marble_artifacts.sh <task> <scene-dir> [placement-idx]
#
# Examples:
#   bash scripts/build_marble_artifacts.sh \
#       UtensilsInMugTask \
#       data/marble_backgrounds/scenes/Designer_Bath_Laundry_Nook
#
#   PLACEMENT_IDX=2 NUM_VIEWS=120 SPP=64 \
#     bash scripts/build_marble_artifacts.sh \
#       UtensilsInMugTask \
#       data/marble_backgrounds/scenes/Contemporary_Studio_Kitchen
#
# Env-var overrides:
#   PLACEMENT_IDX     (0)                          which entry from placements.json to render at
#   NUM_VIEWS         (120)                        hemispheric views per placement
#   RADIUS_LO         (0.8) / RADIUS_HI (1.3)      Fibonacci hemisphere radius range (m)
#   CENTER            "0.4 0.0 0.4"                hemisphere center in world (above the table)
#   RESOLUTION        "512 512"                    rendered RGB+depth resolution (W H)
#   SPP               (32)                         path-tracing samples per pixel (raise for cleaner GT)
#   ARTIFACT_ROOT     (data/diffusion_harmonizer)  where to write the artifact-views mirror
#   ENV_NAME          (auto)                       FastGS env-dir name; default: <task>__<scene>__pl_<idx>
#   FASTGS_REPO       (third_party/FastGS)         FastGS clone path
#   FASTGS_CONDA_ENV  (fastgs)                     conda env to activate for FastGS train+render
#   FULL_ITERS        (30000)                      iterations for the 'full' and 'sparse_arc' strategies
#   TARGET_SOURCE     (gt)                         pair target: 'gt' (captured RGB) or 'reference' ('full' GS render)
#   SKIP_RENDER       (0)                          skip stage 1 (re-use existing views/)
#   SKIP_EXPORT       (0)                          skip stage 2 (re-use existing fastgs/ datasets)
#   SKIP_FASTGS       (0)                          skip stage 3 (renders + pairs)

set -euo pipefail

TASK="${1:?Usage: $0 <task> <scene-dir> [placement-idx]}"
SCENE_DIR="${2:?Usage: $0 <task> <scene-dir> [placement-idx]}"
if [[ ! -d "$SCENE_DIR" ]]; then
    echo "error: scene dir not found: $SCENE_DIR" >&2
    exit 1
fi
PLACEMENT_IDX="${3:-${PLACEMENT_IDX:-0}}"

NUM_VIEWS="${NUM_VIEWS:-120}"
RADIUS_LO="${RADIUS_LO:-0.8}"
RADIUS_HI="${RADIUS_HI:-1.3}"
CENTER="${CENTER:-0.4 0.0 0.4}"
RESOLUTION="${RESOLUTION:-512 512}"
SPP="${SPP:-32}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-data/diffusion_harmonizer}"
SCENE_ID="$(basename "$SCENE_DIR")"
PLACEMENT_PADDED="$(printf "%02d" "$PLACEMENT_IDX")"
ENV_NAME="${ENV_NAME:-${TASK}__${SCENE_ID}__pl_${PLACEMENT_PADDED}}"
FASTGS_REPO="${FASTGS_REPO:-third_party/FastGS}"
FASTGS_CONDA_ENV="${FASTGS_CONDA_ENV:-fastgs}"
FULL_ITERS="${FULL_ITERS:-30000}"
TARGET_SOURCE="${TARGET_SOURCE:-gt}"
SKIP_RENDER="${SKIP_RENDER:-0}"
SKIP_EXPORT="${SKIP_EXPORT:-0}"
SKIP_FASTGS="${SKIP_FASTGS:-0}"

ART_DIR="${ARTIFACT_ROOT}/${ENV_NAME}/01_artifacts_correction"
RUNS_DIR="runs_fastgs/${ENV_NAME}"

echo "============================================================"
echo "Marble artifact-correction pipeline"
echo "  task         : $TASK"
echo "  scene dir    : $SCENE_DIR"
echo "  placement    : $PLACEMENT_IDX"
echo "  env name     : $ENV_NAME"
echo "  artifact dir : $ART_DIR"
echo "  fastgs runs  : $RUNS_DIR"
echo "  views        : $NUM_VIEWS x $RESOLUTION  spp=$SPP  radius=[$RADIUS_LO, $RADIUS_HI]"
echo "  pair target  : $TARGET_SOURCE"
echo "============================================================"

# ------ 1: render with marble + robot, mirror into artifact-views layout ----
if [[ "$SKIP_RENDER" == "1" ]]; then
    echo
    echo "[1/3] === SKIP_RENDER=1; reusing existing views/ under $ART_DIR ==="
else
    echo
    echo "[1/3] === render $NUM_VIEWS views (artifact-views format) ==="
    # shellcheck disable=SC2086
    PYTHONPATH=. python scripts/render_dl3dv_with_robot.py \
        --task "$TASK" \
        --scene-dir "$SCENE_DIR" \
        --placement-idx "$PLACEMENT_IDX" \
        --num-views "$NUM_VIEWS" \
        --radius-range $RADIUS_LO $RADIUS_HI \
        --center $CENTER \
        --resolution $RESOLUTION \
        --spp "$SPP" \
        --artifact-output-root "$ARTIFACT_ROOT" \
        --artifact-env-name "$ENV_NAME"
fi

if [[ ! -d "$ART_DIR/views" ]]; then
    echo "error: $ART_DIR/views not produced — render step failed?" >&2
    exit 1
fi

# ------ 2: views → nerfstudio (depth_init.ply) → FastGS COLMAP datasets ----
# fastgs_export.py SEEDS points3D.txt from <env>/01_artifacts_correction/
# nerfstudio/depth_init.ply (back-projected depth points). Without that
# seed FastGS starts with 0 gaussians and the rasterizer crashes with
# 'CUDA error: invalid configuration argument' on the very first
# training step. So we always run the nerfstudio export first to build
# depth_init.ply, then the fastgs export (which is otherwise a cheap
# JSON / PLY rewrite).
if [[ "$SKIP_EXPORT" == "1" ]]; then
    echo
    echo "[2/3] === SKIP_EXPORT=1; reusing existing $ART_DIR/{nerfstudio,fastgs}/ ==="
else
    echo
    echo "[2/3a] === export views → nerfstudio/ (builds depth_init.ply seed) ==="
    python scripts/export_to_nerfstudio.py \
        --output-root "$ARTIFACT_ROOT" \
        --env "$ENV_NAME"

    if [[ ! -f "$ART_DIR/nerfstudio/depth_init.ply" ]]; then
        echo "[$ENV_NAME] WARNING: nerfstudio/depth_init.ply was NOT produced."
        echo "                     FastGS will start with 0 points and CRASH on iter 0."
        echo "                     Most likely cause: per-view depth.npy missing or all +inf."
        echo "                     Re-run the render step with --spp >= 16 and check that"
        echo "                     $ART_DIR/views/0000/depth.npy exists and has finite values."
    fi

    echo
    echo "[2/3b] === export views → fastgs/{full,underfit,sparse_arc}/ ==="
    python scripts/export_to_fastgs.py \
        --output-root "$ARTIFACT_ROOT" \
        --env "$ENV_NAME" \
        --full-iterations "$FULL_ITERS"

    # Sanity: confirm the seed actually landed in points3D.txt.
    pts_file="$ART_DIR/fastgs/full/train/sparse/0/points3D.txt"
    if [[ -f "$pts_file" ]]; then
        n_pts=$(wc -l < "$pts_file")
        echo "[$ENV_NAME] points3D.txt seeded with $n_pts points"
        if (( n_pts == 0 )); then
            echo "[$ENV_NAME] ERROR: 0 seed points → FastGS would crash. Aborting."
            exit 1
        fi
    fi
fi

# ------ 3: train + render + pair (FastGS conda env) ------------------------
if [[ "$SKIP_FASTGS" == "1" ]]; then
    echo
    echo "[3/3] === SKIP_FASTGS=1; not running FastGS ==="
else
    echo
    echo "[3/3] === FastGS train + render + pair (target=$TARGET_SOURCE) ==="
    # build_artifacts_via_fastgs.sh hardcodes the pair invocation; we
    # call its train/render half ourselves and then re-pair with the
    # target-source we want.
    fastgs_rc=0
    FASTGS_REPO="$FASTGS_REPO" FASTGS_CONDA_ENV="$FASTGS_CONDA_ENV" \
        bash scripts/build_artifacts_via_fastgs.sh "$ENV_NAME" "$FULL_ITERS" || fastgs_rc=$?
    if (( fastgs_rc != 0 )); then
        echo "[$ENV_NAME] FastGS train/render exited with code $fastgs_rc — see logs above"
    fi
    # Concrete success check: did `train.py` actually produce a model?
    if [[ ! -d "$RUNS_DIR/full/point_cloud" ]]; then
        echo "[$ENV_NAME] ERROR: $RUNS_DIR/full/point_cloud/ not produced. Train step failed."
        echo "                 Common causes:"
        echo "                   - $FASTGS_REPO is not a valid FastGS clone (missing train.py / render.py)"
        echo "                   - conda env '$FASTGS_CONDA_ENV' missing or broken"
        echo "                   - CUDA/torch version mismatch inside the env"
        echo "                 Reproduce manually with verbose output:"
        echo "                   conda activate $FASTGS_CONDA_ENV"
        echo "                   cd $FASTGS_REPO && python train.py \\"
        echo "                       --source_path \$(realpath $ART_DIR/fastgs/full/train) \\"
        echo "                       --model_path \$(realpath -m $RUNS_DIR/full) \\"
        echo "                       --iterations $FULL_ITERS"
        exit 1
    fi

    # Re-pair with the requested target source. The build script already
    # paired against 'full' renders; redo it with --target-source so we
    # get GT-as-target by default. It overwrites the existing pair dirs
    # (same numbering scheme) — safe to re-run.
    echo
    echo "[3/3] === re-pair with --target-source=$TARGET_SOURCE ==="
    python scripts/pair_splatfacto_renders.py \
        --ns-root "$ART_DIR/fastgs" \
        --output-dir "$ART_DIR" \
        --strategies sparse_arc underfit \
        --target-source "$TARGET_SOURCE" \
        --gt-views-dir "$ART_DIR/views"
fi

echo
echo "=== DONE ==="
echo "artifact dir : $ART_DIR"
echo "fastgs runs  : $RUNS_DIR"
echo
ls -lh "$ART_DIR"/*.{json,png} 2>/dev/null || true
echo
PAIR_COUNT=$(find "$ART_DIR" -mindepth 1 -maxdepth 1 -type d -regex '.*/[0-9][0-9][0-9][0-9]' 2>/dev/null | wc -l | tr -d ' ')
echo "pair count   : $PAIR_COUNT"
echo

cat <<EOF

Inspect the captures + pairs in rerun:

    PYTHONPATH=. python scripts/visualize_artifact_views_rerun.py \\
        --root $ARTIFACT_ROOT \\
        --env  $ENV_NAME \\
        --stride 8 --max-views 60

  - cameras + frustums + per-view depth points
  - merged depth_init.ply (the 3DGS densification seed)
  - if pairs exist: scrub a frustum to see GT rgb / sparse_arc input / sparse_arc target
    side-by-side at the same viewpoint

Inspect FastGS training runs (FastGS is vanilla-3DGS-style: train.py, render.py, no view.py):

    # 1. Checkpoints + final render PNGs (no extra tooling needed):
    ls $RUNS_DIR/full/point_cloud/                               # iteration_*  dirs
    ls $RUNS_DIR/full/train/ours_${FULL_ITERS}/renders/ | head   # rendered PNGs at training viewpoints
    ls $RUNS_DIR/full/train/ours_${FULL_ITERS}/gt/      | head   # corresponding GT for psnr/lpips comparison

    # 2. Re-render at any captured viewpoint without retraining
    #    (activate the fastgs env, then call render.py from inside the FastGS clone):
    conda activate $FASTGS_CONDA_ENV
    cd $FASTGS_REPO && python render.py \\
        --source_path \$(realpath ../../$ART_DIR/fastgs/full/render) \\
        --model_path  \$(realpath ../../$RUNS_DIR/full) \\
        --iteration   $FULL_ITERS

    # 3. Interactive 3D viewer for the trained Gaussians:
    #    FastGS ships SIBR_viewers (C++); follow your FastGS README to build it,
    #    then point it at $RUNS_DIR/full/point_cloud/iteration_${FULL_ITERS}/point_cloud.ply
    #    Browser alternative (no build): https://playcanvas.com/supersplat/editor — drag the .ply in.

    # 4. Or load the trained Gaussians into a marble-style scene-dir and re-render with the
    #    robot via render_dl3dv_with_robot.py (gives you placement-aligned views):
    #    just rename the trained PLY to point_cloud.ply under fastpgsr/point_cloud/iteration_*/

Inspect a single pair on disk:

    ls $ART_DIR/0000/
    # input.png  target.png  comparison.png  metadata.json
EOF
