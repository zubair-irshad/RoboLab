#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# End-to-end FastGS path: train sparse/underfit + render + pair, mirroring
# build_artifacts_via_nerfstudio.sh but for graphdeco-inria-style 3DGS.
#
# Prereqs (one-time):
#   1. git clone https://github.com/fastgs/FastGS  (default location: third_party/FastGS)
#      override with FASTGS_REPO=/path/to/FastGS
#   2. cd FastGS && conda env create --file environment.yml && conda activate fastgs
#      override env name with FASTGS_CONDA_ENV=<name>
#   3. From the robolab env (NOT fastgs):
#        PYTHONPATH=. python scripts/capture_artifact_views.py --task <task>
#        python scripts/export_to_nerfstudio.py --output-root data/diffusion_harmonizer
#        python scripts/export_to_fastgs.py    --output-root data/diffusion_harmonizer
#
# Usage:
#     bash scripts/build_artifacts_via_fastgs.sh UtensilsInMugTask [iterations]
#
# Optional second arg overrides --iterations for sparse_arc; underfit always uses 300.

set -euo pipefail

ENV_NAME="${1:?Usage: $0 <env_name> [iterations]}"
ITERS="${2:-30000}"
FASTGS_REPO="${FASTGS_REPO:-third_party/FastGS}"
FASTGS_CONDA_ENV="${FASTGS_CONDA_ENV:-fastgs}"

ROOT="data/diffusion_harmonizer/${ENV_NAME}/01_artifacts_correction"
FG="${ROOT}/fastgs"
RUNS="runs_fastgs/${ENV_NAME}"

if [[ ! -d "$FG" ]]; then
    echo "[$ENV_NAME] $FG not found — run scripts/export_to_fastgs.py first." >&2
    exit 1
fi
if [[ ! -d "$FASTGS_REPO" ]]; then
    echo "[$ENV_NAME] FastGS repo not at $FASTGS_REPO. Set FASTGS_REPO=/path/to/FastGS or clone it." >&2
    exit 1
fi
if [[ ! -f "$FASTGS_REPO/train.py" || ! -f "$FASTGS_REPO/render.py" ]]; then
    echo "[$ENV_NAME] $FASTGS_REPO doesn't look like the FastGS repo (no train.py / render.py)." >&2
    exit 1
fi

# Activate the FastGS conda env. Conda activate isn't on PATH inside a
# non-interactive shell by default — source the conda hook explicitly.
if ! command -v conda >/dev/null 2>&1; then
    echo "[$ENV_NAME] conda not on PATH; cannot activate $FASTGS_CONDA_ENV." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$FASTGS_CONDA_ENV"
echo "[$ENV_NAME] using conda env: $(python -c 'import sys; print(sys.executable)')"

mkdir -p "$RUNS"

# All path resolution must happen in the parent shell BEFORE we cd into
# the FastGS repo — otherwise $(realpath …) evaluates with the FastGS
# repo as cwd and our relative ./data/... paths fail to resolve.
train_one() {
    local strategy="$1"
    local iters="$2"
    local src="$FG/$strategy/train"
    local model="$RUNS/$strategy"
    if [[ ! -d "$src" ]]; then
        echo "[$ENV_NAME] skipping $strategy — no $src dir."
        return
    fi
    local src_abs model_abs
    src_abs=$(realpath "$src")
    model_abs=$(realpath -m "$model")
    echo "[$ENV_NAME] >>> fastgs train $strategy ($iters iters)"
    rm -rf "$model"
    # FastGS / 3DGS caches the parsed point cloud as a binary PLY in TWO
    # places: the source dir's sparse/0/points3D.ply (alongside the .txt)
    # and the model dir's input.ply. Both short-circuit re-loads. If a
    # previous run baked them when points3D.txt was empty / stale, the
    # cached PLYs stay at 229 bytes and Number-of-points stays 0. Always
    # nuke them before train so the .txt is re-parsed fresh.
    rm -f "$src/sparse/0/points3D.ply"
    (cd "$FASTGS_REPO" && python train.py \
        --source_path "$src_abs" \
        --model_path "$model_abs" \
        --iterations "$iters")
    # Smoke-check: the cache should now reflect the actual point count.
    if [[ -f "$model/input.ply" ]]; then
        sz=$(stat -c%s "$model/input.ply" 2>/dev/null || stat -f%z "$model/input.ply" 2>/dev/null)
        if (( sz < 1024 )); then
            echo "[$ENV_NAME] WARNING: $model/input.ply is suspiciously small ($sz bytes)" \
                 "— FastGS may have read an empty/stale points3D.txt."
        fi
    fi
}

render_one() {
    local strategy="$1"
    local iters="$2"
    local src="$FG/$strategy/render"
    local model="$RUNS/$strategy"
    if [[ ! -d "$src" || ! -d "$model" ]]; then
        echo "[$ENV_NAME] skipping render $strategy — missing $src or $model."
        return
    fi
    local src_abs model_abs
    src_abs=$(realpath "$src")
    model_abs=$(realpath "$model")
    echo "[$ENV_NAME] >>> fastgs render $strategy"
    # Vanilla 3DGS render.py loads gaussians from --model_path and reads
    # cameras from --source_path. For sparse_arc, export_to_fastgs.py writes
    # only held-out poses under render/; underfit keeps all poses.
    (cd "$FASTGS_REPO" && python render.py \
        --source_path "$src_abs" \
        --model_path "$model_abs" \
        --iteration "$iters")
}

train_one sparse_arc "$ITERS"
train_one underfit   300

render_one sparse_arc "$ITERS"
render_one underfit   300

# Vanilla 3DGS render.py writes to <model>/train/ours_<iter>/{renders,gt}/<NNNNN>.png.
# pair_splatfacto_renders.py expects <ns_root>/<strategy>/renders/*.png, so we
# symlink each strategy's render output into that layout before pairing.
link_renders() {
    local strategy="$1"
    local out
    out=$(ls -d "$RUNS/$strategy"/train/ours_* 2>/dev/null | tail -n1 || true)
    if [[ -z "$out" || ! -d "$out/renders" ]]; then
        echo "[$ENV_NAME] no FastGS renders for $strategy under $RUNS/$strategy/train/ — skipping link."
        return
    fi
    local link="$FG/$strategy/renders"
    rm -rf "$link"
    ln -s "$(realpath "$out/renders")" "$link"
    echo "[$ENV_NAME] linked $link -> $out/renders"
}
link_renders sparse_arc
link_renders underfit

echo "[$ENV_NAME] >>> pair"
python scripts/pair_splatfacto_renders.py \
    --ns-root "$FG" \
    --output-dir "$ROOT" \
    --strategies sparse_arc underfit \
    || echo "[$ENV_NAME] pairing skipped/failed; renders are still under $RUNS"

echo "[$ENV_NAME] DONE -> $ROOT (model dirs under $RUNS)"
