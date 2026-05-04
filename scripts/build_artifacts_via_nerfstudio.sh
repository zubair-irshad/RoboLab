#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# End-to-end nerfstudio path: ns-train (3x) + ns-render (3x) + pair.
# Assumes scripts/capture_artifact_views.py and scripts/export_to_nerfstudio.py
# already ran for this env.
#
# Usage:
#     bash scripts/build_artifacts_via_nerfstudio.sh UtensilsInMugTask
#
# Optional second arg overrides --max-num-iterations for the heavy strategies
# (full + sparse_arc); underfit always uses 1500.

set -euo pipefail

ENV_NAME="${1:?Usage: $0 <env_name> [iterations] [vis]}"
ITERS="${2:-30000}"
VIS="${3:-wandb}"           # nerfstudio --vis target: viewer | wandb | tensorboard | viewer+wandb | none
WANDB_PROJECT="${WANDB_PROJECT:-diffusion-harmonizer}"

ROOT="data/diffusion_harmonizer/${ENV_NAME}/01_artifacts_correction"
NS="${ROOT}/nerfstudio"
RUNS="runs/${ENV_NAME}"

if [[ ! -d "$NS" ]]; then
    echo "[$ENV_NAME] $NS not found — run scripts/export_to_nerfstudio.py first." >&2
    exit 1
fi

mkdir -p "$RUNS"

train_one() {
    local strategy="$1"
    local iters="$2"
    if [[ ! -d "$NS/$strategy" ]]; then
        echo "[$ENV_NAME] skipping $strategy — no $NS/$strategy dir."
        return
    fi
    echo "[$ENV_NAME] >>> ns-train $strategy ($iters iters, --vis $VIS)"
    ns-train splatfacto \
        --max-num-iterations "$iters" \
        --vis "$VIS" \
        --experiment-name "${ENV_NAME}_${strategy}" \
        --project-name "$WANDB_PROJECT" \
        --data "$NS/$strategy" \
        --output-dir "$RUNS/$strategy"
}

render_one() {
    local strategy="$1"
    local cfg
    cfg=$(ls -d "$RUNS/$strategy"/*/splatfacto/*/config.yml 2>/dev/null | tail -n1 || true)
    if [[ -z "$cfg" ]]; then
        echo "[$ENV_NAME] no config.yml under $RUNS/$strategy — skipping render."
        return
    fi
    echo "[$ENV_NAME] >>> ns-render $strategy ($cfg)"
    rm -rf "$NS/$strategy/renders"
    # Render train + eval to the same output dir; pairing script doesn't care about subdir.
    ns-render dataset --load-config "$cfg" \
        --output-path "$NS/$strategy/renders" --split train || true
    ns-render dataset --load-config "$cfg" \
        --output-path "$NS/$strategy/renders" --split val || true
}

train_one full       "$ITERS"
train_one sparse_arc "$ITERS"
train_one underfit   1500

render_one full
render_one sparse_arc
render_one underfit

echo "[$ENV_NAME] >>> pair"
python scripts/pair_splatfacto_renders.py \
    --ns-root "$NS" \
    --output-dir "$ROOT" \
    --strategies sparse_arc underfit

echo "[$ENV_NAME] DONE -> $ROOT"
