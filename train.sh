#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${FEDSEMI_CONDA_ENV:-py3.10}"
GPU_POOL="${FEDSEMI_GPU_POOL:-2,3,4}"
DATA_ROOT="${FEDSEMI_DATA_ROOT:-/data/segmentation}"
STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"

run_local() {
  local task="$1"
  local trainer="$2"
  local client="$3"
  conda run -n "${CONDA_ENV}" python local_train.py \
    --config "configs/${task}/run_conf.yaml" \
    --run_name "${task}_${trainer}_${client}_${STAMP}" \
    --trainer "${trainer}" \
    --train_path "${DATA_ROOT}/${task}/fed_semi/${client}" \
    --gpu_pool "${GPU_POOL}"
}

run_fl() {
  local task="$1"
  local trainer="$2"
  conda run -n "${CONDA_ENV}" python fl_train.py \
    --config "configs/${task}/fl_run_conf.yaml" \
    --run_name "${task}_fl_${trainer}_${STAMP}" \
    --trainer "${trainer}" \
    --train_path "${DATA_ROOT}/${task}/fed_semi" \
    --labeled_clients client_1 client_2 client_3 \
    --unseen_client client_4 \
    --gpu_pool "${GPU_POOL}"
}

# Examples:
# run_local cardiac supervised client_1
# run_local colon semi client_1
# run_fl spine semi
# run_fl bladder semi
