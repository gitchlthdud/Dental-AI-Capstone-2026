#!/bin/bash

#SBATCH -J eval_merged_models
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=16G
#SBATCH -p batch_ugrad
#SBATCH -t 01:00:00
#SBATCH -o /data/thdud23/logs/eval_merged_models_%A.out

set -euo pipefail

USER_NAME="thdud23"

echo "===== JOB INFO ====="
hostname
date

echo "===== PATHS ====="
REPO_DIR="/data/${USER_NAME}/repos/RF-DETR"
EVAL_PY="${REPO_DIR}/eval_merged_models.py"

DATA_DIR="/data/${USER_NAME}/datasets/AlphaDent_2class"
GT_JSON="${DATA_DIR}/test/_annotations.coco.json"

# RF-DETR 예측 결과 경로 (test_merged_RF_DETR.sh 실행 후 생성됨)
RF_PRED="/data/${USER_NAME}/outputs/rfdetr_medium_seg_aug_e50/predictions_test.json"
OUT_BASE="/data/${USER_NAME}/outputs/eval_rfdetr_seg_aug"

echo "EVAL_PY=${EVAL_PY}"
echo "DATA_DIR=${DATA_DIR}"
echo "GT_JSON=${GT_JSON}"
echo "RF_PRED=${RF_PRED}"
echo "OUT_BASE=${OUT_BASE}"

echo "===== SET CACHE DIRS ====="
export TMPDIR="/data/${USER_NAME}/tmp"
export XDG_CACHE_HOME="/data/${USER_NAME}/.cache"
export PIP_CACHE_DIR="/data/${USER_NAME}/.cache/pip"

mkdir -p "$TMPDIR"
mkdir -p "$XDG_CACHE_HOME"
mkdir -p "$PIP_CACHE_DIR"
mkdir -p "/data/${USER_NAME}/logs"
mkdir -p "$OUT_BASE"

echo "===== ACTIVATE CONDA ENV ====="
source "/data/${USER_NAME}/anaconda3/etc/profile.d/conda.sh"
conda activate rfdetr

echo "===== ENV CHECK ====="
which python
python --version

python - <<'PY'
import numpy
print("numpy:", numpy.__version__)

import pycocotools
print("pycocotools import OK")
PY

echo "===== CHECK FILES ====="
if [ ! -f "$EVAL_PY" ]; then
    echo "[ERROR] Missing eval script: $EVAL_PY"
    exit 1
fi

if [ ! -f "$GT_JSON" ]; then
    echo "[ERROR] Missing GT annotation: $GT_JSON"
    exit 1
fi

if [ ! -f "$RF_PRED" ]; then
    echo "[ERROR] Missing RF-DETR prediction json: $RF_PRED"
    echo "Run test_merged_RF_DETR.sh first to generate predictions!"
    exit 1
fi

echo "===== START EVALUATION ====="
echo "Evaluating with default threshold (0.5)..."

python "$EVAL_PY" \
    --gt "$GT_JSON" \
    --pred RF-DETR-CopyPaste="$RF_PRED" \
    --out-dir "$OUT_BASE" \
    --iou-thr 0.5 \
    --score-thr 0.5 \
    --max-dets 100

echo ""
echo "===== OUTPUT FILES ====="
find "$OUT_BASE" -maxdepth 3 -type f | sort

date
echo "===== DONE ====="