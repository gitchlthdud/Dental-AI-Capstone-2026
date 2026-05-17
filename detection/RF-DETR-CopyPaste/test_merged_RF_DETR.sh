#!/bin/bash

#SBATCH -J test_merged_RF_DETR
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=24G
#SBATCH -p batch_ugrad
#SBATCH -t 02:00:00
#SBATCH -o /data/thdud23/logs/test_merged_RF_DETR_%A.out

set -euo pipefail

USER_NAME="thdud23"

echo "===== JOB INFO ====="
hostname
date
nvidia-smi

echo "===== PATHS ====="
REPO_DIR="/data/${USER_NAME}/repos/RF-DETR"
TEST_PY="${REPO_DIR}/test_merged_RF_DETR.py"

DATA_DIR="/data/${USER_NAME}/datasets/AlphaDent_2class"
TRAIN_OUT_DIR="/data/${USER_NAME}/outputs/rfdetr_medium_seg_aug_e50"
TEST_OUT_DIR="${TRAIN_OUT_DIR}"
PRED_JSON="${TEST_OUT_DIR}/predictions_test.json"

CHECKPOINT="${TRAIN_OUT_DIR}/checkpoint_best_total.pth"

echo "TEST_PY=${TEST_PY}"
echo "DATA_DIR=${DATA_DIR}"
echo "TRAIN_OUT_DIR=${TRAIN_OUT_DIR}"
echo "TEST_OUT_DIR=${TEST_OUT_DIR}"
echo "PRED_JSON=${PRED_JSON}"
echo "CHECKPOINT=${CHECKPOINT}"

echo "===== SET CACHE DIRS ====="
export TMPDIR="/data/${USER_NAME}/tmp"
export XDG_CACHE_HOME="/data/${USER_NAME}/.cache"
export HF_HOME="/data/${USER_NAME}/.cache/huggingface"
export TORCH_HOME="/data/${USER_NAME}/.cache/torch"
export PIP_CACHE_DIR="/data/${USER_NAME}/.cache/pip"
export RF_HOME="/data/${USER_NAME}/.cache/roboflow"

mkdir -p "$TMPDIR"
mkdir -p "$XDG_CACHE_HOME"
mkdir -p "$HF_HOME"
mkdir -p "$TORCH_HOME"
mkdir -p "$PIP_CACHE_DIR"
mkdir -p "$RF_HOME"
mkdir -p "/data/${USER_NAME}/logs"
mkdir -p "$TEST_OUT_DIR"

echo "===== ACTIVATE CONDA ENV ====="
source "/data/${USER_NAME}/anaconda3/etc/profile.d/conda.sh"
conda activate rfdetr

echo "===== ENV CHECK ====="
which python
python --version

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

from rfdetr import RFDETRBase
print("RFDETRBase import OK")

from PIL import Image
print("PIL import OK")
PY

echo "===== CHECK FILES ====="
if [ ! -f "$TEST_PY" ]; then
    echo "[ERROR] Missing test script: $TEST_PY"
    exit 1
fi

if [ ! -f "${DATA_DIR}/test/_annotations.coco.json" ]; then
    echo "[ERROR] Missing test annotation: ${DATA_DIR}/test/_annotations.coco.json"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "[ERROR] Missing checkpoint: $CHECKPOINT"
    echo "Files in TRAIN_OUT_DIR:"
    find "$TRAIN_OUT_DIR" -maxdepth 2 -type f | sort || true
    exit 1
fi

echo "===== CHECK TEST DATASET ====="
echo "Found: ${DATA_DIR}/test/_annotations.coco.json"

echo "===== START RF-DETR TEST INFERENCE ====="
python "$TEST_PY" \
    --dataset "$DATA_DIR" \
    --split test \
    --checkpoint "$CHECKPOINT" \
    --out "$PRED_JSON" \
    --num-classes 2 \
    --threshold 0.001 \
    --class-id-base zero

echo "===== OUTPUT FILE ====="
if [ -f "$PRED_JSON" ]; then
    ls -lh "$PRED_JSON"
else
    echo "[ERROR] Prediction JSON was not created: $PRED_JSON"
    exit 1
fi

date
echo "===== DONE ====="