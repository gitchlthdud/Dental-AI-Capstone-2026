#!/bin/bash

#SBATCH -J train_rfdetr_2class
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -t 1-00:00:00
#SBATCH -o /data/thdud23/logs/train_rfdetr_2class_%A.out

set -euo pipefail

USER_NAME="thdud23"

echo "===== JOB INFO ====="
hostname
date
nvidia-smi

echo "===== PATHS ====="
REPO_DIR="/data/${USER_NAME}/repos/RF-DETR"
TRAIN_PY="${REPO_DIR}/train_merged_RF_DETR.py"

DATA_DIR="/data/${USER_NAME}/datasets/AlphaDent_2class_seg_aug"
LOCAL_ROOT="/local_datasets/${USER_NAME}"
LOCAL_DATASET="${LOCAL_ROOT}/AlphaDent_2class_seg_aug"

OUT_DIR="/data/${USER_NAME}/outputs/rfdetr_medium_seg_aug_e50"

echo "REPO_DIR=${REPO_DIR}"
echo "TRAIN_PY=${TRAIN_PY}"
echo "DATA_DIR=${DATA_DIR}"
echo "LOCAL_ROOT=${LOCAL_ROOT}"
echo "LOCAL_DATASET=${LOCAL_DATASET}"
echo "OUT_DIR=${OUT_DIR}"

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
mkdir -p "/data/${USER_NAME}/outputs"
mkdir -p "$LOCAL_ROOT"

echo "===== ACTIVATE CONDA ENV ====="
source /data/${USER_NAME}/anaconda3/etc/profile.d/conda.sh
conda activate rfdetr

echo "===== ENV CHECK ====="
which python
python --version
conda info --envs | grep '*'

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "===== CHECK SCRIPT ====="
if [ ! -f "$TRAIN_PY" ]; then
    echo "[ERROR] Missing train script: $TRAIN_PY"
    exit 1
fi

echo "===== PREPARE DATASET ON COMPUTE NODE ====="
if [ ! -d "$DATA_DIR" ]; then
    echo "[ERROR] Dataset directory not found: $DATA_DIR"
    exit 1
fi

echo "===== CLEAN LOCAL DATASETS ====="
rm -rf "${LOCAL_ROOT}"/*

echo "Copy dataset to local..."
cp -r "$DATA_DIR" "$LOCAL_DATASET"

echo "===== CHECK DATASET FILES ====="
for split in train valid test; do
    ANN="${LOCAL_DATASET}/${split}/_annotations.coco.json"
    if [ ! -f "$ANN" ]; then
        echo "[ERROR] Missing annotation file: $ANN"
        exit 1
    fi
    echo "Found: $ANN"
done

echo "===== CHECK 2-CLASS COCO CATEGORIES ====="
python - <<PY
import json
from pathlib import Path
from collections import Counter

dataset = Path("${LOCAL_DATASET}")
expected = {1: "Abrasion", 2: "Caries"}

for split in ["train", "valid", "test"]:
    ann_path = dataset / split / "_annotations.coco.json"

    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    cats = {int(c["id"]): c["name"] for c in coco.get("categories", [])}
    cnt = Counter(int(a["category_id"]) for a in coco.get("annotations", []))

    print(f"[{split}] categories={cats}")
    print(f"[{split}] annotation counts={dict(sorted(cnt.items()))}")

    if cats != expected:
        raise SystemExit(
            f"[ERROR] {split} category mismatch. Expected {expected}, found {cats}"
        )

    bad = [
        a for a in coco.get("annotations", [])
        if int(a["category_id"]) not in expected
    ]

    if bad:
        raise SystemExit(
            f"[ERROR] {split} has invalid category_id examples: {bad[:5]}"
        )

print("2-class dataset check passed.")
PY

echo "===== START RF-DETR M TRAINING ====="
python "$TRAIN_PY" \
    --dataset "$LOCAL_DATASET" \
    --output-dir "$OUT_DIR" \
    --epochs 50 \
    --batch-size 8 \
    --grad-accum-steps 1 \
    --lr 1e-4 \
    --lr-encoder 1.5e-4 \
    --weight-decay 1e-4 \
    --resolution 576 \
    --num-classes 2 \
    --seed 42 \
    --checkpoint-interval 5 \
    --eval-interval 1

echo "===== OUTPUT FILES ====="
find "$OUT_DIR" -maxdepth 2 -type f | sort

echo "===== FINAL CHECKPOINT CHECK ====="
if [ -f "${OUT_DIR}/checkpoint_best_total.pth" ]; then
    echo "SUCCESS: ${OUT_DIR}/checkpoint_best_total.pth"
else
    echo "[WARN] checkpoint_best_total.pth not found."
fi

date
echo "===== DONE ====="