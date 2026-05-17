#!/bin/bash

#SBATCH -J eval_thresholds
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -p batch_ugrad
#SBATCH -t 0-02:00:00
#SBATCH -o /data/thdud23/logs/eval_thresholds_%A.out

set -euo pipefail

USER_NAME="thdud23"

echo "=========================================="
echo "RF-DETR Threshold Sweep (0.10 ~ 0.50, 0.05 step)"
echo "Start: $(date)"
echo "=========================================="

eval "$(/data/${USER_NAME}/anaconda3/bin/conda shell.bash hook)"
conda activate rfdetr

cd /data/${USER_NAME}/repos/RF-DETR

GT="/data/${USER_NAME}/datasets/AlphaDent_2class/test/_annotations.coco.json"
PRED="/data/${USER_NAME}/outputs/rfdetr_medium_seg_aug_e50/predictions_test.json"
OUT_BASE="/data/${USER_NAME}/outputs/thresh_sweep_seg_aug"

mkdir -p "${OUT_BASE}"

# Threshold sweep
for THRESH in 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50
do
    echo ""
    echo "=========================================="
    echo "Evaluating Threshold: ${THRESH}"
    echo "=========================================="
    
    python eval_merged_models.py \
        --gt "${GT}" \
        --pred RF-DETR-Medium="${PRED}" \
        --out-dir "${OUT_BASE}/thresh_${THRESH}" \
        --iou-thr 0.5 \
        --score-thr ${THRESH} \
        --max-dets 100
done

echo ""
echo "=========================================="
echo "Analyzing Results..."
echo "=========================================="

python analyze_thresholds.py "${OUT_BASE}"

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="