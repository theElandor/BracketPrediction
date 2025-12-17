t s#!/bin/bash
set -e

# Activate conda
source /opt/conda/etc/profile.d/conda.sh
conda activate base

# Run the monitor
python /workspace/application/monitor.py \
    --data-root /workspace/application/data/ \
    --seg-config /workspace/application/configs/Pt_semseg_app.py \
    --seg-weight /workspace/application/weights/segmentator_best.pth \
    --bond-config /workspace/application/configs/Pt_regressor_app.py \
    --bond-weight /workspace/application/weights/regressor_best.pth \
    --check-interval 10