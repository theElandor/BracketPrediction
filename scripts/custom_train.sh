#!/bin/bash

# Training script for Bracket Point Prediction

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0

# Configuration
CONFIG="configs/brackets/SpUnet.py"
EXP_NAME="SpUnet_8kpoints_SGD"
NUM_GPU=1

# Training command
python tools/train.py \
    --config-file ${CONFIG} \
    --num-gpus ${NUM_GPU} \
    --options \
    save_path=exp/${EXP_NAME} \
    # Optionally override config parameters:
    # batch_size=16 \
    # epoch=200 \
    # optimizer.lr=0.0005

# For multi-GPU training, use:
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# python tools/train.py --config-file ${CONFIG} --num-gpus 4 --options save_path=exp/${EXP_NAME}

# For resuming training from checkpoint:
# python tools/train.py --config-file ${CONFIG} --num-gpus ${NUM_GPU} --options save_path=exp/${EXP_NAME} resume=True weight=exp/${EXP_NAME}/model/model_last.pth
