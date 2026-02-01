#!/bin/bash
#SBATCH --job-name=sp_reg_norm_f5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=grana_maxillo
#SBATCH --partition=all_usr_prod
#SBATCH --time=00:30:00
#SBATCH --constraint="gpu_RTX5000_16G|gpu_RTX6000_24G|gpu_RTX_A5000_24G"
#SBATCH --mem=20GB
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Activate your environment
source /homes/mlugli/BracketPrediction/pointcept-brackets-venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH=./

# Configuration
CONFIG="configs/brackets/Sp_regressor_normals.py"
EXP_NAME="sp_reg_norm_f5"
NUM_GPU=1
FOLD=5

# Training command
python tools/test.py \
--config-file ${CONFIG} \
--num-gpus ${NUM_GPU} \
--options \
    weight=exp/brackets/${EXP_NAME}/model/model_best.pth \
    save_path=exp/brackets/${EXP_NAME} \
    data.train.fold=${FOLD} \
    data.val.fold=${FOLD} \
    data.test.fold=${FOLD}
