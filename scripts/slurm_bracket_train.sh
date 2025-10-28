#!/bin/bash
#SBATCH --job-name=point_transformer_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=grana_maxillo
#SBATCH --partition=all_usr_prod
#SBATCH --time=12:00:00
#SBATCH --mem=32GB
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Create necessary directories
mkdir -p logs exp

# Activate your environment
source /homes/mlugli/BracketPrediction/pointcept-brackets-venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH=./

# Configuration
CONFIG="configs/brackets/PointTransformer.py"
EXP_NAME="PointTransformer_test"
NUM_GPU=1

# Training command
python tools/train.py --config-file ${CONFIG} --num-gpus ${NUM_GPU} --options save_path=exp/${EXP_NAME}
