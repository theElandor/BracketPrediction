#!/bin/bash
#SBATCH --job-name=Pt_reg_dilated_1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=grana_maxillo
#SBATCH --partition=all_serial
#SBATCH --nodelist=ailb-login-03
#SBATCH --time=03:00:00
#SBATCH --mem=50GB
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Create necessary directories

mkdir -p logs exp

# Activate your environment
source /homes/mlugli/BracketPrediction/pointcept-brackets-venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH=./

# Configuration
CONFIG="configs/brackets/Pt_regressor.py"
EXP_NAME="Pt_reg_dilated_1"
NUM_GPU=1

# Training command
python tools/train.py --config-file ${CONFIG} --num-gpus ${NUM_GPU} --options save_path=exp/semseg/${EXP_NAME}
