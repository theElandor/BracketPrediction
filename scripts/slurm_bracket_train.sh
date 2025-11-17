#!/bin/bash
#SBATCH --job-name=Pt_bracket_prediction
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=grana_maxillo
#SBATCH --partition=all_serial
#SBATCH --nodelist=ailb-login-03
#SBATCH --time=4:00:00
#SBATCH --mem=20GB
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Create necessary directories
mkdir -p logs exp

# Activate your environment
source /homes/mlugli/BracketPrediction/pointcept-brackets-venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH=./

# Configuration
CONFIG="configs/brackets/Pt.py"
EXP_NAME="Pt_newData_classE"
NUM_GPU=1

# Training command
python tools/train.py --config-file ${CONFIG} --num-gpus ${NUM_GPU} --options save_path=exp/brackets/${EXP_NAME}
