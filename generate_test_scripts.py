#!/usr/bin/env python3
"""
Generate SLURM job files for 5-fold cross-validation across 4 models.
"""

import os

# Define models with their configurations and short names for job naming
MODELS = [
    #{"config": "Pt_map.py", "name": "map"},
    #{"config": "Pt_regressor.py", "name": "reg"},
    #{"config": "Pt_map_normals.py", "name": "map_norm"},
    #{"config": "Pt_regressor_normals.py", "name": "reg_norm"}
    {"config": "Sp_map.py", "name": "sp_map"},
    {"config": "Sp_map_normals.py", "name": "sp_map_norm"},
    {"config": "Sp_regressor.py", "name": "sp_reg"},
    {"config": "Sp_regressor_normals.py", "name": "sp_reg_norm"}
]

FOLDS = range(1, 6)  # 1 to 5

# SLURM template
SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
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
CONFIG="configs/brackets/{config}"
EXP_NAME="{exp_name}"
NUM_GPU=1
FOLD={fold}

# Training command
python tools/test.py \\
--config-file ${{CONFIG}} \\
--num-gpus ${{NUM_GPU}} \\
--options \\
    weight=exp/brackets/${{EXP_NAME}}/model/model_best.pth \\
    save_path=exp/brackets/${{EXP_NAME}} \\
    data.train.fold=${{FOLD}} \\
    data.val.fold=${{FOLD}} \\
    data.test.fold=${{FOLD}}
"""


def generate_slurm_files():
    """Generate SLURM job files for all model-fold combinations."""
    
    # Create output directory if it doesn't exist
    output_dir = "scripts/test"
    os.makedirs(output_dir, exist_ok=True)
    
    generated_files = []
    
    for model in MODELS:
        for fold in FOLDS:
            # Create job name and experiment name
            job_name = f"{model['name']}_f{fold}"
            exp_name = job_name
            
            # Fill in the template
            slurm_content = SLURM_TEMPLATE.format(
                job_name=job_name,
                config=model['config'],
                exp_name=exp_name,
                fold=fold
            )
            
            # Create filename
            filename = f"{output_dir}/{job_name}.sh"
            
            # Write to file
            with open(filename, 'w') as f:
                f.write(slurm_content)
 
            # Make executable
            os.chmod(filename, 0o755)
            
            generated_files.append(filename)
            print(f"Generated: {filename}")
    
    print(f"\nTotal files generated: {len(generated_files)}")
    print(f"All files saved in: {output_dir}/")
    
    # Generate a master submission script in the current directory
    master_script = "submit_all_test_jobs.sh"
    with open(master_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Master script to submit all jobs\n\n")
        for filename in generated_files:
            f.write(f"sbatch {filename}\n")
    
    os.chmod(master_script, 0o755)
    print(f"\nMaster submission script created: {master_script}")
    print("Run './submit_all_jobs.sh' from the project root to submit all jobs at once.")


if __name__ == "__main__":
    generate_slurm_files()