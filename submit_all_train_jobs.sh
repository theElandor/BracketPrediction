#!/bin/bash
# Master script to submit all jobs

sbatch scripts/train/sp_reg_f1.sh
sbatch scripts/train/sp_reg_f2.sh
sbatch scripts/train/sp_reg_f3.sh
sbatch scripts/train/sp_reg_f4.sh
sbatch scripts/train/sp_reg_f5.sh
sbatch scripts/train/sp_reg_norm_f1.sh
sbatch scripts/train/sp_reg_norm_f2.sh
sbatch scripts/train/sp_reg_norm_f3.sh
sbatch scripts/train/sp_reg_norm_f4.sh
sbatch scripts/train/sp_reg_norm_f5.sh
