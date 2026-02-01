#!/bin/bash
# Master script to submit all jobs

sbatch scripts/test/sp_map_f1.sh
sbatch scripts/test/sp_map_f2.sh
sbatch scripts/test/sp_map_f3.sh
sbatch scripts/test/sp_map_f4.sh
sbatch scripts/test/sp_map_f5.sh
sbatch scripts/test/sp_map_norm_f1.sh
sbatch scripts/test/sp_map_norm_f2.sh
sbatch scripts/test/sp_map_norm_f3.sh
sbatch scripts/test/sp_map_norm_f4.sh
sbatch scripts/test/sp_map_norm_f5.sh
sbatch scripts/test/sp_reg_f1.sh
sbatch scripts/test/sp_reg_f2.sh
sbatch scripts/test/sp_reg_f3.sh
sbatch scripts/test/sp_reg_f4.sh
sbatch scripts/test/sp_reg_f5.sh
sbatch scripts/test/sp_reg_norm_f1.sh
sbatch scripts/test/sp_reg_norm_f2.sh
sbatch scripts/test/sp_reg_norm_f3.sh
sbatch scripts/test/sp_reg_norm_f4.sh
sbatch scripts/test/sp_reg_norm_f5.sh
