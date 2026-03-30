#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuH200x8
#SBATCH --account=bepk-delta-gpu
#SBATCH --job-name="soybean_M3"
#SBATCH --mail-user=bgekim@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="soybean_M3_%j.out"
#SBATCH --error="soybean_M3_%j.err"

# Conda environment
source /u/apowadi/miniforge3/etc/profile.d/conda.sh
conda activate isa_yield_env

# to avoid PROJ conflict
unset PROJ_DATA
unset PROJ_LIB
unset SLURM_NTASKS

echo "=========================================="
echo "Job started on $(hostname) at $(date)"
echo "=========================================="

# Config files to run
# CONFIGS=(
#     "conf_M4/M4_2/delta_s12d_6_soybean.yaml"
#     "conf_M4/M4_2/delta_s12d_8_soybean.yaml"
#     "conf_M4/M4_2/delta_s12d_10_soybean.yaml"
# )
CONFIGS=(
    # "conf_M3_stat/s12w_24_soybean.yaml"
    # "conf_M3_stat/s12c_24_soybean.yaml"
    # "conf_M3_stat/s12ws_24_soybean.yaml"
    # "conf_M3_stat/s12dc_24_soybean.yaml"
    "conf_M3_stat/s12wds_24_soybean.yaml"
    "conf_M3_stat/s12wsc_24_soybean.yaml"
)


# Run each config
for config in "${CONFIGS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running: $config"
    echo "Start time: $(date)"
    echo "=========================================="
    
    terratorch fit -c "$config"
    
    echo "Finished: $config at $(date)"
    echo ""
done

echo "=========================================="
echo "All jobs completed at $(date)"
echo "=========================================="