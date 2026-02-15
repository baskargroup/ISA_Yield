#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuH200x8 
#SBATCH --account=bepk-delta-gpu
#SBATCH --job-name="soybean_M4_NEW"
#SBATCH --mail-user=bgekim@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="soybean_M4_NEW_%j.out"
#SBATCH --error="soybean_M4_NEW_%j.err"

# Conda environment
source /u/bkim2/miniforge3/etc/profile.d/conda.sh
conda activate isa_yield_env

# to avoid PROJ conflict
unset PROJ_DATA
unset PROJ_LIB
unset SLURM_NTASKS

echo "=========================================="
echo "Job started on $(hostname) at $(date)"
echo "=========================================="

# Config files to run
CONFIGS=(
    "conf_M4/NEW/s12sd_6_soybean.yaml"
    "conf_M4/NEW/s12sd_8_soybean.yaml"
    "conf_M4/NEW/s12sd_10_soybean.yaml"
    "conf_M4/NEW/s12sd_14_soybean.yaml"
    "conf_M4/NEW/s12sd_24_soybean.yaml"
)

# Run each config
for config in "${CONFIGS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running: $config"
    echo "Start time: $(date)"
    echo "=========================================="
    
    # Set wandb run name based on config file
    config_name=$(basename "$config" .yaml)
    export WANDB_RUN_NAME="${config_name}_${SLURM_JOB_ID}"
    export WANDB_GROUP="M4_soybean_weekly"
    
    terratorch fit -c "$config"
    
    echo "Finished: $config at $(date)"
    echo ""
done

echo "=========================================="
echo "All jobs completed at $(date)"
echo "=========================================="