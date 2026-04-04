#!/bin/bash

#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuH200x8
#SBATCH --account=bepk-delta-gpu
#SBATCH --job-name="M3_test"
#SBATCH --mail-user=bgekim@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="M3_%j.out"
#SBATCH --error="M3_%j.err"

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

for yaml_file in conf_M3_stat/*_test.yaml; do
    echo "Running $yaml_file"
    
    # Extract the base config name (e.g., s12_24_corn from s12_24_corn_test.yaml)
    base_name=$(basename "$yaml_file" _test.yaml)
    
    # Determine the modality code (e.g., S12, S12CDW, S12WD, etc.)
    modality=$(echo "$base_name" | sed 's/_24_corn//' | sed 's/_24_soybean//' | tr '[:lower:]' '[:upper:]')
    
    # Determine crop type
    if [[ "$base_name" == *"corn"* ]]; then
        crop="corn"
    else
        crop="soybean"
    fi
    
    # Find the checkpoint file
    ckpt_file=$(find "output/$crop/M3/differentnorm/${modality}_stat/checkpoints" -name "best-*.ckpt" 2>/dev/null | head -1)
    
    if [ -f "$ckpt_file" ]; then
        echo "Using checkpoint: $ckpt_file"
        terratorch test -c "$yaml_file" --ckpt "$ckpt_file"
    else
        echo "WARNING: No checkpoint found for $yaml_file"
    fi
done



echo "Job finished for M3_test_stat"