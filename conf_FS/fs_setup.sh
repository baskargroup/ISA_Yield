#!/bin/bash
#SBATCH --job-name=fs_prep_soybean
#SBATCH --partition=nova
#SBATCH --array=1-24
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/generate_dataset/fs_soybean/%a.out
#SBATCH --error=logs/generate_dataset/fs_soybean/%a.err

mkdir -p logs/generate_dataset/fs_soybean

# ✅ modify only here!
SELECTED="18 5 12 8 7 11 17 16 9"  # soybean

for w in $SELECTED; do
    if [ "$SLURM_ARRAY_TASK_ID" -eq $w ]; then
        echo "Skipping week $w (already selected)"
        exit 0
    fi
done

source /work/mech-ai-scratch/bgekim/miniconda3/etc/profile.d/conda.sh
conda activate isa_yield_env

python fs_setup.py $SLURM_ARRAY_TASK_ID