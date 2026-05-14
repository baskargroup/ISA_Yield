#!/bin/bash
#SBATCH --job-name=bs_prep_w16
#SBATCH --partition=nova
#SBATCH --array=1-16
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --mail-user=bgekim@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/generate_dataset/bs16/%a.out
#SBATCH --error=logs/generate_dataset/bs16/%a.err

mkdir -p logs/generate_dataset/bs16

# ✅ modify only here!
REMOVED="9 11 5 4 14 16 6 3 13 10 12"  # soybean round8

for w in $REMOVED; do
    if [ "$SLURM_ARRAY_TASK_ID" -eq $w ]; then
        echo "Skipping week $w (removed)"
        exit 0
    fi
done

source /work/mech-ai-scratch/bgekim/miniconda3/etc/profile.d/conda.sh
conda activate isa_yield_env

python bs_setup_soybean.py $SLURM_ARRAY_TASK_ID