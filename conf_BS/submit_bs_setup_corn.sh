#!/bin/bash
#SBATCH --job-name=bs_prep_corn
#SBATCH --partition=nova
#SBATCH --array=1-16
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --mail-user=bgekim@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/generate_dataset/corn/%a.out
#SBATCH --error=logs/generate_dataset/corn/%a.err

mkdir -p logs/generate_dataset/corn

# ✅ modify only here!
REMOVED="11 12 6 14 10 4 2 8 3 15 9"  # corn round10

for w in $REMOVED; do
    if [ "$SLURM_ARRAY_TASK_ID" -eq $w ]; then
        echo "Skipping week $w (removed)"
        exit 0
    fi
done

source /work/mech-ai-scratch/bgekim/miniconda3/etc/profile.d/conda.sh
conda activate isa_yield_env

python bs_setup_corn.py $SLURM_ARRAY_TASK_ID