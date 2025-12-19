#!/bin/bash

# ====================== SLURM DIRECTIVES ==========================
# Adjust node count for multi-node biweekly processing (12 iterations total)
# If you want one iteration per node, request --nodes=12; else fewer nodes will partition iterations.
#SBATCH --time=24:00:00              # walltime limit (HH:MM:SS)
#SBATCH --nodes=1                    # number of nodes (change as needed)
#SBATCH --ntasks-per-node=1          # one task per node (gives a unique NODE_RANK)
#SBATCH --cpus-per-task=16           # CPUs for intra-node multiprocessing
#SBATCH --mem=128G                   # memory per node (adjust if needed)
#SBATCH --partition=nova             # partition name
#SBATCH --account=mech-ai
#SBATCH --job-name="Test_Nova_Finetune"
#SBATCH --mail-user=aapowadi@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="Nova_Test%j.out"
#SBATCH --error="Nova_Test%j.err"

# Process soybean models
for i in {1..12}; do
    CKPT_FILE=$(find regr_soybean/S12wd_${i}/checkpoints -name "*.ckpt" -type f 2>/dev/null | head -n 1)
    if [[ -n "$CKPT_FILE" ]]; then
        echo "Processing soybean biweek $i with checkpoint: $CKPT_FILE"
        terratorch test -c conf_ts_soybean/s12wd_${i}.yaml --ckpt "$CKPT_FILE"
    else
        echo "Warning: No checkpoint found for soybean biweek $i, skipping..."
    fi
done

# Process corn models
for i in {1..12}; do
    CKPT_FILE=$(find regr_corn/S12wd_${i}/checkpoints -name "*.ckpt" -type f 2>/dev/null | head -n 1)
    if [[ -n "$CKPT_FILE" ]]; then
        echo "Processing corn biweek $i with checkpoint: $CKPT_FILE"
        terratorch test -c conf_ts_corn/s12wd_${i}.yaml --ckpt "$CKPT_FILE"
    else
        echo "Warning: No checkpoint found for corn biweek $i, skipping..."
    fi
done
