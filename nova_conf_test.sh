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

# Process all config files with corresponding checkpoints
for CONFIG_FILE in configs/*.yaml; do
    # Extract the base name (e.g., s12wd_11.yaml -> s12wd_11)
    BASE_NAME=$(basename "$CONFIG_FILE" .yaml)
    
    # Convert to the checkpoint directory naming convention (e.g., s12wd_11 -> S12wd_11)
    # Capitalize the first letter
    CKPT_DIR=$(echo "$BASE_NAME" | sed 's/^./\U&/')
    
    # Look for checkpoint file
    CKPT_FILE=$(find "config_test/${CKPT_DIR}/checkpoints" -name "*.ckpt" -type f 2>/dev/null | head -n 1)
    
    if [[ -n "$CKPT_FILE" ]]; then
        echo "Processing config $BASE_NAME with checkpoint: $CKPT_FILE"
        terratorch test -c "$CONFIG_FILE" --ckpt "$CKPT_FILE"
    else
        echo "Warning: No checkpoint found for config $BASE_NAME, skipping..."
    fi
done