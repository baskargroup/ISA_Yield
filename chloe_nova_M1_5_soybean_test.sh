#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --job-name="Test_Native_Soybean"
#SBATCH --mail-user=bgekim@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="Native_Soybean_Test%j.out"
#SBATCH --error="Native_Soybean_Test%j.err"

# conda environment
source /work/mech-ai-scratch/bgekim/miniconda3/etc/profile.d/conda.sh
conda activate isa_yield_env

# Test native soybean model (24 weeks)
CKPT_FILE="output/soybean/M1_5/lr_e5/checkpoints/best-epoch=290.ckpt"


if [[ -n "$CKPT_FILE" ]]; then
    echo "Testing native soybean with checkpoint: $CKPT_FILE"
    terratorch test -c conf_native_soybean/s12d_24_new.yaml --ckpt_path "$CKPT_FILE"
else
    echo "Error: No checkpoint found for native soybean!"
fi

echo "Test finished"