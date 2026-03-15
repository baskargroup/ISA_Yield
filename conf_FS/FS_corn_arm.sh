#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=256G
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=nova-arm
#SBATCH --account=mech-ai
#SBATCH --job-name="FS_test"
#SBATCH --output="logs/fs_corn/test.out"
#SBATCH --error="logs/fs_corn/test.err"

source /work/mech-ai-scratch/bgekim/miniconda3-arm/etc/profile.d/conda.sh
conda activate isa_yield_env_arm

terratorch fit --config fs_yamls_corn/config_fs_corn_week_3.yaml