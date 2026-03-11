#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA100x4
#SBATCH --account=bepk-delta-gpu
#SBATCH --job-name="FS_week19"
#SBATCH --output="logs/fs_corn/test.out"
#SBATCH --error="logs/fs_corn/test.err"

source ~/.bashrc
conda activate isa_yield_env

terratorch fit --config fs_yamls_corn/config_fs_corn_week_19.yaml