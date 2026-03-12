#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA100x4
#SBATCH --account=bepk-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name="round1_setup"
#SBATCH --output="logs/fs_corn/round1_setup.out"
#SBATCH --error="logs/fs_corn/round1_setup.err"

source ~/.bashrc
conda activate isa_yield_env
python round1_setup.py