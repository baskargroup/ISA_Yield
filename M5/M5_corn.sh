#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA100x4
#SBATCH --account=bepk-delta-gpu
#SBATCH --job-name="M5_corn"
#SBATCH --mail-user=bgekim@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="M5_corn_%j.out"
#SBATCH --error="M5_corn_%j.err"

# Conda environment - delta
source /u/bkim2/miniforge3/etc/profile.d/conda.sh
conda activate isa_yield_env

# to avoid PROJ conflict
unset PROJ_DATA
unset PROJ_LIB
unset SLURM_NTASKS

echo "Job is starting on $(hostname) for M5_corn"

cd /scratch/bepk/bkim2/ISA_Yield 

for yaml_file in M5/*_corn.yaml; do    # ← M5/ 추가!
    echo "Running $yaml_file"
    terratorch fit -c "$yaml_file"
done

echo "Job finished for M5_corn"