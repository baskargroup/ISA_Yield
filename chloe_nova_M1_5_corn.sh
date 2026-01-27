#!/bin/bash

#SBATCH --time=5:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # 36 processor core(s) per node 
#SBATCH --mem=256G   # maximum memory per node
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=nova    # gpu node(s)
#SBATCH --account=mech-ai
#SBATCH --job-name="M1.5_corn_1e_5"
#SBATCH --mail-user=bgekim@iastate.edu   # email address
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="M1.5_corn_1e_5%j.out" # job standard output file (%j replaced by job id)
#SBATCH --error="M1.5_corn_1e_5%j.err" # job standard error file (%j replaced by job id)
# SBATCH --cpus-per-task=16   # spread out to use 1 core per numa, set to 64 if tasks is 1
# Environment setup


# conda environment
source /work/mech-ai-scratch/bgekim/miniconda3/etc/profile.d/conda.sh
conda activate isa_yield_env

# Wandb setting
export WANDB_RUN_NAME="S12d_24_${SLURM_JOB_ID}"
export WANDB_GROUP="S12d_24"


echo "Job is starting on `hostname` for s12d-corn"


terratorch fit -c conf_M1_5/s12d_24_corn.yaml

echo "Job finished for s12d-corn"
