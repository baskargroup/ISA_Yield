#!/bin/bash

#SBATCH --time=4:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # 36 processor core(s) per node 
#SBATCH --mem=369G   # maximum memory per node
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=nova    # gpu node(s)
#SBATCH --account=mech-ai
#SBATCH --job-name="M5_Corn_stat"
#SBATCH --mail-user=aapowadi@iastate.edu   # email address
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="M5_Corn_stat%j.out" # job standard output file (%j replaced by job id)
#SBATCH --error="M5_Corn_stat%j.err" # job standard error file (%j replaced by job id)
# SBATCH --cpus-per-task=16   # spread out to use 1 core per numa, set to 64 if tasks is 1
# Environment setup


# conda environment
source /work/mech-ai-scratch/bgekim/miniconda3/etc/profile.d/conda.sh
conda activate isa_yield_env


echo "Job is starting on `hostname` for M5_Corn"

# terratorch fit -c fs_yamls_corn_round6/config_fs_corn_week_16_20_1_12_18_19.yaml

terratorch test -c fs_yamls_corn_round6/config_M5.yaml --ckpt output/corn/M5/week_16_20_1_12_18_19/checkpoints/best-epoch=053.ckpt

echo "Job finished for M5_Corn"