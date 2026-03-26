#!/bin/bash

#SBATCH --time=4:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # 36 processor core(s) per node 
#SBATCH --mem=369G   # maximum memory per node
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=nova    # gpu node(s)
#SBATCH --account=mech-ai
#SBATCH --job-name="M5_Soybean_stat"
#SBATCH --mail-user=aapowadi@iastate.edu   # email address
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="M5_Soybean_stat%j.out" # job standard output file (%j replaced by job id)
#SBATCH --error="M5_Soybean_stat%j.err" # job standard error file (%j replaced by job id)
# SBATCH --cpus-per-task=16   # spread out to use 1 core per numa, set to 64 if tasks is 1
# Environment setup


# conda environment
source /work/mech-ai-scratch/bgekim/miniconda3/etc/profile.d/conda.sh
conda activate isa_yield_env


echo "Job is starting on `hostname` for M5_Soybean_stat"

# terratorch fit -c fs_yamls_soybean_round3/config_fs_soybean_week_16_18_1.yaml

terratorch test -c fs_yamls_soybean_round3/config_fs_soybean_week_16_18_1.yaml --ckpt output/soybean/M5_S12SD/week_16_18_1/checkpoints/best-epoch=116.ckpt


echo "Job finished for M5_Soybean_stat"