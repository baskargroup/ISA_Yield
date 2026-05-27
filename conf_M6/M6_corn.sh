#!/bin/bash

#SBATCH --time=4:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # 36 processor core(s) per node 
#SBATCH --mem=100G   # maximum memory per node
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=nova    # gpu node(s)
#SBATCH --account=mech-ai
#SBATCH --job-name="M6_c"
#SBATCH --mail-user=aapowadi@iastate.edu   # email address
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="M6_corn_%j.out" # job standard output file (%j replaced by job id)
#SBATCH --error="M6_corn_%j.err" # job standard error file (%j replaced by job id)
# SBATCH --cpus-per-task=16   # spread out to use 1 core per numa, set to 64 if tasks is 1
# Environment setup


# conda environment
source /work/mech-ai-scratch/bgekim/miniconda3/etc/profile.d/conda.sh
conda activate isa_yield_env

# terratorch fit -c config_bs_corn_rm_8_19_7_17_12_9_10_2_1.yaml
terratorch test -c config_bs_corn_rm_8_19_7_17_12_9_10_2_1.yaml --ckpt output/corn/BS_S12WDC/bs20/round9/rm_8_19_7_17_12_9_10_2_1/checkpoints/best-epoch=077.ckpt
echo "Job finished for M3_corn_stat seed $seed"
