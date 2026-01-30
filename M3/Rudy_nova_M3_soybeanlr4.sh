#!/bin/bash

#SBATCH --time=168:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # 36 processor core(s) per node 
#SBATCH --mem=369G   # maximum memory per node
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=nova    # gpu node(s)
#SBATCH --account=mech-ai
#SBATCH --job-name="M3_Soybean_lr4"
#SBATCH --mail-user=aapowadi@iastate.edu   # email address
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="M3_Soybean_lr4%j.out" # job standard output file (%j replaced by job id)
#SBATCH --error="M3_Soybean_lr4%j.err" # job standard error file (%j replaced by job id)
# SBATCH --cpus-per-task=16   # spread out to use 1 core per numa, set to 64 if tasks is 1
# Environment setup


# conda environment
source /work/mech-ai-scratch/bgekim/miniconda3/etc/profile.d/conda.sh
conda activate isa_yield_env


echo "Job is starting on `hostname` for M3_Soybean_lr4"

# for yaml_file in conf_M3_lre4/*_soybean.yaml; do
#     echo "Running $yaml_file"
#     terratorch fit -c "$yaml_file"
# done

terratorch fit -c conf_M3_lre4/s12cdws_24_soybean.yaml


echo "Job finished for M3_Soybean_lr4"