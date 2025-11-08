#!/bin/bash

#SBATCH --time=24:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # 36 processor core(s) per node 
#SBATCH --mem=369G   # maximum memory per node
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=nova    # gpu node(s)
#SBATCH --account=mech-ai
#SBATCH --job-name="TeraFineConfig"
#SBATCH --mail-user=aapowadi@iastate.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="FineConfig%j.out" # job standard output file (%j replaced by job id)
#SBATCH --error="FineConfig%j.err" # job standard error file (%j replaced by job id)

# SBATCH --cpus-per-task=16   # spread out to use 1 core per numa, set to 64 if tasks is 1
# Environment setup
# module purge
# module load cuda/12.1
# module load python/3.10

echo "Job is starting on `hostname`"

# Loop through all YAML config files in ./configs folder
# for config_file in ./configs/*.yaml; do
#     if [ -f "$config_file" ]; then
#         echo "Running terratorch fit with config: $config_file"
#         terratorch fit -c "$config_file"
#     fi
# done

terratorch fit -c configs/config_s12ws.yaml

echo "All config files processed"

