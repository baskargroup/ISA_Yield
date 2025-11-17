#!/bin/bash

#SBATCH --time=24:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # 36 processor core(s) per node 
#SBATCH --mem=369G   # maximum memory per node
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=nova    # gpu node(s)
#SBATCH --account=mech-ai
#SBATCH --job-name="S12_Finetune"
#SBATCH --mail-user=aapowadi@iastate.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="Finetune%j.out" # job standard output file (%j replaced by job id)
#SBATCH --error="Finetune%j.err" # job standard error file (%j replaced by job id)

# SBATCH --cpus-per-task=16   # spread out to use 1 core per numa, set to 64 if tasks is 1
# Environment setup
# module purge
# module load cuda/12.1
# module load python/3.10


echo "Job is starting on `hostname` for s12"

terratorch fit -c conf_ts/s12_1.yaml
terratorch fit -c conf_ts/s12_2.yaml
terratorch fit -c conf_ts/s12_3.yaml
terratorch fit -c conf_ts/s12_4.yaml
terratorch fit -c conf_ts/s12_5.yaml
terratorch fit -c conf_ts/s12_6.yaml
terratorch fit -c conf_ts/s12_7.yaml
terratorch fit -c conf_ts/s12_8.yaml
terratorch fit -c conf_ts/s12_9.yaml
terratorch fit -c conf_ts/s12_10.yaml
terratorch fit -c conf_ts/s12_11.yaml
terratorch fit -c conf_ts/s12_12.yaml

echo "Job finished for s12"