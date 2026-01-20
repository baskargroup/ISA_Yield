#!/bin/bash

#SBATCH --time=24:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # 36 processor core(s) per node 
#SBATCH --mem=369G   # maximum memory per node
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=nova    # gpu node(s)
#SBATCH --account=mech-ai
#SBATCH --job-name="Corn_Finetune"
#SBATCH --mail-user=bgekim@iastate.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="Finetune%j.out" # job standard output file (%j replaced by job id)
#SBATCH --error="Finetune%j.err" # job standard error file (%j replaced by job id)

# SBATCH --cpus-per-task=16   # spread out to use 1 core per numa, set to 64 if tasks is 1
# Environment setup


# conda environment
source /work/mech-ai-scratch/bgekim/miniconda3/etc/profile.d/conda.sh
conda activate isa_yield_env

# Wandb setting
export WANDB_RUN_NAME="S12d_24_${SLURM_JOB_ID}"
export WANDB_GROUP="S12d_24"


echo "Job is starting on `hostname` for s12wd-corn"

# terratorch fit -c conf_ts_corn/s12wd_1.yaml
# terratorch fit -c conf_ts_corn/s12wd_2.yaml
# terratorch fit -c conf_ts_corn/s12wd_3.yaml
# terratorch fit -c conf_ts_corn/s12wd_4.yaml
# terratorch fit -c conf_ts_corn/s12wd_5.yaml
# terratorch fit -c conf_ts_corn/s12wd_6.yaml
# terratorch fit -c conf_ts_corn/s12wd_7.yaml
# terratorch fit -c conf_ts_corn/s12wd_8.yaml
# terratorch fit -c conf_native_corn/s12d_9.yaml
# terratorch fit -c conf_native_corn/s12d_10.yaml
# terratorch fit -c conf_native_corn/s12d_11.yaml
# terratorch fit -c conf_native_corn/s12d_12.yaml

terratorch fit -c conf_native_corn/s12d_24_new.yaml

echo "Job finished for s12wd-corn"
