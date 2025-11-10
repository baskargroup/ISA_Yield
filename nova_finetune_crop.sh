#!/bin/bash

#SBATCH --time=24:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # 36 processor core(s) per node 
#SBATCH --mem=369G   # maximum memory per node
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=nova    # gpu node(s)
#SBATCH --account=mech-ai
#SBATCH --job-name="Finetune_Crop"
#SBATCH --mail-user=aapowadi@iastate.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="Finetune_Crop%j.out" # job standard output file (%j replaced by job id)
#SBATCH --error="Finetune_Crop%j.err" # job standard error file (%j replaced by job id)

# SBATCH --cpus-per-task=16   # spread out to use 1 core per numa, set to 64 if tasks is 1
# Environment setup
# module purge
# module load cuda/12.1
# module load python/3.10

echo "Job is starting on `hostname` for s12wd"
export WANDB_HTTP_TIMEOUT=120
# terratorch fit -c conf_crop/s12wd_12_Corn.yaml
# terratorch fit -c conf_crop/s12wd_12_Soybean.yaml

terratorch fit -c conf_year/s12wc_12_2021_corn.yaml
terratorch fit -c conf_year/s12wc_12_2021_soybean.yaml
terratorch fit -c conf_year/s12wd_12_2021_corn.yaml
terratorch fit -c conf_year/s12wd_12_2021_soybean.yaml
terratorch fit -c conf_year/s12cdw_12_2021_corn.yaml
terratorch fit -c conf_year/s12cdw_12_2021_soybean.yaml

echo "Job finished for s12wd"
