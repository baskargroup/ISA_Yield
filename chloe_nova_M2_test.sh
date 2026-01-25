#!/bin/bash
#SBATCH --time=4:00:00   # test는 짧게
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=369G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --job-name="soybean_s12wds_Test"
#SBATCH --mail-user=bgekim@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="soybean_s12wds_Test%j.out"
#SBATCH --error="soybean_s12wds_Test%j.err"

# Environment setup
source /work/mech-ai-scratch/bgekim/miniconda3/etc/profile.d/conda.sh
conda activate isa_yield_env

# Wandb setting
export WANDB_RUN_NAME="S12wds_24_test_${SLURM_JOB_ID}"
export WANDB_GROUP="S12wds_24_test"

echo "Job is starting on `hostname` for s12wds-soybean TEST"
terratorch test -c conf_crop/s12wds_24_Soybean_new.yaml --ckpt_path output/soybean/M2/S12wds_le_5/checkpoints/best-epoch=201.ckpt

echo "Job finished for s12wds-soybean TEST"