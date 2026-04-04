#!/bin/bash

#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuH200x8
#SBATCH --account=bepk-delta-gpu
#SBATCH --job-name="M3_test"
#SBATCH --mail-user=bgekim@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="M3_%j.out"
#SBATCH --error="M3_%j.err"

# Conda environment
source /u/apowadi/miniforge3/etc/profile.d/conda.sh
conda activate isa_yield_env

# to avoid PROJ conflict
unset PROJ_DATA
unset PROJ_LIB
unset SLURM_NTASKS

echo "=========================================="
echo "Job started on $(hostname) at $(date)"
echo "=========================================="

terratorch test -c conf_M3_stat/s12ws_24_corn.yaml --ckpt output/corn/M3/differentnorm/S12WS_stat/checkpoints/best-epoch=139.ckpt

# terratorch test -c conf_M3_stat/s12wdsc_24_soybean.yaml --ckpt output/soybean/M3/differentnorm/S12WDSC_stat/checkpoints/best-epoch=036.ckpt

echo "Job finished for M3_test_stat"