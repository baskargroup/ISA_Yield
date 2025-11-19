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

# terratorch fit -c conf_ts/s12_1.yaml
# terratorch fit -c conf_ts/s12_2.yaml
# terratorch fit -c conf_ts/s12_3.yaml
# terratorch fit -c conf_ts/s12_4.yaml
# terratorch fit -c conf_ts/s12_5.yaml
# terratorch fit -c conf_ts/s12_6.yaml
# terratorch fit -c conf_ts/s12_7.yaml
# terratorch fit -c conf_ts/s12_8.yaml
# terratorch fit -c conf_ts/s12_9.yaml
# terratorch fit -c conf_ts/s12_10.yaml
# terratorch fit -c conf_ts/s12_11.yaml
# terratorch fit -c conf_ts/s12_12.yaml

terratorch test -c conf_ts/s12_1.yaml --ckpt regr_mean/S12_1/checkpoints/epoch=57-step=1566.ckpt
terratorch test -c conf_ts/s12_2.yaml --ckpt regr_mean/S12_2/checkpoints/epoch=63-step=1728.ckpt
terratorch test -c conf_ts/s12_3.yaml --ckpt regr_mean/S12_3/checkpoints/epoch=36-step=999.ckpt
terratorch test -c conf_ts/s12_4.yaml --ckpt regr_mean/S12_4/checkpoints/epoch=49-step=1350.ckpt
terratorch test -c conf_ts/s12_5.yaml --ckpt regr_mean/S12_5/checkpoints/epoch=60-step=1647.ckpt
terratorch test -c conf_ts/s12_6.yaml --ckpt regr_mean/S12_6/checkpoints/epoch=68-step=1863.ckpt
terratorch test -c conf_ts/s12_7.yaml --ckpt regr_mean/S12_7/checkpoints/epoch=74-step=2025.ckpt
terratorch test -c conf_ts/s12_8.yaml --ckpt regr_mean/S12_8/checkpoints/epoch=60-step=1647.ckpt
terratorch test -c conf_ts/s12_9.yaml --ckpt regr_mean/S12_9/checkpoints/epoch=104-step=2835.ckpt
terratorch test -c conf_ts/s12_10.yaml --ckpt regr_mean/S12_10/checkpoints/epoch=96-step=2619.ckpt
terratorch test -c conf_ts/s12_11.yaml --ckpt regr_mean/S12_11/checkpoints/epoch=72-step=1971.ckpt
terratorch test -c conf_ts/s12_12.yaml --ckpt regr_mean/S12_12/checkpoints/epoch=119-step=3240.ckpt

echo "Job finished for s12"