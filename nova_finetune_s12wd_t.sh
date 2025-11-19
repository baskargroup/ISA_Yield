#!/bin/bash

#SBATCH --time=24:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # 36 processor core(s) per node 
#SBATCH --mem=369G   # maximum memory per node
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=nova    # gpu node(s)
#SBATCH --account=mech-ai
#SBATCH --job-name="S12WD_Finetune"
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

echo "Job is starting on `hostname` for s12wd"

# terratorch fit -c conf_ts/s12wd_1.yaml
# terratorch fit -c conf_ts/s12wd_2.yaml
# terratorch fit -c conf_ts/s12wd_3.yaml
# terratorch fit -c conf_ts/s12wd_4.yaml
# terratorch fit -c conf_ts/s12wd_5.yaml
# terratorch fit -c conf_ts/s12wd_6.yaml
# terratorch fit -c conf_ts/s12wd_7.yaml
# terratorch fit -c conf_ts/s12wd_8.yaml
# terratorch fit -c conf_ts/s12wd_9.yaml
# terratorch fit -c conf_ts/s12wd_10.yaml
# terratorch fit -c conf_ts/s12wd_11.yaml
# terratorch fit -c conf_ts/s12wd_12.yaml

terratorch test -c conf_ts/s12wd_1.yaml --ckpt regr_mean/S12wd_1/checkpoints/epoch=83-step=2268.ckpt
terratorch test -c conf_ts/s12wd_2.yaml --ckpt regr_mean/S12wd_2/checkpoints/epoch=76-step=2079.ckpt
terratorch test -c conf_ts/s12wd_3.yaml --ckpt regr_mean/S12wd_3/checkpoints/epoch=62-step=1701.ckpt
terratorch test -c conf_ts/s12wd_4.yaml --ckpt regr_mean/S12wd_4/checkpoints/epoch=56-step=1539.ckpt
terratorch test -c conf_ts/s12wd_5.yaml --ckpt regr_mean/S12wd_5/checkpoints/epoch=70-step=1917.ckpt
terratorch test -c conf_ts/s12wd_6.yaml --ckpt regr_mean/S12wd_6/checkpoints/epoch=95-step=2592.ckpt
terratorch test -c conf_ts/s12wd_7.yaml --ckpt regr_mean/S12wd_7/checkpoints/epoch=95-step=2592.ckpt
terratorch test -c conf_ts/s12wd_8.yaml --ckpt regr_mean/S12wd_8/checkpoints/epoch=97-step=2646.ckpt
terratorch test -c conf_ts/s12wd_9.yaml --ckpt regr_mean/S12wd_9/checkpoints/epoch=88-step=2403.ckpt
terratorch test -c conf_ts/s12wd_10.yaml --ckpt regr_mean/S12wd_10/checkpoints/epoch=118-step=3213.ckpt
terratorch test -c conf_ts/s12wd_11.yaml --ckpt regr_mean/S12wd_11/checkpoints/epoch=84-step=2295.ckpt
terratorch test -c conf_ts/s12wd_12.yaml --ckpt regr_mean/S12wd_12/checkpoints/epoch=88-step=2403.ckpt

echo "Job finished for s12wd"
