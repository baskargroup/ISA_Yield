#!/bin/bash

#SBATCH --time=24:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # 36 processor core(s) per node 
#SBATCH --mem=369G   # maximum memory per node
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=nova    # gpu node(s)
#SBATCH --account=mech-ai
#SBATCH --job-name="TerraFinetune"
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

echo "Job is starting on `hostname` for s12cd"

for yaml_file in conf_ts/s12cd_{1..12}.yaml; do
    base_name=$(basename "$yaml_file" .yaml)
    ckpt_dir="regr_mean/$(tr '[:lower:]' '[:upper:]' <<< ${base_name:0:1})${base_name:1}/checkpoints"
    ckpt_file=$(ls "$ckpt_dir"/*.ckpt 2>/dev/null | head -n 1)
    if [[ -n "$ckpt_file" ]]; then
        terratorch test -c "$yaml_file" --ckpt "$ckpt_file"
    else
        echo "Checkpoint not found for $yaml_file"
    fi
done

echo "Job finished for s12cd"

echo "Job is starting on `hostname` for s12cdw"

for yaml_file in conf_ts/s12cdw_{1..12}.yaml; do
    base_name=$(basename "$yaml_file" .yaml)
    ckpt_dir="regr_mean/$(tr '[:lower:]' '[:upper:]' <<< ${base_name:0:1})${base_name:1}/checkpoints"
    ckpt_file=$(ls "$ckpt_dir"/*.ckpt 2>/dev/null | head -n 1)
    if [[ -n "$ckpt_file" ]]; then
        terratorch test -c "$yaml_file" --ckpt "$ckpt_file"
    else
        echo "Checkpoint not found for $yaml_file"
    fi
done

echo "Job finished for s12cdw"

