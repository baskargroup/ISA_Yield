#!/bin/bash

#SBATCH --time=48:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # 36 processor core(s) per node 
#SBATCH --mem=100G   # maximum memory per node
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=scavenger    # gpu node(s)
#SBATCH --reservation=mech-ai
#SBATCH --job-name="M3_Soybean_stat_seed2048"
#SBATCH --mail-user=aapowadi@iastate.edu   # email address
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="M3_Soybean_stat_seed2048_%j.out" # job standard output file (%j replaced by job id)
#SBATCH --error="M3_Soybean_stat_seed2048_%j.err" # job standard error file (%j replaced by job id)
# SBATCH --cpus-per-task=16   # spread out to use 1 core per numa, set to 64 if tasks is 1
# Environment setup


# conda environment
source /work/mech-ai-scratch/bgekim/miniconda3/etc/profile.d/conda.sh
conda activate isa_yield_env


echo "Job is starting on `hostname` for M3_Soybean_stat seed 2048"

seed=2048
for yaml_file in conf_M3_stat/*_soybean.yaml; do
    # Extract base wandb name and checkpoint dir from yaml
    wandb_name=$(grep -A 3 'WandbLogger' "$yaml_file" | grep 'name:' | sed 's/.*name: //')
    ckpt_dir=$(grep 'dirpath:' "$yaml_file" | sed 's/.*dirpath: //')

    # Skip if already completed
    if [[ -f "${ckpt_dir}/seed_${seed}/last.ckpt" ]]; then
        echo "Skipping $yaml_file with seed $seed (already completed)"
        continue
    fi

    # Create temp yaml with seed-specific overrides
    tmp_yaml=$(mktemp /tmp/terratorch_XXXXXX.yaml)
    sed -e "s|seed_everything: .*|seed_everything: ${seed}|" \
        -e "s|name: ${wandb_name}|name: ${wandb_name}_seed${seed}|" \
        -e "s|dirpath: ${ckpt_dir}|dirpath: ${ckpt_dir}/seed_${seed}|" \
        "$yaml_file" > "$tmp_yaml"

    echo "Running $yaml_file with seed $seed (wandb: ${wandb_name}_seed${seed})"
    terratorch fit -c "$tmp_yaml"
    rm -f "$tmp_yaml"
done
echo "Job finished for M3_soybean_stat seed $seed"
