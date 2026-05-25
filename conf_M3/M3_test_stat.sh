#!/bin/bash

#SBATCH --time=168:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # 36 processor core(s) per node 
#SBATCH --mem=369G   # maximum memory per node
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=nova    # gpu node(s)
#SBATCH --account=mech-ai
#SBATCH --job-name="M3_test_stat"
#SBATCH --mail-user=aapowadi@iastate.edu   # email address
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="M3_test_stat%j.out" # job standard output file (%j replaced by job id)
#SBATCH --error="M3_test_stat%j.err" # job standard error file (%j replaced by job id)
# SBATCH --cpus-per-task=16   # spread out to use 1 core per numa, set to 64 if tasks is 1
# Environment setup


# conda environment
source /work/mech-ai-scratch/bgekim/miniconda3/etc/profile.d/conda.sh
conda activate isa_yield_env


echo "=========================================="
echo "Job started on $(hostname) at $(date)"
echo "=========================================="

for seed in 42 123 456 789; do
    for yaml_file in conf_M3_stat/*_test.yaml; do
        # Extract the base config name (e.g., s12_24_corn from s12_24_corn_test.yaml)
        base_name=$(basename "$yaml_file" _test.yaml)

        # Determine the modality code (e.g., S12, S12CDW, S12WD, etc.)
        modality=$(echo "$base_name" | sed 's/_24_corn//' | sed 's/_24_soybean//' | tr '[:lower:]' '[:upper:]')

        # Determine crop type
        if [[ "$base_name" == *"corn"* ]]; then
            crop="corn"
        else
            crop="soybean"
        fi

        # Find the best checkpoint for this seed
        ckpt_dir="output/$crop/M3/differentnorm/${modality}_stat/checkpoints/seed_${seed}"
        ckpt_file=$(find "$ckpt_dir" -name "best-*.ckpt" 2>/dev/null | head -1)

        if [ -f "$ckpt_file" ]; then
            echo "Running $yaml_file with seed $seed"
            echo "Using checkpoint: $ckpt_file"

            # Extract wandb name from yaml and create seed-specific temp yaml
            wandb_name=$(grep -A 3 'WandbLogger' "$yaml_file" | grep 'name:' | sed 's/.*name: //')
            tmp_yaml=$(mktemp /tmp/terratorch_XXXXXX.yaml)
            sed -e "s|seed_everything: .*|seed_everything: ${seed}|" \
                -e "s|name: ${wandb_name}|name: ${wandb_name}_seed${seed}|" \
                "$yaml_file" > "$tmp_yaml"

            terratorch test -c "$tmp_yaml" --ckpt "$ckpt_file"
            rm -f "$tmp_yaml"
        else
            echo "WARNING: No checkpoint found for $yaml_file seed $seed"
        fi
    done
done
echo "Job finished for M3_test_stat"