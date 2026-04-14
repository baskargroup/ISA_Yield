#!/bin/bash

#SBATCH --time=48:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # 36 processor core(s) per node
#SBATCH --mem=100G   # maximum memory per node
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=scavenger    # gpu node(s)
#SBATCH --reservation=mech-ai
#SBATCH --job-name="M3_test_stat_by_seed"
#SBATCH --mail-user=aapowadi@iastate.edu   # email address
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="M3_test_stat_by_seed_%j.out" # job standard output file (%j replaced by job id)
#SBATCH --error="M3_test_stat_by_seed_%j.err" # job standard error file (%j replaced by job id)

# conda environment
source /work/mech-ai-scratch/bgekim/miniconda3/etc/profile.d/conda.sh
conda activate isa_yield_env

set -euo pipefail

echo "=========================================="
echo "Job started on $(hostname) at $(date)"
echo "=========================================="

mkdir -p predictions plots

for seed in 42 123 456 789 101 234 567 890 1024 2048; do
    seed_prediction_dir="predictions/seed_${seed}"
    seed_plot_dir="plots/seed_${seed}"
    mkdir -p "$seed_prediction_dir" "$seed_plot_dir"

    for yaml_file in conf_M3_stat/*soybean_test.yaml; do
        base_name=$(basename "$yaml_file" _test.yaml)
        modality=$(echo "$base_name" | sed 's/_24_corn//' | sed 's/_24_soybean//' | tr '[:lower:]' '[:upper:]')

        if [[ "$base_name" == *"corn"* ]]; then
            crop="corn"
            crop_label="Corn"
        else
            crop="soybean"
            crop_label="Soybean"
        fi

        ckpt_dir="output/$crop/M3/differentnorm/${modality}_stat/checkpoints/seed_${seed}"
        ckpt_file=$(find "$ckpt_dir" -name "best-*.ckpt" 2>/dev/null | head -1)

        if [[ ! -f "$ckpt_file" ]]; then
            echo "WARNING: No checkpoint found for $yaml_file seed $seed"
            continue
        fi

        echo "Running $yaml_file with seed $seed"
        echo "Using checkpoint: $ckpt_file"

        wandb_name=$(grep -A 3 'WandbLogger' "$yaml_file" | grep 'name:' | sed 's/.*name: //')
        tmp_yaml=$(mktemp /tmp/terratorch_XXXXXX.yaml)
        before_predictions=$(mktemp /tmp/m3_predictions_before_XXXXXX.txt)
        before_plots=$(mktemp /tmp/m3_plots_before_XXXXXX.txt)

        find predictions -maxdepth 1 -type f | sort > "$before_predictions"
        find plots -maxdepth 1 -type f | sort > "$before_plots"

        sed -e "s|seed_everything: .*|seed_everything: ${seed}|" \
            -e "s|name: ${wandb_name}|name: ${wandb_name}_seed${seed}|" \
            "$yaml_file" > "$tmp_yaml"

        terratorch test -c "$tmp_yaml" --ckpt "$ckpt_file"

        while IFS= read -r new_prediction; do
            [[ -n "$new_prediction" ]] || continue
            mv "$new_prediction" "$seed_prediction_dir/$(basename "$new_prediction")"
            echo "Saved prediction to $seed_prediction_dir/$(basename "$new_prediction")"
        done < <(find predictions -maxdepth 1 -type f | sort | comm -13 "$before_predictions" -)

        while IFS= read -r new_plot; do
            [[ -n "$new_plot" ]] || continue
            mv "$new_plot" "$seed_plot_dir/$(basename "$new_plot")"
            echo "Saved plot to $seed_plot_dir/$(basename "$new_plot")"
        done < <(find plots -maxdepth 1 -type f | sort | comm -13 "$before_plots" -)

        expected_prediction=$(find "$seed_prediction_dir" -maxdepth 1 -type f -name "*${crop_label}.csv" | grep "${modality}" | tail -1 || true)
        if [[ -z "$expected_prediction" ]]; then
            echo "WARNING: No prediction CSV captured for $yaml_file seed $seed"
        fi

        rm -f "$tmp_yaml" "$before_predictions" "$before_plots"
    done
done

echo "Job finished for M3_test_stat_by_seed"