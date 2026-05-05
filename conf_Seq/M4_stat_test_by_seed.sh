#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=scavenger
#SBATCH --reservation=mech-ai
#SBATCH --job-name="M4_test_seq_by_seed"
#SBATCH --mail-user=bgekim@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="M4_test_seq_by_seed_%j.out"
#SBATCH --error="M4_test_seq_by_seed_%j.err"

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

    for yaml_file in conf_M4_stat/*.yaml; do
        base_name=$(basename "$yaml_file" .yaml)
        modality=$(echo "$base_name" | sed 's/_[0-9]*_corn//' | sed 's/_[0-9]*_soybean//' | tr '[:lower:]' '[:upper:]')

        if [[ "$base_name" == *"corn"* ]]; then
            crop="corn"
            crop_label="Corn"
        else
            crop="soybean"
            crop_label="Soybean"
        fi

        week=$(echo "$base_name" | grep -oP '(?<=_)\d+(?=_)' || true)
        ckpt_dir="output/$crop/seq/${modality}/week${week}/checkpoints/seed_${seed}"
        ckpt_file=$(find "$ckpt_dir" -name "best-*.ckpt" 2>/dev/null | head -1 || true)

        if [[ ! -f "$ckpt_file" ]]; then
            echo "WARNING: No checkpoint found for $yaml_file seed $seed"
            continue
        fi

        echo "Running $yaml_file with seed $seed"
        echo "Using checkpoint: $ckpt_file"

        wandb_name=$(grep -A 3 'WandbLogger' "$yaml_file" | grep 'name:' | sed 's/.*name: //')
        tmp_yaml=$(mktemp /tmp/terratorch_XXXXXX.yaml)
        before_predictions=$(mktemp /tmp/predictions_before_XXXXXX.txt)
        before_plots=$(mktemp /tmp/plots_before_XXXXXX.txt)

        find predictions -maxdepth 1 -type f 2>/dev/null | sort > "$before_predictions" || true
        find plots -maxdepth 1 -type f 2>/dev/null | sort > "$before_plots" || true

        sed -e "s|seed_everything: .*|seed_everything: ${seed}|" \
            -e "s|name: ${wandb_name}|name: ${wandb_name}_seed${seed}|" \
            "$yaml_file" > "$tmp_yaml"

        terratorch test -c "$tmp_yaml" --ckpt "$ckpt_file"

        # ✅ 수정: base_name + seed로 rename
        for f in $(find predictions -maxdepth 1 -type f | sort | comm -13 "$before_predictions" - || true); do
            ext="${f##*.}"
            mv "$f" "$seed_prediction_dir/${base_name}_seed${seed}.${ext}"
            echo "Saved prediction to $seed_prediction_dir/${base_name}_seed${seed}.${ext}"
        done

        for f in $(find plots -maxdepth 1 -type f | sort | comm -13 "$before_plots" - || true); do
            ext="${f##*.}"
            mv "$f" "$seed_plot_dir/${base_name}_seed${seed}.${ext}"
            echo "Saved plot to $seed_plot_dir/${base_name}_seed${seed}.${ext}"
        done

        rm -f "$tmp_yaml" "$before_predictions" "$before_plots"
    done
done

echo "Job finished for M4_test_seq_by_seed"