#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --exclude=nova21-gpu-1,nova21-gpu-2
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --job-name="FS_Test_Corn"
#SBATCH --mail-user=bgekim@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="logs/fs_corn/test_round10.out"
#SBATCH --error="logs/fs_corn/test_round10.err"

# ✅ only modify here per each round!
SELECTED=(16 20 1 12 18 19 17 15 5)

# ========== Automatic Calculation ==========
selected_str=$(IFS=_; echo "${SELECTED[*]}")
round=$((${#SELECTED[@]} + 1))
yaml_dir="fs_yamls_corn_round${round}"

echo "🔄 Round ${round} Testing: weeks (${SELECTED[*]}) + remaining"

source /work/mech-ai-scratch/bgekim/miniconda3/etc/profile.d/conda.sh
conda activate isa_yield_env

# remaining weeks loop
for week in {1..24}; do
    skip=0
    for sw in "${SELECTED[@]}"; do
        if [ "$week" -eq "$sw" ]; then skip=1; break; fi
    done
    if [ "$skip" -eq 1 ]; then continue; fi

    combo="${selected_str}_${week}"
    yaml_path="${yaml_dir}/config_fs_corn_week_${combo}.yaml"
    ckpt_dir="output/corn/FS_S12CDW/week_${combo}/checkpoints"
    out_csv="predictions/S12cdw_Corn_week_${combo}.csv"

    if [ ! -f "${yaml_path}" ]; then
        echo "❌ YAML not found, skip: ${yaml_path}"
        continue
    fi

    if [ -f "${out_csv}" ]; then
        echo "⏭️  Already done, skip: week ${combo}"
        continue
    fi

    CKPT=$(ls ${ckpt_dir}/best-*.ckpt 2>/dev/null | tail -1)
    if [ -z "$CKPT" ]; then
        echo "❌ No best checkpoint found, skip: week ${combo}"
        continue
    fi

    echo "🚀 Testing week ${combo}: $(date)"
    terratorch test --config "${yaml_path}" --ckpt_path "${CKPT}"

    if [ $? -ne 0 ]; then
        echo "❌ terratorch failed for week ${combo}, skip"
        sleep 10
        continue
    fi

    # CSV rename
    OLD_CSV=$(ls predictions/*_Corn.csv 2>/dev/null | tail -1)
    if [ -n "$OLD_CSV" ]; then
        mv "$OLD_CSV" "${out_csv}"
        echo "📄 Saved: ${out_csv}"
    else
        echo "⚠️  No CSV found for week ${combo}"
    fi

    # Plot rename
    OLD_PLOT=$(ls plots/*_r2_plot.png 2>/dev/null | tail -1)
    if [ -n "$OLD_PLOT" ]; then
        mv "$OLD_PLOT" "plots/S12cdw_Corn_week_${combo}.png"
    fi

    echo "✅ Finished week ${combo}: $(date)"
done

echo "🎉 Round ${round} Testing complete!"