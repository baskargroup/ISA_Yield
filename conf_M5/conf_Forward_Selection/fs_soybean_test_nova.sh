#!/bin/bash
#SBATCH --time=24:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # 36 processor core(s) per node 
#SBATCH --mem=100G   # maximum memory per node
#SBATCH --cpus-per-task=16
#SBATCH --exclude=nova21-gpu-1,nova21-gpu-2
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --job-name="FS_Test_Soybean_NEW"
#SBATCH --mail-user=bgekim@iastate.edu   # email address
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="logs/fs_soybean/test_soybean_round10.out"
#SBATCH --error="logs/fs_soybean/test_soybean_round10.err"

# ✅ only modify here per each round!
SELECTED=(18 5 12 8 7 11 17 16 9) # should be the same as training

# ========== automatic calculate ==========
round=$((${#SELECTED[@]} + 1))
if [ ${#SELECTED[@]} -eq 0 ]; then
    selected_str=""
else
    selected_str=$(IFS=_; echo "${SELECTED[*]}")
fi

echo "🔄 Round ${round} Soybean S12W Testing"

source /work/mech-ai-scratch/bgekim/miniconda3/etc/profile.d/conda.sh
conda activate isa_yield_env

for week in {1..24}; do
    skip=0
    for sw in "${SELECTED[@]}"; do
        if [ "$week" -eq "$sw" ]; then skip=1; break; fi
    done
    if [ "$skip" -eq 1 ]; then continue; fi

    if [ ${#SELECTED[@]} -eq 0 ]; then
        combo="${week}"
    else
        combo="${selected_str}_${week}"
    fi

    yaml_path="fs_yamls_soybean_round${round}/config_fs_soybean_week_${combo}.yaml"
    ckpt_dir="output/soybean/FS_S12W/week_${combo}/checkpoints"
    out_csv="predictions/S12w_Soybean_week_${combo}.csv"

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
    OLD_CSV=$(ls predictions/*_Soybean.csv 2>/dev/null | tail -1)
    if [ -n "$OLD_CSV" ]; then
        mv "$OLD_CSV" "${out_csv}"
        echo "📄 Saved: ${out_csv}"
    else
        echo "⚠️  No CSV found for week ${combo}"
    fi

    # Plot rename
    OLD_PLOT=$(ls plots/*_r2_plot.png 2>/dev/null | tail -1)
    if [ -n "$OLD_PLOT" ]; then
        mv "$OLD_PLOT" "plots/S12w_Soybean_week_${combo}.png"
    fi

    echo "✅ Finished week ${combo}: $(date)"
done

echo "🎉 Round ${round} Soybean S12W Testing complete!"