#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --job-name="BS_Test_Soybean_S12W"
#SBATCH --mail-user=bgekim@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="logs/bs_soybean/test_soybean_bs16_round12.out"
#SBATCH --error="logs/bs_soybean/test_soybean_bs16_round12.err"

# ✅ only modify here per each round!
START_WEEKS=16        # starting point 
REMOVED_WEEKS=(9 11 5 4 14 16 6 3 13 10 12)      # Round 1: (), Round 2: (best_removed_week), ...

# ========== automatic calculate ==========
start_str="${START_WEEKS}"
round_num=$((${#REMOVED_WEEKS[@]} + 1))

if [ ${#REMOVED_WEEKS[@]} -eq 0 ]; then
    removed_str=""
else
    removed_str=$(IFS=_; echo "${REMOVED_WEEKS[*]}")
fi

# Current weeks = 1~START_WEEKS minus REMOVED_WEEKS
CURRENT_WEEKS=()
for w in $(seq 1 ${START_WEEKS}); do
    skip=0
    for rw in "${REMOVED_WEEKS[@]}"; do
        if [ "$w" -eq "$rw" ]; then skip=1; break; fi
    done
    if [ "$skip" -eq 0 ]; then
        CURRENT_WEEKS+=($w)
    fi
done

echo "🔄 Backward Round ${round_num} Soybean S12W Testing (from Week ${start_str})"
echo "✅ Current weeks: ${CURRENT_WEEKS[@]}"

source /work/mech-ai-scratch/bgekim/miniconda3/etc/profile.d/conda.sh
conda activate isa_yield_env

mkdir -p logs/bs_soybean
mkdir -p predictions/bs_soybean/w16/round${round_num}
mkdir -p plots/bs_soybean/w16/round${round_num}

for remove_week in "${CURRENT_WEEKS[@]}"; do
    if [ ${#REMOVED_WEEKS[@]} -eq 0 ]; then
        combo="${remove_week}"
    else
        combo="${removed_str}_${remove_week}"
    fi

    yaml_path="bs_yamls_soybean_bs${start_str}_round${round_num}/config_bs_soybean_rm_${combo}.yaml"
    ckpt_dir="output/soybean/BS_S12W/bs${start_str}/round${round_num}/rm_${combo}/checkpoints"
    out_csv="predictions/bs_soybean/w16/round${round_num}/S12w_Soybean_bs${start_str}_round${round_num}_rm_${combo}.csv"

    if [ ! -f "${yaml_path}" ]; then
        echo "❌ YAML not found, skip: ${yaml_path}"
        continue
    fi

    if [ -f "${out_csv}" ]; then
        echo "⏭️  Already done, skip: rm_${combo}"
        continue
    fi

    CKPT=$(ls ${ckpt_dir}/best-*.ckpt 2>/dev/null | tail -1)
    if [ -z "$CKPT" ]; then
        echo "❌ No best checkpoint found, skip: rm_${combo}"
        continue
    fi

    echo "🚀 Testing rm_${combo}: $(date)"
    terratorch test --config "${yaml_path}" --ckpt_path "${CKPT}"

    if [ $? -ne 0 ]; then
        echo "❌ terratorch failed for rm_${combo}, skip"
        sleep 10
        continue
    fi

    # CSV rename
    OLD_CSV=$(ls predictions/*_Soybean.csv 2>/dev/null | tail -1)
    if [ -n "$OLD_CSV" ]; then
        mv "$OLD_CSV" "predictions/bs_soybean/w16/round${round_num}/S12w_Soybean_bs${start_str}_round${round_num}_rm_${combo}.csv"
        echo "📄 Saved: ${out_csv}"
    else
        echo "⚠️  No CSV found for rm_${combo}"
    fi

    # Plot rename
    OLD_PLOT=$(ls plots/*_r2_plot.png 2>/dev/null | tail -1)
    if [ -n "$OLD_PLOT" ]; then
        mv "$OLD_PLOT" "plots/bs_soybean/w16/round${round_num}/S12w_Soybean_bs${start_str}_round${round_num}_rm_${combo}.png"
    fi

    echo "✅ Finished rm_${combo}: $(date)"
done

echo "🎉 Backward Round ${round_num} Soybean S12W Testing complete!"