#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --exclude=nova21-gpu-1,nova21-gpu-2
#SBATCH --gres=gpu:1
#SBATCH --partition=scavenger
#SBATCH --reservation=mech-ai-1
#SBATCH --job-name="BS_Corn_Test_R12"
#SBATCH --mail-user=bgekim@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="logs/bs_corn/round12/BS_test_round12_%j.out"
#SBATCH --error="logs/bs_corn/round12/BS_test_round12_%j.err"

# ========================================================
# ✅ Please edit only this section for each round!
# ========================================================
round=12
REMOVED_WEEKS=(11 12 6 14 10 4 2 8 3 15 9)  # previously removed weeks
# ========================================================

removed_str=$(IFS=_; echo "${REMOVED_WEEKS[*]}")

source /work/mech-ai-scratch/bgekim/miniconda3/etc/profile.d/conda.sh
conda activate isa_yield_env

mkdir -p logs/bs_corn/round${round}
mkdir -p predictions/bs_corn/w16/round${round}
mkdir -p plots/bs_corn/w16/round${round}

for rm_week in $(seq 1 16); do
    # Skip already removed weeks
    skip=0
    for p_rm in "${REMOVED_WEEKS[@]}"; do
        if [ "$rm_week" -eq "$p_rm" ]; then skip=1; break; fi
    done
    if [ "$skip" -eq 1 ]; then
        echo "🚫 Week ${rm_week} already removed, skipping."
        continue
    fi

    if [ ${#REMOVED_WEEKS[@]} -eq 0 ]; then
        combo="${rm_week}"
        YAML_FILE="bs_yamls_corn_bs16_round${round}/config_bs_corn_rm_${rm_week}.yaml"
        CKPT_DIR="output/corn/BS_S12WDC/bs16/round${round}/rm_${rm_week}/checkpoints"
    else
        combo="${removed_str}_${rm_week}"
        YAML_FILE="bs_yamls_corn_bs16_round${round}/config_bs_corn_rm_${removed_str}_${rm_week}.yaml"
        CKPT_DIR="output/corn/BS_S12WDC/bs16/round${round}/rm_${removed_str}_${rm_week}/checkpoints"
    fi

    out_csv="predictions/bs_corn/w16/round${round}/S12wdc_Corn_bs16_round${round}_rm_${combo}.csv"

    if [ ! -f "${YAML_FILE}" ]; then
        echo "❌ YAML not found, skip: ${YAML_FILE}"
        continue
    fi

    if [ -f "${out_csv}" ]; then
        echo "⏭️ Already done, skip: rm_${combo}"
        continue
    fi

    CKPT=$(ls ${CKPT_DIR}/best-*.ckpt 2>/dev/null | tail -1)
    if [ -z "$CKPT" ]; then
        echo "❌ No best checkpoint found, skip: rm_${combo}"
        continue
    fi

    echo "🚀 Testing rm_${combo}: $(date)"
    terratorch test --config "${YAML_FILE}" --ckpt_path "${CKPT}"

    if [ $? -ne 0 ]; then
        echo "❌ terratorch failed for rm_${combo}, skip"
        sleep 10
        continue
    fi

    # CSV rename
    OLD_CSV=$(ls predictions/*_Corn.csv 2>/dev/null | tail -1)
    if [ -n "$OLD_CSV" ]; then
        mv "$OLD_CSV" "predictions/bs_corn/w16/round${round}/S12wdc_Corn_bs16_round${round}_rm_${combo}.csv"
        echo "📄 Saved: ${out_csv}"
    else
        echo "⚠️ No CSV found for rm_${combo}"
    fi

    # Plot rename
    OLD_PLOT=$(ls plots/*_r2_plot.png 2>/dev/null | tail -1)
    if [ -n "$OLD_PLOT" ]; then
        mv "$OLD_PLOT" "plots/bs_corn/w16/round${round}/S12wdc_Corn_bs16_round${round}_rm_${combo}.png"
    fi

    echo "✅ Finished rm_${combo}: $(date)"
done

echo "🎉 Backward Round ${round} Corn Testing complete!"