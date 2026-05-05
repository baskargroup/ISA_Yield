#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=nova21-gpu-1,nova21-gpu-2
#SBATCH --cpus-per-task=16
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --job-name="Rep_Test_Corn"
#SBATCH --mail-user=bgekim@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="logs/replication/corn/test.out"
#SBATCH --error="logs/replication/corn/test.err"

source /work/mech-ai-scratch/bgekim/miniconda3/etc/profile.d/conda.sh
conda activate isa_yield_env

for seed in 0 1 2; do
    CKPT_DIR="output/corn/replication/week_16_20_1_12_18_19/checkpoints_seed${seed}"
    out_csv="predictions/S12cdw_Corn_week_16_20_1_12_18_19_seed${seed}.csv"
    echo "======================================"
    echo "Seed ${seed} test start: $(date)"
    echo "======================================"

    if [ -f "${out_csv}" ]; then
        echo "⏭️ Seed ${seed} already tested, skip!"
        continue
    fi

    CKPT=$(ls ${CKPT_DIR}/best-*.ckpt 2>/dev/null | tail -1)
    if [ -z "$CKPT" ]; then
        echo "❌ No best checkpoint found for seed ${seed}, skip!"
        continue
    fi

    echo "🚀 Testing seed ${seed}: $(date)"
    terratorch test --config fs_yamls_corn_round6_rep/config_fs_corn_week_16_20_1_12_18_19_seed${seed}.yaml \
        --ckpt_path "${CKPT}"

    # CSV rename
    OLD_CSV=$(ls predictions/*_Corn.csv 2>/dev/null | tail -1)
    if [ -n "$OLD_CSV" ]; then
        mv "$OLD_CSV" "${out_csv}"
        echo "📄 Saved: ${out_csv}"
    else
        echo "⚠️ No CSV found for seed ${seed}"
    fi

    # Plot rename
    OLD_PLOT=$(ls plots/*_r2_plot.png 2>/dev/null | tail -1)
    if [ -n "$OLD_PLOT" ]; then
        mv "$OLD_PLOT" "plots/S12cdw_Corn_week_16_20_1_12_18_19_seed${seed}.png"
    fi

    echo "✅ Finished seed ${seed}: $(date)"
done
echo "🎉 Corn replication testing complete!"