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
#SBATCH --job-name="Rep_Test_soybean"
#SBATCH --mail-user=bgekim@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="logs/replication/soybean/test.out"
#SBATCH --error="logs/replication/soybean/test.err"

source /work/mech-ai-scratch/bgekim/miniconda3/etc/profile.d/conda.sh
conda activate isa_yield_env

for seed in 10 20 30; do
    CKPT_DIR="output/soybean/replication/week_16_18_1/checkpoints_seed${seed}"
    out_csv="predictions/soybean/replication/week_16_18_1_seed${seed}.csv"
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
    terratorch test --config fs_yamls_soybean_round3_rep/config_fs_soybean_week_16_18_1_seed${seed}.yaml \
        --ckpt_path "${CKPT}"

    # CSV rename
    OLD_CSV=$(ls predictions/*_Soybean.csv 2>/dev/null | tail -1)
    if [ -n "$OLD_CSV" ]; then
        mv "$OLD_CSV" "${out_csv}"
        echo "📄 Saved: ${out_csv}"
    else
        echo "⚠️ No CSV found for seed ${seed}"
    fi

    # Plot rename
    OLD_PLOT=$(ls plots/*_r2_plot.png 2>/dev/null | tail -1)
    if [ -n "$OLD_PLOT" ]; then
        mv "$OLD_PLOT" "plots/S12sd_soybean_week_16_18_1_seed${seed}.png"
    fi

    echo "✅ Finished seed ${seed}: $(date)"
done
echo "🎉 soybean replication testing complete!"