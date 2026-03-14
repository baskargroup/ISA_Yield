#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA100x4
#SBATCH --account=bepk-delta-gpu
#SBATCH --job-name="FS_Round2_Test"
#SBATCH --mail-user=bgekim@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="logs/fs_corn/round2_test_all.out"
#SBATCH --error="logs/fs_corn/round2_test_all.err"

source ~/.bashrc
conda activate isa_yield_env

unset PROJ_DATA
unset PROJ_LIB
unset SLURM_NTASKS

for week in {1..24}; do
    if [ "$week" -eq 16 ]; then continue; fi

    CKPT_DIR="output/corn/FS_S12CDW/week_16_${week}/checkpoints"

    echo "======================================"
    echo "Test Week 16+${week} 시작: $(date)"
    echo "======================================"

    # 이미 완료된 경우 skip
    if [ -f "predictions/S12cdw_Corn_week_16_${week}.csv" ]; then
        echo "⏭️ Week 16+${week} test 이미 완료, skip!"
        continue
    fi

    # best checkpoint 없으면 skip
    CKPT=$(ls ${CKPT_DIR}/best-*.ckpt 2>/dev/null)
    if [ -z "$CKPT" ]; then
        echo "❌ Week 16+${week} best checkpoint 없음, skip!"
        continue
    fi

    terratorch test --config fs_yamls_corn_round2/config_fs_corn_week_16_${week}.yaml --ckpt_path ${CKPT}

    # CSV rename
    OLD_CSV=$(ls predictions/*_Corn.csv 2>/dev/null | tail -1)
    if [ -n "$OLD_CSV" ]; then
        mv "$OLD_CSV" "predictions/S12cdw_Corn_week_16_${week}.csv"
        echo "✅ CSV 저장: predictions/S12cdw_Corn_week_16_${week}.csv"
    fi

    # Plot rename
    OLD_PLOT=$(ls plots/*_r2_plot.png 2>/dev/null | tail -1)
    if [ -n "$OLD_PLOT" ]; then
        mv "$OLD_PLOT" "plots/S12cdw_Corn_week_16_${week}.png"
        echo "✅ Plot 저장: plots/S12cdw_Corn_week_16_${week}.png"
    fi

    echo "✅ Finished Test Week 16+${week}: $(date)"
done

echo "🎉 Round 2 Test 전체 완료!"
