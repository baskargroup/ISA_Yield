#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA100x4
#SBATCH --account=bepk-delta-gpu
#SBATCH --job-name="FS_Test"
#SBATCH --mail-user=bgekim@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="logs/fs_corn/test_all.out"
#SBATCH --error="logs/fs_corn/test_all.err"

# ✅ 매 라운드마다 여기만 수정!
SELECTED=(16 20)

# ========== 자동 계산 ==========
selected_str=$(IFS=_; echo "${SELECTED[*]}")
round=$((${#SELECTED[@]} + 1))
yaml_dir="fs_yamls_corn_round${round}"

echo "🔄 Round ${round} Testing: weeks (${SELECTED[*]}) + remaining"

source ~/.bashrc
conda activate isa_yield_env

unset PROJ_DATA
unset PROJ_LIB
unset SLURM_NTASKS

# remaining weeks 루프
for week in {1..24}; do
    # SELECTED에 포함된 week이면 skip
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

    # CSV rename
    OLD_CSV=$(ls predictions/*_Corn.csv 2>/dev/null | tail -1)
    if [ -n "$OLD_CSV" ]; then
        mv "$OLD_CSV" "${out_csv}"
        echo "📄 Saved: ${out_csv}"
    fi

    # Plot rename
    OLD_PLOT=$(ls plots/*_r2_plot.png 2>/dev/null | tail -1)
    if [ -n "$OLD_PLOT" ]; then
        mv "$OLD_PLOT" "plots/S12cdw_Corn_week_${combo}.png"
    fi

    echo "✅ Finished week ${combo}: $(date)"
done

echo "🎉 Round ${round} Testing complete!"