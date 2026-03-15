#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=256G
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=nova21-gpu-1,nova21-gpu-2
#SBATCH --cpus-per-task=16
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --job-name="FS_Train"
#SBATCH --mail-user=bgekim@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="logs/fs_soybean/train_soybean_round3.out"
#SBATCH --error="logs/fs_soybean/train_soybean_round3.err"

# ✅ Only modify here per each round!
SELECTED=(16 18)

# ========== Automatic Calculation ==========
round=$((${#SELECTED[@]} + 1))

if [ ${#SELECTED[@]} -eq 0 ]; then
    selected_str=""
else
    selected_str=$(IFS=_; echo "${SELECTED[*]}")
fi

echo "🔄 Round ${round} Soybean Training"

source /work/mech-ai-scratch/bgekim/miniconda3/etc/profile.d/conda.sh
conda activate isa_yield_env

for week in {1..24}; do
    # Skip if the week is included in SELECTED
    skip=0
    for sw in "${SELECTED[@]}"; do
        if [ "$week" -eq "$sw" ]; then skip=1; break; fi
    done
    if [ "$skip" -eq 1 ]; then continue; fi

    # combo setting
    if [ ${#SELECTED[@]} -eq 0 ]; then
        combo="${week}"
    else
        combo="${selected_str}_${week}"
    fi

    CKPT_DIR="output/soybean/FS_S12SD/week_${combo}/checkpoints"

    echo "======================================"
    echo "Week ${combo} start: $(date)"
    echo "======================================"

    if [ -f "${CKPT_DIR}/last.ckpt" ]; then
        echo "⏭️ Week ${combo} already completed, skip!"
    else
        terratorch fit --config fs_yamls_soybean_round${round}/config_fs_soybean_week_${combo}.yaml
        echo "✅ Finished Week ${combo}: $(date)"
    fi
done

echo "🎉 Round ${round} Soybean total test complete!"