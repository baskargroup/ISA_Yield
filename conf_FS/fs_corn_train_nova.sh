#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=nova21-gpu-1,nova21-gpu-2
#SBATCH --cpus-per-task=16
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --job-name="FS_Train_Corn"
#SBATCH --mail-user=bgekim@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="logs/fs_corn/train_round10.out"
#SBATCH --error="logs/fs_corn/train_round10.err"

# ✅ only modify here per each round!
SELECTED=(16 20 1 12 18 19 17 15 5)

selected_str=$(IFS=_; echo "${SELECTED[*]}")
round=$((${#SELECTED[@]} + 1))

source /work/mech-ai-scratch/bgekim/miniconda3/etc/profile.d/conda.sh
conda activate isa_yield_env


for week in {1..24}; do
    skip=0
    for sw in "${SELECTED[@]}"; do
        if [ "$week" -eq "$sw" ]; then skip=1; break; fi
    done
    if [ "$skip" -eq 1 ]; then continue; fi

    CKPT_DIR="output/corn/FS_S12CDW/week_${selected_str}_${week}/checkpoints"

    echo "======================================"
    echo "Week ${selected_str}+${week} start: $(date)"
    echo "======================================"

    if [ -f "${CKPT_DIR}/last.ckpt" ]; then
        echo "⏭️ Week ${selected_str}+${week} already completed, skip!"
    else
        terratorch fit --config fs_yamls_corn_round${round}/config_fs_corn_week_${selected_str}_${week}.yaml
        echo "✅ Finished Week ${selected_str}+${week}: $(date)"
    fi
done

echo "🎉 Round ${round} total test complete!"