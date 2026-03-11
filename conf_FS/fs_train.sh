#!/bin/bash
#SBATCH --time=14:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuH200x8
#SBATCH --account=bepk-delta-gpu
#SBATCH --job-name="FS_Train"
#SBATCH --mail-user=bgekim@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="logs/fs_corn/train_all.out"
#SBATCH --error="logs/fs_corn/train_all.err"

# ✅ 매 라운드마다 여기만 수정!
SELECTED=(16 20)

# ========== 자동 계산 ==========
selected_str=$(IFS=_; echo "${SELECTED[*]}")
round=$((${#SELECTED[@]} + 1))

source ~/.bashrc
conda activate isa_yield_env

unset PROJ_DATA
unset PROJ_LIB
unset SLURM_NTASKS

for week in {1..24}; do
    skip=0
    for sw in "${SELECTED[@]}"; do
        if [ "$week" -eq "$sw" ]; then skip=1; break; fi
    done
    if [ "$skip" -eq 1 ]; then continue; fi

    CKPT_DIR="output/corn/FS_S12CDW/week_${selected_str}_${week}/checkpoints"

    echo "======================================"
    echo "Week ${selected_str}+${week} 시작: $(date)"
    echo "======================================"

    if [ -f "${CKPT_DIR}/last.ckpt" ]; then
        echo "⏭️ Week ${selected_str}+${week} 이미 완료, skip!"
    else
        terratorch fit --config fs_yamls_corn_round${round}/config_fs_corn_week_${selected_str}_${week}.yaml
        echo "✅ Finished Week ${selected_str}+${week}: $(date)"
    fi
done

echo "🎉 Round ${round} 전체 완료!"