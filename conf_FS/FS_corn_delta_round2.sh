#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA100x4
#SBATCH --account=bepk-delta-gpu
#SBATCH --job-name="FS_Round2"
#SBATCH --mail-user=bgekim@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="logs/fs_corn/round2_all.out"
#SBATCH --error="logs/fs_corn/round2_all.err"

source /work/mech-ai-scratch/bgekim/miniconda3/etc/profile.d/conda.sh
conda activate isa_yield_env

for week in {1..24}; do
    if [ "$week" -eq 16 ]; then continue; fi
    
    CKPT_DIR="output/corn/FS_S12CDW/week_16_${week}/checkpoints"
    
    echo "======================================"
    echo "Week 16+${week} 시작: $(date)"
    echo "======================================"

    if [ -f "${CKPT_DIR}/last.ckpt" ]; then
        echo "⏭️ Week 16+${week} 이미 완료, skip!"
    else
        terratorch fit --config fs_yamls_corn_round2/config_fs_corn_week_16_${week}.yaml
        echo "✅ Finished Week 16+${week}: $(date)"
    fi

done

echo "🎉 Round 2 total compltete!"