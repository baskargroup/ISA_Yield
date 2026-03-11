#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --job-name="FS_test_all"
#SBATCH --output="logs/fs_corn/test_all.out"
#SBATCH --error="logs/fs_corn/test_all.err"

source /work/mech-ai-scratch/bgekim/miniconda3/etc/profile.d/conda.sh
conda activate isa_yield_env

for week in {1..24}; do
    echo "=========================================="
    echo "Testing Week ${week} at $(date)"
    echo "=========================================="
    
    # best checkpoint 찾기
    CKPT=$(ls output/corn/FS_S12CDW/week_${week}/checkpoints/best-*.ckpt 2>/dev/null | head -1)
    
    if [ -z "$CKPT" ]; then
        echo "⚠️ Week ${week}: No checkpoint found, skipping..."
        continue
    fi
    
    echo "Using checkpoint: $CKPT"
    
    terratorch test --config fs_yamls_corn/config_fs_corn_week_${week}.yaml --ckpt_path $CKPT
    
    echo "Finished Week ${week} at $(date)"
    echo ""
done

echo "All weeks tested!"