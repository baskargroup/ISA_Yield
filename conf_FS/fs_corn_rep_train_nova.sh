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
#SBATCH --job-name="Rep_Train_Corn"
#SBATCH --mail-user=bgekim@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="logs/replication/corn/train.out"
#SBATCH --error="logs/replication/corn/train.err"

source /work/mech-ai-scratch/bgekim/miniconda3/etc/profile.d/conda.sh
conda activate isa_yield_env

for seed in 0 1 2; do
    CKPT_DIR="output/corn/replication/week_16_20_1_12_18_19/checkpoints_seed${seed}"
    echo "======================================"
    echo "Seed ${seed} start: $(date)"
    echo "======================================"
    if [ -f "${CKPT_DIR}/last.ckpt" ]; then
        echo "⏭️ Seed ${seed} already completed, skip!"
    else
        terratorch fit --config fs_yamls_corn_round6_rep/config_fs_corn_week_16_20_1_12_18_19_seed${seed}.yaml
        echo "✅ Finished Seed ${seed}: $(date)"
    fi
done
echo "🎉 Corn replication training complete!"