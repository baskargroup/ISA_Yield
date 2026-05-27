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
#SBATCH --job-name="Rep_Train_soybean"
#SBATCH --mail-user=bgekim@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="logs/replication/soybean/train.out"
#SBATCH --error="logs/replication/soybean/train.err"

source /work/mech-ai-scratch/bgekim/miniconda3/etc/profile.d/conda.sh
conda activate isa_yield_env

for seed in 10 20 30; do
    CKPT_DIR="output/soybean/replication/week_16_18_1/checkpoints_seed${seed}"
    echo "======================================"
    echo "Seed ${seed} start: $(date)"
    echo "======================================"
    if [ -f "${CKPT_DIR}/last.ckpt" ]; then
        echo "⏭️ Seed ${seed} already completed, skip!"
    else
        terratorch fit --config fs_yamls_soybean_round3_rep/config_fs_soybean_week_16_18_1_seed${seed}.yaml
        echo "✅ Finished Seed ${seed}: $(date)"
    fi
done
echo "🎉 soybean replication training complete!"