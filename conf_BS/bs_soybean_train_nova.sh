#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --exclude=nova21-gpu-1,nova21-gpu-2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=scavenger
#SBATCH --reservation=mech-ai-1
#SBATCH --array=1-16
#SBATCH --job-name="BS_Soy_Multi"
#SBATCH --mail-user=bgekim@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="logs/bs_soybean/round12/BS_R%a.out"
#SBATCH --error="logs/bs_soybean/round12/BS_R%a.err"
# ========================================================
# ✅ Please edit only this section for each round!
# ========================================================
round=12
REMOVED_WEEKS=(9 11 5 4 14 16 6 3 13 10 16)  # previously removed weeks
# ========================================================
# Check if the current array ID is in the list of removed weeks
for p_rm in "${REMOVED_WEEKS[@]}"; do
    if [ "$SLURM_ARRAY_TASK_ID" -eq "$p_rm" ]; then
        echo "🚫 Week $SLURM_ARRAY_TASK_ID was already removed in previous rounds. Exit."
        exit 0
    fi
done
rm_week=$SLURM_ARRAY_TASK_ID
# Build removed_str dynamically
removed_str=$(IFS=_; echo "${REMOVED_WEEKS[*]}")
source /work/mech-ai-scratch/bgekim/miniconda3/etc/profile.d/conda.sh
conda activate isa_yield_env
if [ ${#REMOVED_WEEKS[@]} -eq 0 ]; then
    YAML_FILE="bs_yamls_soybean_bs16_round${round}/config_bs_soybean_rm_${rm_week}.yaml"
    CKPT_DIR="output/soybean/BS_S12W/bs16/round${round}/rm_${rm_week}/checkpoints"
else
    YAML_FILE="bs_yamls_soybean_bs16_round${round}/config_bs_soybean_rm_${removed_str}_${rm_week}.yaml"
    CKPT_DIR="output/soybean/BS_S12W/bs16/round${round}/rm_${removed_str}_${rm_week}/checkpoints"
fi
echo "🚀 Backward Selection Round ${round}: Removing Week ${rm_week} (previously removed: ${removed_str})"
if [ -f "${CKPT_DIR}/last.ckpt" ]; then
    echo "⏭️ rm_${rm_week} already done, skipping."
else
    terratorch fit --config "$YAML_FILE"
    echo "✅ Finished rm_${rm_week} at $(date)"
fi