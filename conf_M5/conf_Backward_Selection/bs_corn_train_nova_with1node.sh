#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=nova   
#SBATCH --account=mech-ai
#SBATCH --mail-user=bgekim@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name="BS_Corn_R3"
#SBATCH --output="logs/bs_corn/round3/BS_round3_%j.out"
#SBATCH --error="logs/bs_corn/round3/BS_round3_%j.err"


# ========================================================
# ✅ Please edit only this section for each round!
# ========================================================
round=3
REMOVED_WEEKS=(8 19)
# ========================================================

removed_str=$(IFS=_; echo "${REMOVED_WEEKS[*]}")

source /work/mech-ai-scratch/bgekim/miniconda3/etc/profile.d/conda.sh
conda activate isa_yield_env

mkdir -p logs/bs_corn/round3

for rm_week in $(seq 1 20); do
    skip=0
    for p_rm in "${REMOVED_WEEKS[@]}"; do
        if [ "$rm_week" -eq "$p_rm" ]; then skip=1; break; fi
    done
    if [ "$skip" -eq 1 ]; then
        echo "🚫 Week ${rm_week} already removed, skipping."
        continue
    fi

    if [ ${#REMOVED_WEEKS[@]} -eq 0 ]; then
        YAML_FILE="bs_yamls_corn_bs20_round${round}/config_bs_corn_rm_${rm_week}.yaml"
        CKPT_DIR="output/corn/BS_S12WDC/bs20/round${round}/rm_${rm_week}/checkpoints"
    else
        YAML_FILE="bs_yamls_corn_bs20_round${round}/config_bs_corn_rm_${removed_str}_${rm_week}.yaml"
        CKPT_DIR="output/corn/BS_S12WDC/bs20/round${round}/rm_${removed_str}_${rm_week}/checkpoints"
    fi

    if [ ! -f "${YAML_FILE}" ]; then
        echo "❌ YAML not found, skip: ${YAML_FILE}"
        continue
    fi

    echo "🚀 Backward Round ${round}: Removing Week ${rm_week}"

    if [ -f "${CKPT_DIR}/last.ckpt" ]; then
        echo "⏭️ rm_${rm_week} already done, skipping."
    else
        terratorch fit --config "$YAML_FILE"
        echo "✅ Finished rm_${rm_week} at $(date)"
    fi
done

echo "🎉 Backward Round ${round} complete!"