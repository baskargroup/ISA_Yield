#!/bin/bash

#SBATCH --time=168:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # 36 processor core(s) per node 
#SBATCH --mem=369G   # maximum memory per node
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=nova    # gpu node(s)
#SBATCH --account=mech-ai
#SBATCH --job-name="M3_test_stat"
#SBATCH --mail-user=aapowadi@iastate.edu   # email address
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="M3_test_stat%j.out" # job standard output file (%j replaced by job id)
#SBATCH --error="M3_test_stat%j.err" # job standard error file (%j replaced by job id)
# SBATCH --cpus-per-task=16   # spread out to use 1 core per numa, set to 64 if tasks is 1
# Environment setup


# conda environment
source /work/mech-ai-scratch/bgekim/miniconda3/etc/profile.d/conda.sh
conda activate isa_yield_env


echo "Job is starting on `hostname` for M3_test_stat"

for yaml_file in \
    conf_M3_stat/s12w_24_corn_test.yaml \
    conf_M3_stat/s12c_24_corn_test.yaml \
    conf_M3_stat/s12ws_24_corn_test.yaml \
    conf_M3_stat/s12dc_24_corn_test.yaml \
    conf_M3_stat/s12wds_24_corn_test.yaml \
    conf_M3_stat/s12wsc_24_corn_test.yaml; do
    echo "Running $yaml_file"
    
    # Extract the base config name (e.g., s12_24_corn from s12_24_corn_test.yaml)
    base_name=$(basename "$yaml_file" _test.yaml)
    
    # Determine the modality code (e.g., S12, S12CDW, S12WD, etc.)
    modality=$(echo "$base_name" | sed 's/_24_corn//' | sed 's/_24_soybean//' | tr '[:lower:]' '[:upper:]')
    
    # Determine crop type
    if [[ "$base_name" == *"corn"* ]]; then
        crop="corn"
    else
        crop="soybean"
    fi
    
    # Find the checkpoint file
    ckpt_file=$(find "output/$crop/M3/differentnorm/${modality}_stat/checkpoints" -name "best-*.ckpt" 2>/dev/null | head -1)
    
    if [ -f "$ckpt_file" ]; then
        echo "Using checkpoint: $ckpt_file"
        terratorch test -c "$yaml_file" --ckpt "$ckpt_file"
    else
        echo "WARNING: No checkpoint found for $yaml_file"
    fi
done

echo "Job finished for M3_test_stat"