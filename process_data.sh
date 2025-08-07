#!/bin/bash

#SBATCH --time=48:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=64   # 64 processor core(s) per node 
#SBATCH --mem=369G   # maximum memory per node
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --job-name="Data_Processing"
#SBATCH --mail-user=aapowadi@iastate.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="data_saving%j.out" # job standard output file (%j replaced by job id)
#SBATCH --error="data_saving%j.err" # job standard error file (%j replaced by job id)

# Capture the number of nodes from SLURM
NUM_NODES=$SLURM_NNODES

module load openmpi/4.1.5
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/work/mech-ai-scratch/aapowadi/miniforge3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/work/mech-ai-scratch/aapowadi/miniforge3/etc/profile.d/conda.sh" ]; then
        . "/work/mech-ai-scratch/aapowadi/miniforge3/etc/profile.d/conda.sh"
    else
        export PATH="/work/mech-ai-scratch/aapowadi/miniforge3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate prc
cd /work/mech-ai-scratch/aapowadi/ISA_Yield

echo "Start time for LULC: $(date)"
python save_parquet_chunks.py #--year 2014
python combine_chunks.py #--year 2014