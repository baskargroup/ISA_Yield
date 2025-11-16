#!/bin/bash

# ====================== SLURM DIRECTIVES ==========================
# Adjust node count for multi-node biweekly processing (12 iterations total)
# If you want one iteration per node, request --nodes=12; else fewer nodes will partition iterations.
#SBATCH --time=24:00:00              # walltime limit (HH:MM:SS)
#SBATCH --nodes=4                    # number of nodes (change as needed)
#SBATCH --ntasks-per-node=1          # one task per node (gives a unique NODE_RANK)
#SBATCH --cpus-per-task=16           # CPUs for intra-node multiprocessing
#SBATCH --mem=128G                   # memory per node (adjust if needed)
#SBATCH --partition=nova             # partition name
#SBATCH --account=mech-ai
#SBATCH --job-name="biweek_processing"
#SBATCH --mail-user=aapowadi@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="biweek_processing_%j.out"
#SBATCH --error="biweek_processing_%j.err"

# ====================== ENV / MODULE SETUP ========================
echo "[INFO] Starting job on $(hostname) at $(date)"
module purge
# Load required modules (example; adjust to cluster environment)
# module load cuda/12.1
# module load python/3.10

# ====================== CONFIGURATION =============================
PYTHON=${PYTHON:-python}
SCRIPT=process_ts_biweek.py
UNPROCESSED_ROOT=./unprocessed_data
WORKERS_PER_NODE=${WORKERS_PER_NODE:-14}   # leave some cores for overhead

# ====================== PREPROCESS (MULTI-NODE) ===================
echo "[INFO] Launching multi-node biweekly preprocessing"
echo "[INFO] SLURM_JOB_NUM_NODES=$SLURM_JOB_NUM_NODES SLURM_NODEID=$SLURM_NODEID"

# Each node will automatically pick its assigned iterations via SLURM_NODEID and total nodes.
# --all triggers distribution across 12 iterations (1..12) inside the script.
srun --kill-on-bad-exit=1 $PYTHON $SCRIPT --all --workers $WORKERS_PER_NODE

PREPROCESS_STATUS=$?
if [ $PREPROCESS_STATUS -ne 0 ]; then
	echo "[ERROR] Preprocessing failed with status $PREPROCESS_STATUS"
	exit $PREPROCESS_STATUS
fi
echo "[INFO] Preprocessing complete at $(date)"
