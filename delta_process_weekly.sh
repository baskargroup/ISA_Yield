#!/bin/bash
#SBATCH --job-name="Finetune"
#SBATCH --partition=gpuH200x8
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16   # spread out to use 1 core per numa, set to 64 if tasks is 1  
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bepk-delta-gpu   # <- match to a "Project" returned by the "accounts" command
#SBATCH --exclusive  # dedicated node for this job
#SBATCH --no-requeue 
#SBATCH -t 06:00:00
#SBATCH -e Finetune-%j.err
#SBATCH -o Finetune-%j.out

# SBATCH --cpus-per-task=16   # spread out to use 1 core per numa, set to 64 if tasks is 1
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo "Head node: $head_node"
echo "Head node IP: $head_node_ip"

export LOGLEVEL=INFO

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn
module load nccl # loads the nccl built with the AWS nccl plugin for Slingshot11
module list
echo "Job is starting on `hostname`"

terratorch fit -c conf_native_corn/s12d_12_new.yaml
# terratorch fit -c configs/config_s12.yaml 
# terratorch fit -c configs/config_s12c.yaml 
# terratorch fit -c configs/config_s12cs.yaml
# terratorch fit -c configs/config_s1cd.yaml
# # terratorch fit -c configs/config_s1csd.yaml 
# # terratorch fit -c configs/config_s12s.yaml 
# terratorch fit -c configs/config_s1sd.yaml 
# terratorch fit -c configs/config_s1d.yaml 
