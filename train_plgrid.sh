#!/bin/bash
# Athena GPU node specs: 128 cores, 1024GB RAM, 8x A100-40GB per node
#SBATCH --job-name=t150_2nd_3_5_masks
#SBATCH --account=CHANGEME
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=400GB

cd "${HOME}/cnn3"

module purge

module load GCC/11.3.0  OpenMPI/4.1.4 PyTorch/1.13.1-CUDA-11.7.0

module unload Pillow
source ~/venv/bin/activate

# Set DDP master address using SLURM hostnames
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

# --- Dataset and output paths (using $SCRATCH) ---
DATASET_ROOT="${SCRATCH}/datasets"
CHECKPOINTS_ROOT="${SCRATCH}/checkpoints"
RUNS_ROOT="${SCRATCH}/runs"

SCENEFLOW_DIR="${DATASET_ROOT}/sceneflow"                # <-- CHANGEME if your folder names differ

# Export paths so they are inherited by srun remote tasks
export SCENEFLOW_DIR CHECKPOINTS_ROOT RUNS_ROOT

echo "Phase 1: Pre-training on SceneFlow..."
srun bash -c 'python -m torch.distributed.run \
    --nproc_per_node=4 \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py \
    --sceneflow_dir "$SCENEFLOW_DIR" \
    --epochs 150 \
    --batch_size 16 \
    --save_dir "$CHECKPOINTS_ROOT/sceneflow_2nd_3_5" \
    --log_dir "$RUNS_ROOT/sceneflow_2nd_3_5"'
echo "Job finished."
