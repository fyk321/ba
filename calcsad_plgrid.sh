#!/bin/bash
# Athena GPU node specs: 128 cores, 1024GB RAM, 8x A100-40GB per node
#SBATCH --job-name=s2nd_3_5_masks
#SBATCH --account=CHANGEME-gpu-a100          # <-- CHANGEME: your grant ID (check with `hpc-grants`)
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=400GB

cd "${HOME}/cnn3"

module purge

module load GCC/11.3.0  OpenMPI/4.1.4 PyTorch/1.13.1-CUDA-11.7.0

# The bundled Pillow module lacks support for WebP, using one from venv
module unload Pillow
source ~/venv/bin/activate 

# Set DDP master address using SLURM hostnames
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

# --- Dataset and output paths (using $SCRATCH) ---
DATASET_ROOT="${SCRATCH}/datasets"
CHECKPOINTS_ROOT="${SCRATCH}/checkpoints"
RUNS_ROOT="${SCRATCH}/runs"

SCENEFLOW_DIR="${DATASET_ROOT}/sceneflow"

# Export paths so they are inherited by srun remote tasks
export SCENEFLOW_DIR CHECKPOINTS_ROOT RUNS_ROOT

# SceneFlow (uses --sceneflow_dir mode)
echo "Computing SAD disparities for SceneFlow..."
python sad_compute.py --sceneflow_dir "$SCENEFLOW_DIR" --max_disp 192
