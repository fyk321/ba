#!/bin/bash
# Athena GPU node specs: 128 cores, 1024GB RAM, 8x A100-40GB per node
#SBATCH --job-name=i2nd_infer_images
#SBATCH --account=CHANGEME-gpu-a100
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=plgrid-gpu-a100           # <-- CHANGEME: partition (e.g. plgrid-gpu-a100, plgrid-gpu)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100GB

cd "${HOME}/cnn3"

module purge

module load GCC/11.3.0  OpenMPI/4.1.4 PyTorch/1.13.1-CUDA-11.7.0

module unload Pillow
source ~/venv/bin/activate

# Force PyTorch DDP to use IPv4 loopback (avoiding IPv6 socket family errors on PLGrid compute nodes)
export MASTER_ADDR=127.0.0.1

# --- Dataset and output paths (using $SCRATCH) ---
DATASET_ROOT="${SCRATCH}/datasets"
CHECKPOINTS_ROOT="${SCRATCH}/checkpoints"

SCENEFLOW_DIR="${DATASET_ROOT}/sceneflow"                # <-- CHANGEME if your folder names differ

OUTPUT_DIR="${HOME}/img/sceneflow_2nd_3_5_epoch96_flying"

python infer_images.py \
    --sceneflow_dir "${SCRATCH}/datasets/sceneflow" \
    --checkpoint "${SCRATCH}/checkpoints2/sceneflow_2nd_3_5/best_model.pth" \
    --subdataset "frames" \
    --sample_idx 42 \
    --output_dir "${OUTPUT_DIR}"
