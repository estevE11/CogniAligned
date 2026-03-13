#!/usr/bin/env bash
#SBATCH --job-name=cognialigned
#SBATCH --output=/home/usuaris/veussd/roger.esteve.sanchez/CogniAligned/logs/slurm/%x_%j.txt
#SBATCH -A veu
#SBATCH -p veu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --nodelist=veuc10

set -euo pipefail

REPO_DIR="/home/usuaris/veussd/roger.esteve.sanchez/CogniAligned"
cd "$REPO_DIR"

# Create and activate virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# NCCL Stability Fix (Prevents timeouts on nodes like veuc12)
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Redirect cache to project directory to avoid /tmp space issues
export HF_HOME="/home/usuaris/veussd/roger.esteve.sanchez/CogniAligned/.cache/huggingface"
export WANDB_DIR="/home/usuaris/veussd/roger.esteve.sanchez/CogniAligned/.cache/wandb"
mkdir -p "$HF_HOME" "$WANDB_DIR"

# Create logs directory for SLURM outputs
mkdir -p "/home/usuaris/veussd/roger.esteve.sanchez/CogniAligned/logs/slurm"

date

# ===================================
# Configuration
# ===================================
# Default config file (can be overridden by passing as first argument)
CONFIG_FILE="${1:-modules/configs/default.yaml}"

# W&B Configuration (override via environment variables)
WANDB_PROJECT="${WANDB_PROJECT:-CogniAligned}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

export WANDB_PROJECT
export WANDB_ENTITY
export WANDB_MODE

# Optional: Set SLURM_JOB_ID for W&B run naming
export SLURM_JOB_ID="${SLURM_JOB_ID:-local}"

# ===================================
# Training Execution
# ===================================
echo "Starting training with config: $CONFIG_FILE"
echo "W&B Project: $WANDB_PROJECT"
echo "SLURM Job ID: $SLURM_JOB_ID"

srun python -u modules/main.py --config "$CONFIG_FILE"

date
echo "Training completed!"
