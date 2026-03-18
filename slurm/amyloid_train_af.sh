#!/usr/bin/env bash
#SBATCH --job-name=cogni_amyloid_af
#SBATCH --output=/home/usuaris/veussd/roger.esteve.sanchez/CogniAligned/logs/slurm/%x_%j.txt
#SBATCH -A veu
#SBATCH -p veu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --exclude=veuc05
#SBATCH --time=08:00:00

set -euo pipefail

REPO_DIR="/home/usuaris/veussd/roger.esteve.sanchez/CogniAligned"
cd "$REPO_DIR"

source .venv/bin/activate

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

export HF_HOME="/home/usuaris/veussd/roger.esteve.sanchez/CogniAligned/.cache/huggingface"
export WANDB_DIR="/home/usuaris/veussd/roger.esteve.sanchez/CogniAligned/.cache/wandb"
mkdir -p "$HF_HOME" "$WANDB_DIR"
mkdir -p "/home/usuaris/veussd/roger.esteve.sanchez/CogniAligned/logs/slurm"

date

CONFIG_FILE="modules/configs/amyloid/default.yaml"

echo "Starting CogniAlign+AF Amyloid training with config: $CONFIG_FILE"
echo "use_acoustic_features: $(grep use_acoustic_features $CONFIG_FILE)"

python -u modules/amyloid/main.py --config "$CONFIG_FILE"

date
echo "Training completed!"
