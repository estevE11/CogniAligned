#!/usr/bin/env bash
#SBATCH --job-name=ppa_prep
#SBATCH --output=logs/slurm/%x_%j.txt
#SBATCH -A veu
#SBATCH -p veu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --nodelist=veuc10

set -euo pipefail

REPO_DIR="/home/usuaris/veussd/roger.esteve.sanchez/CogniAligned"
cd "$REPO_DIR"

if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found."
    exit 1
fi

source .venv/bin/activate
export PYTHONUNBUFFERED=1

echo "========================================"
echo "Starting PPA Preprocessing"
echo "========================================"

python -u modules/ppa/preprocess.py

echo "Preprocessing completed!"