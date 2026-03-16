#!/usr/bin/env bash
#SBATCH --job-name=cogni_test
#SBATCH --output=/home/usuaris/veussd/roger.esteve.sanchez/CogniAligned/logs/slurm/%x_%j.txt
#SBATCH -A veu
#SBATCH -p veu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
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
export TOKENIZERS_PARALLELISM=false

echo "========================================"
echo "Evaluating Model on Test Set"
echo "========================================"

python -u modules/test.py --config modules/configs/default.yaml

echo "Testing completed!"