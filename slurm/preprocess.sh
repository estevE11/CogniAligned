#!/usr/bin/env bash
#SBATCH --job-name=cogni_preprocess
#SBATCH --output=/home/usuaris/veussd/roger.esteve.sanchez/CogniAligned/logs/slurm/%x_%j.txt
#SBATCH -A veu
#SBATCH -p veu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --nodelist=veuc12

set -euo pipefail

# Use GPU for faster preprocessing (Whisper + DistilBERT)
# No CUDA_VISIBLE_DEVICES restriction - use GPU!

REPO_DIR="/home/usuaris/veussd/roger.esteve.sanchez/CogniAligned"
cd "$REPO_DIR"

# Activate virtual environment
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found. Please run training script first to create it."
    exit 1
fi

source .venv/bin/activate

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Redirect cache to project directory
export HF_HOME="/home/usuaris/veussd/roger.esteve.sanchez/CogniAligned/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

date

# ===================================
# Data Paths Configuration
# ===================================
ADRESSO_ROOT="/home/usuaris/veussd/roger.esteve.sanchez/adresso/ADReSSo21"
AUDIO_PATH="$ADRESSO_ROOT/diagnosis/train/audio"
OUTPUT_PATH="$ADRESSO_ROOT/diagnosis/train"

echo "========================================"
echo "CogniAlign Data Preprocessing"
echo "========================================"
echo "Audio Path: $AUDIO_PATH"
echo "Output Path: $OUTPUT_PATH"
echo ""

# Create output directories
if [ "${1:-}" == "--test" ]; then
    echo "Running in TEST mode..."
    mkdir -p "$OUTPUT_PATH/text"
else
    mkdir -p "$OUTPUT_PATH/text/ad"
    mkdir -p "$OUTPUT_PATH/text/cn"
fi

echo "Running preprocessing pipeline..."
echo "========================================"

# Export path for the preprocessing script
export ADRESSO_ROOT="$ADRESSO_ROOT"

# Run preprocessing
if [ "${1:-}" == "--test" ]; then
    python -u modules/preprocess/run_preprocessing.py --test
else
    python -u modules/preprocess/run_preprocessing.py
fi

date
echo "Preprocessing completed!"
echo ""
echo "Preprocessed data saved to: $OUTPUT_PATH/text/"
echo "Cross-validation splits saved to: $OUTPUT_PATH/splits/"
