#!/usr/bin/env bash
#SBATCH --job-name=extract_features
#SBATCH --output=/home/usuaris/veussd/roger.esteve.sanchez/CogniAligned/logs/slurm/%x_%j.txt
#SBATCH -A veu
#SBATCH -p veu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=04:00:00

set -euo pipefail

REPO_DIR="/home/usuaris/veussd/roger.esteve.sanchez/CogniAligned"
cd "$REPO_DIR"

source .venv/bin/activate

echo "Extracting ADReSSo features..."
python modules/extract_acoustic_features.py /home/usuaris/veussd/roger.esteve.sanchez/adresso/ADReSSo21/diagnosis/train/audio --output_csv /home/usuaris/veussd/roger.esteve.sanchez/adresso/ADReSSo21/diagnosis/train/acoustic_features.csv

echo "Extracting WAB features..."
python modules/extract_acoustic_features.py /home/usuaris/veussd/roger.esteve.sanchez/WAB_samples --output_csv /home/usuaris/veussd/roger.esteve.sanchez/WAB_samples/acoustic_features.csv

echo "Extraction completed!"
