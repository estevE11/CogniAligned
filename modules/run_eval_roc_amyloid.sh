#!/bin/bash
#SBATCH --output /home/usuaris/veussd/roger.esteve.sanchez/ad-detection/wandb/slurm_%x_%j.txt
#SBATCH -A veu
#SBATCH -p veu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --ntasks=1
#SBATCH --job-name=eval_roc_cognialign_amyloid

date

cd /home/usuaris/veussd/roger.esteve.sanchez/CogniAligned
source .venv/bin/activate

export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

mkdir -p roc_data

CKPT_DIR="logs/amyloid_distil_wav2vec2_P_crossgated_mean"

for fold in 0 1 2 3 4; do
    ckpt="${CKPT_DIR}/model_fold_${fold}.pth"
    [ -f "$ckpt" ] || continue
    echo "=== CogniAlign+AF fold${fold} ==="
    python modules/eval_roc.py \
        --config modules/configs/amyloid/default.yaml \
        --task amyloid --fold $fold \
        --checkpoint "$ckpt" \
        --label "CogniAlign + AF" \
        --output "roc_data/amyloid_CogniAlign_plus_AF_fold${fold}.json" \
        --cpu
done

echo "Done with CogniAlign+AF amyloid eval"
date
