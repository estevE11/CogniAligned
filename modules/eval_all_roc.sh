#!/bin/bash
# Evaluate all saved CogniAlign checkpoints and generate ROC data JSONs.
# Run from the CogniAligned root: bash modules/eval_all_roc.sh

set -e
cd /home/usuaris/veussd/roger.esteve.sanchez/CogniAligned
source .venv/bin/activate

export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

mkdir -p roc_data

# --- ADReSSo (binary, with AF) ---
# Config: modules/configs/default.yaml (use_acoustic_features: True)
CKPT_DIR="logs/adresso_distil_wav2vec2_P_crossgated_mean"
for fold in 0 1 2 3 4; do
    ckpt="${CKPT_DIR}/model_fold_${fold}.pth"
    [ -f "$ckpt" ] || continue
    echo "=== ADReSSo CogniAlign+AF fold${fold} ==="
    python modules/eval_roc.py --config modules/configs/default.yaml \
        --task adresso --fold $fold --checkpoint "$ckpt" \
        --label "CogniAlign + AF" --output "roc_data/adresso_CogniAlign_plus_AF_fold${fold}.json"
done

# --- ADReSSo (binary, without AF) ---
# If you have a config without acoustic features, use it here
# CKPT_DIR="logs/distil_wav2vec2_P_crossgated_mean"
# for fold in 0 1 2 3 4; do
#     ckpt="${CKPT_DIR}/model_fold_${fold}.pth"
#     [ -f "$ckpt" ] || continue
#     echo "=== ADReSSo CogniAlign fold${fold} ==="
#     python modules/eval_roc.py --config modules/configs/default_no_af.yaml \
#         --task adresso --fold $fold --checkpoint "$ckpt" \
#         --label "CogniAlign" --output "roc_data/adresso_CogniAlign_fold${fold}.json"
# done

# --- Amyloid (binary, with AF) ---
CKPT_DIR="logs/amyloid_distil_wav2vec2_P_crossgated_mean"
for fold in 0 1 2 3 4; do
    ckpt="${CKPT_DIR}/model_fold_${fold}.pth"
    [ -f "$ckpt" ] || continue
    echo "=== Amyloid CogniAlign+AF fold${fold} ==="
    python modules/eval_roc.py --config modules/configs/amyloid/default.yaml \
        --task amyloid --fold $fold --checkpoint "$ckpt" \
        --label "CogniAlign + AF" --output "roc_data/amyloid_CogniAlign_plus_AF_fold${fold}.json"
done

# --- PPA (3-class) ---
CKPT_DIR="logs/ppa_distil_wav2vec2_P_crossgated_mean"
for fold in 0 1 2 3 4; do
    ckpt="${CKPT_DIR}/model_fold_${fold}.pth"
    [ -f "$ckpt" ] || continue
    echo "=== PPA CogniAlign fold${fold} ==="
    python modules/eval_roc.py --config modules/configs/ppa/default.yaml \
        --task ppa --fold $fold --checkpoint "$ckpt" \
        --label "CogniAlign" --output "roc_data/ppa_CogniAlign_fold${fold}.json"
done

echo ""
echo "Done! ROC data saved in roc_data/"
echo ""
echo "Generate comparison plots (combine with SER2025 data from ad-detection):"
echo "  python modules/plot_roc.py roc_data/adresso_*.json ../ad-detection/roc_data/adresso_*.json -o figures/roc_adresso.pdf"
echo "  python modules/plot_roc.py roc_data/amyloid_*.json ../ad-detection/roc_data/amyloid_*.json -o figures/roc_amyloid.pdf"
echo "  python modules/plot_roc.py roc_data/ppa_*.json ../ad-detection/roc_data/ppa_*.json -o figures/roc_ppa.pdf --per-class"
