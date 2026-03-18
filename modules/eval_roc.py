#!/usr/bin/env python3
"""
Evaluate a saved CogniAlign checkpoint on its validation fold and save ROC curve data to JSON.

Usage examples:
    # ADReSSo (binary, with AF)
    python modules/eval_roc.py \
        --config modules/configs/default.yaml \
        --task adresso --fold 0 \
        --checkpoint logs/adresso_distil_wav2vec2_P_crossgated_mean/model_fold_0.pth \
        --label "CogniAlign + AF"

    # Amyloid (binary, with AF)
    python modules/eval_roc.py \
        --config modules/configs/amyloid/default.yaml \
        --task amyloid --fold 0 \
        --checkpoint logs/amyloid_distil_wav2vec2_P_crossgated_mean/model_fold_0.pth \
        --label "CogniAlign + AF"

    # PPA (3-class)
    python modules/eval_roc.py \
        --config modules/configs/ppa/default.yaml \
        --task ppa --fold 0 \
        --checkpoint logs/ppa_distil_wav2vec2_P_crossgated_mean/model_fold_0.pth \
        --label "CogniAlign"

Output: JSON file compatible with plot_roc.py from ad-detection project.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from datetime import datetime

# Add modules directory to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from utils import set_seed, get_config, save_config
from model import (CrossAttentionTransformerEncoder, MyTransformerEncoder,
                   BidirectionalCrossAttentionTransformerEncoder,
                   ElementWiseFusionEncoder)
from sklearn.metrics import roc_auc_score


TASK_CLASS_NAMES = {
    'adresso': ['CN', 'AD'],
    'amyloid': ['Negative', 'Positive'],
    'ppa': ['lvPPA', 'nfPPA', 'svPPA'],
}


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate CogniAlign checkpoint and save ROC data.')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML.')
    parser.add_argument('--task', type=str, required=True, choices=['adresso', 'amyloid', 'ppa'],
                        help='Task to evaluate.')
    parser.add_argument('--fold', type=int, required=True, help='Fold index (0-4).')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pth).')
    parser.add_argument('--label', type=str, default=None,
                        help='Human-readable model label for plots (e.g. "CogniAlign + AF").')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path. Default: roc_data/<task>_<label>_fold<N>.json')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU inference even if CUDA is available.')
    return parser.parse_args()


def build_model(config, device):
    """Instantiate the right model architecture from config."""
    if config.model.multimodality:
        if 'bicross' in config.model.fusion:
            model = BidirectionalCrossAttentionTransformerEncoder(config.model).to(device)
        elif 'cross' in config.model.fusion:
            model = CrossAttentionTransformerEncoder(config.model).to(device)
        else:
            model = ElementWiseFusionEncoder(config.model).to(device)
    else:
        model = MyTransformerEncoder(config.model).to(device)
    return model


def get_dataloader(config, task, fold):
    """Get validation dataloader for the given task and fold."""
    if task == 'adresso':
        from dataset import get_dataloaders
    elif task == 'amyloid':
        from amyloid.dataset import get_dataloaders
    elif task == 'ppa':
        from ppa.dataset import get_dataloaders

    _, val_dataloader = get_dataloaders(config, kfold_number=fold)
    return val_dataloader


def main():
    args = parse_args()

    set_seed(43)
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load config
    config = get_config(args.config)
    save_config(config)

    # Set multimodality flag (same as save_config does)
    config.model.multimodality = config.model.textual_model != '' and config.model.audio_model != ''

    # Build model name (same logic as save_config)
    textual_data = config.model.textual_model + '_' if config.model.textual_model != '' else ''
    audio_data = config.model.audio_model + '_' if config.model.audio_model != '' else ''
    pauses_data = 'P_' if config.model.pauses else ''
    config.model_name = f"{textual_data}{audio_data}{pauses_data}{config.model.fusion}"
    config.model.model_name = config.model_name
    config.path_name = f"{config.model_name}_{config.model.pooling}"

    num_classes = config.model.num_classes
    is_binary = (num_classes == 1)  # BCEWithLogitsLoss style
    class_names = TASK_CLASS_NAMES[args.task]

    # Build model and load checkpoint
    model = build_model(config, device)
    print(f"Loading checkpoint: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    # Get validation dataloader
    print(f"Loading validation data for {args.task} fold {args.fold}...")
    val_dataloader = get_dataloader(config, args.task, args.fold)
    print(f"Validation samples: {len(val_dataloader.dataset)}")

    # Inference
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for features, labels in val_dataloader:
            # Move features to device
            if isinstance(features, (list, tuple)):
                features = [f.to(device) if isinstance(f, torch.Tensor) else f for f in features]
            else:
                features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)

            if is_binary:
                # BCEWithLogitsLoss: single logit output → sigmoid → [p_neg, p_pos]
                logits = outputs.view(-1)
                p_pos = torch.sigmoid(logits)
                probs = torch.stack([1 - p_pos, p_pos], dim=1)  # (B, 2)
                targets = labels.view(-1).long()
            else:
                # CrossEntropyLoss: multi-class logits → softmax
                probs = torch.softmax(outputs, dim=1)
                targets = labels.view(-1).long()

            all_targets.extend(targets.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Compute AUC
    actual_n_classes = all_probs.shape[1]
    if actual_n_classes == 2:
        auc = roc_auc_score(all_targets, all_probs[:, 1])
    else:
        auc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='macro')

    print(f"AUC-ROC: {auc:.4f}")

    # Build output
    label = args.label or f"CogniAlign_{args.task}_fold{args.fold}"
    result = {
        'task': args.task,
        'fold': args.fold,
        'label': label,
        'checkpoint': os.path.abspath(args.checkpoint),
        'num_classes': actual_n_classes,
        'class_names': class_names,
        'has_acoustic_features': getattr(config.model, 'use_acoustic_features', False),
        'auc_roc': float(auc),
        'num_samples': len(all_targets),
        'targets': all_targets.tolist(),
        'probs': all_probs.tolist(),
        'timestamp': datetime.now().isoformat(),
    }

    # Output path
    if args.output:
        out_path = args.output
    else:
        os.makedirs('roc_data', exist_ok=True)
        safe_label = label.replace(' ', '_').replace('+', 'plus')
        out_path = f"roc_data/{args.task}_{safe_label}_fold{args.fold}.json"

    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Saved ROC data to {out_path}")


if __name__ == '__main__':
    main()
