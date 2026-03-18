#!/usr/bin/env python3
"""
Generate ROC curve comparison plots from JSON files produced by eval_roc.py.

Aggregates folds per model label: computes mean ROC curve with std-dev band.

Usage examples:
    # Compare all models on ADReSSo
    python scripts/plot_roc.py roc_data/adresso_*.json -o figures/roc_adresso.pdf

    # Compare specific models
    python scripts/plot_roc.py \
        roc_data/adresso_SER2025_fold*.json \
        roc_data/adresso_SER2025_plus_AF_fold*.json \
        roc_data/adresso_CogniAlign_fold*.json \
        -o figures/roc_adresso.pdf --title "ADReSSo: AD vs CN"

    # Single fold, no aggregation
    python scripts/plot_roc.py roc_data/amyloid_SER2025_fold0.json -o roc_single.pdf
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='Plot ROC curves from JSON data files.')
    parser.add_argument('files', nargs='+', help='JSON files from eval_roc.py')
    parser.add_argument('-o', '--output', type=str, default='roc_comparison.pdf',
                        help='Output figure path (pdf/png/svg).')
    parser.add_argument('--title', type=str, default=None,
                        help='Plot title. Auto-generated from task if not provided.')
    parser.add_argument('--figsize', type=float, nargs=2, default=[7, 6],
                        help='Figure size (width height).')
    parser.add_argument('--no-std', action='store_true',
                        help='Do not show standard deviation bands.')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--per-class', action='store_true',
                        help='For multiclass tasks, plot per-class OVR curves instead of macro.')
    return parser.parse_args()


def interpolate_roc(targets, probs_positive, n_points=200):
    """Compute ROC and interpolate to common FPR grid."""
    fpr, tpr, _ = roc_curve(targets, probs_positive)
    mean_fpr = np.linspace(0, 1, n_points)
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    return mean_fpr, interp_tpr, roc_auc


def load_files(file_paths):
    """Load JSON files and group by label."""
    groups = defaultdict(list)
    for path in file_paths:
        with open(path) as f:
            data = json.load(f)
        groups[data['label']].append(data)
    return groups


def plot_binary_roc(groups, args):
    """Plot ROC for binary classification (one curve per model label)."""
    fig, ax = plt.subplots(1, 1, figsize=args.figsize)
    n_points = 200
    mean_fpr = np.linspace(0, 1, n_points)

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for idx, (label, entries) in enumerate(sorted(groups.items())):
        all_tprs = []
        all_aucs = []

        for entry in entries:
            targets = np.array(entry['targets'])
            probs = np.array(entry['probs'])
            probs_pos = probs[:, 1]

            _, interp_tpr, fold_auc = interpolate_roc(targets, probs_pos, n_points)
            all_tprs.append(interp_tpr)
            all_aucs.append(fold_auc)

        mean_tpr = np.mean(all_tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(all_aucs)
        std_auc = np.std(all_aucs)

        n_folds = len(entries)
        if n_folds > 1:
            auc_str = f"AUC = {mean_auc:.3f} $\\pm$ {std_auc:.3f} ({n_folds} folds)"
        else:
            auc_str = f"AUC = {mean_auc:.3f}"

        ax.plot(mean_fpr, mean_tpr, color=colors[idx % 10], lw=2,
                label=f"{label} ({auc_str})")

        if not args.no_std and n_folds > 1:
            std_tpr = np.std(all_tprs, axis=0)
            ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                            color=colors[idx % 10], alpha=0.15)

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Chance')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_multiclass_roc(groups, args):
    """Plot ROC for multiclass (macro-averaged OVR, one curve per model label)."""
    fig, ax = plt.subplots(1, 1, figsize=args.figsize)
    n_points = 200
    mean_fpr = np.linspace(0, 1, n_points)

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Get class names from first entry
    first_entry = next(iter(groups.values()))[0]
    class_names = first_entry.get('class_names', [f'Class {i}' for i in range(first_entry['num_classes'])])
    n_classes = first_entry['num_classes']

    if args.per_class:
        # One subplot per class
        fig, axes = plt.subplots(1, n_classes, figsize=(args.figsize[0] * n_classes / 2, args.figsize[1]))
        if n_classes == 1:
            axes = [axes]

        for cls_idx in range(n_classes):
            ax_cls = axes[cls_idx]
            for idx, (label, entries) in enumerate(sorted(groups.items())):
                all_tprs = []
                all_aucs = []
                for entry in entries:
                    targets = np.array(entry['targets'])
                    probs = np.array(entry['probs'])
                    binary_targets = (targets == cls_idx).astype(int)
                    _, interp_tpr, fold_auc = interpolate_roc(binary_targets, probs[:, cls_idx], n_points)
                    all_tprs.append(interp_tpr)
                    all_aucs.append(fold_auc)

                mean_tpr = np.mean(all_tprs, axis=0)
                mean_tpr[-1] = 1.0
                mean_auc = np.mean(all_aucs)
                n_folds = len(entries)
                auc_str = f"{mean_auc:.3f}" if n_folds == 1 else f"{mean_auc:.3f}$\\pm${np.std(all_aucs):.3f}"
                ax_cls.plot(mean_fpr, mean_tpr, color=colors[idx % 10], lw=2,
                            label=f"{label} ({auc_str})")
                if not args.no_std and n_folds > 1:
                    std_tpr = np.std(all_tprs, axis=0)
                    ax_cls.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                                        color=colors[idx % 10], alpha=0.15)

            ax_cls.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
            ax_cls.set_title(class_names[cls_idx], fontsize=11)
            ax_cls.set_xlabel('FPR', fontsize=10)
            if cls_idx == 0:
                ax_cls.set_ylabel('TPR', fontsize=10)
            ax_cls.legend(loc='lower right', fontsize=7)
            ax_cls.grid(True, alpha=0.3)

        return fig, axes

    else:
        # Macro-average: average per-class OVR curves
        for idx, (label, entries) in enumerate(sorted(groups.items())):
            all_tprs = []
            all_aucs = []

            for entry in entries:
                targets = np.array(entry['targets'])
                probs = np.array(entry['probs'])

                # Compute macro-average ROC by averaging per-class OVR curves
                class_tprs = []
                for cls_idx in range(n_classes):
                    binary_targets = (targets == cls_idx).astype(int)
                    _, interp_tpr, _ = interpolate_roc(binary_targets, probs[:, cls_idx], n_points)
                    class_tprs.append(interp_tpr)
                macro_tpr = np.mean(class_tprs, axis=0)
                macro_auc = auc(mean_fpr, macro_tpr)
                all_tprs.append(macro_tpr)
                all_aucs.append(macro_auc)

            mean_tpr = np.mean(all_tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = np.mean(all_aucs)
            std_auc = np.std(all_aucs)

            n_folds = len(entries)
            if n_folds > 1:
                auc_str = f"AUC = {mean_auc:.3f} $\\pm$ {std_auc:.3f} ({n_folds} folds)"
            else:
                auc_str = f"AUC = {mean_auc:.3f}"

            ax.plot(mean_fpr, mean_tpr, color=colors[idx % 10], lw=2,
                    label=f"{label} ({auc_str})")

            if not args.no_std and n_folds > 1:
                std_tpr = np.std(all_tprs, axis=0)
                ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                                color=colors[idx % 10], alpha=0.15)

        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Chance')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)

        return fig, ax


def main():
    args = parse_args()

    groups = load_files(args.files)
    if not groups:
        print("No data loaded.")
        sys.exit(1)

    # Determine if binary or multiclass from first entry
    first_entry = next(iter(groups.values()))[0]
    n_classes = first_entry['num_classes']
    task = first_entry['task']

    if n_classes == 2:
        fig, ax = plot_binary_roc(groups, args)
    else:
        fig, ax = plot_multiclass_roc(groups, args)

    title = args.title
    if title is None:
        task_titles = {
            'adresso': 'ADReSSo: AD vs CN',
            'amyloid': 'Beta-Amyloid: Positive vs Negative',
            'ppa': 'PPA: 3-class Classification',
        }
        title = task_titles.get(task, task)

    if isinstance(ax, np.ndarray):
        fig.suptitle(title, fontsize=13, y=1.02)
    else:
        ax.set_title(title, fontsize=13)

    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches='tight')
    print(f"Saved figure to {args.output}")

    # Print summary table
    print(f"\n{'Model':<25} {'AUC (mean ± std)':>20} {'Folds':>6}")
    print("-" * 55)
    for label, entries in sorted(groups.items()):
        aucs = [e['auc_roc'] for e in entries]
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        print(f"{label:<25} {mean_auc:>8.4f} ± {std_auc:<8.4f} {len(entries):>4}")


if __name__ == '__main__':
    main()
