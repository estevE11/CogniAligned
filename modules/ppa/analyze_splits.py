import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
import os
import sys

# Load data
csv_path = "/home/usuaris/veussd/roger.esteve.sanchez/WAB_samples/labels.csv"
if not os.path.exists(csv_path):
    print(f"File not found: {csv_path}")
    sys.exit(1)

df = pd.read_csv(csv_path)
target_classes = ['lvPPA', 'nfPPA', 'svPPA']
df = df[df['DX_Pilar'].isin(target_classes)].reset_index(drop=True)

X = df.index.values
y = df['DX_Pilar']
groups = df['UT ID']

def analyze_folds(splitter, name):
    print(f"\n--- Analyzing {name} ---")
    try:
        splits = list(splitter.split(X, y, groups))
    except Exception as e:
        print(f"Splitter failed: {e}")
        return

    for i, (train_idx, val_idx) in enumerate(splits):
        val_y = y.iloc[val_idx]
        counts = val_y.value_counts().reindex(target_classes, fill_value=0)
        total = len(val_idx)
        
        print(f"Fold {i}: Total={total}")
        for cls in target_classes:
            pct = counts[cls] / total * 100 if total > 0 else 0
            print(f"  {cls}: {counts[cls]} ({pct:.1f}%)")
            
        # Check standard deviation of class counts to see balance
        print(f"  Std Dev of Counts: {np.std(counts.values):.2f}")

print(f"Total Samples: {len(df)}")
print("Class Distribution:")
print(df['DX_Pilar'].value_counts())

# Check GroupKFold (Current)
analyze_folds(GroupKFold(n_splits=5), "GroupKFold (Current)")

# Check StratifiedGroupKFold (Proposed)
try:
    analyze_folds(StratifiedGroupKFold(n_splits=5), "StratifiedGroupKFold")
except ImportError:
    print("\nStratifiedGroupKFold not available in this sklearn version.")
