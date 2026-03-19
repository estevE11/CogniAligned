# PPA Experiments Log

| Number | Explanation | Split | Loss | Acc | F1 | Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Baseline (Fast)**: Batch 32, LR 2e-5, Dropout 0.3, MLP 256. `CrossEntropyLoss` with weights. | Validation | - | `47.5%` (F0), `67.5%` (F4) | `0.21` (F0), `0.56` (F4) | `0.33` (F0), `0.55` (F4) |
| **2** | **Slower / Regularized**: Batch 8, LR 1e-5, Dropout 0.5, MLP 128. Increased patience. | Validation | - | `47.5%` (F0), `77.5%` (F4) | `0.21` (F0), `0.68` (F4) | `0.33` (F0), `0.63` (F4) |
| **3** | **Confusion Matrix Analysis**: Rerun of Exp 2 with confusion matrix logging. Confirmed majority class collapse in Folds 0-3. | Validation | - | `47.5%` (F0), `77.5%` (F4) | `0.21` (F0), `0.68` (F4) | `0.33` (F0), `0.63` (F4) |
| **4** | **Balanced Sampling**: Added `WeightedRandomSampler` to oversample minority classes (`nfPPA`, `svPPA`) during training. Same "Slower" config otherwise. | Validation | - | `47.5%` (F0), `77.5%` (F4) | `0.21` (F0), `0.68` (F4) | `0.33` (F0), `0.63` (F4) |
| **5** | **Balanced No Weights**: `WeightedRandomSampler` + `CrossEntropyLoss` (No Weights) + Batch Size 16. | Validation | - | `47.5%` (F0), `77.5%` (F4) | `0.21` (F0), `0.65` (F4) | `0.33` (F0), `0.60` (F4) |
| **6** | **Stratified Group Split**: Implemented `StratifiedGroupKFold` to ensure balanced class distribution across all folds. `WeightedRandomSampler` + Batch 16 + No Loss Weights. | Validation | - | - | - | - |
| **7** | **Acoustic Features**: Added 55 acoustic features (concatenated to classifier input). | Validation | - | Failed | Failed | Initial run (TypeError) |
| **8** | **Acoustic Features (Fixed)**: Added 55 acoustic features (concatenated to classifier input). | Validation | - | `70.3%` | `0.61` | `0.62` |
| **9** | **Reduced Underfitting**: dropout 0.5→0.3, weight_decay 0.1→0.01. | Validation | - | `71.3%` | `0.647` | `0.670` |

## Detailed Notes

### Experiment 9: Reduced Underfitting (Job 2398099)
- **Goal**: Fix underfitting in Exp 8 (train acc ~94% but dropout=0.5 and weight_decay=0.1 slowed convergence). Reduce regularization.
- **Config**: `balanced_noweights.yaml` with `dropout: 0.3` (was 0.5), `weight_decay: 0.01` (was 0.1).
- **Results**:
    - **Fold 0**: Acc 82.1%, F1 0.814, Recall 0.843, Prec 0.802
    - **Fold 1**: Acc 64.1%, F1 0.499, Recall 0.539, Prec 0.475
    - **Fold 2**: Acc 61.0%, F1 0.589, Recall 0.614, Prec 0.596
    - **Fold 3**: Acc 82.1%, F1 0.774, Recall 0.788, Prec 0.780
    - **Fold 4**: Acc 67.5%, F1 0.560, Recall 0.567, Prec 0.624
    - **Average**: Acc 71.3%, F1 0.647, Recall 0.670, Prec 0.655
- **Observation**: Modest improvement over Exp 8 (+1% Acc, +0.4% F1). Folds 1 and 2 remain challenging — the high variance across folds is driven by the data distribution, not the regularization settings.

### Experiment 8: Acoustic Features (Fixed) (Job 2398092)
- **Goal**: Integrate 55 acoustic features (pause time, jitter, shimmer, etc.) to improve classification.
- **Config**: `balanced_noweights.yaml` + `use_acoustic_features: True`.
- **Results**:
    - **Fold 0**: Acc 82.1%, F1 0.805, Recall 0.843
    - **Fold 1**: Acc 64.1%, F1 0.545, Recall 0.570
    - **Fold 2**: Acc 61.0%, F1 0.579, Recall 0.560
    - **Fold 3**: Acc 79.5%, F1 0.753, Recall 0.771
    - **Fold 4**: Acc 65.0%, F1 0.531, Recall 0.528
    - **Average**: Acc 70.3%, F1 0.643, Recall 0.654
- **Observation**: The integration of acoustic features shows mixed results across folds. Folds 0 and 3 show strong performance, while Folds 1, 2, and 4 remain challenging.

### Experiment 1: Baseline (Job 2396959)
- **Config**: Default PPA config.
- **Observation**: Training was very fast (16-34 epochs).
- **Results**:
    - Folds 0-3 collapsed to predicting the majority class (Recall ~0.33, F1 ~0.21).
    - Fold 4 showed learning (Acc 67.5%, Recall 0.55).

### Experiment 2: Slower / Regularized (Job 2396962)
- **Config**: Reduced batch size (32->8) to increase steps/epoch. Lowered LR (2e-5 -> 1e-5). Increased Dropout (0.3 -> 0.5). Reduced MLP size (256 -> 128).
- **Observation**:
    - Folds 0-3 still collapsed (Recall ~0.33).
    - Fold 4 performance improved significantly (Acc 77.5%, Recall 0.63).
- **Hypothesis**: The dataset split for Fold 4 might be easier or contain more representative samples. Folds 0-3 might have a difficult distribution or specific "hard" samples in the validation set.

### Experiment 3: Confusion Matrix Analysis (Job 2396963)
- **Goal**: Visualize the predictions to confirm if the model is predicting only one class in the failing folds.
- **Results**:
    - Confirmed exact same metrics as Experiment 2.
    - Folds 0-3: **Collapsed** (Acc 47.5%, Recall 33.3%).
    - Fold 4: **Success** (Acc 77.5%, Recall 62.6%).
- **Deep Dive Analysis**:
    - **Fold 0 (Collapsed)**:
      - **Confusion Matrix**: 100% of samples predicted as `lvPPA` (Majority Class).
        - `lvPPA`: 19/19 Correct (100% Recall)
        - `nfPPA`: 0/18 Correct (0% Recall)
        - `svPPA`: 0/3 Correct (0% Recall)
      - **Reason**: The model completely ignores `nfPPA` and `svPPA` despite class weighting. The validation set has a high proportion of `nfPPA` (18 vs 19 `lvPPA`), but the model still fails.
    - **Fold 4 (Success)**:
      - **Confusion Matrix**: Shows learning but heavily biased.
        - `lvPPA`: 24/26 Correct (92% Recall)
        - `nfPPA`: 2/5 Correct (40% Recall)
        - `svPPA`: 5/9 Correct (55% Recall)
      - **Reason**: Fold 4 validation set is dominated by `lvPPA` (26 samples vs 5 `nfPPA` + 9 `svPPA`). The high accuracy (77.5%) is largely driven by correctly predicting the majority class `lvPPA`. However, it *did* learn to distinguish `svPPA` reasonably well (5/9).
- **Key Finding**: The model suffers from severe class imbalance issues. `lvPPA` is over-represented and "easier" to learn or just safer to predict. The current `ClassWeights` are insufficient.
- **Proposed Fixes**:
  1. **Stratified Group Split**: Ensure folds have balanced class distributions (currently Fold 0 has 18 `nfPPA`, Fold 4 has 5).
  2. **Oversampling**: Use `WeightedRandomSampler` to balance batches during training.
  3. **Focal Loss**: Replace CrossEntropy with Focal Loss to focus on hard misclassified examples.

### Experiment 4: Balanced Sampling (Job 2397002)
- **Goal**: Force the model to see an equal distribution of classes during training using `WeightedRandomSampler`.
- **Config**: Based on `slower.yaml` but enabled `weighted_sampling`.
- **Results**:
    - **Identical to Exp 2 & 3**. Folds 0-3 still collapsed to majority class.
    - `WeightedRandomSampler` did not break the collapse.
- **Possible Reasons**:
    - **Double Weighting**: We are using both `WeightedRandomSampler` AND `CrossEntropyLoss(weight=...)`. This might be distorting the loss surface strangely.
    - **Batch Size**: Batch size 8 might be too small for effective sampling (high variance).
    - **Model Capacity**: The model might just be converging to the "mean" (lvPPA) because it can't find features for the others.
- **Next Steps**:
    - Remove `class_weights` from Loss when using Sampler.
    - Increase Batch Size (e.g. 16 or 32).

### Experiment 5: Balanced No Weights (Job 2397007)
- **Goal**: Fix the potential "double weighting" issue from Exp 4.
- **Config**: `balanced_noweights.yaml`.
  - `weighted_sampling`: True (keeps batches balanced).
  - `CrossEntropyLoss`: No weights (since batches are already balanced).
  - `batch_size`: 16 (increased from 8 to stabilize gradients).
- **Results**:
  - **No Change**. Folds 0-3 still completely collapsed (Recall 0.33).
  - Fold 4 still works (Acc 77.5%), proving the model *can* learn but chooses not to for most folds.
- **Conclusion**: The collapse is likely not due to the training dynamics (sampling/weights) but due to the **data distribution** in the folds.
  - Fold 0 Val: 19 `lvPPA`, 18 `nfPPA`, 3 `svPPA`. The model predicts 100% `lvPPA`. It completely fails to generalize to `nfPPA` despite balanced training batches.
  - This suggests the `nfPPA` samples in Fold 0 Validation are very different from the `nfPPA` samples in Fold 0 Train (or just hard to distinguish).
- **Next Step**: Implement **Stratified Group K-Fold**. The current `GroupKFold` is creating unbalanced splits (e.g., Fold 0 has 18 `nfPPA` in val, Fold 4 has 5). We need to ensure every fold has a representative distribution of classes.

### Experiment 6: Stratified Group Split (Job 2397018)
- **Goal**: Ensure consistent class distribution across all folds to prevent "unlucky" splits like Fold 0.
- **Analysis**:
  - `GroupKFold` (Old): Fold 0 (18 `nfPPA`), Fold 4 (5 `nfPPA`). High variance.
  - `StratifiedGroupKFold` (New): All folds have ~13 `nfPPA` and ~7 `svPPA`. Perfectly balanced.
- **Config**: Same as Exp 5 (`balanced_noweights.yaml`) but with new splits.
- **Hypothesis**: With balanced validation sets AND balanced training batches, the model should achieve consistent performance across all folds.

### Experiment 7: Acoustic Features (Job 2397189)
- **Goal**: Integrate 47 acoustic features (pause time, jitter, shimmer, etc.) to improve classification.
- **Config**: `balanced_noweights.yaml` + `use_acoustic_features: True`.
- **Status**: Running.
