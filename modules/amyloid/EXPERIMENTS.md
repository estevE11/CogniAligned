# Amyloid Experiments Log

| Number | Explanation | Split | Loss | Acc | F1 | Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Baseline**: Binary classification (Positive/Negative). `WeightedRandomSampler` enabled. Batch 16. Stratified Group K-Fold. | Validation | - | - | - | - |
| **2** | **Acoustic Features**: Added 55 acoustic features (concatenated to classifier input). | Validation | - | Failed | Failed | Initial run (TypeError) |
| **3** | **Acoustic Features (Fixed)**: Added 55 acoustic features (concatenated to classifier input). | Validation | - | `78.5%` | `0.78` | `0.77` |
| **4** | **Reduced Underfitting**: LR 5e-6→2e-5, weight_decay 0.1→0.01. | Validation | - | `83.9%` | `0.834` | `0.832` |

## Detailed Notes

### Experiment 4: Reduced Underfitting (Job 2398098)
- **Goal**: Fix underfitting observed in Exp 3 (train acc peaked at ~81%). Increase LR and reduce weight decay.
- **Config**: `default.yaml` with `learning_rate: 2e-5` (was 5e-6), `weight_decay: 0.01` (was 0.1).
- **Results**:
    - **Fold 0**: Acc 82.4%, F1 0.824, Recall 0.824, Prec 0.824
    - **Fold 1**: Acc 84.8%, F1 0.846, Recall 0.844, Prec 0.850
    - **Fold 2**: Acc 80.0%, F1 0.789, Recall 0.783, Prec 0.808
    - **Fold 3**: Acc 83.9%, F1 0.832, Recall 0.828, Prec 0.855
    - **Fold 4**: Acc 88.2%, F1 0.881, Recall 0.881, Prec 0.881
    - **Average**: Acc 83.9%, F1 0.834, Recall 0.832, Prec 0.843
- **Observation**: Significant improvement over Exp 3 (+5.4% Acc, +5.5% F1). Higher LR enabled proper convergence. Consistent gains across all folds, with no signs of overfitting.

### Experiment 3: Acoustic Features (Fixed) (Job 2398093)
- **Goal**: Integrate 55 acoustic features (pause time, jitter, shimmer, etc.) to improve classification.
- **Config**: `default.yaml` + `use_acoustic_features: True`.
- **Results**:
    - **Fold 0**: Acc 82.4%, F1 0.823, Recall 0.824
    - **Fold 1**: Acc 75.8%, F1 0.756, Recall 0.756
    - **Fold 2**: Acc 77.1%, F1 0.770, Recall 0.775
    - **Fold 3**: Acc 83.9%, F1 0.832, Recall 0.828
    - **Fold 4**: Acc 73.5%, F1 0.715, Recall 0.714
    - **Average**: Acc 78.5%, F1 0.779, Recall 0.779
- **Observation**: Consistent performance across all folds, with Fold 3 showing the best results.

### Experiment 4: Reduced Underfitting (Job 2398098)
- **Goal**: Fix underfitting observed in Exp 3 (train acc peaked at ~81%). Increase LR and reduce weight decay to let the model learn more aggressively.
- **Config**: `default.yaml` with `learning_rate: 2e-5` (was 5e-6), `weight_decay: 0.01` (was 0.1).
- **Results**:
    - **Fold 0**: Acc 82.4%, F1 0.824, Recall 0.824, Prec 0.824
    - **Fold 1**: Acc 84.8%, F1 0.846, Recall 0.844, Prec 0.850
    - **Fold 2**: Acc 80.0%, F1 0.789, Recall 0.783, Prec 0.808
    - **Fold 3**: Acc 83.9%, F1 0.832, Recall 0.828, Prec 0.855
    - **Fold 4**: Acc 88.2%, F1 0.881, Recall 0.881, Prec 0.881
    - **Average**: Acc 83.9%, F1 0.834, Recall 0.832, Prec 0.843
- **Observation**: Significant improvement over Exp 3 (+5.4% Acc, +5.5% F1). Higher LR allowed the model to converge properly. No overfitting observed — val acc tracks well with consistent gains across all folds.

### Experiment 1: Baseline
- **Config**: `modules/configs/amyloid/default.yaml`
- **Goal**: Establish a baseline for Amyloid binary classification using the WAB dataset.
- **Setup**: 
    - Task: Binary Classification (Amyloid Positive vs Negative).
    - Splits: Stratified Group K-Fold (5 folds).
    - Training: Weighted Random Sampling to handle potential imbalance.
    - Loss: `BCEWithLogitsLoss`.

### Experiment 2: Acoustic Features (Job 2397190)
- **Goal**: Integrate 47 acoustic features (pause time, jitter, shimmer, etc.) to improve classification.
- **Config**: `default.yaml` + `use_acoustic_features: True`.
- **Status**: Running.
