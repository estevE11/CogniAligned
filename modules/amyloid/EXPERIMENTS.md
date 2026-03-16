# Amyloid Experiments Log

| Number | Explanation | Split | Loss | Acc | F1 | Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Baseline**: Binary classification (Positive/Negative). `WeightedRandomSampler` enabled. Batch 16. Stratified Group K-Fold. | Validation | - | - | - | - |
| **2** | **Acoustic Features**: Added 55 acoustic features (concatenated to classifier input). | Validation | - | Failed | Failed | Initial run (TypeError) |
| **3** | **Acoustic Features (Fixed)**: Added 55 acoustic features (concatenated to classifier input). | Validation | - | `79.7%` | `0.79` | `0.78` |

## Detailed Notes

### Experiment 3: Acoustic Features (Fixed) (Job 2397217)
- **Goal**: Integrate 55 acoustic features (pause time, jitter, shimmer, etc.) to improve classification.
- **Config**: `default.yaml` + `use_acoustic_features: True`.
- **Results**:
    - **Fold 0**: Acc 85.3%, F1 0.853, Recall 0.853
    - **Fold 1**: Acc 75.8%, F1 0.756, Recall 0.756
    - **Fold 2**: Acc 80.0%, F1 0.789, Recall 0.783
    - **Fold 3**: Acc 83.9%, F1 0.832, Recall 0.828
    - **Fold 4**: Acc 73.5%, F1 0.715, Recall 0.714
    - **Average**: Acc 79.7%, F1 0.789, Recall 0.787
- **Observation**: Strong performance across all folds, with Fold 0 and Fold 3 showing particularly good results. The acoustic features provide a solid boost to Amyloid detection.

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
