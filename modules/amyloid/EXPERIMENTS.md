# Amyloid Experiments Log

| Number | Explanation | Split | Loss | Acc | F1 | Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Baseline**: Binary classification (Positive/Negative). `WeightedRandomSampler` enabled. Batch 16. Stratified Group K-Fold. | Validation | - | - | - | - |

## Detailed Notes

### Experiment 1: Baseline
- **Config**: `modules/configs/amyloid/default.yaml`
- **Goal**: Establish a baseline for Amyloid binary classification using the WAB dataset.
- **Setup**: 
    - Task: Binary Classification (Amyloid Positive vs Negative).
    - Splits: Stratified Group K-Fold (5 folds).
    - Training: Weighted Random Sampling to handle potential imbalance.
    - Loss: `BCEWithLogitsLoss`.
