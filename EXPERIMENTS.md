# ADReSSo Experiments

| Job ID | Model | Features | Validation Accuracy | F1 Score | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 2397188 | CrossAttention | Audio + Text + Acoustic (47) | Failed | Failed | Initial run with acoustic features (TypeError) |
| 2397215 | CrossAttention | Audio + Text + Acoustic (55) | 87.2% | 0.871 | Fixed acoustic features count and config access |

## Detailed Results (Job 2397215)
- **Fold 0**: Acc 91.2%, F1 0.908
- **Fold 1**: Acc 87.9%, F1 0.879
- **Fold 2**: Acc 72.7%, F1 0.726
- **Fold 3**: Acc 84.4%, F1 0.840
- **Fold 4**: Acc 100.0%, F1 1.000
- **Average**: Acc 87.2%, F1 0.871
