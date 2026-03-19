# ADReSSo Experiments

| Job ID | Model | Features | Validation Accuracy | F1 Score | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 2397188 | CrossAttention | Audio + Text + Acoustic (47) | Failed | Failed | Initial run with acoustic features (TypeError) |
| 2398091 | CogniAlign | Audio + Text + Acoustic (55) | 87.8% | 0.876 | Concatenation fusion (late fusion) |

## Detailed Results (Job 2398091)
- **Fold 0**: Acc 91.2%, F1 0.901
- **Fold 1**: Acc 87.9%, F1 0.879
- **Fold 2**: Acc 75.8%, F1 0.756
- **Fold 3**: Acc 84.4%, F1 0.842
- **Fold 4**: Acc 100.0%, F1 1.000
- **Average**: Acc 87.8%, F1 0.876
