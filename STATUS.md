# Status of CogniAligned Pipeline

## Recent Updates
- **Mamba Regularization**: Reduced Mamba fusion encoder size (to ~6.7M params) and added dropout to mitigate overfitting.
- **Ensemble Evaluation**: `modules/test.py` now calculates Accuracy, F1, Precision, Recall, and Loss against `task1.csv` ground truth.
- **W&B Final Test Run**: Test results are now logged as a 6th run in the W&B group with `job_type="test_evaluation"`.
- **SOTA Config Setup**: Successfully ran the pipeline with Wav2Vec2, pause tokens, and `crossgated` fusion.

## Active SLURM Jobs
- **Training (Job 2396178):** Regularized Mamba fusion training (5-fold CV).
- **Evaluation/Testing (Job 2396179):** Final ensemble test evaluation on Mamba model.

## Results Summary
- **SOTA Config (Wav2Vec2 + P + crossgated)**: 
  - Test Accuracy: **80.28%**
  - Group: `distil_wav2vec2_P_crossgated`
- **Mamba Experiment (First Attempt)**: Overfitted severely (100% train accuracy, ~60% val accuracy).
- **Mamba Experiment (Regularized)**: Currently running...

---
*Monitoring squeue for job updates.*