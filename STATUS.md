# Status of CogniAligned Pipeline

## Recent Updates
- **PPA Experiment**: Created a new training pipeline for PPA dataset (multiclass: lvPPA, nfPPA, svPPA) in `modules/ppa/`.
- **Mamba Regularization**: Reduced Mamba fusion encoder size (to ~6.7M params) and added dropout to mitigate overfitting.
- **Ensemble Evaluation**: `modules/test.py` now calculates Accuracy, F1, Precision, Recall, and Loss against `task1.csv` ground truth.
- **W&B Final Test Run**: Test results are now logged as a 6th run in the W&B group with `job_type="test_evaluation"`.
- **SOTA Config Setup**: Successfully ran the pipeline with Wav2Vec2, pause tokens, and `crossgated` fusion.

## Active SLURM Jobs
- **PPA Preprocessing (Job 2396923):** Running Whisper and embedding extraction for WAB samples.
- **PPA Training (Job 2396924):** Pending (dependent on preprocessing).
- **Training (Job 2396866):** SOTA config rerun with seed 43 and new splits (5-fold CV) - Completed.
- **Evaluation/Testing (Job 2396867):** Final ensemble test evaluation on SOTA model - Completed.

## Results Summary
- **SOTA Config (Wav2Vec2 + P + crossgated)**: 
  - Latest Test Accuracy: **83.10%** (Seed 43, Job 2396866/7)
  - Previous Test Accuracy: **80.28%** (Seed 42, Job 2395731)
- **Mamba Experiment (First Attempt)**: Overfitted severely (100% train accuracy, ~60% val accuracy).
- **Mamba Experiment (Regularized - Job 2396178):** Results pending...

---
*Monitoring squeue for job updates.*