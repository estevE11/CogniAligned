# Status of CogniAligned Pipeline

## Recent Updates
- **PPA Experiment**: Created a new training pipeline for PPA dataset (multiclass: lvPPA, nfPPA, svPPA) in `modules/ppa/`.
- **Confusion Matrix**: Added confusion matrix logging to W&B for PPA experiments to analyze class-wise performance.
- **PPA Slower Config**: Created `modules/configs/ppa/slower.yaml` (Batch size 8, LR 1e-5, Dropout 0.5, smaller MLP).
- **Balanced Sampling**: Added `WeightedRandomSampler` to force balanced class distribution during training (Experiment 4).
- **Balanced No Weights**: Experiment 5 removes class weights from loss (since sampling is balanced) and increases batch size to 16.
- **Stratified Group Split**: Implemented `StratifiedGroupKFold` to ensure every fold has a representative distribution of `nfPPA` and `svPPA`.
- **Mamba Regularization**: Reduced Mamba fusion encoder size (to ~6.7M params) and added dropout.
- **Ensemble Evaluation**: `modules/test.py` calculates detailed metrics against ground truth.

## Active SLURM Jobs
- **PPA Stratified Training (Job 2397018):** Running with `StratifiedGroupKFold` splits + Balanced Sampling + No Weights.

## Results Summary
- **PPA Balanced No Weights (Job 2397007)**:
  - **Result**: **Collapse persists** in Folds 0-3. Identified root cause as **poor data distribution** in `GroupKFold` splits (Fold 0 had 18 `nfPPA` vs Fold 4 with 5).
- **PPA Slower Config (Job 2396963 / 2396962)**:
  - **Fold 4**: **77.5% Accuracy**, **0.68 F1**, **0.63 Recall**.
  - **Folds 0-3**: ~48% Accuracy, ~0.21 F1, **0.33 Recall** (majority class collapse confirmed via Confusion Matrix).
- **SOTA Config (Wav2Vec2 + P + crossgated)**: 
  - Latest Test Accuracy: **83.10%** (Seed 43)
- **Mamba Experiment**: Overfitted.

---
*Monitoring squeue for job updates.*
