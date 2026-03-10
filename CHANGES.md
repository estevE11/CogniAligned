# CogniAligned - SLURM & W&B Integration Changes

This document summarizes all changes made to adapt CogniAligned for HPC/SLURM execution and enhanced W&B logging.

---

## 📝 Summary of Changes

### New Files Created

1. **`slurm/train.sh`** - SLURM batch script
   - Configures job resources (2 GPUs, 32GB RAM, 8h)
   - Sets up virtual environment automatically
   - Configures environment variables (HF_HOME, WANDB_DIR, NCCL settings)
   - Supports config file override via command-line argument

2. **`requirements.txt`** - Python dependencies
   - Lists all required packages with version constraints
   - Includes torch, transformers, wandb, whisper, librosa, etc.

3. **`modules/model_utils.py`** - Model utilities
   - `count_parameters()` - Count total trainable parameters
   - `count_parameters_by_component()` - Parameter breakdown by model component
   - `get_model_architecture_summary()` - Comprehensive architecture details
   - `get_training_config_summary()` - Training hyperparameters summary
   - `log_model_summary_to_wandb()` - Log everything to W&B

4. **`HPC_README.md`** - Comprehensive HPC usage guide
   - Quick start instructions
   - Setup and configuration details
   - W&B integration explanation
   - Monitoring and troubleshooting tips
   - Example workflows

5. **`CHANGES.md`** - This file

---

## 🔧 Modified Files

### 1. `modules/configs/default.yaml`

**Added:**
- `data:` section with configurable dataset paths
  - `train_dir`, `root_text_path`, `root_audio_path`, `csv_labels_path`, `splits_path`
- `wandb:` section with W&B configuration
  - `project`, `entity`, `mode`

### 2. `modules/main.py`

**Changes:**
- Added import for `model_utils.py`
- Enhanced `set_up()` function:
  - Reads W&B config from YAML or environment variables
  - Creates descriptive run names with SLURM job ID
  - Comprehensive W&B initialization with model architecture
  - Calls `log_model_summary_to_wandb()` to log model details
  - Enhanced `wandb.watch()` with gradient logging
- Updated fold logging to avoid duplicates

### 3. `modules/utils.py`

**Changes:**
- `train()` function:
  - Removed duplicate `wandb.init()` (now called in main.py)
  - Enhanced per-epoch logging:
    - Train: loss, accuracy, F1, recall, precision
    - Val: loss, accuracy, F1, recall, precision
    - Learning rate
  - Added best metrics logging at end of training
- `evaluation()` function:
  - Enhanced logging for both train and validation
  - Separate namespaces: `train/`, `val/`, `test/`

### 4. `modules/dataset.py`

**Changes:**
- Added global variables for default paths with `splits_path`
- Modified `read_CSV()` to use config paths if available
- Modified `get_dataloaders()` to use config splits path
- Updated `set_splits()` to accept config and use config paths
- Updated `get_splits_stats()` to accept config and use config paths
- Fixed path construction to use `os.path.join()` consistently
- Initialized `text_embeddings_path` and `audio_embeddings_path` to avoid unbound errors

---

## 📊 W&B Logging Enhancements

### What Gets Logged Now

#### Model Architecture (logged once at initialization)
```python
{
    "model_type": "BidirectionalCrossAttentionTransformerEncoder",
    "model_name": "distil_egemaps_P_bicrossgated",
    "total_parameters": 12453889,
    "total_parameters_millions": 12.45,
    "parameters_by_component": {
        "layers_1": 2359296,
        "layers_2": 2359296,
        "norm_layers_1": 1536,
        "norm_layers_2": 1536,
        "classifier": 394241,
        "total": 12453889
    },
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_layers": 1,
    "num_heads": 12,
    "dropout": 0.3,
    "pooling_strategy": "mean",
    "fusion_type": "bicrossgated",
    "textual_model": "distil",
    "audio_model": "egemaps",
    ...
}
```

#### Training Configuration (logged once at initialization)
```python
{
    "batch_size": 32,
    "num_epochs": 200,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "early_stopping": True,
    "early_stopping_patience": 20,
    "optimizer": "AdamW",
    "lr_scheduler": "cosine",
    "dataset": "ADReSSo",
    ...
}
```

#### Per-Epoch Metrics
```python
{
    "epoch": 1,
    "train/loss": 0.6234,
    "train/accuracy": 0.7123,
    "train/f1": 0.6891,
    "train/recall": 0.6754,
    "train/precision": 0.7032,
    "val/loss": 0.5821,
    "val/accuracy": 0.7456,
    "val/f1": 0.7234,
    "val/recall": 0.7012,
    "val/precision": 0.7456,
    "learning_rate": 1.98e-5
}
```

#### Best Performance (logged at end)
```python
{
    "best/val_accuracy": 0.8234,
    "best/val_f1": 0.8012,
    "best/val_recall": 0.7891,
    "best/val_precision": 0.8156,
    "best_epoch": 87
}
```

---

## 🚀 Usage

### Quick Start
```bash
cd /home/usuaris/veussd/roger.esteve.sanchez/CogniAligned
sbatch slurm/train.sh
```

### With Custom Config
```bash
sbatch slurm/train.sh modules/configs/qwen.yaml
```

### Environment Variables
```bash
WANDB_MODE=offline sbatch slurm/train.sh
WANDB_PROJECT=MyExperiment sbatch slurm/train.sh
```

---

## ✅ Testing Checklist

Before running your first job, verify:

- [ ] Virtual environment is set up or will be created automatically
- [ ] W&B login completed: `wandb login`
- [ ] Data paths are correct in `modules/configs/default.yaml`
- [ ] ADReSSo dataset is accessible at the configured path
- [ ] SLURM partitions are available: `sinfo -p veu`
- [ ] Disk space is sufficient in project directory

---

## 🎯 Key Features

1. **Automatic Environment Setup**: SLURM script creates venv if needed
2. **Configurable Paths**: All data paths in config files, no hardcoding
3. **Comprehensive W&B Logging**: Model architecture, all metrics, hyperparameters
4. **SLURM Integration**: Job management, resource allocation, logging
5. **Environment Variable Overrides**: Easy W&B configuration without editing files
6. **Detailed Documentation**: Step-by-step HPC_README.md guide

---

## 📚 Files to Review Tomorrow

1. **`HPC_README.md`** - Start here for usage instructions
2. **`slurm/train.sh`** - Review SLURM configuration
3. **`modules/configs/default.yaml`** - Verify data paths
4. **`requirements.txt`** - Check dependencies

---

## 🔗 Resources

- W&B Dashboard: https://wandb.ai/YOUR_USERNAME/CogniAligned
- SLURM Logs: `logs/slurm/cognialigned_*.txt`
- Training Logs: `logs/<model_name>/train_stats*.txt`
- Model Checkpoints: `logs/<model_name>/model_fold_*.pth`

---

**All changes have been implemented and are ready for testing! 🎉**
