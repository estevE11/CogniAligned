# CogniAligned HPC Training Guide

This guide explains how to run CogniAligned training on the HPC cluster using SLURM.

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Setup](#setup)
3. [Running Training](#running-training)
4. [Configuration](#configuration)
5. [W&B Integration](#wb-integration)
6. [Monitoring](#monitoring)
7. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

For the impatient, here's how to get started:

```bash
# Navigate to the project directory
cd /home/usuaris/veussd/roger.esteve.sanchez/CogniAligned

# Submit training job with default configuration
sbatch slurm/train.sh

# Check job status
squeue -u $USER

# Monitor training output
tail -f logs/slurm/cognialigned_<JOB_ID>.txt
```

---

## 🛠️ Setup

### 1. First-Time Setup

The SLURM script will automatically create a virtual environment and install dependencies on first run. However, you can set it up manually if preferred:

```bash
cd /home/usuaris/veussd/roger.esteve.sanchez/CogniAligned

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Verify Data Paths

Make sure your ADReSSo dataset is located at the expected path. The default configuration expects:

```
/home/usuaris/veussd/roger.esteve.sanchez/adresso/processed_data/diagnosis/train/
├── text/                              # Text embeddings
│   ├── ad/                           # Alzheimer's Disease samples
│   └── cn/                           # Control samples
├── audio/                             # Audio files
│   ├── ad/
│   └── cn/
├── splits/                            # Cross-validation splits
│   ├── train_uids0.npy
│   ├── val_uids0.npy
│   └── ...
└── adresso-train-mmse-scores.csv     # Labels
```

If your data is in a different location, update `modules/configs/default.yaml` (see [Configuration](#configuration)).

### 3. W&B Login

Make sure you're logged into Weights & Biases:

```bash
wandb login
```

You'll be prompted to enter your W&B API key (find it at https://wandb.ai/authorize).

---

## 🎯 Running Training

### Basic Training

Submit a training job with the default configuration:

```bash
sbatch slurm/train.sh
```

### Custom Configuration

To use a different config file:

```bash
sbatch slurm/train.sh modules/configs/qwen.yaml
```

### Environment Variable Overrides

You can override W&B settings via environment variables:

```bash
# Run in offline mode (no W&B syncing)
WANDB_MODE=offline sbatch slurm/train.sh

# Use a different W&B project
WANDB_PROJECT=MyExperiment sbatch slurm/train.sh

# Specify W&B entity/team
WANDB_ENTITY=my-team sbatch slurm/train.sh
```

### Interactive Testing (for debugging)

For testing without SLURM:

```bash
# Activate virtual environment
source .venv/bin/activate

# Run training directly
python -u modules/main.py --config modules/configs/default.yaml
```

---

## ⚙️ Configuration

### Configuration Files

Configuration files are located in `modules/configs/`. The main config file is `default.yaml`.

#### Key Configuration Sections

**1. Training Parameters** (`train:`)
```yaml
train:
  batch_size: 32                      # Batch size per GPU
  num_epochs: 200                     # Maximum number of epochs
  learning_rate: 0.00002              # Learning rate (2e-5)
  weight_decay: 0.01                  # Weight decay for regularization
  early_stopping: True                # Enable early stopping
  early_stopping_patience: 20         # Stop after N epochs without improvement
  cross_validation: True              # Enable 5-fold cross-validation
  cross_validation_folds: 5           # Number of folds
```

**2. Model Architecture** (`model:`)
```yaml
model:
  pooling: 'mean'                     # Pooling strategy: 'mean', 'cls', 'attn', 'gatedattn'
  n_layers: 1                         # Number of transformer layers
  dropout: 0.3                        # Dropout rate
  hidden_size: 768                    # Hidden dimension size
  intermediate_size: 3072             # FFN intermediate size
  n_heads: 12                         # Number of attention heads
  num_classes: 1                      # Output classes (1 for binary)
  hidden_mlp_size: 256                # MLP hidden layer size
  textual_model: 'distil'             # Text model: 'bert', 'distil', 'roberta', 'mistral', 'qwen', 'stella'
  audio_model: 'egemaps'              # Audio model: 'wav2vec2', 'egemaps', 'mel'
  pauses: False                       # Include pause information
  fusion: 'cross'                     # Fusion type: 'cross', 'crossgated', 'bicross', 'bicrossgated', 'concat', 'sum', 'mul', 'mean'
```

**3. Data Paths** (`data:`)
```yaml
data:
  train_dir: "/path/to/adresso/train"
  root_text_path: "/path/to/adresso/train/text/"
  root_audio_path: "/path/to/adresso/train/audio/"
  csv_labels_path: "/path/to/adresso/train/adresso-train-mmse-scores.csv"
  splits_path: "/path/to/adresso/train/splits/"
```

**4. W&B Settings** (`wandb:`)
```yaml
wandb:
  project: "CogniAligned"             # W&B project name
  entity: ""                          # W&B team/entity (optional)
  mode: "online"                      # 'online', 'offline', or 'disabled'
```

### Creating Custom Configurations

To create a new configuration:

```bash
# Copy default config
cp modules/configs/default.yaml modules/configs/my_experiment.yaml

# Edit the new config
nano modules/configs/my_experiment.yaml

# Run training with your config
sbatch slurm/train.sh modules/configs/my_experiment.yaml
```

---

## 📊 W&B Integration

### What Gets Logged

The enhanced W&B integration logs comprehensive metrics and model information:

#### Model Architecture
- Total parameter count
- Parameters per component (encoder, classifier, etc.)
- Model hyperparameters (hidden size, layers, heads, dropout, etc.)
- Fusion strategy and modality information

#### Training Metrics (per epoch)
- **Train**: loss, accuracy, F1, recall, precision
- **Validation**: loss, accuracy, F1, recall, precision
- Learning rate
- Epoch number

#### Best Performance
- Best validation accuracy, F1, recall, precision
- Best epoch number

#### Training Configuration
- All hyperparameters (batch size, learning rate, etc.)
- Optimizer and scheduler details
- Early stopping settings
- Cross-validation configuration

### Viewing Results

Access your training runs at: https://wandb.ai/YOUR_USERNAME/CogniAligned

The W&B dashboard provides:
- Real-time training curves
- Model architecture visualization
- Hyperparameter comparison
- System metrics (GPU, memory, etc.)

### W&B Modes

- **`online`** (default): Sync data in real-time to W&B cloud
- **`offline`**: Save data locally, sync later with `wandb sync`
- **`disabled`**: Disable W&B logging completely

To change mode:
```bash
WANDB_MODE=offline sbatch slurm/train.sh
```

---

## 👀 Monitoring

### Check GPU Availability

Before submitting jobs, check which GPUs are available:

```bash
# Quick GPU check for veu partition
./slurm/check_gpus.sh

# Or use SLURM commands directly
sinfo -p veu -o "%n %G %a %T"

# Check specific node details
scontrol show node veuc12
```

**VEU Partition GPU Summary:**
- `veuc01`: 8 GPUs, 40 CPUs, 256GB RAM
- `veuc05`: 3 GPUs, 48 CPUs, 256GB RAM
- `veuc09`: 7 GPUs, 16 CPUs, 257GB RAM
- `veuc10`: 7 GPUs, 16 CPUs, 515GB RAM (most RAM!)
- `veuc11`: 7 GPUs, 40 CPUs, 385GB RAM
- `veuc12`: 8 GPUs, 64 CPUs, 257GB RAM

**Node States:**
- `IDLE`: Fully available
- `MIXED`: Partially allocated (some resources free)
- `ALLOCATED`: Fully in use
- `DRAINED`: Offline/unavailable

### Check Job Status

```bash
# View your running/pending jobs
squeue -u $USER

# View detailed job info
scontrol show job <JOB_ID>

# Cancel a job
scancel <JOB_ID>
```

### View Training Logs

SLURM output is saved to `logs/slurm/cognialigned_<JOB_ID>.txt`

```bash
# Follow training progress in real-time
tail -f logs/slurm/cognialigned_<JOB_ID>.txt

# View entire log
cat logs/slurm/cognialigned_<JOB_ID>.txt

# Search for errors
grep -i error logs/slurm/cognialigned_<JOB_ID>.txt
```

### Monitor GPU Usage

If you have SSH access to the compute node:

```bash
# Check which node your job is running on
squeue -u $USER

# SSH to that node (e.g., veuc12)
ssh veuc12

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### Check W&B Dashboard

Monitor training in real-time at: https://wandb.ai/YOUR_USERNAME/CogniAligned

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM) Errors

**Symptoms**: CUDA out of memory error, job killed

**Solutions**:
- Reduce batch size in config: `train.batch_size: 16` (or even 8)
- Use gradient accumulation (requires code modification)
- Request more GPU memory: edit `slurm/train.sh` and change `#SBATCH --gres=gpu:2` to use GPUs with more memory

#### 2. Job Not Starting

**Symptoms**: Job stays in PENDING state

**Solutions**:
```bash
# Check why job is pending
squeue -u $USER --start

# Check partition availability
sinfo -p veu

# Try a different partition if available
# Edit slurm/train.sh and change #SBATCH -p veu
```

#### 3. Module/Import Errors

**Symptoms**: `ModuleNotFoundError` or import errors

**Solutions**:
```bash
# Recreate virtual environment
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### 4. Data Not Found Errors

**Symptoms**: `FileNotFoundError` for data files

**Solutions**:
- Verify data path exists:
  ```bash
  ls -la /home/usuaris/veussd/roger.esteve.sanchez/adresso/processed_data/diagnosis/train/
  ```
- Update paths in `modules/configs/default.yaml` if data is elsewhere
- Check permissions: ensure you can read the data directory

#### 5. W&B Login Issues

**Symptoms**: W&B authentication errors

**Solutions**:
```bash
# Re-login to W&B
wandb login --relogin

# Or set API key directly
export WANDB_API_KEY=<your_api_key>
```

#### 6. NCCL/GPU Communication Errors

**Symptoms**: NCCL timeout, GPU communication failures

**Solutions**:
- The SLURM script already sets `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1`
- If issues persist, try running on a single GPU:
  ```bash
  # Edit slurm/train.sh
  #SBATCH --gres=gpu:1
  ```

### Getting Help

If you encounter issues:

1. **Check logs**: Review SLURM output in `logs/slurm/`
2. **Check W&B**: Look for errors in the W&B dashboard
3. **Test locally**: Try running without SLURM to isolate the issue
4. **Check resources**: Use `sinfo`, `squeue` to verify cluster availability

---

## 📁 File Structure

```
CogniAligned/
├── slurm/
│   └── train.sh                      # SLURM batch script
├── modules/
│   ├── main.py                       # Main training script
│   ├── model.py                      # Model architectures
│   ├── dataset.py                    # Data loading
│   ├── utils.py                      # Training utilities
│   ├── model_utils.py                # Model statistics and logging
│   └── configs/
│       ├── default.yaml              # Default configuration
│       ├── qwen.yaml                 # Qwen model config
│       ├── mistral.yaml              # Mistral model config
│       └── stella.yaml               # Stella model config
├── logs/
│   ├── slurm/                        # SLURM output logs
│   └── <model_name>/                 # Training logs per model
├── .cache/
│   ├── huggingface/                  # HuggingFace model cache
│   └── wandb/                        # W&B cache
├── .venv/                            # Virtual environment
├── requirements.txt                  # Python dependencies
├── README.md                         # Project README
└── HPC_README.md                     # This file
```

---

## 🎓 Tips and Best Practices

### 1. Start with a Small Test Run

Before running a full 200-epoch training:
```yaml
# In your config file
train:
  num_epochs: 5
  early_stopping: False
```

### 2. Use Cross-Validation Wisely

Cross-validation runs 5 separate trainings. For quick experiments, disable it:
```yaml
train:
  cross_validation: False
```

### 3. Monitor W&B Early

Check W&B after the first few epochs to ensure:
- Metrics are being logged correctly
- Loss is decreasing
- No NaN values appear

### 4. Save Intermediate Checkpoints

Models are automatically saved in `logs/<model_name>/model_fold_<N>.pth`

### 5. Use Descriptive Run Names

The run name includes:
- Model name
- Fold number (if cross-validation)
- SLURM job ID

This makes it easy to identify runs later.

### 6. Experiment Tracking

Use W&B tags and notes to organize experiments:
```python
# In the W&B dashboard, add tags like:
# - "baseline"
# - "qwen-fusion"
# - "hyperparameter-search"
```

---

## 📝 Example Workflows

### Workflow 1: Baseline Experiment

```bash
cd /home/usuaris/veussd/roger.esteve.sanchez/CogniAligned

# Submit baseline training with default config
sbatch slurm/train.sh

# Monitor progress
tail -f logs/slurm/cognialigned_*.txt
```

### Workflow 2: Hyperparameter Search

```bash
# Create configs for different learning rates
cp modules/configs/default.yaml modules/configs/lr_1e-5.yaml
# Edit lr_1e-5.yaml: learning_rate: 0.00001

cp modules/configs/default.yaml modules/configs/lr_5e-5.yaml
# Edit lr_5e-5.yaml: learning_rate: 0.00005

# Submit both
sbatch slurm/train.sh modules/configs/lr_1e-5.yaml
sbatch slurm/train.sh modules/configs/lr_5e-5.yaml

# Compare in W&B
```

### Workflow 3: Model Comparison

```bash
# Test different fusion strategies
sbatch slurm/train.sh modules/configs/default.yaml        # cross fusion
sbatch slurm/train.sh modules/configs/qwen.yaml           # qwen model

# Compare results in W&B under the CogniAligned project
```

---

## 🎯 Next Steps

After your first successful run:

1. ✅ Review results in W&B
2. ✅ Compare different model configurations
3. ✅ Experiment with fusion strategies
4. ✅ Try different embedding models (BERT, RoBERTa, Qwen, etc.)
5. ✅ Optimize hyperparameters based on validation performance

---

## 📚 Additional Resources

- **Original Paper**: [CogniAlign on arXiv](https://arxiv.org/abs/2506.01890)
- **W&B Documentation**: https://docs.wandb.ai/
- **SLURM Cheat Sheet**: https://slurm.schedmd.com/pdfs/summary.pdf
- **ADReSSo Challenge**: https://dementia.talkbank.org/ADReSSo-2021/

---

**Good luck with your experiments! 🚀**

If you encounter any issues not covered in this guide, please check the main README.md or contact the project maintainers.
