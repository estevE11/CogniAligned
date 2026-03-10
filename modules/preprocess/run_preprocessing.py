#!/usr/bin/env python3
"""
Wrapper script to run preprocessing with configurable paths.
This avoids modifying the original preprocessing scripts.
"""

import os
import sys

# Set paths before importing preprocessing modules
ADRESSO_ROOT = os.environ.get(
    'ADRESSO_ROOT', 
    '/home/usuaris/veussd/roger.esteve.sanchez/adresso/ADReSSo21'
)

# Configure paths
AUDIO_PATH = f"{ADRESSO_ROOT}/diagnosis/train/audio/"
TEXT_OUTPUT_PATH = f"{ADRESSO_ROOT}/diagnosis/train/text/"
CSV_LABELS_PATH = f"{ADRESSO_ROOT}/diagnosis/train/adresso-train-mmse-scores.csv"
SPLITS_PATH = f"{ADRESSO_ROOT}/diagnosis/train/splits/"

print(f"Using ADReSSo Root: {ADRESSO_ROOT}")
print(f"Audio Path: {AUDIO_PATH}")
print(f"Text Output Path: {TEXT_OUTPUT_PATH}")
print(f"CSV Labels Path: {CSV_LABELS_PATH}")
print(f"Splits Path: {SPLITS_PATH}")

# Create output directories
os.makedirs(f"{TEXT_OUTPUT_PATH}/ad", exist_ok=True)
os.makedirs(f"{TEXT_OUTPUT_PATH}/cn", exist_ok=True)
os.makedirs(SPLITS_PATH, exist_ok=True)

# Verify paths exist
if not os.path.exists(AUDIO_PATH):
    print(f"ERROR: Audio path does not exist: {AUDIO_PATH}")
    sys.exit(1)

if not os.path.exists(CSV_LABELS_PATH):
    print(f"ERROR: CSV labels file does not exist: {CSV_LABELS_PATH}")
    sys.exit(1)

print("\n" + "="*60)
print("Starting Preprocessing")
print("="*60 + "\n")

# Import and patch the preprocessing modules
print("Step 1/3: Running Whisper transcription...")
print("-" * 60)

import preprocesswhisper
# Monkey patch the paths
preprocesswhisper.root_path = AUDIO_PATH
preprocesswhisper.textual_data = f"{TEXT_OUTPUT_PATH.rstrip('/')}/text_transcriptions.csv"

# Run whisper preprocessing
preprocesswhisper.preprocess_whisper()

print("\n" + "="*60)
print("Step 2/3: Generating embeddings...")
print("-" * 60)

import preprocessembeddings
# Monkey patch the paths
preprocessembeddings.root_path = AUDIO_PATH
preprocessembeddings.root_text_path = TEXT_OUTPUT_PATH
preprocessembeddings.textual_data = f"{TEXT_OUTPUT_PATH.rstrip('/')}/text_transcriptions.csv"

# Run embeddings preprocessing
preprocessembeddings.preprocess_text()

print("\n" + "="*60)
print("Step 3/3: Creating cross-validation splits...")
print("-" * 60)

# Create splits using the dataset module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dataset import set_splits
from dotmap import DotMap

# Create minimal config for split creation
config = DotMap({
    'data': {
        'csv_labels_path': CSV_LABELS_PATH,
        'splits_path': SPLITS_PATH
    }
})

set_splits(config)

print("\n" + "="*60)
print("Preprocessing Complete!")
print("="*60)
print(f"\nOutput saved to: {TEXT_OUTPUT_PATH}")
print(f"Splits saved to: {SPLITS_PATH}")
