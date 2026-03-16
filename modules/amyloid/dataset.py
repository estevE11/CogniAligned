from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
import os
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AmyloidDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

name_mapping_text = {
    'bert': '',
    'distil': 'distil',
    'roberta': 'roberta',
    'mistral': 'mistral',
    'qwen': 'qwen',
    'stella': 'stella'
}

name_mapping_audio = {
    'wav2vec2': 'audio',
    'egemaps': 'egemaps',
    'mel': 'mel'
}

def read_CSV(config):
    labels_path = config.data.csv_labels_path
    # Read CSV
    labels_pd = pd.read_csv(labels_path)
    
    # Filter for valid amyloid labels (exclude 'exclude')
    # Convert column to string to handle mixed types if any, then strip whitespace
    labels_pd = labels_pd[labels_pd['DX_PILAR_amyloid'].astype(str).str.strip() != 'exclude'].reset_index(drop=True)

    uids = []
    features = []
    labels = []

    pauses_data = '_pauses' if config.model.pauses else ''
    audio_data = '_' + name_mapping_audio[config.model.audio_model] if config.model.audio_model != '' else ''
    
    text_path = config.data.root_text_path

    for index, row in labels_pd.iterrows():
        # Filename is in the first column
        filename = row.iloc[0] 
        file_id = os.path.splitext(filename)[0] # Remove extension
        
        # If there's a second extension (like .alac.pt), remove it too
        file_id = os.path.splitext(file_id)[0] if file_id.endswith('.alac') else file_id
        
        text_embeddings_path = None
        audio_embeddings_path = None

        if config.model.textual_model != '':
            text_embeddings_path = os.path.join(text_path, file_id + 
                                                    name_mapping_text[config.model.textual_model] + pauses_data + '.pt')
            
        if config.model.audio_model != '':
            textual_data = name_mapping_text[config.model.textual_model] if config.model.textual_model != '' else 'distil'
            audio_embeddings_path = os.path.join(text_path, file_id + textual_data 
                                                 + pauses_data + audio_data + '.pt')
        
        # Get label (0 or 1)
        try:
            label_val = float(row['DX_PILAR_amyloid'])
        except ValueError:
            print(f"Skipping invalid label {row['DX_PILAR_amyloid']} for {file_id}")
            continue

        # Check if files exist
        if config.model.multimodality:
            if os.path.exists(audio_embeddings_path) and os.path.exists(text_embeddings_path):
                audio_feat = torch.load(audio_embeddings_path, map_location='cpu')
                text_feat = torch.load(text_embeddings_path, map_location='cpu')
                
                if torch.isnan(audio_feat).any() or torch.isnan(text_feat).any():
                    print(f"DEBUG: NaN detected in saved file for {file_id}! Skipping.")
                    continue
                    
                features.append((audio_feat.to(device), text_feat.to(device)))
                uids.append(file_id)
                labels.append(torch.tensor(label_val).to(device).float()) # Float for BCEWithLogitsLoss
            else:
                # print(f"Skipping missing files for {file_id}")
                continue
        else:
            if config.model.textual_model != '' and os.path.exists(text_embeddings_path):
                text_feat = torch.load(text_embeddings_path, map_location='cpu')
                if torch.isnan(text_feat).any():
                    print(f"DEBUG: NaN detected in saved file for {file_id}! Skipping.")
                    continue
                features.append(text_feat.to(device))
                uids.append(file_id)
                labels.append(torch.tensor(label_val).to(device).float())
            elif config.model.audio_model != '' and os.path.exists(audio_embeddings_path):
                audio_feat = torch.load(audio_embeddings_path, map_location='cpu')
                if torch.isnan(audio_feat).any():
                    print(f"DEBUG: NaN detected in saved file for {file_id}! Skipping.")
                    continue
                features.append(audio_feat.to(device))
                uids.append(file_id)
                labels.append(torch.tensor(label_val).to(device).float())
            else:
                # print(f"Skipping missing files for {file_id}")
                continue

    return uids, features, labels, labels_pd

def set_splits(config):
    labels_path = config.data.csv_labels_path
    splits_dir = config.data.splits_path
    
    labels_pd = pd.read_csv(labels_path)
    # Filter for valid amyloid labels
    labels_pd = labels_pd[labels_pd['DX_PILAR_amyloid'].astype(str).str.strip() != 'exclude'].reset_index(drop=True)
    
    groups = labels_pd['UT ID']
    X = labels_pd.index.values
    y = labels_pd['DX_PILAR_amyloid'].astype(float) # Ensure numeric for stratification
    
    # Try StratifiedGroupKFold first, fallback to GroupKFold if not available
    try:
        gkf = StratifiedGroupKFold(n_splits=5)
        print("Using StratifiedGroupKFold for splitting.")
    except NameError:
        gkf = GroupKFold(n_splits=5)
        print("Using GroupKFold for splitting (StratifiedGroupKFold not found).")
    
    os.makedirs(splits_dir, exist_ok=True)
    
    for i, (train_index, test_index) in enumerate(gkf.split(X, y, groups)):
        # Save the filenames (IDs) for consistency with existing pipeline
        # Remove .alac extension if present before saving to split
        def get_clean_id(idx):
            fname = labels_pd.iloc[idx].iloc[0]
            # Use the same logic as read_CSV to get the file_id
            file_id = os.path.splitext(fname)[0] # Remove extension
            file_id = os.path.splitext(file_id)[0] if file_id.endswith('.alac') else file_id
            return file_id

        train_uids = [get_clean_id(idx) for idx in train_index]
        val_uids = [get_clean_id(idx) for idx in test_index]
        
        np.save(os.path.join(splits_dir, f'train_uids{i}'), np.array(train_uids))
        np.save(os.path.join(splits_dir, f'val_uids{i}'), np.array(val_uids))
        print(f"Fold {i} created with {len(train_uids)} train and {len(val_uids)} val samples.")

def get_dataloaders(config, kfold_number=0):
    uids, features, labels, _ = read_CSV(config)
    
    splits_dir = config.data.splits_path
    validation_split = np.load(os.path.join(splits_dir, f'val_uids{kfold_number}.npy'))
    
    batch_size = config.train.batch_size
    
    train_features = []
    train_labels = []
    validation_features = []
    validation_labels = []
    
    # Convert validation_split to set for faster lookup
    val_set = set(validation_split)
    
    # print(f"DEBUG: Fold {kfold_number} - val_set size: {len(val_set)}")
    # print(f"DEBUG: Fold {kfold_number} - total uids: {len(uids)}")

    for i, uid in enumerate(uids):
        if uid in val_set:
            validation_features.append(features[i])
            validation_labels.append(labels[i])
        else:
            train_features.append(features[i])
            train_labels.append(labels[i])
            
    print(f"Fold {kfold_number}: {len(train_features)} train samples, {len(validation_features)} validation samples.")
    
    if len(train_features) == 0 or len(validation_features) == 0:
        print(f"WARNING: Fold {kfold_number} has empty splits! Check if UID formats match.")
        if len(uids) > 0 and len(val_set) > 0:
            print(f"Sample UID from data: {uids[0]}")
            print(f"Sample UID from split: {list(val_set)[0]}")
            
    train_dataset = AmyloidDataset(train_features, train_labels)
    validation_dataset = AmyloidDataset(validation_features, validation_labels)
    
    # For amyloid, the class distribution is already reasonably balanced.
    # To avoid potential numerical issues from extreme sampling weights,
    # we start without WeightedRandomSampler.
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, validation_dataloader
