from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
import os
import re
from sklearn.model_selection import KFold

# Default paths (will be overridden by config)
root_text_path = '/dataset/diagnosis/train/text/'
root_audio_path = '/dataset/diagnosis/train/audio/'
csv_labels_path = '/dataset/diagnosis/train/adresso-train-mmse-scores.csv'
splits_path = '/dataset/diagnosis/train/splits/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_length_wav2vec = 4000

class AdressoDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    # Read CSV with labels (use config path if available, otherwise use default)
    labels_path = config.data.csv_labels_path if hasattr(config, 'data') else csv_labels_path
    labels_pd = pd.read_csv(labels_path)

    uids = []
    features = []
    labels = []

    pauses_data = '_pauses' if config.model.pauses else ''
    audio_data = '_' + name_mapping_audio[config.model.audio_model] if config.model.audio_model != '' else ''
    
    # Use config paths if available, otherwise use defaults
    text_path = config.data.root_text_path if hasattr(config, 'data') else root_text_path

    print(f"DEBUG: Reading CSV from {labels_path}")
    print(f"DEBUG: Feature root path: {text_path}")
    
    # Cache directory listing for faster lookups
    dir_contents = os.listdir(text_path) if os.path.exists(text_path) else []

    # Load acoustic features if enabled
    acoustic_features_df = None
    feature_cols = []
    if hasattr(config.model, 'use_acoustic_features') and config.model.use_acoustic_features:
        af_path = config.data.acoustic_features_path
        if os.path.exists(af_path):
            print(f"DEBUG: Loading acoustic features from {af_path}")
            acoustic_features_df = pd.read_csv(af_path)
            # Normalize features (Z-score)
            feature_cols = [c for c in acoustic_features_df.columns if c not in ['file_id', 'class', 'patient_id', 'Unnamed: 0']]
            # Handle NaNs
            acoustic_features_df[feature_cols] = acoustic_features_df[feature_cols].fillna(0)
            
            # Z-score normalization
            mean = acoustic_features_df[feature_cols].mean()
            std = acoustic_features_df[feature_cols].std()
            # Avoid division by zero
            std = std.replace(0, 1)
            acoustic_features_df[feature_cols] = (acoustic_features_df[feature_cols] - mean) / std
            
            # Set index for faster lookup
            if 'file_id' in acoustic_features_df.columns:
                acoustic_features_df.set_index('file_id', inplace=True)
            elif 'patient_id' in acoustic_features_df.columns:
                acoustic_features_df.set_index('patient_id', inplace=True)
        else:
            print(f"WARNING: Acoustic features file not found at {af_path}")

    for index, row in labels_pd.iterrows():
        text_embeddings_path = None
        audio_embeddings_path = None

        # Handle both PPA and ADReSSo/Amyloid column names
        dx_col = 'dx' if 'dx' in row else ('DX_Pilar' if 'DX_Pilar' in row else 'DX_PILAR_amyloid')
        uid_col = 'adressfname' if 'adressfname' in row else ('UT ID' if 'UT ID' in row else labels_pd.columns[0])

        # Clean file_id (remove .alac and other extensions)
        filename = str(row[uid_col])
        file_id = os.path.splitext(filename)[0]
        if file_id.endswith('.alac'):
            file_id = os.path.splitext(file_id)[0]
            
        # Get acoustic features
        acoustic_feat = None
        if acoustic_features_df is not None:
            if file_id in acoustic_features_df.index:
                vals = acoustic_features_df.loc[file_id, feature_cols].values.astype(float)
                acoustic_feat = torch.tensor(vals, dtype=torch.float32).to(device)
            else:
                # print(f"WARNING: No acoustic features for {file_id}")
                vals = np.zeros(len(feature_cols))
                acoustic_feat = torch.tensor(vals, dtype=torch.float32).to(device)

        if config.model.textual_model != '':
            suffix = name_mapping_text[config.model.textual_model] + pauses_data + '.pt'
            # Try direct path options first
            path_options = [
                os.path.join(text_path, filename + suffix),
                os.path.join(text_path, file_id + suffix),
                os.path.join(text_path, str(row[dx_col]), filename + suffix),
                os.path.join(text_path, str(row[dx_col]), file_id + suffix)
            ]
            for p in path_options:
                if os.path.exists(p):
                    text_embeddings_path = p
                    break
            
            if not text_embeddings_path:
                # Regex fallback: find any file containing the numeric ID and matching suffix
                match = re.search(r'(\d+)', filename)
                if match:
                    num_id = match.group(1)
                    for f in dir_contents:
                        if num_id in f and f.endswith(suffix):
                            text_embeddings_path = os.path.join(text_path, f)
                            break
            
        if config.model.audio_model != '':
            textual_data = name_mapping_text[config.model.textual_model] if config.model.textual_model != '' else 'distil'
            suffix = textual_data + pauses_data + audio_data + '.pt'
            path_options = [
                os.path.join(text_path, filename + suffix),
                os.path.join(text_path, file_id + suffix),
                os.path.join(text_path, str(row[dx_col]), filename + suffix),
                os.path.join(text_path, str(row[dx_col]), file_id + suffix)
            ]
            for p in path_options:
                if os.path.exists(p):
                    audio_embeddings_path = p
                    break
            
            if not audio_embeddings_path:
                match = re.search(r'(\d+)', filename)
                if match:
                    num_id = match.group(1)
                    for f in dir_contents:
                        if num_id in f and f.endswith(suffix):
                            audio_embeddings_path = os.path.join(text_path, f)
                            break
        
        # Check if files exist
        if config.model.multimodality:
            if audio_embeddings_path and text_embeddings_path:
                feats = (torch.load(audio_embeddings_path, map_location=device), 
                         torch.load(text_embeddings_path, map_location=device))
                if acoustic_feat is not None:
                    feats = feats + (acoustic_feat,)
                features.append(feats)
                uids.append(file_id)
                label_val = 0 if str(row[dx_col]).lower() in ['cn', '0', 'lvppa'] else (1 if str(row[dx_col]).lower() in ['ad', '1', 'nfppa'] else 2)
                labels.append(torch.tensor(label_val).to(device).float() if config.model.num_classes == 1 else torch.tensor(label_val).to(device).long())
            else:
                continue
        else:
            if config.model.textual_model != '' and text_embeddings_path:
                feat = torch.load(text_embeddings_path, map_location=device)
                if acoustic_feat is not None:
                    feat = (feat, acoustic_feat)
                features.append(feat)
                uids.append(file_id)
                label_val = 0 if str(row[dx_col]).lower() in ['cn', '0', 'lvppa'] else (1 if str(row[dx_col]).lower() in ['ad', '1', 'nfppa'] else 2)
                labels.append(torch.tensor(label_val).to(device).float() if config.model.num_classes == 1 else torch.tensor(label_val).to(device).long())
            elif config.model.audio_model != '' and audio_embeddings_path:
                feat = torch.load(audio_embeddings_path, map_location=device)
                if acoustic_feat is not None:
                    feat = (feat, acoustic_feat)
                features.append(feat)
                uids.append(file_id)
                label_val = 0 if str(row[dx_col]).lower() in ['cn', '0', 'lvppa'] else (1 if str(row[dx_col]).lower() in ['ad', '1', 'nfppa'] else 2)
                labels.append(torch.tensor(label_val).to(device).float() if config.model.num_classes == 1 else torch.tensor(label_val).to(device).long())
            else:
                continue

    print(f"DEBUG: Successfully loaded {len(features)} samples.")
    return uids, features, labels

def get_dataloaders(config, kfold_number = 0):
    uids, features, labels = read_CSV(config)
    
    # Use config path if available, otherwise use default
    splits_dir = config.data.splits_path if hasattr(config, 'data') else splits_path
    validation_split = np.load(os.path.join(splits_dir, f'val_uids{kfold_number}.npy'))

    batch_size=config.train.batch_size
    # Split those lists into training and validation
    train_uids = []
    train_features = []
    train_labels = []

    validation_uids = []
    validation_features = []
    validation_labels = []

    # Convert validation_split to set for faster lookup
    val_set = set(validation_split)

    for i in range(len(uids)):
        if uids[i] in val_set:
            validation_uids.append(uids[i])
            validation_features.append(features[i])
            validation_labels.append(labels[i])
        else:
            train_uids.append(uids[i])
            train_features.append(features[i])
            train_labels.append(labels[i])

    train_dataset = AdressoDataset(train_features, train_labels)
    validation_dataset = AdressoDataset(validation_features, validation_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, validation_dataloader

def set_splits(config=None):
    # Use config paths if provided
    labels_path = config.data.csv_labels_path if (config and hasattr(config, 'data')) else csv_labels_path
    splits_dir = config.data.splits_path if (config and hasattr(config, 'data')) else splits_path
    
    labels_pd = pd.read_csv(labels_path)
    uids = []

    uid_col = 'adressfname' if 'adressfname' in labels_pd.columns else ('UT ID' if 'UT ID' in labels_pd.columns else labels_pd.columns[0])

    for index, row in labels_pd.iterrows():
        # Clean UID format
        filename = str(row[uid_col])
        file_id = os.path.splitext(filename)[0]
        if file_id.endswith('.alac'):
            file_id = os.path.splitext(file_id)[0]
        uids.append(file_id)

    # Split the uids into 5 folds with kfold from sklearn
    kfold = KFold(n_splits=5, shuffle=True, random_state=43)

    os.makedirs(splits_dir, exist_ok=True)
    for i, (train_index, test_index) in enumerate(kfold.split(uids)):
        print("TRAIN:", train_index, "TEST:", test_index)
        np.save(os.path.join(splits_dir, f'train_uids{i}'), np.array(uids)[train_index])
        np.save(os.path.join(splits_dir, f'val_uids{i}'), np.array(uids)[test_index])

def get_splits_stats(config=None):
    # Use config paths if provided
    labels_path = config.data.csv_labels_path if (config and hasattr(config, 'data')) else csv_labels_path
    splits_dir = config.data.splits_path if (config and hasattr(config, 'data')) else splits_path
    
    labels_pd = pd.read_csv(labels_path)
    uids = []

    uid_col = 'adressfname' if 'adressfname' in labels_pd.columns else ('UT ID' if 'UT ID' in labels_pd.columns else labels_pd.columns[0])

    for index, row in labels_pd.iterrows():
        # Clean UID format
        filename = str(row[uid_col])
        file_id = os.path.splitext(filename)[0]
        if file_id.endswith('.alac'):
            file_id = os.path.splitext(file_id)[0]
        uids.append(file_id)

    for i in range(5):
        training_split = np.load(os.path.join(splits_dir, f'train_uids{i}.npy'))
        validation_split = np.load(os.path.join(splits_dir, f'val_uids{i}.npy'))
        n_cn_train = 0
        n_ad_train = 0
        n_cn_val = 0
        n_ad_val = 0

        dx_col = 'dx' if 'dx' in labels_pd.columns else ('DX_Pilar' if 'DX_Pilar' in labels_pd.columns else 'DX_PILAR_amyloid')

        for uid in training_split:
            if str(labels_pd[labels_pd[uid_col] == uid][dx_col].values[0]).lower() in ['cn', '0']:
                n_cn_train += 1
            else:
                n_ad_train += 1

        for uid in validation_split:
            if str(labels_pd[labels_pd[uid_col] == uid][dx_col].values[0]).lower() in ['cn', '0']:
                n_cn_val += 1
            else:
                n_ad_val += 1

        print(f"Fold {i}:")
        print(f"Training CN: {n_cn_train}, Training AD: {n_ad_train}")
        print(f"Validation CN: {n_cn_val}, Validation AD: {n_ad_val}")
