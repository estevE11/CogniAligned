import sys
import os
import torch
import numpy as np
from sklearn.metrics import confusion_matrix

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ppa.dataset import get_dataloaders
from utils import get_config
from model import CrossAttentionTransformerEncoder, MyTransformerEncoder, BidirectionalCrossAttentionTransformerEncoder, ElementWiseFusionEncoder, MambaFusionEncoder

def load_model(config, fold, device):
    if config.model.multimodality:
        if 'mamba' in config.model.fusion:
            model = MambaFusionEncoder(config.model).to(device)
        elif 'bicross' in config.model.fusion:
            model = BidirectionalCrossAttentionTransformerEncoder(config.model).to(device)
        elif 'cross' in config.model.fusion:
            model = CrossAttentionTransformerEncoder(config.model).to(device)
        else:
            model = ElementWiseFusionEncoder(config.model).to(device)
    else:
         model = MyTransformerEncoder(config.model).to(device)

    log_path = os.path.join('logs', config.path_name)
    model_path = os.path.join(log_path, f'model_fold_{fold}.pth')
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return None
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def evaluate_fold(config, fold, device):
    print(f"\n=== Evaluating Fold {fold} ===")
    _, val_loader = get_dataloaders(config, kfold_number=fold)
    model = load_model(config, fold, device)
    
    if model is None:
        return

    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            features, labels = batch
            
            if config.model.multimodality:
                # features is [audio_batch, text_batch]
                audio_emb = features[0].to(device)
                text_emb = features[1].to(device)
                outputs = model((audio_emb, text_emb))
            else:
                features = features.to(device)
                outputs = model(features)
            
            labels = labels.to(device)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    
    # Class mapping from dataset.py: 'lvPPA': 0, 'nfPPA': 1, 'svPPA': 2
    class_names = ['lvPPA', 'nfPPA', 'svPPA']
    
    print("\nConfusion Matrix (Rows=True, Cols=Pred):")
    print(f"{'':>10} {'lvPPA':>10} {'nfPPA':>10} {'svPPA':>10}")
    for i, row in enumerate(cm):
        print(f"{class_names[i]:>10} {row[0]:>10} {row[1]:>10} {row[2]:>10}")
        
    print("\nPer-class Recall (Sensitivity):")
    for i, class_name in enumerate(class_names):
        total = np.sum(cm[i])
        correct = cm[i][i]
        recall = correct / total if total > 0 else 0.0
        print(f"{class_name}: {correct}/{total} ({recall:.2%})")

    print("\nPredictions breakdown:")
    total_preds = np.sum(cm)
    for i, class_name in enumerate(class_names):
        count = np.sum(cm[:, i])
        pct = count / total_preds if total_preds > 0 else 0
        print(f"Predicted {class_name}: {count} times ({pct:.2%})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default to slower config if not provided
        config_path = "modules/configs/ppa/slower.yaml"
        print(f"No config provided, using default: {config_path}")
    else:
        config_path = sys.argv[sys.argv.index('--config') + 1]
    
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)
        
    config = get_config(config_path)
    
    # Manually set derived attributes as save_config would
    config.model.multimodality = config.model.textual_model != '' and config.model.audio_model != ''
    
    textual_data = config.model.textual_model if config.model.textual_model else ""
    audio_data = "_" + config.model.audio_model if config.model.audio_model else ""
    pauses_data = "_P" if config.model.pauses else ""
    
    config.model_name = f"{textual_data}{audio_data}{pauses_data}_{config.model.fusion}"
    config.model.model_name = config.model_name
    config.path_name = f"{config.model_name}_{config.model.pooling}"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Evaluate all folds
    for fold in range(5):
        evaluate_fold(config, fold, device)
