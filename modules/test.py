import os
import sys
import torch
import pandas as pd
import wandb
import numpy as np
from dotmap import DotMap
from dataset import AdressoDataset
from main import set_up
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_test_dataloader(config):
    # Read test labels for ground truth comparison
    # The user mentioned task1.csv has the ground truth
    ground_truth_path = 'task1.csv'
    if os.path.exists(ground_truth_path):
        gt_df = pd.read_csv(ground_truth_path)
        # Map "Control" -> 0, "ProbableAD" -> 1
        gt_mapping = {"Control": 0, "ProbableAD": 1}
        gt_df['label'] = gt_df['Dx'].map(gt_mapping)
        gt_dict = dict(zip(gt_df['ID'], gt_df['label']))
    else:
        print(f"Warning: Ground truth file {ground_truth_path} not found.")
        gt_dict = {}

    # Read test CSV from config (this is usually the one with IDs for the competition)
    labels_pd = pd.read_csv(config.data.test_csv_labels_path)
    
    uids = []
    features = []
    labels = []
    
    pauses_data = '_pauses' if config.model.pauses else ''
    name_mapping_text = {
        'bert': '', 'distil': 'distil', 'roberta': 'roberta',
        'mistral': 'mistral', 'qwen': 'qwen', 'stella': 'stella'
    }
    name_mapping_audio = {
        'wav2vec2': 'audio', 'egemaps': 'egemaps', 'mel': 'mel'
    }
    audio_data = '_' + name_mapping_audio[config.model.audio_model] if config.model.audio_model != '' else ''
    
    text_path = config.data.test_root_text_path

    for index, row in labels_pd.iterrows():
        uid = row['ID']
        
        text_embeddings_path = None
        audio_embeddings_path = None

        if config.model.textual_model != '':
            text_embeddings_path = os.path.join(text_path, uid + 
                                                    name_mapping_text[config.model.textual_model] + pauses_data + '.pt')
            
        if config.model.audio_model != '':
            textual_data = name_mapping_text[config.model.textual_model] if config.model.textual_model != '' else 'distil'
            audio_embeddings_path = os.path.join(text_path, uid + textual_data 
                                                 + pauses_data + audio_data + '.pt')
        
        # Check if files exist
        found = False
        if config.model.multimodality:
            if os.path.exists(audio_embeddings_path) and os.path.exists(text_embeddings_path):
                feat = (torch.load(audio_embeddings_path, map_location=device), 
                        torch.load(text_embeddings_path, map_location=device))
                found = True
            else:
                print(f"Skipping missing files for {uid}")
        else:
            if config.model.textual_model != '' and os.path.exists(text_embeddings_path):
                feat = torch.load(text_embeddings_path, map_location=device)
                found = True
            elif config.model.audio_model != '' and os.path.exists(audio_embeddings_path):
                feat = torch.load(audio_embeddings_path, map_location=device)
                found = True
            else:
                print(f"Skipping missing files for {uid}")

        if found:
            features.append(feat)
            uids.append(uid)
            # Use ground truth label if available, otherwise dummy 0
            label_val = gt_dict.get(uid, 0)
            labels.append(torch.tensor(label_val).to(device).float())

    test_dataset = AdressoDataset(features, labels)
    test_dataloader = DataLoader(test_dataset, batch_size=config.train.batch_size, shuffle=False)
    
    return test_dataloader, uids, labels


def test(config):
    test_dataloader, uids, true_labels_tensors = get_test_dataloader(config)
    true_labels = [l.item() for l in true_labels_tensors]
    
    log_path = os.path.join('logs', config.path_name)
    
    # Initialize array to store probabilities from all folds
    all_fold_probs = []

    if config.train.cross_validation:
        for fold in range(config.train.cross_validation_folds):
            model, _, _, _ = set_up(config, test_dataloader, device, fold)
            # We need to finish the wandb run started by set_up since we only want one test run
            if wandb.run:
                wandb.finish()
                
            model_path = os.path.join(log_path, f'model_fold_{fold}.pth')
            
            if not os.path.exists(model_path):
                print(f"Model weights for fold {fold} not found. Ensure training is completed.")
                return

            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            fold_probs = []
            with torch.no_grad():
                for features, _ in test_dataloader:
                    outputs = model(features).squeeze(-1)
                    probs = torch.sigmoid(outputs)
                    fold_probs.extend(probs.cpu().numpy())
            
            all_fold_probs.append(fold_probs)
            print(f"Computed predictions for fold {fold}")

        # Average probabilities across folds for ensemble
        avg_probs = np.mean(all_fold_probs, axis=0)
        final_predictions = [1 if p >= 0.5 else 0 for p in avg_probs]
    else:
        model, _, _, _ = set_up(config, test_dataloader, device)
        if wandb.run:
            wandb.finish()
            
        model_path = os.path.join(log_path, 'model.pt')
        
        if not os.path.exists(model_path):
            print("Model weights not found. Ensure training is completed.")
            return
            
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        fold_probs = []
        with torch.no_grad():
            for features, _ in test_dataloader:
                outputs = model(features).squeeze(-1)
                probs = torch.sigmoid(outputs)
                fold_probs.extend(probs.cpu().numpy())
                
        avg_probs = np.array(fold_probs)
        final_predictions = [1 if p >= 0.5 else 0 for p in fold_probs]

    # Calculate Metrics
    accuracy = accuracy_score(true_labels, final_predictions)
    f1 = f1_score(true_labels, final_predictions)
    precision = precision_score(true_labels, final_predictions)
    recall = recall_score(true_labels, final_predictions)
    
    # Calculate binary cross entropy loss
    avg_probs_t = torch.tensor(avg_probs)
    true_labels_t = torch.tensor(true_labels).float()
    test_loss = F.binary_cross_entropy(avg_probs_t, true_labels_t).item()

    print("\n" + "="*30)
    print("TEST SET EVALUATION (Ensemble)")
    print("="*30)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Loss:      {test_loss:.4f}")
    print("-" * 30)
    print(classification_report(true_labels, final_predictions, target_names=['cn', 'ad']))
    print("="*30)

    # Log to W&B as the 6th run
    slurm_job_id = os.environ.get('SLURM_JOB_ID', '')
    wandb_project = config.wandb.project if hasattr(config, 'wandb') else 'CogniAligned'
    wandb_entity = config.wandb.entity if hasattr(config, 'wandb') and config.wandb.entity else None
    
    run_name = f"{config.model_name}_final_test"
    if slurm_job_id:
        run_name = f"{run_name}_job{slurm_job_id}"

    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=run_name,
        group=slurm_job_id if slurm_job_id else config.model_name,
        job_type="test_evaluation",
        config={
            "model_name": config.model_name,
            "evaluation_type": "ensemble_test",
            "slurm_job_id": slurm_job_id
        }
    )
    
    wandb.log({
        "test/accuracy": accuracy,
        "test/f1": f1,
        "test/precision": precision,
        "test/recall": recall,
        "test/loss": test_loss,
    })
    
    wandb.finish()

    # Map numeric predictions back to strings for CSV save
    pred_labels = ['ad' if p == 1 else 'cn' for p in final_predictions]
    results_df = pd.DataFrame({'ID': uids, 'Prediction': pred_labels})
    output_csv = os.path.join(log_path, 'test_predictions.csv')
    results_df.to_csv(output_csv, index=False)
    print(f"Predictions successfully saved to: {output_csv}")


import yaml

if __name__ == '__main__':
    config_path = sys.argv[sys.argv.index('--config') + 1] if '--config' in sys.argv else 'modules/configs/default.yaml'
    
    with open(config_path, 'r') as f:
        config_yaml = yaml.safe_load(f)
    config = DotMap(config_yaml)
    
    # Re-build dynamic config fields
    config.model.multimodality = config.model.textual_model != '' and config.model.audio_model != ''
    textual_data = config.model.textual_model + '_' if config.model.textual_model != '' else ''
    audio_data = config.model.audio_model + '_' if config.model.audio_model != '' else ''
    pauses_data = 'P_' if config.model.pauses else ''
    config.model_name = f"{textual_data}{audio_data}{pauses_data}{config.model.fusion}"
    config.model.model_name = config.model_name
    config.path_name = f"{config.model_name}_{config.model.pooling}"
    
    test(config)