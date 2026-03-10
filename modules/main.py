from dataset import get_dataloaders
from utils import set_seed, get_config, train, save_config
from model import CrossAttentionTransformerEncoder, MyTransformerEncoder, BidirectionalCrossAttentionTransformerEncoder, ElementWiseFusionEncoder
from model_utils import log_model_summary_to_wandb
import torch
import wandb
import sys
import torch.nn as nn
from transformers import get_scheduler
from torch.optim import AdamW
import os

wandb.login()


def set_up(config, train_dataloader, device, fold=0):
    """Set up model, optimizer, loss function, and scheduler."""
    set_seed(42)
    
    if config.model.multimodality:
        if 'bicross' in config.model.fusion:
            model = BidirectionalCrossAttentionTransformerEncoder(config.model).to(device)
        elif 'cross' in config.model.fusion:
            model = CrossAttentionTransformerEncoder(config.model).to(device)
        else:
            model = ElementWiseFusionEncoder(config.model).to(device)
    else:
         model = MyTransformerEncoder(config.model).to(device)


    optimizer = AdamW(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    lossfn = nn.BCEWithLogitsLoss()
    
    num_training_steps = config.train.num_epochs * len(train_dataloader)

    lr_scheduler = get_scheduler(
        name="cosine", optimizer=optimizer, num_warmup_steps=20, num_training_steps=num_training_steps
    )

    # Get W&B configuration from config or environment
    wandb_project = os.environ.get('WANDB_PROJECT', config.wandb.project if hasattr(config, 'wandb') else 'CogniAligned')
    wandb_entity = os.environ.get('WANDB_ENTITY', config.wandb.entity if hasattr(config, 'wandb') and config.wandb.entity else None)
    wandb_mode = os.environ.get('WANDB_MODE', config.wandb.mode if hasattr(config, 'wandb') else 'online')
    
    # Create run name with SLURM job ID if available
    slurm_job_id = os.environ.get('SLURM_JOB_ID', '')
    run_name = f"{config.model_name}_fold{fold}" if config.train.cross_validation else config.model_name
    if slurm_job_id:
        run_name = f"{run_name}_job{slurm_job_id}"
    
    # Initialize W&B with comprehensive config
    wandb_run = wandb.init(
        project=wandb_project,
        entity=wandb_entity if wandb_entity else None,
        name=run_name,
        mode=wandb_mode,
        config={
            # Basic info
            "model_name": config.model_name,
            "dataset": "ADReSSo",
            "fold": fold if config.train.cross_validation else None,
            "slurm_job_id": slurm_job_id if slurm_job_id else None,
        }
    )
    
    # Log comprehensive model architecture and training config
    log_model_summary_to_wandb(model, config, wandb_run)
    
    # Watch model for gradients
    wandb.watch(model, log='all', log_freq=100)
    
    return model, optimizer, lossfn, lr_scheduler

def main(config):
    """Main function to train and save model, supporting cross-validation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_path = os.path.join('logs', config.path_name)
    os.makedirs(log_path, exist_ok=True)
    
    if config.train.cross_validation:
        log_file = os.path.join(log_path, 'cross_fold_summary.txt')
        with open(log_file, "w") as log:
            for fold in range(config.train.cross_validation_folds):
                train_dataloader, validation_dataloader = get_dataloaders(config, kfold_number=fold)
                
                model, optimizer, lossfn, lr_scheduler = set_up(config, train_dataloader, device, fold)
                model, best_value, rest_best_values = train(
                    model, train_dataloader, validation_dataloader, lossfn, optimizer, lr_scheduler,
                    config.train.num_epochs, config.path_name, config.train.early_stopping, 
                    config.train.early_stopping_patience, config.train.cross_validation, fold
                )
                
                log.write(f'Fold {fold}: Best Value = {best_value}\n')
                log.write(f'Best F1: {rest_best_values[0]}\nBest Recall: {rest_best_values[1]}\nBest Precision: {rest_best_values[2]}\n')
                
                torch.save(model.state_dict(), os.path.join(log_path, f'model_fold_{fold}.pth'))
                print(f'Model for fold {fold} saved')
                
                # Log fold summary to W&B (final summary metrics already logged in train())
                wandb.log({
                    f"fold_{fold}/final_accuracy": best_value,
                    f"fold_{fold}/final_f1": rest_best_values[0],
                    f"fold_{fold}/final_recall": rest_best_values[1],
                    f"fold_{fold}/final_precision": rest_best_values[2],
                })
                wandb.finish()
    else:
        train_dataloader, validation_dataloader = get_dataloaders(config)
        
        model, optimizer, lossfn, lr_scheduler = set_up(config, train_dataloader, device)
        model, best_value, rest_best_values = train(
            model, train_dataloader, validation_dataloader, lossfn, optimizer, lr_scheduler, 
            config.train.num_epochs, config.path_name, config.train.early_stopping, 
            config.train.early_stopping_patience
        )
        
        model_save_path = os.path.join(log_path, 'model.pt')
        torch.save(model.state_dict(), model_save_path)
        print('Model saved')
        wandb.finish()


if __name__ == '__main__':

    config_path = sys.argv[sys.argv.index('--config') + 1]
    config = get_config(config_path)
    """
    for model_name in ['qwen']:
            config.model_name = model_name
            config.model.model_name = config.model_name

            for fusion in ['crossgated']:
                config.model.fusion = fusion

                for pooling in ['mean', 'cls']:
                    config.model.pooling = pooling
    """
    save_config(config)    
    main(config)