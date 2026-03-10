"""Utility functions for model statistics and W&B logging."""

import torch
import torch.nn as nn
from typing import Dict, Any


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_by_component(model: nn.Module) -> Dict[str, int]:
    """Count parameters for each component of the model.
    
    Args:
        model: The model to analyze
        
    Returns:
        Dictionary with parameter counts for each named component
    """
    param_counts = {}
    
    for name, module in model.named_children():
        param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
        param_counts[name] = param_count
    
    param_counts['total'] = count_parameters(model)
    
    return param_counts


def get_model_architecture_summary(model: nn.Module, config) -> Dict[str, Any]:
    """Generate a comprehensive summary of model architecture for W&B logging.
    
    Args:
        model: The model to summarize
        config: Model configuration object
        
    Returns:
        Dictionary with model architecture details
    """
    param_counts = count_parameters_by_component(model)
    
    summary = {
        # Architecture type
        'model_type': model.__class__.__name__,
        'model_name': config.model_name if hasattr(config, 'model_name') else 'unknown',
        
        # Parameter counts
        'total_parameters': param_counts['total'],
        'total_parameters_millions': round(param_counts['total'] / 1e6, 2),
        
        # Component-wise parameter counts
        'parameters_by_component': param_counts,
        
        # Model hyperparameters
        'hidden_size': config.hidden_size,
        'intermediate_size': config.intermediate_size,
        'num_layers': config.n_layers,
        'num_heads': config.n_heads,
        'dropout': config.dropout,
        'pooling_strategy': config.pooling,
        'num_classes': config.num_classes,
        'hidden_mlp_size': config.hidden_mlp_size,
        
        # Fusion configuration (if applicable)
        'fusion_type': config.fusion if hasattr(config, 'fusion') else 'none',
        'textual_model': config.textual_model if hasattr(config, 'textual_model') else 'none',
        'audio_model': config.audio_model if hasattr(config, 'audio_model') else 'none',
        'pauses': config.pauses if hasattr(config, 'pauses') else False,
        'multimodality': config.multimodality if hasattr(config, 'multimodality') else False,
    }
    
    return summary


def get_training_config_summary(config) -> Dict[str, Any]:
    """Extract training configuration for W&B logging.
    
    Args:
        config: Training configuration object
        
    Returns:
        Dictionary with training hyperparameters
    """
    train_config = config.train
    
    summary = {
        # Training hyperparameters
        'batch_size': train_config.batch_size,
        'num_epochs': train_config.num_epochs,
        'learning_rate': train_config.learning_rate,
        'weight_decay': train_config.weight_decay,
        
        # Early stopping
        'early_stopping': train_config.early_stopping,
        'early_stopping_patience': train_config.early_stopping_patience,
        
        # Cross-validation
        'cross_validation': train_config.cross_validation,
        'cross_validation_folds': train_config.cross_validation_folds,
        
        # Optimizer details
        'optimizer': 'AdamW',
        'lr_scheduler': 'cosine',
        'lr_warmup_steps': 20,
        'gradient_clip_norm': 1.0,
        
        # Loss function
        'loss_function': 'BCEWithLogitsLoss',
    }
    
    return summary


def log_model_summary_to_wandb(model: nn.Module, config, wandb_run) -> None:
    """Log comprehensive model summary to W&B.
    
    Args:
        model: The model to log
        config: Configuration object
        wandb_run: Active W&B run object
    """
    import wandb
    
    # Get architecture summary
    arch_summary = get_model_architecture_summary(model, config.model)
    
    # Get training config summary
    train_summary = get_training_config_summary(config)
    
    # Combine all summaries
    full_summary = {
        **arch_summary,
        **train_summary,
        'dataset': 'ADReSSo',
    }
    
    # Update W&B config
    wandb_run.config.update(full_summary)
    
    # Log parameter counts as a table
    param_table = wandb.Table(
        columns=['Component', 'Parameters', 'Parameters (M)'],
        data=[
            [name, count, round(count / 1e6, 2)]
            for name, count in arch_summary['parameters_by_component'].items()
        ]
    )
    wandb_run.log({'model_parameters': param_table})
    
    print(f"\n{'='*60}")
    print(f"Model Architecture Summary")
    print(f"{'='*60}")
    print(f"Model Type: {arch_summary['model_type']}")
    print(f"Total Parameters: {arch_summary['total_parameters']:,} ({arch_summary['total_parameters_millions']}M)")
    print(f"\nParameter Breakdown:")
    for name, count in arch_summary['parameters_by_component'].items():
        if name != 'total':
            print(f"  {name:25s}: {count:12,} ({count/1e6:6.2f}M)")
    print(f"{'='*60}\n")
