import os
import logging
import torch
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

def get_checkpoint_path(config: DictConfig) -> str:
    """
    Construct the checkpoint path based on experiment configuration.
    
    Args:
        config: Hydra configuration object
        
    Returns:
        Full path to the checkpoint file
    """
    output_dir = config.exp.output_dir
    checkpoint_name = config.exp.get("checkpoint_name", "checkpoint.pt")
    return os.path.join(output_dir, checkpoint_name)

def save_checkpoint(
    path: str, 
    state: dict, 
    fabric: any, 
    is_final: bool = False
) -> None:
    """
    Save a checkpoint using Fabric's save functionality.
    
    Args:
        path: Path to save the checkpoint
        state: Dictionary containing model state
        fabric: Fabric instance for distributed saving
        is_final: Flag indicating if this is the final checkpoint
    """
    if not path:
        log.warning("Checkpoint path is empty, skipping save")
        return
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    try:
        fabric.save(path, state)
        log.info(f"Saved {'final ' if is_final else ''}checkpoint to {path}")
    except Exception as e:
        log.error(f"Failed to save checkpoint to {path}: {str(e)}")

def load_checkpoint(path: str, fabric: any) -> dict:
    """
    Load a checkpoint using Fabric's load functionality.
    
    Args:
        path: Path to the checkpoint file
        fabric: Fabric instance for distributed loading
        
    Returns:
        Loaded state dictionary
    """
    if not path or not os.path.exists(path):
        log.warning(f"Checkpoint path {path} does not exist")
        return None
    
    try:
        state = fabric.load(path)
        log.info(f"Loaded checkpoint from {path}")
        return state
    except Exception as e:
        log.error(f"Failed to load checkpoint from {path}: {str(e)}")
        return None

def calculate_loaded_run_percentage(
    start_epoch: int, 
    total_epochs: int, 
    batches_completed: int, 
    total_batches: int
) -> float:
    """
    Calculate the percentage of the training run that has been completed.
    
    Args:
        start_epoch: Epoch to resume from
        total_epochs: Total epochs in training
        batches_completed: Number of batches completed in current epoch
        total_batches: Total batches per epoch
        
    Returns:
        Percentage of training completed (0.0-1.0)
    """
    # Calculate progress in current epoch
    epoch_progress = batches_completed / total_batches if total_batches > 0 else 0
    
    # Calculate overall progress
    completed_epochs = start_epoch
    overall_progress = (completed_epochs + epoch_progress) / total_epochs
    
    return max(0.0, min(1.0, overall_progress))