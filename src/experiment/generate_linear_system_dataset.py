import numpy as np
import torch
from pathlib import Path
from omegaconf import DictConfig
from hydra.utils import instantiate

import utils
import logging

from dataset.linear import LinearSystemDataset

log = logging.getLogger(__name__)


def run(config: DictConfig):
    """Generate and save synthetic time series dataset"""
    utils.hydra.preprocess_config(config)
    utils.wandb.setup_wandb(config)
    
    # Create output directory
    output_dir = Path(config.exp.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log.info(f"Generating dataset with parameters: {config.dataset}")
    
    # Generate full dataset
    full_data = LinearSystemDataset._generate_trajectory_dataset(
        num_trajectories=config.dataset.num_trajectories,
        system_type=config.dataset.system_type,
        num_timepoints=config.dataset.num_timepoints,
        t_span=tuple(config.dataset.t_span),
        x0_range=tuple(config.dataset.x0_range),
        seed=config.exp.seed
    )
    
    log.info(f"Generated dataset with shape: {full_data.shape}")
    
    # Split into train/test
    num_train = int(len(full_data) * config.dataset.train_ratio)
    rng = np.random.default_rng(config.exp.seed)
    indices = np.arange(len(full_data))
    rng.shuffle(indices)
    
    train_data = full_data[indices[:num_train]]
    test_data = full_data[indices[num_train:]]
    
    log.info(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    
    # Save datasets
    np.save(output_dir / "train.npy", train_data)
    np.save(output_dir / "test.npy", test_data)
    
    log.info(f"Saved datasets to {output_dir}")
    
    # Save config for reproducibility
    utils.hydra.save_config(config, output_dir / "config.yaml")
    
    # Log to WandB if enabled
    if config.wandb.enable:
        import wandb
        artifact = wandb.Artifact(config.exp.dataset_name, type="dataset")
        artifact.add_dir(str(output_dir))
        wandb.log_artifact(artifact)
        wandb.finish()

    log.info("Dataset generation complete!")
