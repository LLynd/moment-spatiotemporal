import torch
import os
from omegaconf import DictConfig
from hydra.utils import instantiate
import utils
from functools import partial
import logging
from utils.checkpointing import (
    get_checkpoint_path,
    save_checkpoint,
    load_checkpoint,
    calculate_loaded_run_percentage,
)
import torch.multiprocessing as mp
from models.moment_gcn import MOMENTGCNReconstructor
from dataset.linear import time_series_collate_fn
mp.set_start_method("spawn", force=True)

log = logging.getLogger(__name__)
torch.set_float32_matmul_precision("medium")

# Hydra Integration (Main script)
def get_fabric(config):
    """Instantiate Fabric object"""
    fabric = instantiate(config.fabric)
    fabric.seed_everything(config.exp.seed + fabric.global_rank)
    fabric.launch()
    return fabric

def get_components(config, fabric):
    """Instantiate model, loss, and optimizer"""
    # Instantiate model
    model = instantiate(config.models)
    model = fabric.setup_module(model)  # Returns a single wrapped model
    
    # Instantiate loss
    loss_fn = instantiate(config.loss)
    
    # Instantiate optimizer
    optimizer_class = instantiate(config.optimizer)
    optimizer = optimizer_class(model.parameters())
    optimizer = fabric.setup_optimizers(optimizer)  # Setup optimizer separately
    
    return model, loss_fn, optimizer  # Return three components

def get_dataloader(config, fabric):
    """Instantiate dataset and dataloader"""
    # Instantiate dataset on CPU
    dataset = instantiate(config.dataloader.dataset)
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.dataloader.batch_size,
        shuffle=False,
        #num_workers=config.dataloader.num_workers,
        pin_memory=True,  # Important for GPU performance
        collate_fn=time_series_collate_fn  # Use your custom collate function
    )
    
    # Setup with Fabric
    return fabric.setup_dataloaders(dataloader)

def run(config: DictConfig):
    log.info("Launching Fabric")
    fabric = get_fabric(config)
    utils.hydra.preprocess_config(config)
    utils.wandb.setup_wandb(config)

    with fabric.init_tensor():
        log.info("Initializing components")
        model, loss_fn, optimizer = get_components(config, fabric)
        log.info("Initializing dataloader")
        dataloader = get_dataloader(config, fabric)

        # Checkpoint handling
        checkpoint_path = get_checkpoint_path(config)
        start_epoch = 0
        if checkpoint_path and config.exp.resume_from_checkpoint:
            checkpoint = load_checkpoint(checkpoint_path, fabric)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            log.info(f"Resuming from epoch {start_epoch}")

        log.info("Beginning training loop")
        for epoch in range(start_epoch, config.exp.epochs):
            fabric.call("on_epoch_start")
            model.train()
            
            for batch_idx, batch in enumerate(dataloader):
                optimizer.zero_grad()
                
                print(batch)  # Debugging: print batch shape
                # Forward pass
                reconstructed = model(batch)
                
                # Loss calculation
                loss = loss_fn(reconstructed, batch)
                
                # Backward pass
                fabric.backward(loss)
                optimizer.step()
                
                # Logging
                if batch_idx % config.logging.interval == 0:
                    fabric.log_dict({
                        "train/loss": loss.item(),
                        "epoch": epoch,
                        "batch": batch_idx
                    })
            
            # Checkpointing
            if epoch % config.exp.checkpoint_freq == 0:
                save_checkpoint(
                    path=checkpoint_path,
                    state={
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'config': config
                    },
                    fabric=fabric
                )
            
            fabric.call("on_epoch_end")
        
        log.info("Training completed")

















'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from moment import MOMENT
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
import matplotlib.pyplot as plt
from tqdm import tqdm
    
    
import torch
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
from hydra.utils import instantiate
from typing import Callable

import utils

import logging

log = logging.getLogger(__name__)

torch.set_float32_matmul_precision("medium")


def get_fabric(config):
    """Instantiate Fabric object, set seed and launch."""
    fabric = instantiate(config.fabric)
    fabric.seed_everything(config.exp.seed)
    fabric.launch()
    return fabric


def get_components(config, fabric):
    loss_fn: ... = instantiate(config.loss)
    metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = fabric.setup(
        instantiate(config.metric)
    )
    models: ... = fabric.setup(instantiate(config.models))

    return loss_fn, metric, models


def get_dataloader(config, fabric):
    """Instantiate the dataloader and setup with Fabric."""
    return fabric.setup_dataloaders(instantiate(config.dataloader))


def run(config: DictConfig):
    utils.hydra.preprocess_config(config)
    utils.wandb.setup_wandb(config)

    log.info("Launching Fabric")
    fabric = get_fabric(config)

    with fabric.init_tensor():
        log.info("Initializing components")
        fid = get_components(config, fabric)

        log.info("Initializing dataloader")
        dataloader = get_dataloader(config, fabric)

        log.info("Beginning optimization loop")

        # storage for features
        features = []

        # path for saving
        path_output = Path("data/assets/metric/fid")

        # loop over the dataloader
        for batch_idx, batch_input in enumerate(dataloader):
            # append features for this batch
            features.append(fid.precompute(batch_input.image))

        # concatenate from all batches
        features = torch.cat(features).numpy(force=True)

        # save under assets
        np.save(path_output / config.exp.dataset_name / f"precomputed.npy", features)



def prepare_patches(trajectories, patch_size=10, stride=5):
    """Prepare patches in MOMENT format and graph format"""
    # Standardize each trajectory
    scaler = StandardScaler()
    scaled_trajs = []
    for traj in trajectories:
        scaled_trajs.append(scaler.fit_transform(traj))
    
    # Create MOMENT-style patches [num_patches, 2, patch_size]
    moment_patches = []
    graph_data = []
    
    for traj in scaled_trajs:
        T = traj.shape[0]
        for i in range(0, T - patch_size + 1, stride):
            patch = traj[i:i+patch_size].T  # [2, patch_size]
            moment_patches.append(patch)
            
            # Create graph data for this patch
            x = torch.tensor(patch.T, dtype=torch.float32)  # [patch_size, 2]
            
            # Fully connected spatial graph
            edge_index = []
            for j in range(patch_size):
                for k in range(patch_size):
                    if j != k:
                        edge_index.append([j, k])
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            
            graph_data.append(Data(x=x, edge_index=edge_index))
    
    moment_patches = np.array(moment_patches)
    return moment_patches, graph_data

def run(config: DictConfig):
    utils.hydra.preprocess_config(config)
    utils.wandb.setup_wandb(config)

    log.info("Launching Fabric")
    fabric = get_fabric(config)

    with fabric.init_tensor():
        log.info("Initializing components")
        fid = get_components(config, fabric)

        log.info("Initializing dataloader")
        dataloader = get_dataloader(config, fabric)
    \
        # Hyperparameters
    PATCH_SIZE = 10
    STRIDE = 5
    BATCH_SIZE = 64
    EPOCHS = 20
    LR = 0.001
    SYSTEM_TYPE = 'stable_spiral'
    NUM_TRAJECTORIES = 10000
    
    # Generate dataset
    print(f"Generating {NUM_TRAJECTORIES} {SYSTEM_TYPE} trajectories...")
    trajectories = generate_trajectory_dataset(NUM_TRAJECTORIES, SYSTEM_TYPE)
    
    # Prepare patches
    print("Preparing patches...")
    moment_patches, graph_data = prepare_patches(trajectories, PATCH_SIZE, STRIDE)
    
    # Convert to tensors
    moment_patches = torch.tensor(moment_patches, dtype=torch.float32)
    
    # Split data
    indices = np.arange(len(moment_patches))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    train_moment = moment_patches[train_idx]
    test_moment = moment_patches[test_idx]
    train_graph = [graph_data[i] for i in train_idx]
    test_graph = [graph_data[i] for i in test_idx]
    
    # Create loaders
    train_loader = torch.utils.data.DataLoader(
        list(zip(train_moment, train_graph)), batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        list(zip(test_moment, test_graph)), batch_size=BATCH_SIZE, shuffle=False
    )
    
    # Initialize models
    print("Loading MOMENT model...")
    moment_model = MOMENT.load_pretrained()
    moment_model.eval()
    
    gnn_model = SpatialGNN(input_dim=2, hidden_dim=64, latent_dim=32)
    reconstructor = PatchReconstructor(moment_model, gnn_model, patch_size=PATCH_SIZE)
    
    # Freeze MOMENT
    for param in reconstructor.moment.parameters():
        param.requires_grad = False
    
    # Setup training
    optimizer = torch.optim.Adam(reconstructor.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    # Training loop
    print("Starting training...")
    train_losses, test_losses = [], []
    
    for epoch in range(EPOCHS):
        reconstructor.train()
        epoch_train_loss = 0
        
        for moment_batch, graph_batch in tqdm(train_loader):
            # Create batch for graph data
            graph_batch = Batch.from_data_list(graph_batch)
            
            optimizer.zero_grad()
            
            # Forward pass
            recon_x, recon_y = reconstructor(moment_batch, graph_batch)
            
            # Compute loss
            loss_x = criterion(recon_x, graph_batch.x[:, 0])
            loss_y = criterion(recon_y, graph_batch.x[:, 1])
            loss = loss_x + loss_y
            
            # Backprop
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        # Validation
        reconstructor.eval()
        epoch_test_loss = 0
        with torch.no_grad():
            for moment_batch, graph_batch in test_loader:
                graph_batch = Batch.from_data_list(graph_batch)
                recon_x, recon_y = reconstructor(moment_batch, graph_batch)
                
                loss_x = criterion(recon_x, graph_batch.x[:, 0])
                loss_y = criterion(recon_y, graph_batch.x[:, 1])
                loss = loss_x + loss_y
                
                epoch_test_loss += loss.item()
        
        # Record losses
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_test_loss = epoch_test_loss / len(test_loader)
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        
        print(f'Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f}')
    
    # Save model
    torch.save({
        'gnn_state_dict': gnn_model.state_dict(),
        'reconstructor_state_dict': reconstructor.state_dict(),
    }, 'patch_reconstructor.pth')
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.savefig('training_history.png')
    plt.show()

    # After training, visualize results
    checkpoint = torch.load('patch_reconstructor.pth')
    
    # Initialize models
    moment_model = MOMENT.load_pretrained()
    gnn_model = SpatialGNN(input_dim=2, hidden_dim=64, latent_dim=32)
    reconstructor = PatchReconstructor(moment_model, gnn_model, patch_size=10)
    
    # Load trained weights
    gnn_model.load_state_dict(checkpoint['gnn_state_dict'])
    reconstructor.load_state_dict(checkpoint['reconstructor_state_dict'])
    
    # Create test set
    trajectories = generate_trajectory_dataset(100, 'stable_spiral')
    _, test_graph = prepare_patches(trajectories[:1], 10, 5)
    test_moment = torch.tensor([d.x.numpy().T for d in test_graph], dtype=torch.float32)
    test_loader = torch.utils.data.DataLoader(
        list(zip(test_moment, test_graph)), batch_size=1, shuffle=False
    )
    
    visualize_reconstruction(reconstructor, test_loader)
'''